"""Qwen-Image 2.1 int8 prefix cache: correctness and timing of the int8 path.

The cache entry is built exactly the way PoseBranchCache builds it (stacked
[K; V] rows, TensorWiseINT8Layout convrot per-channel), then the same attention
call is run three ways:

  shipped   dequantize the cached K/V, cat with the current step, fp16 hip kernel
  int8      cache_attention: cached rows stay int8, Q/K/V of this step are
            rotated+quantized, output rotated back
  reference fp32 softmax over the dequantized cache and the fp16 current step

relL2 against `reference` isolates what the int8 path costs beyond the storage
precision the cache already paid for.

Usage: python tools/verify_qwen21_int8_cache.py [--bench]
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/home/liu/ComfyUI-0.39.0")
import hip_attn_i8  # noqa: E402
import hip_attention  # noqa: E402
from comfy.quant_ops import TensorWiseINT8Layout  # noqa: E402


def convrot_group(d):
    g = 256
    while g > 4 and d % g:
        g //= 4
    return g


def make_cache(k_pre, v_pre):
    """Same call PoseBranchCache.put makes for dtype="int8"."""
    b, s, h, d = k_pre.shape
    stacked = torch.stack([k_pre, v_pre], dim=1).reshape(-1, d)
    qdata, params = TensorWiseINT8Layout.quantize(
        stacked, is_weight=True, per_channel=True, convrot=True,
        convrot_groupsize=convrot_group(d))
    kv8 = qdata.reshape(b, 2, s, h, d)
    scale = params.scale.reshape(b, 2, s, h).float()
    return kv8, scale, params, qdata


def reference(q, k, v):
    """fp32 attention on [B, S, H, D]; k/v already include the prefix."""
    qh = q.float().transpose(1, 2)
    kh = k.float().transpose(1, 2)
    vh = v.float().transpose(1, 2)
    p = (qh @ kh.transpose(-1, -2) * q.shape[-1] ** -0.5).softmax(-1)
    return (p @ vh).transpose(1, 2)


def rel(a, b):
    a = a.float()
    b = b.float()
    return (a - b).norm() / b.norm()


def case(b, sp, st, h, d, bench=False):
    torch.manual_seed(11)
    k_pre = torch.randn(b, sp, h, d, device="cuda", dtype=torch.float16)
    v_pre = torch.randn(b, sp, h, d, device="cuda", dtype=torch.float16)
    q = torch.randn(b, st, h, d, device="cuda", dtype=torch.float16)
    k_t = torch.randn(b, st, h, d, device="cuda", dtype=torch.float16)
    v_t = torch.randn(b, st, h, d, device="cuda", dtype=torch.float16)

    kv8, scale, params, qdata = make_cache(k_pre, v_pre)
    deq = TensorWiseINT8Layout.dequantize(qdata, params).reshape(b, 2, sp, h, d).to(torch.float16)

    out_i8 = hip_attn_i8.cache_attention(q, k_t, v_t, kv8, scale)

    if not bench:
        k_all = torch.cat([deq[:, 0], k_t], dim=1)
        v_all = torch.cat([deq[:, 1], v_t], dim=1)
        e = rel(out_i8, reference(q, k_all, v_all))
        out_fp16 = hip_attention.attention_fwd(q, k_all.contiguous(), v_all.contiguous())
        e_fp16 = rel(out_fp16, reference(q, k_all, v_all))
        print(f"sp={sp} st={st} h={h} d={d}: int8 relL2 {e:.2e}   "
              f"fp16-on-dequant relL2 {e_fp16:.2e}")
        return e

    def timed(fn, n=3):
        fn()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(n):
            s = torch.cuda.Event(True)
            e = torch.cuda.Event(True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
        return best

    def shipped():
        dq = TensorWiseINT8Layout.dequantize(qdata, params).reshape(b, 2, sp, h, d).to(torch.float16)
        k_all = torch.cat([dq[:, 0], k_t], dim=1).contiguous()
        v_all = torch.cat([dq[:, 1], v_t], dim=1).contiguous()
        return hip_attention.attention_fwd(q, k_all, v_all)

    t_ship = timed(shipped)
    t_i8 = timed(lambda: hip_attn_i8.cache_attention(q, k_t, v_t, kv8, scale))
    print(f"sp={sp} st={st} h={h} d={d}: shipped {t_ship:.2f} ms, int8 {t_i8:.2f} ms, "
          f"{t_ship / t_i8:.2f}x")
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if not hip_attn_i8.rotation_ready():
        print("int8 attention / rotation kernels are not built")
        return 1
    hip_attention.available()
    worst = 0.0
    for shape in ((1, 130, 96, 2, 128), (1, 200, 128, 3, 64), (2, 66, 70, 2, 128)):
        worst = max(worst, case(*shape))
    print(f"worst relL2 {worst:.2e}")
    if args.bench:
        for shape in ((1, 4058, 4032, 32, 128),):
            case(*shape, bench=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
