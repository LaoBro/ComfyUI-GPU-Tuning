"""Numerical check for the INT8 attention kernel.

Two references, because two different errors are worth separating:

* dequantized: softmax over exactly the int8 Q/K/V the kernel sees. Any gap
  here is a kernel bug (P quantization shows up too, as the only difference
  left: the kernel quantizes P to 8 bits, the reference does not).
* original: softmax over the fp32 inputs. The gap is what the quantization
  costs, which is the number that decides whether the mode is usable.

Usage: python tools/verify_int8_attn.py [--shapes small,mid,big]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hip_attn_i8  # noqa: E402

SHAPES = {
    "small": (1, 100, 100, 3, 64),
    "mid": (1, 256, 320, 4, 128),
    "tail": (1, 300, 130, 2, 128),
    "gqa": (1, 192, 256, 4, 128),     # heads -> run with hkv=2
    "big": (1, 1024, 1024, 8, 128),
}


def ref_attention(q, k, v, scale, hkv=None):
    """fp32 reference on [B, S, H, D]; q/k/v already dequantized."""
    b, sq, h, d = q.shape
    if hkv is None:
        hkv = k.shape[2]
    k = k.repeat_interleave(h // hkv, dim=2)
    v = v.repeat_interleave(h // hkv, dim=2)
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", p, v.float())


def rel(a, b):
    a = a.float()
    b = b.float()
    return (a - b).norm() / b.norm()


def run(name, shape, rotate, hkv=None, outlier=0.0):
    b, sq, sk, h, d = shape
    torch.manual_seed(7)
    q = torch.randn(b, sq, h, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, sk, h if hkv is None else hkv, d, device="cuda", dtype=torch.float32)
    v = torch.randn(b, sk, h if hkv is None else hkv, d, device="cuda", dtype=torch.float32)
    if outlier:
        # one channel 40x the rest: the case the rotation switch exists for
        q[..., 0] *= outlier
        k[..., 0] *= outlier
    scale = d ** -0.5

    out = hip_attn_i8.attention(q, k, v, rotate=rotate)
    assert out.shape == q.shape and out.dtype == torch.float16

    q8, qs = hip_attn_i8.quantize_qk(q, rotate)
    k8, ks = hip_attn_i8.quantize_qk(k, rotate)
    vq, vs = hip_attn_i8.quantize_v(v)
    dq = q8.float() * qs.unsqueeze(-1)
    dk = k8.float() * ks.unsqueeze(-1)
    # invert the rotation for the reference only; the kernel already ran on
    # the rotated forms and that is the point of the transform
    if rotate:
        g = min(rotate, d)
        hv = hip_attn_i8._hadamard(g, q.device, torch.float32)
        dq = hip_attn_i8._rotate(dq, g, hv)
        dk = hip_attn_i8._rotate(dk, g, hv)
    n = vq.shape[2] * 4
    dv = (vq.view(torch.int8).reshape(b, v.shape[2], n // 4, d, 4)
          .permute(0, 1, 2, 4, 3).reshape(b, v.shape[2], n, d)
          .permute(0, 2, 1, 3)[:, :sk].float() * vs.unsqueeze(1))

    e_deq = rel(out, ref_attention(dq, dk, dv, scale, hkv))
    e_orig = rel(out, ref_attention(q, k, v, scale, hkv))
    print(f"{name:6s} rot={rotate:3d} hkv={h if hkv is None else hkv}  "
          f"dequant relL2 {e_deq:.2e}   original relL2 {e_orig:.2e}")
    return e_deq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="small,mid,tail,gqa")
    args = parser.parse_args()
    if not hip_attn_i8.available():
        print("INT8 attention kernel is not built or not on gfx906")
        return 1
    worst = 0.0
    for name in args.shapes.split(","):
        shape = SHAPES[name]
        for rotate in (0, 64, 128):
            hkv = shape[3] // 2 if name == "gqa" else None
            worst = max(worst, run(name, shape, rotate, hkv))
    print(f"worst dequantized relL2 {worst:.2e}")
    print("\noutlier channel (q/k channel 0 x40), mid shape:")
    for rotate in (0, 64, 128):
        run("outlr", SHAPES["mid"], rotate, outlier=40.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
