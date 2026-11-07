"""INT8 attention vs the fp16 HIP kernel, on shapes that model attention takes.

Times the kernel alone (pre-quantized inputs) and the whole fp16 -> int8 ->
kernel -> fp16 path, since that is what a hook would pay. The fp16 baseline is
hip_attn.py, the pack's fused dot2 kernel.

Usage: python tools/bench_int8_attn.py [--shapes 1024x1024x16x128,...]
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import hip_attn_i8  # noqa: E402
import hip_attention  # noqa: E402

SHAPES = ("1024x1024x16x128", "256x1024x8x128", "512x512x20x64", "4096x256x4x128")


def parse(text):
    sq, sk, h, d = (int(v) for v in text.split("x"))
    return sq, sk, h, d


def timeit(fn, runs=3):
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(runs):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default=",".join(SHAPES))
    args = parser.parse_args()
    fp16_ok = hip_attention.available()
    if not hip_attn_i8.available():
        print("INT8 attention kernel is not built")
        return 1
    print(f"{'shape':22s} {'fp16 hip':>9s} {'int8 kern':>9s} {'int8 total':>10s} "
          f"{'kern x':>7s} {'total x':>7s}")
    for spec in args.shapes.split(","):
        sq, sk, h, d = parse(spec)
        q = torch.randn(1, sq, h, d, device="cuda")
        k = torch.randn(1, sk, h, d, device="cuda")
        v = torch.randn(1, sk, h, d, device="cuda")
        q8, qs = hip_attn_i8.quantize_qk(q)
        k8, ks = hip_attn_i8.quantize_qk(k)
        vq, vs = hip_attn_i8.quantize_v(v)
        t_i8 = timeit(lambda: hip_attn_i8.attention_fwd(q8, qs, k8, ks, vq, vs))
        t_all = timeit(lambda: hip_attn_i8.attention(q, k, v))
        t_fp16 = timeit(lambda: hip_attention.attention_fwd(q.half(), k.half(), v.half())) if fp16_ok else float("nan")
        print(f"{spec:22s} {t_fp16:9.2f} {t_i8:9.2f} {t_all:10.2f} "
              f"{t_fp16 / t_i8:7.2f} {t_fp16 / t_all:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
