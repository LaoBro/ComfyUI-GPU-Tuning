"""INT8 attention vs the fp16 HIP kernel, on shapes that model attention takes.

Times the kernel alone (pre-quantized inputs) and the whole fp16 -> int8 ->
kernel -> fp16 path, since that is what a hook would pay. The fp16 baseline is
hip_attn.py, the pack's fused dot2 kernel.

Every row is also reported as a ratio to a clock reference, one 4096^3 fp16
matmul sampled right before and right after that row (the same reference the
profiler prints). Without it two runs minutes apart read as 3% apart on this
card purely because the clocks moved, which is larger than most of the
differences worth measuring here.

Usage: python tools/bench_int8_attn.py [--shapes 1024x1024x16x128,...]
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kernels.hip import attention_i8 as hip_attn_i8  # noqa: E402
from kernels.hip import attention as hip_attention  # noqa: E402

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


def clock_ref(runs=4):
    """One 4096^3 fp16 matmul, second fastest of four: the profiler's reference."""
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    samples = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        a @ b
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    del a, b
    return sorted(samples)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default=",".join(SHAPES))
    args = parser.parse_args()
    fp16_ok = hip_attention.available()
    if not hip_attn_i8.available():
        print("INT8 attention kernel is not built")
        return 1
    print(f"{'shape':20s} {'fp16 hip':>9s} {'int8 kern':>10s} {'int8 total':>11s} "
          f"{'kern x':>7s} {'total x':>8s} {'ref ms':>7s} {'kern/ref':>9s} {'total/ref':>10s}")
    first_ref = last_ref = None
    for spec in args.shapes.split(","):
        sq, sk, h, d = parse(spec)
        q = torch.randn(1, sq, h, d, device="cuda")
        k = torch.randn(1, sk, h, d, device="cuda")
        v = torch.randn(1, sk, h, d, device="cuda")
        q8, qs = hip_attn_i8.quantize_qk(q)
        k8, ks = hip_attn_i8.quantize_qk(k)
        vq, vs = hip_attn_i8.quantize_v(v)
        before = clock_ref()
        t_i8 = timeit(lambda: hip_attn_i8.attention_fwd(q8, qs, k8, ks, vq, vs))
        t_all = timeit(lambda: hip_attn_i8.attention(q, k, v))
        t_fp16 = timeit(lambda: hip_attention.attention_fwd(q.half(), k.half(), v.half())) if fp16_ok else float("nan")
        after = clock_ref()
        ref = 0.5 * (before + after)
        first_ref = before if first_ref is None else first_ref
        last_ref = after
        print(f"{spec:20s} {t_fp16:9.2f} {t_i8:10.2f} {t_all:11.2f} "
              f"{t_fp16 / t_i8:7.2f} {t_fp16 / t_all:8.2f} {ref:7.2f} "
              f"{t_i8 / ref:9.4f} {t_all / ref:10.4f}")
    if first_ref is not None:
        print(f"\nclock ref {first_ref:.2f} -> {last_ref:.2f} ms (4096^3 fp16, before -> after; "
              "kern/ref and total/ref are the numbers to compare across runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
