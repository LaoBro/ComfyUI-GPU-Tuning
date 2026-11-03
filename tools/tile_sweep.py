"""Sweep int8 GEMM tiles on this GPU and print the winners.

    python custom_nodes/ComfyUI-gfx906-Tuning/tools/tile_sweep.py
    python ... --m 1,2,8,128,4032 --shapes 4096x4096,4096x12288
    python ... --candidate block_m=128,block_n=128,block_k=64,group_size_m=8,warps=8,stages=3

The script does not call apply(). It drives comfy-kitchen's int8_linear with
exactly one candidate in the list at a time, so autotune compiles and measures
only that tile. Results are per (shape, m), plus one suggested large tile
(m > 8) and one small tile (m <= 8).

Pass --emit to print the two COMFY_GFX906_*_TILE_CONFIG startup variables and a
ready-to-paste INT8_TILE_TABLE entry for another card.
"""

import argparse
import os
import statistics
import sys
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NODE_DIR))

import torch
import triton


def parse_spec(text):
    values = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        values[key.strip()] = int(value)
    spec = {k: values[k] for k in ("block_m", "block_n", "block_k")}
    spec["group_size_m"] = values.get("group_size_m", 8)
    warps = values.get("warps", values.get("num_warps", 4))
    stages = values.get("stages", values.get("num_stages", 2))
    return spec, warps, stages


def spec_text(spec):
    params, warps, stages = spec
    body = ",".join(f"{k}={v}" for k, v in params.items())
    return f"{body},warps={warps},stages={stages}"


def fmt(spec):
    params, warps, stages = spec
    body = "x".join(str(params[k]) for k in ("block_m", "block_n", "block_k"))
    return f"{body} w{warps} s{stages}"


def spec_label(spec):
    """Bare-key-safe name for a tile, for the manifest's ms table: 128x256x32g8w4s2."""
    params, warps, stages = spec
    body = "x".join(str(params[k]) for k in ("block_m", "block_n", "block_k"))
    return f"{body}g{params['group_size_m']}w{warps}s{stages}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", default="1,2,8,128,4032", help="comma separated batch widths")
    parser.add_argument("--shapes", default="4096x4096,24576x4096,4096x12288",
                        help="comma separated NxK shapes")
    parser.add_argument("--candidate", action="append", default=[],
                        help="tile spec; repeat to replace the default candidate set")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--emit", action="store_true", help="print env variables and a table entry")
    return parser.parse_args()


def default_candidates(kernels):
    from gfx906_tuning import INT8_SMALL_STAGES, INT8_SMALL_WARPS, INT8_STAGES, INT8_TILE, INT8_TILE_SMALL, INT8_WARPS

    items = []
    for config in kernels[0].configs:
        items.append((dict(config.kwargs), config.num_warps, config.num_stages))
    items.append((dict(INT8_TILE), INT8_WARPS, INT8_STAGES))
    items.append((dict(INT8_TILE_SMALL), INT8_SMALL_WARPS, INT8_SMALL_STAGES))
    unique = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def sweep(kernels, shapes, batch_widths, candidates, iters=3, warmup=2, top=3):
    """Time every candidate tile on every (shape, batch width).

    Each candidate is installed alone in the config list, so comfy-kitchen's
    autotune compiles and measures only that tile. Returns
    {(n, k, m, spec_text): ms}; the original list is restored on the way out,
    including after a compile failure.
    """
    from comfy_kitchen.backends.triton import quantization

    original = [list(kernel.configs) for kernel in kernels]
    results = {}
    try:
        for n, k in shapes:
            weight = torch.randint(-127, 127, (n, k), device="cuda", dtype=torch.int8)
            scale = torch.rand(n, 1, device="cuda", dtype=torch.float32) * 0.01
            for m in batch_widths:
                x = torch.randn(m, k, device="cuda", dtype=torch.float32)
                rows = []
                for spec in candidates:
                    for kernel in kernels:
                        kernel.configs = [triton.Config(dict(spec[0]), num_warps=spec[1], num_stages=spec[2])]
                        kernel.cache.clear()
                    run = lambda: quantization.int8_linear(x, weight, scale, None, out_dtype=torch.float32, convrot=True)
                    try:
                        elapsed = bench(run, iters, warmup)
                        results[(n, k, m, spec_text(spec))] = elapsed
                        rows.append((elapsed, spec))
                    except Exception as exc:
                        print(f"  {n}x{k} m={m} {fmt(spec)} failed: {type(exc).__name__}: {exc}")
                if not rows:
                    continue
                rows.sort(key=lambda item: item[0])
                best = ", ".join(f"{fmt(spec)} {elapsed:.2f} ms" for elapsed, spec in rows[:top])
                print(f"{n}x{k} m={m}: {best}")
    finally:
        for kernel, configs in zip(kernels, original):
            kernel.configs = configs
            kernel.cache.clear()
    return results


def winner(results, want_small):
    """(spec_text, median ms) for the fastest tile in one batch-width bucket."""
    scores = {}
    for (n, k, m, key), elapsed in results.items():
        if (m <= 8) == want_small:
            scores.setdefault(key, []).append(elapsed)
    if not scores:
        return None, None
    key = min(scores, key=lambda name: statistics.median(scores[name]))
    return key, statistics.median(scores[key])


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("no CUDA/ROCm device")
        return 1

    import gfx906_tuning as tuning

    tuning.enable_triton_backend()
    kernels = list(tuning._int8_kernels())
    if not kernels:
        print("comfy-kitchen int8 triton kernels not found")
        return 1

    candidates = [parse_spec(c) for c in args.candidate] if args.candidate else default_candidates(kernels)
    ms = [int(x) for x in args.m.split(",")]
    shapes = []
    for item in args.shapes.split(","):
        n, _, k = item.partition("x")
        shapes.append((int(n), int(k)))
    for n, k in shapes:
        if k % 256:
            print(f"skip {n}x{k}: K must be a multiple of 256 for convrot")
            return 1

    arch = tuning.arch_name() or torch.cuda.get_device_name(0)
    print(f"arch {arch} | {len(candidates)} candidates | {len(shapes)} shapes | m={ms}")
    print("each candidate is compiled once per (m,n,k); the first run can take seconds")

    results = sweep(kernels, shapes, ms, candidates, args.iters, args.warmup, args.top)

    large_key, large_ms = winner(results, False)
    small_key, small_ms = winner(results, True)
    if large_key is None:
        print("no large-m results")
        return 1
    print(f"\nlarge tile (m>8): {large_key}  median {large_ms:.2f} ms")
    if small_key is not None:
        print(f"small tile (m<=8): {small_key}  median {small_ms:.2f} ms")
    else:
        print("small tile (m<=8): no results")

    if args.emit:
        print("\nstartup variables (node defaults stay 'default'):")
        print(f'COMFY_GFX906_TILE_CONFIG="{large_key}"')
        if small_key is not None:
            print(f'COMFY_GFX906_SMALL_TILE_CONFIG="{small_key}"')
        print("TILE_CONFIG makes this card 'measured' (single config, no autotune).")
        print("SMALL_TILE_CONFIG alone only changes the candidate / forced small tile.")
        print("\nTo store both for this card instead of exporting variables, run")
        print("tools/bench.py --all --write: it fills arch/<gcn>/manifest.toml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
