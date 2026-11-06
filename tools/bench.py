"""Measure a site's implementations on this GPU and record the result.

    # dry run: print the table, touch nothing
    .venv/bin/python custom_nodes/ComfyUI-gfx906-Tuning/tools/bench.py

    # one site, custom shapes
    ... --site w4a8.m1 --shapes 4096x4096@m1,4096x4096@m8

    # everything one card needs: GEMV backends, int8 GEMM tiles, attention tiles,
    # DeltaNet / VAE / ConvRot against their real fallbacks
    ... --all

    # write it back (local override, does not touch the shipped file)
    ... --all --write

    # target the file that ships with the node (for a PR)
    ... --write --out custom_nodes/ComfyUI-gfx906-Tuning/arch/gfx906/manifest.toml

--all is what fills a card's measurement gates: after one run the int8 linear
list holds a single tile, attention has a tile per head dim and a measured
kernel mode, the GEMV sites have a winner per shape, and every feature whose
fastest form differs per card (gemv, w4a8, deltanet, vae, convrot, vision) has an
on/off record against its real fallback.

The pipeline per shape is fixed, in this order:

1. build inputs and the torch reference from tools/refs.py;
2. run each implementation once and compare against the reference -- a kernel
   that is off by more than --max-rel is reported and *not* timed, so a broken
   fast path can never win a benchmark;
3. time what survived (the fastest of --repeats rounds, each the median of
   --iters after --warmup), keep every time in
   the record, and mark the fastest as the winner -- unless the runner-up is
   within --min-gain, in which case the probe order wins: this card moves ~10%
   between runs, so a 0.4% "win" is the clock, not the kernel.

Numbers are measured with convrot off (group size 0). The rotation pass is a
separate kernel shared by both implementations, so leaving it out makes the
comparison cleaner; the manifest is about which GEMV to call, not about the
quantiser.

This tool allocates the weights itself: expect ~1 GB of VRAM at the larger
shapes. Do not run it while ComfyUI is generating.
"""

import argparse
import datetime
import gc
import os
import statistics
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
NODE_DIR = TOOLS_DIR.parent
sys.path.insert(0, str(NODE_DIR))
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(NODE_DIR.parent.parent))          # ComfyUI root, for the qwen35 check

# Same pin as the pack's __init__.py, which these top-level imports bypass.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch                                                           # noqa: E402

import build_hip_attn                                                  # noqa: E402
import refs                                                            # noqa: E402
from kernels import manifest as manifest_mod                           # noqa: E402
from kernels import manifest_write                                     # noqa: E402

SITES = {
    "int8_gemv.m1": {
        "module": "int8_gemv",
        "impls": ("asm", "triton"),
        "default": "4096x4096@m1,4096x12288@m1,12288x4096@m1,152064x4096@m1",
        "max_m": 1,
        # baseline: the stock int8 linear at m=1, i.e. what runs with gemv off
        "feature": "gemv",
    },
    "w4a8.m1": {
        "module": "w4a8_gemv",
        "impls": ("asm", "triton"),
        "default": "4096x4096@m1,4096x4096@m8,12288x4096@m1,12288x4096@m8",
        "max_m": 8,
        # the site answers "which w4a8 backend"; the baseline answers "w4a8 at all"
        "feature": "w4a8",
    },
    # The two register-dequant GEMVs (te=default: fp16 activations, no activation
    # quant). Evidence only: no "feature" key, because feature.te_dequant is decided
    # against the per-call dequantise rather than against another GEMV. The site's
    # registry names are timed next to `dequant` on purpose -- the int8 GEMV is the
    # near-ceiling kernel at these shapes, so the ratio says how far the register
    # dequant is from the hardware limit instead of whether it is more accurate.
    "w8a16.m1": {
        "module": "int8_gemv",
        "impls": ("dequant",),
        "default": "4096x4096@m1,4096x4096@m8,12288x4096@m1,12288x4096@m8",
        "max_m": 8,
    },
    "w4a16.m1": {
        "module": "w4a8_gemv",
        "impls": ("dequant",),
        "default": "4096x4096@m1,4096x4096@m8,12288x4096@m1,12288x4096@m8",
        "max_m": 8,
    },
}
W4A8_GROUP = 128
# Qwen3.5 DeltaNet decode: 16 value heads of 128 in, 128 out, seq 1 and 4.
# 2048 is the 2B hidden size; the gate kernel walks it once per (token, head).
DELTANET_HIDDEN = 2048
DELTANET_SHAPES = "1x16x128x128x1,1x16x128x128x4"
DELTANET_SITE = "deltanet.decode"
# BxHVxDKxDVxS: one Qwen3.5 DeltaNet prefill at the 2B and 9B head counts.
DELTANET_PREFILL_SITE = "deltanet.prefill"
DELTANET_PREFILL_SHAPES = "1x32x128x128x1024,1x32x128x128x2223,1x32x128x128x4096"
# Depth-1 Conv3d (channels x spatial size) and ConvRot prep (rows x k): the two
# tier-3 features whose fastest form differs per card without being kernels.
VAE_SITE = "vae.decode"
VAE_SHAPES = "512x64,512x128"
CONVROT_SITE = "convrot.mode"
CONVROT_SHAPES = "1x4096,128x4096"
# Qwen3.5 image tower patch projection: patches x embed dim. 3024x1152 is what a
# 1160x679 image gives the 9B; the rest are a small and a large image.
VISION_SITE = "vision.patch_embed"
VISION_SHAPES = "768x1152,3024x1152,12288x1152"
# Qwen3.5: the class swap only touches Python-level copies and launches, so the
# numbers here are a proxy. The point of the site is the numeric check -- both
# sides run fp32, so 1e-3 is a real equivalence gate, not the 5% quantisation one.
QWEN35_SITE = "qwen35.block"
QWEN35_MAX_REL = 1e-3
QWEN35_LAYERS = 4
QWEN35_HIDDEN = 2048
QWEN35_SEQ = 16

TILE_SITE = "int8_gemm.tile"
# Shapes one bench run covers by default. 4096x4096 is the DiT qkv projection,
# 4096x12288 its ffn and 12288x4096 the text encoder's, so the table has an
# entry for both families without needing a model to be loaded.
TILE_SHAPES = "4096x4096,4096x12288,12288x4096"
TILE_BATCH = "1,8,128"
# Qwen-Image (d128, long self-attention and a short cross-attention) and SDXL
# (d64) cover the two head dims the kernels were compiled for.
ATTENTION_SHAPES = "1x24x4608x4608@d128,1x24x4608x256@d128,2x20x4096x4096@d64"

# Why a single timed round used to flip winners: after ~0.5 s of GPU idle the
# first kernels measure 2-3x slow (asm GEMV on 4096x4096@m1: 113.9 us back to
# back, 272.2 us after a 0.5 s gap, 318.7 us after 1 s), and pp_dpm_sclk reports
# the same 1556 MHz level the whole time, so the DPM file shows nothing. The JIT
# compile between shapes is exactly that kind of GPU-idle gap. warm_clocks() runs
# after the warmup so the samples do not start in the ramp; the fastest of
# REPEATS rounds is kept on top of that because a noisier card can still disturb
# one round.
REPEATS = 2

_WARM = None


def warm_clocks(ms=100):
    """Run the GPU up to working clocks, after the warmup compiled the kernel."""
    global _WARM
    if _WARM is None:
        _WARM = (torch.randn(4096, 4096, device="cuda", dtype=torch.float16),
                 torch.randn(4096, 4096, device="cuda", dtype=torch.float16))
    stop = time.perf_counter() + ms / 1000
    while time.perf_counter() < stop:
        for _ in range(8):
            _WARM[0] @ _WARM[1]
        torch.cuda.synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", action="append", default=[],
                        help=f"repeatable; default all of {sorted(SITES)}")
    parser.add_argument("--shapes", default="", help="NxK@mM list, overrides the site default")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help="timed rounds per implementation; the fastest round is kept "
                             "(one round is disturbed often enough to flip a winner)")
    parser.add_argument("--max-rel", type=float, default=0.05,
                        help="numeric gate against tools/refs.py (default 5%%, activation quantisation is ~1%%)")
    parser.add_argument("--min-gain", type=float, default=0.03,
                        help="below this margin the faster implementation is not declared a winner "
                             "(default 3%%: the same shape moves ~10%% between runs)")
    parser.add_argument("--tile", action="store_true", help="also sweep the int8 GEMM tiles")
    parser.add_argument("--attention", action="store_true", help="also sweep the attention tiles")
    parser.add_argument("--deltanet", action="store_true",
                        help="also time the DeltaNet decode against comfy-kitchen's fused op, "
                             "and the prefill chunk rule against core's UT loop")
    parser.add_argument("--vae", action="store_true",
                        help="also time depth-1 Conv3d: torch vs the conv2d squeeze vs kitchen")
    parser.add_argument("--convrot", action="store_true",
                        help="also time ConvRot activation prep: kitchen rotation vs ours")
    parser.add_argument("--vision", action="store_true",
                        help="also time the image tower patch projection: shipped conv3d vs the GEMM")
    parser.add_argument("--qwen35", action="store_true",
                        help="also compare the Qwen3.5 block against the shipped methods")
    parser.add_argument("--qwen35-layers", type=int, default=QWEN35_LAYERS,
                        help="layers in the synthetic Qwen3.5 (default 4: 3 DeltaNet + 1 attention)")
    parser.add_argument("--qwen35-hidden", type=int, default=QWEN35_HIDDEN,
                        help="hidden size of the synthetic model")
    parser.add_argument("--qwen35-seq", type=int, default=QWEN35_SEQ,
                        help="prefill length used for the comparison")
    parser.add_argument("--all", action="store_true",
                        help="the whole table in one run: --tile + --attention + --deltanet + "
                             "--vae + --convrot + --vision + --qwen35 + the GEMV sites")
    parser.add_argument("--deltanet-shapes", default=DELTANET_SHAPES,
                        help="BxHVxDKxDVxS list for the DeltaNet decode benchmark")
    parser.add_argument("--deltanet-prefill-shapes", default=DELTANET_PREFILL_SHAPES,
                        help="BxHVxDKxDVxS list for the DeltaNet prefill chunk-rule benchmark")
    parser.add_argument("--vae-shapes", default=VAE_SHAPES,
                        help="channels x size list for the depth-1 Conv3d benchmark")
    parser.add_argument("--vision-shapes", default=VISION_SHAPES,
                        help="patches x embed list for the image tower patch projection benchmark")
    parser.add_argument("--convrot-shapes", default=CONVROT_SHAPES,
                        help="rows x k list for the ConvRot prep benchmark (k mod 256 == 0)")
    parser.add_argument("--deltanet-hidden", type=int, default=DELTANET_HIDDEN,
                        help="hidden size the gate GEMV walks (default 2048, Qwen3.5-2B)")
    parser.add_argument("--deltanet-key-heads", type=int, default=0,
                        help="key heads behind the value heads; 0 means no GQA (Qwen3.5-2B)")
    parser.add_argument("--w4a8-uniform", action="store_true",
                        help="bench 4-bit weights without a codebook; the shipped checkpoints use one")
    parser.add_argument("--w4a8-group", type=int, default=W4A8_GROUP,
                        help="weight group size; below 16 exercises the tl.sum floor")
    parser.add_argument("--tile-shapes", default=TILE_SHAPES,
                        help="NxK list for the int8 tile sweep")
    parser.add_argument("--tile-m", default=TILE_BATCH,
                        help="batch widths for the tile sweep; <=8 is the small-tile bucket")
    parser.add_argument("--attention-shapes", default=ATTENTION_SHAPES,
                        help="BxHxSQxSK@dD list for the attention tile sweep")
    parser.add_argument("--attention-max-rel", type=float, default=0.3,
                        help="attention numeric gate (default 30%%: the fused fp16 paths sit at "
                             "~3e-4 relL2, so this only rejects a kernel computing something else)")
    parser.add_argument("--write", action="store_true", help="update the manifest (dry run otherwise)")
    parser.add_argument("--out", default="", help="manifest path to write; default is the local override")
    return parser.parse_args()


def parse_shapes(text):
    shapes = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        size, _, m = item.partition("@m")
        n, _, k = size.partition("x")
        shapes.append((int(n), int(k), int(m or 1)))
    return shapes


def parse_attention_shapes(text):
    """BxHxSQxSK@dD list -> [(batch, heads, seq_q, seq_k, head_dim)]."""
    shapes = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        size, _, head_dim = item.partition("@d")
        batch, heads, seq_q, seq_k = (int(value) for value in size.split("x"))
        shapes.append((batch, heads, seq_q, seq_k, int(head_dim)))
    return shapes


def bench_ms(fn, iters, warmup, repeat=None):
    """Median of one round, and the fastest round of `repeat` of them.

    Each round warms the clocks after its warmup, so a compile pause cannot put
    the timed samples into the idle ramp (see the REPEATS note above).
    """
    rounds = []
    for _ in range(REPEATS if repeat is None else repeat):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        warm_clocks()
        samples = []
        for _ in range(iters):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        rounds.append(statistics.median(samples))
    return min(rounds)


def int8_case(module, n, k, m, args, spec):
    """int8 weights, one row of activations, group size 0 (no convrot).

    The baseline is the stock int8 linear at the same shape: what runs with the
    GEMV off. That is the decision feature.gemv records.
    """
    gen = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randint(-127, 127, (n, k), device="cuda", dtype=torch.int8, generator=gen)
    scale = torch.rand(n, device="cuda", dtype=torch.float32, generator=gen) * 0.01 + 1e-4
    x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=gen)
    reference = refs.int8_reference(x, weight, scale)

    def run(fn):
        return fn(x, weight, scale, None, torch.float32, 0)

    from comfy_kitchen.backends.triton import quantization

    baseline = ("int8", lambda: quantization.int8_linear(x, weight, scale, None,
                                                         out_dtype=torch.float32, convrot=False))
    return run, reference, baseline


def w4a8_case(module, n, k, m, args, spec):
    """Packed 4-bit weights, no convrot, group size from --w4a8-group.

    Returns the site's callable, the fp32 answer, and the int8 path that runs
    when w4a8 is off: the weights decoded once, outside the timed call. Timing
    the decode inside would compare "unpack + matmul" against "matmul" and make
    the baseline look worse than the thing it replaces.

    A group below 16 is what exercises the tl.sum floor: the portable tl.dot
    kernel declines those shapes, so before it exists the whole site was None.
    """
    group = args.w4a8_group
    if k % group:
        raise ValueError(f"k={k} is not a multiple of --w4a8-group {group}")
    gen = torch.Generator(device="cuda").manual_seed(0)
    qdata = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8, generator=gen)
    # s_rel lives in 1..15.5 so (code - 8) * s_rel fills the int8 grid both
    # decode paths share; a smaller split silently rounds the weights to zero.
    s_rel = torch.rand(n, k // group, device="cuda", dtype=torch.float32, generator=gen) * 14.5 + 1.0
    s_channel = torch.rand(n, device="cuda", dtype=torch.float32, generator=gen) * 0.01 + 1e-3
    codebook = None
    if not args.w4a8_uniform:
        codebook = torch.sort(torch.randn(16, device="cuda", dtype=torch.float32,
                                          generator=gen)).values * 3.0
    x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=gen)
    reference = x @ refs.w4a8_weight(qdata, s_rel, s_channel, group, codebook).T

    levels, inv_level = (module.codebook_words(codebook) if codebook is not None
                         else (None, 1.0))

    def run(fn):
        return fn(x, qdata, s_rel, s_channel, None, torch.float32, group, 0,
                  levels=levels, inv_level=inv_level)

    weight8 = module.decode_int8(qdata, s_rel, group, codebook)
    if weight8 is None:
        baseline = None
    else:
        baseline = ("int8", lambda: int8_baseline(x, weight8, s_channel))
    return run, reference, baseline


def w4a16_case(module, n, k, m, args, spec):
    """Packed 4-bit weights read by fp16/fp32 activations, no activation quant.

    `linear_dequant` takes the per-group scale as fp8 because that is what the
    checkpoints carry and what it bit-decodes in registers; the activation-quantising
    kernels take the same *values* in fp32, which is what w4a8.m1 already does. The
    scale is rounded through fp8 before either side sees it, so all rows of the
    table decode the same numbers and one reference covers them.
    """
    group = args.w4a8_group
    if k % group:
        raise ValueError(f"k={k} is not a multiple of --w4a8-group {group}")
    gen = torch.Generator(device="cuda").manual_seed(0)
    qdata = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8, generator=gen)
    s_rel = (torch.rand(n, k // group, device="cuda", dtype=torch.float32, generator=gen)
             * 14.5 + 1.0).to(torch.float8_e4m3fn).to(torch.float32).contiguous()
    s_channel = torch.rand(n, device="cuda", dtype=torch.float32, generator=gen) * 0.01 + 1e-3
    codebook = None
    if not args.w4a8_uniform:
        codebook = torch.sort(torch.randn(16, device="cuda", dtype=torch.float32,
                                          generator=gen)).values * 3.0
    x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=gen)
    reference = x @ refs.w4a8_weight(qdata, s_rel, s_channel, group, codebook).T

    levels, inv_level = (module.codebook_words(codebook) if codebook is not None
                         else (None, 1.0))
    dequant = module.linear_dequant

    def run(fn):
        if fn is dequant:
            return fn(x, qdata, s_rel.to(torch.float8_e4m3fn), s_channel, None,
                      torch.float32, group, 0, codebook)
        return fn(x, qdata, s_rel, s_channel, None, torch.float32, group, 0,
                  levels=levels, inv_level=inv_level)

    weight8 = module.decode_int8(qdata, s_rel, group, codebook)
    baseline = None if weight8 is None else ("int8", lambda: int8_baseline(x, weight8, s_channel))
    return run, reference, baseline


def int8_baseline(x, weight, scale):
    """What runs with w4a8 off: the int8 GEMV at m=1, the int8 GEMM above it."""
    from comfy_kitchen.backends.triton import quantization

    if x.shape[0] <= 1:
        import int8_gemv

        out = int8_gemv.linear(x, weight, scale, None, torch.float32, 0)
        if out is not None:
            return out
    return quantization.int8_linear(x, weight, scale, None, out_dtype=torch.float32, convrot=False)


CASES = {"int8_gemv.m1": int8_case, "w4a8.m1": w4a8_case,
         "w8a16.m1": int8_case, "w4a16.m1": w4a16_case}


def call_for(module, impl):
    """The callable behind one implementation name.

    The two built-ins have private entry points; anything else is a module from
    arch/<gcn>/kernels and is called through its `linear`, the same contract the
    site uses. Returns None when the name cannot be resolved.
    """
    private = {"asm": "_linear_asm", "sum": "_linear_sum", "triton": "_linear_triton",
               "dequant": "linear_dequant"}
    if impl in private and hasattr(module, private[impl]):
        return getattr(module, private[impl])
    if impl == "triton" and hasattr(module, "_linear_dot"):
        return module._linear_dot
    registry = getattr(module, "_REGISTRY", None)
    candidate = registry.candidate(impl) if registry is not None else None
    loaded = candidate.loader() if candidate is not None and candidate.loader else None
    return getattr(loaded, "linear", None)


def implementations(module, declared):
    """Everything this site could run, in the registry's chain order."""
    registry = getattr(module, "_REGISTRY", None)
    names = list(registry.names()) if registry is not None else []
    names += [name for name in declared if name not in names]
    return names


def pick_winner(times, min_gain, preferred=None):
    """Fastest implementation, unless the runner-up is inside the noise floor.

    The same shape moves by ~10% between runs on this card (clocks, shared
    display), so a 0.4% "win" is not a result. Below the floor the probe order
    decides -- that keeps the table stable across runs and stops a noisy shape
    from blocking the site default. Returns (winner, margin_over_runner_up).
    """
    order = sorted(times, key=times.get)
    winner = order[0]
    if len(order) < 2:
        return winner, 1.0
    margin = 1.0 - times[winner] / times[order[1]]
    if margin < min_gain and preferred in times:
        return preferred, margin
    return winner, margin


def probe_order(module, candidates):
    """Which candidate the registry would pick with no table, if it was timed."""
    registry = getattr(module, "_REGISTRY", None)
    for name in (registry.names() if registry is not None else []):
        if name in candidates:
            return name
    return None


def run_site(site, spec, shapes, args, today):
    module = __import__(spec["module"])
    build = CASES[site]
    impls = implementations(module, spec["impls"])
    records, winners, ratios = {}, {}, []
    for n, k, m in shapes:
        key = f"{n}x{k}@m{m}"
        if m > spec["max_m"]:
            print(f"  {key:<18} skipped: this site handles m <= {spec['max_m']}")
            continue
        try:
            run, reference, baseline = build(module, n, k, m, args, spec)
        except ValueError as exc:                        # a flag that cannot apply to this shape
            print(f"  {key:<18} {exc}")
            continue
        times, errors = {}, {}
        for impl in impls:
            registry = getattr(module, "_REGISTRY", None)
            candidate = registry.candidate(impl) if registry is not None else None
            if candidate is not None and not registry.available(candidate):
                print(f"  {key:<18} {impl}: not available on this device")
                continue
            fn = call_for(module, impl)
            if fn is None:
                print(f"  {key:<18} {impl}: no callable, skipped")
                continue
            try:
                out = run(fn)
            except Exception as exc:                        # noqa: BLE001 - a candidate may not compile here
                print(f"  {key:<18} {impl} unavailable: {type(exc).__name__}: {exc}")
                continue
            if out is None:
                print(f"  {key:<18} {impl} declined this shape")
                continue
            error = refs.rel(out, reference)
            if error > args.max_rel:
                print(f"  {key:<18} {impl} REJECTED: rel {error:.2e} > {args.max_rel:.0e}")
                continue
            times[impl] = bench_ms(lambda: run(fn), args.iters, args.warmup)
            errors[impl] = error
        baseline_ms = None
        if baseline is not None:
            label, fn = baseline
            try:
                error = refs.rel(fn(), reference)
            except Exception as exc:                        # noqa: BLE001 - the fallback may not run here
                print(f"  {key:<18} {label} baseline unavailable: {type(exc).__name__}: {exc}")
            else:
                if error > args.max_rel:
                    print(f"  {key:<18} {label} baseline REJECTED: rel {error:.2e} > {args.max_rel:.0e}")
                else:
                    baseline_ms = bench_ms(fn, args.iters, args.warmup)
                    times[label] = baseline_ms
                    errors[label] = error
        if not times:
            print(f"  {key:<18} nothing measured")
            out = None
            del run, reference
            gc.collect()
            torch.cuda.empty_cache()
            continue
        candidates = {impl: times[impl] for impl in impls if impl in times}
        if not candidates:
            print(f"  {key:<18} no candidate ran, only the baseline")
            del run, reference
            gc.collect()
            torch.cuda.empty_cache()
            continue
        winner, margin = pick_winner(candidates, args.min_gain, probe_order(module, candidates))
        if baseline_ms is not None:
            ratios.append(candidates[winner] / baseline_ms)
        cells = "  ".join(f"{impl} {times[impl] * 1000:7.1f} us" for impl in sorted(times))
        rel_cell = f"rel {errors[winner]:.1e}"
        verdict = winner if margin >= args.min_gain else f"{winner} (margin {margin:.1%}, keeping probe order)"
        print(f"  {key:<18} {cells}  -> {verdict}  ({rel_cell})")
        note = ("convrot off" if site == "int8_gemv.m1"
                else f"convrot off, weight group {args.w4a8_group}")
        records[key] = manifest_write.render(key, winner, times, rel=errors[winner], date=today,
                                             note=note)
        winners[key] = winner
        out = None
        del run, reference
        gc.collect()
        torch.cuda.empty_cache()
    return records, winners, ratios


def run_tile_site(args, today):
    """Sweep the int8 GEMM tiles; one winner per batch-width bucket.

    The tile is not a registry site: its alternatives are configs, not kernel
    names, so it reuses tile_sweep's loop instead of run_site's.
    """
    import tile_sweep

    import gfx906_tuning as tuning

    tuning.enable_triton_backend()
    try:
        kernels = list(tuning._int8_kernels())
    except ImportError as exc:
        print(f"\ntile site\n  comfy-kitchen triton backend unavailable: {exc}")
        return {}
    if not kernels:
        print("\ntile site\n  comfy-kitchen int8 triton kernels not found")
        return {}

    candidates = tile_sweep.default_candidates(kernels)
    shapes = []
    for item in args.tile_shapes.split(","):
        n, _, k = item.strip().partition("x")
        shapes.append((int(n), int(k)))
    widths = [int(value) for value in args.tile_m.split(",")]
    print(f"\ntile site ({len(candidates)} candidates, {len(shapes)} shapes, m={widths})")
    results = tile_sweep.sweep(kernels, shapes, widths, candidates, args.iters, args.warmup,
                               repeat=REPEATS)
    by_text = {tile_sweep.spec_text(spec): spec for spec in candidates}

    records = {}
    for bucket, want_small in (("large", False), ("small", True)):
        scores = {}
        for (n, k, m, text), elapsed in results.items():
            if (m <= tuning.INT8_SMALL_MAX_M) == want_small:
                scores.setdefault(text, []).append(elapsed)
        if not scores:
            print(f"  {bucket:<6} nothing measured in this bucket")
            continue
        best = min(scores, key=lambda text: statistics.median(scores[text]))
        medians = {tile_sweep.spec_label(by_text[text]): statistics.median(values)
                   for text, values in scores.items()}
        records[bucket] = manifest_write.render(bucket, best, medians, date=today,
                                                note=f"{len(scores[best])} cell(s), convrot on")
        print(f"  {bucket:<6} {best}  median {statistics.median(scores[best]):.2f} ms")

    # evidence only: the HIP dot4 GEMM against the Triton path, same shapes
    try:
        import hip_int8_gemm as hipg
    except ImportError:
        hipg = None
    if hipg is None or not hipg.available():
        return records
    from comfy_kitchen.backends.triton import quantization
    print("\n  hip int8 gemm (evidence only, not written to the manifest)")
    for n_gemm, k_gemm in shapes:
        for m_gemm in widths:
            if m_gemm < hipg.MIN_ROWS:
                print(f"  {n_gemm}x{k_gemm}@m{m_gemm}: skipped (m < {hipg.MIN_ROWS})")
                continue
            x = torch.randn(m_gemm, k_gemm, device="cuda", dtype=torch.float16)
            w = torch.randint(-127, 127, (n_gemm, k_gemm), device="cuda", dtype=torch.int8)
            ws = torch.rand(n_gemm, device="cuda", dtype=torch.float32) * 1e-3 + 1e-4
            bias = torch.randn(n_gemm, device="cuda", dtype=torch.float32)
            tuning.configure(int8_gemm="off")
            triton_ms = bench_ms(lambda: quantization.int8_linear(
                x, w, ws, bias, out_dtype=torch.float16), args.iters, args.warmup)
            tuning.configure(int8_gemm="default")
            hip_ms = bench_ms(lambda: hipg.linear(x, w, ws, bias, torch.float16, False),
                              args.iters, args.warmup)
            print(f"  {n_gemm}x{k_gemm}@m{m_gemm}: hip {hip_ms:.2f} ms  triton {triton_ms:.2f} ms  "
                  f"{triton_ms / hip_ms:.2f}x")
            del x, w, ws, bias
            gc.collect()
            torch.cuda.empty_cache()
    return records


def run_attention_site(args, today):
    """Sweep the hand/hip/gemm attention paths against SDPA and record the result.

    Writes a tile per head dim, so a later mode change still has
    data), the mode that won, and whether our kernel should be installed at all.
    """
    import attention_kernel as attn
    import gfx906_tuning as tuning
    try:
        import hip_attention as hip
    except ImportError:
        hip = None

    shapes = parse_attention_shapes(args.attention_shapes)
    gen = torch.Generator(device="cuda").manual_seed(0)
    samples = {}                                     # (shape index, mode, label) -> ms
    for index, (batch, heads, seq_q, seq_k, head_dim) in enumerate(shapes):
        q = torch.randn(batch, seq_q, heads, head_dim, device="cuda", dtype=torch.float16,
                        generator=gen)
        k = torch.randn(batch, seq_k, heads, head_dim, device="cuda", dtype=torch.float16,
                        generator=gen)
        v = torch.randn(batch, seq_k, heads, head_dim, device="cuda", dtype=torch.float16,
                        generator=gen)
        reference = refs.attention_reference(q, k, v)
        out = torch.empty_like(q)
        scale = attn._scale(None, head_dim)
        print(f"\n  d{head_dim} b{batch} h{heads} {seq_q}x{seq_k}")

        def run_fp16(config):
            attn._launch(q, k, v, out, scale, config)
            return out

        candidates = [("hand", attn.config_text(config), lambda c=config: run_fp16(c))
                      for config in attn.CANDIDATES]
        if tuning._gemm_ok(q, k, v):
            candidates.append(("gemm", "gemm", lambda: tuning._gemm_attention(q, k, v, None)))
        if hip is not None and hip.available() and hip.supported(q, k, v):
            candidates.append(("hip", "hip", lambda: hip.attention_fwd(q, k, v, None)))
        candidates.append(("sdpa", "sdpa", lambda: tuning._sdpa(q, k, v, None)))

        for mode, label, run in candidates:
            try:
                result = run()
                torch.cuda.synchronize()
            except Exception as exc:                    # noqa: BLE001 - a tile may not compile here
                print(f"    {mode:<5} {label:<10} unavailable: {type(exc).__name__}: {exc}")
                continue
            error = refs.rel(result, reference)
            if error > args.attention_max_rel:
                print(f"    {mode:<5} {label:<10} REJECTED: rel {error:.2e} > {args.attention_max_rel:.0%}")
                continue
            # same election as bench_ms: the fastest round is the undisturbed one.
            # _time_call compiles inside its own warmup, so the clocks are also
            # pinned up here; the second round covers the ramp the compile leaves.
            warm_clocks()
            samples[(index, mode, label)] = min(attn._time_call(run, runs=args.iters)
                                                 for _ in range(REPEATS))
            print(f"    {mode:<5} {label:<10} {samples[(index, mode, label)] * 1000:8.1f} us  "
                  f"rel {error:.1e}")
        del q, k, v, reference, out
        gc.collect()
        torch.cuda.empty_cache()

    def best(index, mode):
        values = [(ms, label) for (i, m, label), ms in samples.items() if i == index and m == mode]
        return min(values) if values else (None, None)

    # gemm is timed as evidence but never elected: it is shape-lucky plain
    # torch, not a kernel of ours, and its prefill form holds ~1 GB of
    # contiguous copies per call -- users hit OOM under memory pressure that the
    # fused kernels (hip 10-27 MiB, hand the same) survive.
    ratios = {}
    for mode in ("hand", "hip"):
        per_shape = []
        for index in range(len(shapes)):
            ours, _ = best(index, mode)
            sdpa, _ = best(index, "sdpa")
            if ours is not None and sdpa:
                per_shape.append(ours / sdpa)
        if per_shape:
            ratios[mode] = statistics.median(per_shape)
    if best(0, "gemm")[0] is not None:
        print("\n  (gemm timed as evidence only; it is never the elected mode -- "
              "its prefill copies OOM under memory pressure)")
    if not ratios:
        print("\n  none of our attention kernels ran here; leaving the mode alone")
        return {}, {}
    mode = min(ratios, key=ratios.get)
    if mode != "sdpa" and 1.0 - ratios[mode] < args.min_gain:
        print(f"\n  best kernel {mode} is {1.0 - ratios[mode]:+.1%} against SDPA, "
              f"below the {args.min_gain:.0%} floor")
        mode = "sdpa"

    records = {}
    for head_dim in sorted({shape[4] for shape in shapes}):
        site = f"attention.tile.d{head_dim}"
        # the sample shares names with the node's modes, where the fp16 Triton
        # kernel is called "hand"
        for key, sample_mode in (("fp16", "hand"),):
            scores = {}
            for (index, mode_name, label), ms in samples.items():
                if mode_name == sample_mode and shapes[index][4] == head_dim:
                    scores.setdefault(label, []).append(ms)
            if not scores:
                continue
            medians = {label: statistics.median(values) for label, values in scores.items()}
            winner = min(medians, key=medians.get)
            records.setdefault(site, {})[key] = manifest_write.render(
                key, winner, medians, date=today, note=f"head_dim {head_dim}")
            print(f"  {site}/{key:<4} {winner}  median {medians[winner] * 1000:.1f} us")

    # evidence only: the attention mode is a quality choice the user makes in
    # the node (int8 PV rel ~2e-2 vs hand 2.5e-4), so nothing here is written
    # to the manifest -- and the TE bypasses these kernels entirely anyway.
    print(f"\n  attention: fastest kernel {mode}; mode stays a manual node/env choice")
    return {}, {}


def parse_deltanet_shapes(text):
    """BxHVxDKxDVxS list -> [(batch, value_heads, key_dim, value_dim, seq)]."""
    shapes = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        batch, heads, dk, dv, seq = (int(value) for value in item.split("x"))
        shapes.append((batch, heads, dk, dv, seq))
    return shapes


def run_deltanet_site(args, today):
    """Time the DeltaNet decode against comfy-kitchen's fused op.

    The replacement exists because that op has no code object for gfx906. On a
    card where it does run it may well be the faster one, and this is what
    decides `feature.deltanet` there. Output and recurrent state are both
    compared with refs.py before either side is timed.
    """
    import comfy_kitchen
    import deltanet

    records, ratios = {}, []
    for batch, heads, dk, dv, seq in parse_deltanet_shapes(args.deltanet_shapes):
        sub = args.deltanet_key_heads or heads
        key = f"b{batch}xhv{heads}xdk{dk}xdv{dv}xs{seq}"
        if heads % sub or not deltanet._pow2(dk) or not deltanet._pow2(dv) or dv % 64:
            print(f"\n  {key} skipped: the kernels want power-of-two head dims and dv % 64")
            continue

        key_dim = sub * dk
        hidden = args.deltanet_hidden
        gen = torch.Generator(device="cuda").manual_seed(0)
        mixed = torch.randn(batch, 2 * key_dim + heads * dv, seq, device="cuda",
                            dtype=torch.float32, generator=gen)
        x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float32,
                        generator=gen) * 0.1
        w_a = torch.randn(heads, hidden, device="cuda", dtype=torch.float32, generator=gen) * 0.02
        w_b = torch.randn(heads, hidden, device="cuda", dtype=torch.float32, generator=gen) * 0.02
        dt_bias = torch.randn(heads, device="cuda", dtype=torch.float32, generator=gen) * 0.1
        g_decay = -torch.rand(heads, device="cuda", dtype=torch.float32, generator=gen) - 1.0
        z = torch.randn(batch, seq, heads * dv, device="cuda", dtype=torch.float32, generator=gen)
        norm_weight = torch.rand(dv, device="cuda", dtype=torch.float32, generator=gen) + 0.5
        start = torch.randn(batch, heads, dk, dv, device="cuda", dtype=torch.float32,
                            generator=gen) * 0.1
        scale = dk ** -0.5
        eps = 1e-6

        want_state = start.clone()
        expected = refs.delta_decode_reference(mixed, x, w_a, w_b, dt_bias, g_decay, want_state,
                                               key_dim, sub, scale, z, norm_weight, eps)
        print(f"\n  {key} hidden={hidden} key_heads={sub}")

        times = {}
        for label, fn in (("triton", deltanet.gated_delta_decode_fused),
                          ("kitchen", comfy_kitchen.gated_delta_decode_fused)):
            state = start.clone()
            try:
                out = fn(mixed, x, w_a, w_b, dt_bias, g_decay, state, key_dim, sub, scale, z,
                         norm_weight, eps)
                torch.cuda.synchronize()
            except Exception as exc:            # noqa: BLE001 - the baseline may not exist here
                print(f"    {label:<8} unavailable: {type(exc).__name__}: {exc}")
                continue
            error = max(refs.rel(out, expected), refs.rel(state, want_state))
            if error > args.max_rel:
                print(f"    {label:<8} REJECTED: rel {error:.2e} > {args.max_rel:.0e}")
                continue
            state.copy_(start)
            times[label] = bench_ms(
                lambda f=fn: f(mixed, x, w_a, w_b, dt_bias, g_decay, state, key_dim, sub, scale,
                               z, norm_weight, eps),
                args.iters, args.warmup)
            print(f"    {label:<8} {times[label] * 1000:8.1f} us  rel {error:.1e}")

        if not times:
            continue
        winner = min(times, key=times.get)
        records.setdefault(DELTANET_SITE, {})[key] = manifest_write.render(
            key, winner, times, date=today, note=f"hidden {hidden}, key heads {sub}")
        if "triton" in times and "kitchen" in times:
            ratios.append(times["triton"] / times["kitchen"])

    defaults = {}
    if ratios:
        gain = 1.0 - statistics.median(ratios)
        on = gain >= args.min_gain
        defaults["feature.deltanet"] = "on" if on else "off"
        print(f"\n  deltanet: {gain:+.1%} against comfy-kitchen, "
              f"feature.deltanet={defaults['feature.deltanet']}")
    else:
        print("\n  deltanet: no comfy-kitchen baseline here, leaving feature.deltanet alone")
    return records, defaults


def run_deltanet_prefill_site(args, today):
    """Time the Qwen3.5 prefill chunk rule: core's UT loop vs the block recursion.

    Both run the identical algorithm on identical inputs; the only difference is
    the 63-step forward-substitution loop (qwen35.py:143-146) against the batched
    block inversion in deltanet.ut_inverse. The numeric gate compares against
    core's own output, so a record can only be written for a variant that answers
    the same question. feature.deltanet_prefill is written from the gain.

    Each shape is checked on two input regimes and the gate is the worse of the
    two; timing uses the mild one. Marginals alone are not enough to catch an
    unstable transform: with independent randn keys, mild or heavy decay, the
    Neumann doubling that used to ship here matched core to 1e-7 while it produced
    4.29e+16 against core's 0.2672 on the model's own prefill tensors. What
    reproduces that is key-row coherence: with every key row within 1e-3 of one
    shared direction, k_beta @ key^T approaches the strictly-lower matrix of ones,
    whose UT inverse is the +-1 difference matrix (absmax 1) while its powers
    explode. On that regime the doubling measures rel 4.65e+03 and this
    implementation 1.6e-07.
    """
    import deltanet
    from comfy.text_encoders.qwen35 import torch_chunk_gated_delta_rule

    records, ratios = {}, []
    for item in args.deltanet_prefill_shapes.split(","):
        batch, heads, dk, dv, seq = (int(x) for x in item.strip().split("x"))
        key = f"b{batch}xh{heads}xdk{dk}xdv{dv}xs{seq}"

        def inputs(coherent):
            gen = torch.Generator(device="cuda").manual_seed(0)
            q = torch.randn(batch, seq, heads, dk, device="cuda", dtype=torch.float32, generator=gen)
            k = torch.randn(batch, seq, heads, dk, device="cuda", dtype=torch.float32, generator=gen)
            if coherent:
                direction = torch.randn(1, 1, 1, dk, device="cuda", dtype=torch.float32, generator=gen)
                direction = direction / direction.norm(dim=-1, keepdim=True)
                k = 0.999 * direction + 0.001 * k / k.norm(dim=-1, keepdim=True)
            v = torch.randn(batch, seq, heads, dv, device="cuda", dtype=torch.float32, generator=gen) * 0.1
            g = (-torch.rand(batch, seq, heads, device="cuda", generator=gen) - 1.0) * 0.02
            beta = torch.rand(batch, seq, heads, device="cuda", generator=gen).sigmoid()
            return q, k, v, g, beta

        sets = {"mild": inputs(False), "coherent": inputs(True)}
        print(f"\n  {key}")

        times, errors, reference = {}, {}, {}
        for label, fn in (("loop", torch_chunk_gated_delta_rule), ("block", deltanet.chunk_rule)):
            worst = 0.0
            for name, ins in sets.items():
                out, state = fn(*ins)
                torch.cuda.synchronize()
                if label == "loop":
                    reference[name] = (out, state)
                    continue
                want_out, want_state = reference[name]
                error = refs.rel(out, want_out)
                if want_state is not None:
                    error = max(error, refs.rel(state, want_state))
                worst = max(worst, error)
                print(f"    {label:<9} vs loop on {name:<9} rel {error:.2e}")
            if label != "loop":
                if worst > args.max_rel:
                    print(f"    {label:<9} REJECTED: worst rel {worst:.2e} > {args.max_rel:.0e}")
                    continue
                errors[label] = worst
            times[label] = bench_ms(lambda f=fn: f(*sets["mild"]), args.iters, args.warmup)
            print(f"    {label:<9} {times[label] * 1000:8.1f} us  "
                  f"rel {errors.get(label, 0.0):.1e}")

        if len(times) < 2:
            print(f"  {key}: both implementations are the shipped loop, nothing to compare")
            continue
        winner = min(times, key=times.get)
        ratios.append(times["block"] / times["loop"])
        records.setdefault(DELTANET_PREFILL_SITE, {})[key] = manifest_write.render(
            key, winner, times, rel=errors["block"], date=today,
            note="fp32 chunk rule, UT transform only difference, worst of mild+coherent")

    defaults = {}
    if ratios:
        gain = 1.0 - statistics.median(ratios)
        defaults["feature.deltanet_prefill"] = "on" if gain >= args.min_gain else "off"
        print(f"\n  deltanet prefill: block {gain:+.1%} against core's loop, "
              f"feature.deltanet_prefill={defaults['feature.deltanet_prefill']}")
    else:
        print("\n  deltanet prefill: nothing comparable measured, "
              "leaving feature.deltanet_prefill alone")
    return records, defaults


def run_qwen35_site(args, today):
    """Compare the Qwen3.5 class swap against the shipped methods.

    One synthetic model and one set of weights: the gate only decides which
    forward runs, so both sides see identical parameters and inputs. That makes
    this a real numeric check (both run fp32, hence the 1e-3 gate) of the parts
    the swap touches -- RoPE slicing, the conv tap, the gated norm, the chunked
    delta rule -- without needing a checkpoint. The timing is a proxy: at four
    layers and no cache it measures the Python-level saves, not a 24-layer TE.
    """
    import comfy.ops
    from comfy.text_encoders.qwen35 import Qwen35Config, _qwen35_layer_types

    import qwen35_model

    layers, hidden, seq = args.qwen35_layers, args.qwen35_hidden, args.qwen35_seq
    heads = max(1, hidden // 256)
    cfg = Qwen35Config(
        vocab_size=256,
        hidden_size=hidden,
        intermediate_size=hidden * 3,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=max(1, hidden // 1024),
        head_dim=128,
        partial_rotary_factor=0.25,
        mrope_section=[6, 5, 5],
        linear_num_key_heads=max(1, hidden // 128),
        linear_num_value_heads=max(1, hidden // 128),
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=_qwen35_layer_types(layers),
    )
    model = qwen35_model.Qwen35Transformer(cfg, device="cuda", dtype=torch.float32,
                                           ops=comfy.ops.disable_weight_init).cuda().float()
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("A_log"):
                param.copy_(torch.rand(param.shape, generator=gen, device="cuda") + 1.0)
            elif name.endswith("dt_bias"):
                param.normal_(0, 0.1, generator=gen)
            else:
                param.normal_(0, 0.05, generator=gen)
    emb = torch.randn(1, seq, hidden, device="cuda", dtype=torch.float32, generator=gen)
    key = f"b1xs{seq}xh{hidden}xl{layers}"
    print(f"\n  {key} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    def call(gate_on):
        # the block does in-place adds into its input, which needs no_grad like
        # any other forward; the clone keeps one run from mutating the next
        qwen35_model._gate = (lambda: True) if gate_on else (lambda: False)
        with torch.no_grad():
            return model(None, embeds=emb.clone())

    reference, _ = call(False)
    torch.cuda.synchronize()
    out, _ = call(True)
    torch.cuda.synchronize()
    error = refs.rel(out, reference)
    print(f"    swap vs shipped: rel {error:.2e}")
    if error > QWEN35_MAX_REL:
        print(f"    REJECTED: rel {error:.2e} > {QWEN35_MAX_REL:.0e}")
        return {}, {}

    times = {"stock": bench_ms(lambda: call(False), args.iters, args.warmup),
             "ours": bench_ms(lambda: call(True), args.iters, args.warmup)}
    gain = 1.0 - times["ours"] / times["stock"]
    winner = min(times, key=times.get)
    print(f"    shipped {times['stock'] * 1000:8.1f} us   swap {times['ours'] * 1000:8.1f} us "
          f"({gain:+.1%})")
    records = {QWEN35_SITE: {key: manifest_write.render(
        key, winner, times, date=today, note="prefill, fp32, gate on/off")}}
    # no feature.qwen35 here: the case covers prefill only, while the swap's win
    # is mostly on the decode path, so this timing cannot decide the feature.
    # The env switch decides (install_ready); this site stays as evidence.
    print(f"    feature.qwen35 not written: the case covers prefill only")
    return records, {}


def run_vae_site(args, today):
    """Time a depth-1 Conv3d: torch, the conv2d squeeze the patch does, kitchen.

    The VAE patch ships no kernel: it rewrites ComfyUI's Conv3d call into conv2d
    when the depth axis carries a single frame. The alternative on this card is
    torch's conv3d, or comfy-kitchen's fp16_conv3d where a native backend serves
    it -- whichever exists is the baseline feature.vae needs. All candidates run
    with padding 0 so the kitchen kernel (which takes no padding argument) stays
    in the comparison; the conv2d/conv3d ratio at padding 1 is the same story.
    """
    import comfy_kitchen
    from comfy_kitchen import registry

    records, ratios = {}, []
    backend = "hip" if torch.version.hip is not None else "cuda"
    caps = registry.list_backends().get(backend, {}).get("capabilities") or []
    for item in args.vae_shapes.split(","):
        channels, _, size = item.strip().partition("x")
        channels, size = int(channels), int(size)
        key = f"c{channels}x{size}"
        gen = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(1, channels, 1, size, size, device="cuda", dtype=torch.float16,
                        generator=gen)
        weight = torch.randn(channels, channels, 1, 3, 3, device="cuda", dtype=torch.float16,
                             generator=gen) * 0.02
        expected = torch.nn.functional.conv3d(x, weight, stride=1)
        print(f"\n  {key} (depth-1 conv3d, fp16)")

        candidates = [("torch", lambda: torch.nn.functional.conv3d(x, weight, stride=1)),
                      ("conv2d", lambda: torch.nn.functional.conv2d(
                          x[:, :, 0], weight[:, :, 0], None, 1, 0, 1, 1).unsqueeze(2))]
        if "fp16_conv3d" in caps:
            candidates.append(("kitchen", lambda: comfy_kitchen.fp16_conv3d(x, weight, stride=1)))

        times = {}
        for name, fn in candidates:
            try:
                out = fn()
                torch.cuda.synchronize()
            except Exception as exc:            # noqa: BLE001 - the native kernel may decline
                print(f"    {name:<8} unavailable: {type(exc).__name__}: {exc}")
                continue
            error = refs.rel(out, expected)
            if error > args.max_rel:
                print(f"    {name:<8} REJECTED: rel {error:.2e} > {args.max_rel:.0e}")
                continue
            times[name] = bench_ms(fn, args.iters, args.warmup)
            print(f"    {name:<8} {times[name] * 1000:8.1f} us  rel {error:.1e}")
        if "conv2d" not in times or len(times) < 2:
            continue
        baseline = min(ms for name, ms in times.items() if name != "conv2d")
        ratios.append(times["conv2d"] / baseline)
        records.setdefault(VAE_SITE, {})[key] = manifest_write.render(
            key, min(times, key=times.get), times, date=today,
            note="depth-1 conv3d, fp16, padding 0")

    defaults = {}
    if ratios:
        gain = 1.0 - statistics.median(ratios)
        defaults["feature.vae"] = "on" if gain >= args.min_gain else "off"
        print(f"\n  vae: conv2d squeeze {gain:+.1%} against its fallback, "
              f"feature.vae={defaults['feature.vae']}")
    else:
        print("\n  vae: nothing comparable measured, leaving feature.vae alone")
    return records, defaults


def run_vision_site(args, today):
    """Time the Qwen3.5 image tower's patch projection three ways.

    Core hands the tower an fp32 input while its weights are the text-encoder
    dtype, and the patch Conv3d has kernel == stride, so a card without a conv3d
    code object lowers it to im2col + bmm. The three rows separate the two costs:

      conv3d_fp32  what ships today (fp32 input, conv3d) -- the baseline
      conv3d       same conv at the weight dtype, so the dtype change alone
      gemm         the patch: one GEMM over the flattened patch

    feature.vision is written from gemm against conv3d_fp32, which is what the
    switch controls. On a card with a real conv3d kernel and no upcast cost the
    gain is under --min-gain and the verdict stays off.
    """
    import torch.nn.functional as F

    records, ratios = {}, []
    for item in args.vision_shapes.split(","):
        rows, _, dim = item.strip().partition("x")
        rows, dim = int(rows), int(dim)
        key = f"p{rows}xd{dim}"
        gen = torch.Generator(device="cuda").manual_seed(0)
        # the real module's shape: 3 channels x temporal 2 x 16 x 16 per patch
        weight = (torch.randn(dim, 3, 2, 16, 16, device="cuda", dtype=torch.bfloat16,
                              generator=gen) * 0.05)
        bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.01
        x = torch.randn(rows, 3, 2, 16, 16, device="cuda", dtype=torch.bfloat16, generator=gen)
        flat_w = weight.reshape(dim, -1)
        expected = F.conv3d(x.float(), weight.float(), bias.float()).flatten(1)
        print(f"\n  {key} (kernel==stride conv3d, bf16 weight)")

        candidates = [
            ("conv3d_fp32", lambda: F.conv3d(x.float(), weight.float(),
                                             bias.float()).flatten(1)),
            ("conv3d", lambda: F.conv3d(x, weight, bias).flatten(1)),
            ("gemm", lambda: F.linear(x.reshape(rows, -1), flat_w, bias)),
        ]
        times, errors = {}, {}
        for name, fn in candidates:
            try:
                out = fn()
                torch.cuda.synchronize()
            except Exception as exc:                            # noqa: BLE001 - a card may have no conv3d
                print(f"    {name:<12} unavailable: {type(exc).__name__}: {exc}")
                continue
            error = refs.rel(out.float(), expected)
            if error > args.max_rel:
                print(f"    {name:<12} REJECTED: rel {error:.2e} > {args.max_rel:.0e}")
                continue
            times[name] = bench_ms(fn, args.iters, args.warmup)
            errors[name] = error
            print(f"    {name:<12} {times[name] * 1000:8.1f} us  rel {error:.1e}")
        if "conv3d_fp32" not in times or "gemm" not in times:
            continue
        winner = min(times, key=times.get)
        ratios.append(times["gemm"] / times["conv3d_fp32"])
        records.setdefault(VISION_SITE, {})[key] = manifest_write.render(
            key, winner, times, rel=errors[winner], date=today,
            note="image tower patch projection, bf16 weight")

    defaults = {}
    if ratios:
        gain = 1.0 - statistics.median(ratios)
        defaults["feature.vision"] = "on" if gain >= args.min_gain else "off"
        print(f"\n  vision: patch over the shipped path {gain:+.1%}, "
              f"feature.vision={defaults['feature.vision']}")
    else:
        print("\n  vision: nothing comparable measured, leaving feature.vision alone")
    return records, defaults


def run_convrot_site(args, today):
    """Time the ConvRot activation prep: kitchen's matmul rotation vs ours.

    "off" is what runs without the patch (kitchen rotates with a matmul, then
    quantizes); "butterfly" is our two-pass rotate + quantize; "fused" is one
    kernel that rotates and quantizes together. The winner becomes
    convrot.mode, and feature.convrot says whether ours is worth installing.
    """
    import convrot_kernel as rot
    from comfy_kitchen.backends.triton import quantization as tq
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

    hadamard = _build_hadamard(rot.GROUP_SIZE)

    records, per_shape = {}, {}
    for item in args.convrot_shapes.split(","):
        rows, _, k = item.strip().partition("x")
        rows, k = int(rows), int(k)
        if k % rot.GROUP_SIZE:
            print(f"\n  m{rows}x{k} skipped: k must be a multiple of {rot.GROUP_SIZE}")
            continue
        key = f"m{rows}x{k}"
        gen = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(rows, k, device="cuda", dtype=torch.float16, generator=gen)
        reference = rot.hadamard_rotate(x).float()
        print(f"\n  {key}")

        def off_path():
            return tq.triton_quantize_rowwise(_rotate_activation(x, hadamard, rot.GROUP_SIZE))

        def butterfly_path():
            return tq.triton_quantize_rowwise(rot.hadamard_rotate(x))

        times = {}
        for name, fn, check in (("off", off_path, False),
                                ("butterfly", butterfly_path, True),
                                ("fused", lambda: rot.convrot_quantize(x), True)):
            try:
                out = fn()
                torch.cuda.synchronize()
            except Exception as exc:            # noqa: BLE001 - a path may not compile here
                print(f"    {name:<10} unavailable: {type(exc).__name__}: {exc}")
                continue
            if check:
                data, scale = (out.data, out.scale) if hasattr(out, "data") else out
                error = refs.rel(data.float() * scale, reference)
                if error > args.max_rel:
                    print(f"    {name:<10} REJECTED: rel {error:.2e} > {args.max_rel:.0e}")
                    continue
            times[name] = bench_ms(fn, args.iters, args.warmup)
            print(f"    {name:<10} {times[name] * 1000:8.1f} us")

        if not times:
            continue
        ours = {name: ms for name, ms in times.items() if name != "off"}
        if ours:
            records.setdefault(CONVROT_SITE, {})[key] = manifest_write.render(
                key, min(ours, key=ours.get), times, date=today,
                note="rotate + quantize, fp16")
            per_shape[key] = times

    defaults = {}
    ours_keys = ("butterfly", "fused")
    comparable = [times for times in per_shape.values()
                  if "off" in times and any(name in times for name in ours_keys)]
    medians = {name: statistics.median([times[name] for times in per_shape.values()
                                        if name in times])
               for name in ours_keys
               if any(name in times for times in per_shape.values())}
    if comparable and medians:
        ratios = [min(times[name] for name in ours_keys if name in times) / times["off"]
                  for times in comparable]
        gain = 1.0 - statistics.median(ratios)
        mode = min(medians, key=medians.get)
        defaults["feature.convrot"] = "on" if gain >= args.min_gain else "off"
        defaults["convrot.mode"] = mode
        print(f"\n  convrot: ours {gain:+.1%} against kitchen's rotation, "
              f"feature.convrot={defaults['feature.convrot']}, convrot.mode={mode}")
    else:
        print("\n  convrot: nothing comparable measured, leaving convrot.mode alone")
    return records, defaults


def target_path(args, arch):
    if args.out:
        return Path(args.out).expanduser()
    return Path(manifest_mod.default_paths(arch)[1])


def write_manifest(arch, fingerprint, written, winners, defaults, path, today):
    text = path.read_text(encoding="utf-8") if path.exists() else manifest_write.skeleton(arch, fingerprint)
    text = manifest_write.set_arch_identity(text, arch, fingerprint)
    for site, impl in defaults.items():
        text = manifest_write.set_default(text, site, impl)
    for site, records in written.items():
        agreed = winners.get(site) or set()
        if len(agreed) == 1:
            # every measured shape of this site picked the same implementation,
            # so the same answer is a fair generalisation for unmeasured ones
            text = manifest_write.set_default(text, site, next(iter(agreed)))
        elif agreed:
            print(f"  {site}: winners disagree ({', '.join(sorted(agreed))}), leaving default alone")
        text = manifest_write.set_shapes(text, site, records)
    try:
        saved = manifest_write.save(path, text)
    except ValueError as exc:
        print(f"\n{exc}")
        return None
    print(f"\nwrote {saved}")
    if saved == Path(manifest_mod.default_paths(arch)[0]):
        print("that is the file that ships with the node: review the diff before committing")
    else:
        print("local override: it wins over the shipped table, delete it to go back to derived data")
    return saved


def ensure_hip_kernels():
    """Build the node's HIP kernels when this card can use them and they are missing.

    The tile and attention sites have no hip row without the .so files, and they
    say nothing about it, so a fresh clone silently benchmarks the slow half.
    """
    if not build_hip_attn.missing():
        return
    if torch.version.hip is None or not torch.cuda.get_device_properties(
            0).gcnArchName.startswith("gfx906"):
        return
    if build_hip_attn.find_hipcc() is None:
        print("\nhip kernels not built and hipcc not found: run tools/build_hip_attn.py "
              "from the ROCm venv to get the hip rows")
        return
    print(f"\nbuilding the hip kernels ({', '.join(build_hip_attn.missing())})")
    build_hip_attn.build()


def main():
    global REPEATS

    args = parse_args()
    REPEATS = args.repeats
    if not torch.cuda.is_available():
        print("no device: torch.cuda.is_available() is False (see tools/verify_backends.py for the same check)")
        return 2
    import gfx906_tuning as tuning

    arch = manifest_mod.device_arch()
    fingerprint = manifest_mod.runtime_fingerprint(arch)
    # the tile records matter where triton actually runs int8_linear, which on
    # a covered card is only the native backend's constraint fallback; the
    # attention records matter wherever the node installs the attention patch
    tile_unused = tuning.native_covers("int8_linear")
    attn_unused = not tuning.is_amd()
    sites = args.site or list(SITES)
    unknown = [site for site in sites if site not in SITES]
    if unknown:
        print(f"unknown site(s) {unknown}; known: {sorted(SITES)}")
        return 2

    print(f"{arch} | {fingerprint}")
    print(f"{args.iters} iters, {args.warmup} warmup, {args.repeats} round(s), "
          f"numeric gate {args.max_rel:.0%}")
    today = datetime.date.today().isoformat()

    written, winners, defaults = {}, {}, {}
    for site in sites:
        spec = SITES[site]
        shapes = parse_shapes(args.shapes or spec["default"])
        print(f"\nsite {site} ({', '.join(spec['impls'])})")
        records, site_winners, ratios = run_site(site, spec, shapes, args, today)
        if records:
            written[site] = records
            winners[site] = set(site_winners.values())
        if ratios and spec.get("feature"):
            name = f"feature.{spec['feature']}"
            gain = 1.0 - statistics.median(ratios)
            defaults[name] = "on" if gain >= args.min_gain else "off"
            print(f"  {spec['feature']}: {gain:+.1%} against its fallback, {name}={defaults[name]}")

    want_tile = args.tile or (args.all and not tile_unused)
    want_attn = args.attention or (args.all and not attn_unused)
    if want_tile or want_attn:
        ensure_hip_kernels()
    if args.all and (tile_unused or attn_unused):
        parts = [name for name, skip in (("tile", tile_unused), ("attention", attn_unused)) if skip]
        print(f"\n--all skips the {', '.join(parts)} site(s) here: on this device the records "
              f"would almost never be read (pass --tile / --attention to measure anyway)")
    if want_tile:
        records = run_tile_site(args, today)
        if records:
            # the two buckets are different tiles by design, so no site default
            written[TILE_SITE] = records
    if want_attn:
        run_attention_site(args, today)
    if args.deltanet or args.all:
        records, choices = run_deltanet_site(args, today)
        written.update(records)
        defaults.update(choices)
        records, choices = run_deltanet_prefill_site(args, today)
        written.update(records)
        defaults.update(choices)
    if args.vae or args.all:
        records, choices = run_vae_site(args, today)
        written.update(records)
        defaults.update(choices)
    if args.vision or args.all:
        records, choices = run_vision_site(args, today)
        written.update(records)
        defaults.update(choices)
    if args.convrot or args.all:
        records, choices = run_convrot_site(args, today)
        written.update(records)
        defaults.update(choices)
    if args.qwen35 or args.all:
        records, choices = run_qwen35_site(args, today)
        written.update(records)
        defaults.update(choices)

    if not written and not defaults:
        print("\nnothing measured")
        return 1
    path = target_path(args, arch)
    shapes_written = sum(len(records) for records in written.values())
    if not args.write:
        print(f"\ndry run: would update {path} with {shapes_written} record(s) and "
              f"{len(defaults)} default(s); pass --write")
        return 0
    return 0 if write_manifest(arch, fingerprint, written, winners, defaults, path, today) else 1


if __name__ == "__main__":
    sys.exit(main())
