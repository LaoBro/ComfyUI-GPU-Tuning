"""Measure a site's implementations on this GPU and record the result.

    # dry run: print the table, touch nothing
    .venv/bin/python custom_nodes/ComfyUI-gfx906-Tuning/tools/bench.py

    # one site, custom shapes
    ... --site w4a8.m1 --shapes 4096x4096@m1,4096x4096@m8

    # write it back (local override, does not touch the shipped file)
    ... --write

    # target the file that ships with the node (for a PR)
    ... --write --out custom_nodes/ComfyUI-gfx906-Tuning/arch/gfx906/manifest.toml

The pipeline per shape is fixed, in this order:

1. build inputs and the torch reference from tools/refs.py;
2. run each implementation once and compare against the reference -- a kernel
   that is off by more than --max-rel is reported and *not* timed, so a broken
   fast path can never win a benchmark;
3. time what survived (median of --iters after --warmup), keep every time in
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
import statistics
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
NODE_DIR = TOOLS_DIR.parent
sys.path.insert(0, str(NODE_DIR))
sys.path.insert(0, str(TOOLS_DIR))

import torch                                                           # noqa: E402

import refs                                                            # noqa: E402
from kernels import manifest as manifest_mod                           # noqa: E402
from kernels import manifest_write                                     # noqa: E402

SITES = {
    "int8_gemv.m1": {
        "module": "int8_gemv",
        "impls": ("asm", "triton"),
        "default": "4096x4096@m1,4096x12288@m1,12288x4096@m1,152064x4096@m1",
        "max_m": 1,
    },
    "w4a8.m1": {
        "module": "w4a8_gemv",
        "impls": ("asm", "triton"),
        "default": "4096x4096@m1,4096x4096@m8,12288x4096@m1,12288x4096@m8",
        "max_m": 8,
    },
}
W4A8_GROUP = 128


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", action="append", default=[],
                        help=f"repeatable; default all of {sorted(SITES)}")
    parser.add_argument("--shapes", default="", help="NxK@mM list, overrides the site default")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-rel", type=float, default=0.05,
                        help="numeric gate against tools/refs.py (default 5%%, activation quantisation is ~1%%)")
    parser.add_argument("--min-gain", type=float, default=0.03,
                        help="below this margin the faster implementation is not declared a winner "
                             "(default 3%%: the same shape moves ~10%% between runs)")
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


def bench_ms(fn, iters, warmup):
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


def int8_case(module, n, k, m):
    """int8 weights, one row of activations, group size 0 (no convrot)."""
    weight = torch.randint(-127, 127, (n, k), device="cuda", dtype=torch.int8)
    scale = torch.rand(n, device="cuda", dtype=torch.float32) * 0.01 + 1e-4
    x = torch.randn(m, k, device="cuda", dtype=torch.float32)
    reference = refs.int8_reference(x, weight, scale)

    def run(fn):
        return fn(x, weight, scale, None, torch.float32, 0)

    return run, reference


def w4a8_case(module, n, k, m):
    """Packed 4-bit weights, group size 128, no convrot."""
    qdata = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    s_rel = torch.rand(n, k // W4A8_GROUP, device="cuda", dtype=torch.float32) * 0.02 + 1e-4
    s_channel = torch.rand(n, device="cuda", dtype=torch.float32) * 0.5 + 0.5
    x = torch.randn(m, k, device="cuda", dtype=torch.float32)
    reference = x @ refs.w4a8_uniform_weight(qdata, s_rel, s_channel, W4A8_GROUP).T

    def run(fn):
        return fn(x, qdata, s_rel, s_channel, None, torch.float32, W4A8_GROUP, 0)

    return run, reference


CASES = {"int8_gemv.m1": int8_case, "w4a8.m1": w4a8_case}


def call_for(module, impl):
    """The callable behind one implementation name.

    The two built-ins have private entry points; anything else is a module from
    arch/<gcn>/kernels and is called through its `linear`, the same contract the
    site uses. Returns None when the name cannot be resolved.
    """
    private = {"asm": "_linear_asm", "triton": "_linear_triton"}
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
    records, winners = {}, {}
    for n, k, m in shapes:
        key = f"{n}x{k}@m{m}"
        if m > spec["max_m"]:
            print(f"  {key:<18} skipped: this site handles m <= {spec['max_m']}")
            continue
        run, reference = build(module, n, k, m)
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
        if not times:
            print(f"  {key:<18} nothing measured")
            out = None
            del run, reference
            gc.collect()
            torch.cuda.empty_cache()
            continue
        winner, margin = pick_winner(times, args.min_gain, probe_order(module, times))
        cells = "  ".join(f"{impl} {times[impl] * 1000:7.1f} us" for impl in sorted(times))
        rel_cell = f"rel {errors[winner]:.1e}"
        verdict = winner if margin >= args.min_gain else f"{winner} (margin {margin:.1%}, keeping probe order)"
        print(f"  {key:<18} {cells}  -> {verdict}  ({rel_cell})")
        note = "convrot off" if site == "int8_gemv.m1" else f"convrot off, weight group {W4A8_GROUP}"
        records[key] = manifest_write.render(key, winner, times, rel=errors[winner], date=today,
                                             note=note)
        winners[key] = winner
        out = None
        del run, reference
        gc.collect()
        torch.cuda.empty_cache()
    return records, winners


def target_path(args, arch):
    if args.out:
        return Path(args.out).expanduser()
    return Path(manifest_mod.default_paths(arch)[1])


def write_manifest(arch, fingerprint, written, winners, path, today):
    text = path.read_text(encoding="utf-8") if path.exists() else manifest_write.skeleton(arch, fingerprint)
    text = manifest_write.set_arch_identity(text, arch, fingerprint)
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


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("no device: torch.cuda.is_available() is False (see tools/verify_backends.py for the same check)")
        return 2
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).split(":")[0]
    fingerprint = manifest_mod.runtime_fingerprint(arch)
    sites = args.site or list(SITES)
    unknown = [site for site in sites if site not in SITES]
    if unknown:
        print(f"unknown site(s) {unknown}; known: {sorted(SITES)}")
        return 2

    print(f"{arch} | {fingerprint}")
    print(f"{args.iters} iters, {args.warmup} warmup, numeric gate {args.max_rel:.0%}")
    today = datetime.date.today().isoformat()

    written, winners = {}, {}
    for site in sites:
        spec = SITES[site]
        shapes = parse_shapes(args.shapes or spec["default"])
        print(f"\nsite {site} ({', '.join(spec['impls'])})")
        records, site_winners = run_site(site, spec, shapes, args, today)
        if records:
            written[site] = records
            winners[site] = set(site_winners.values())

    if not written:
        print("\nnothing measured")
        return 1
    path = target_path(args, arch)
    if not args.write:
        print(f"\ndry run: would update {path} with {sum(len(v) for v in written.values())} shape(s); pass --write")
        return 0
    return 0 if write_manifest(arch, fingerprint, written, winners, path, today) else 1


if __name__ == "__main__":
    sys.exit(main())
