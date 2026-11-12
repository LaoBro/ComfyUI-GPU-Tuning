"""End-to-end attribution for the text-generation chain.

patches/profiler.py covers the DiT side: BaseModel.apply_model, the module level
attention entry, and kitchen's int8_linear. The text encoder goes through the
CLIP chain instead, and its attention comes from
optimized_attention_for_device(small_input=True), so none of those hooks see it.
This runs one TextGenerate call and attributes it with nested CUDA events over
the model's own classes, found structurally instead of by name:

  * the cond stage model, the module that owns the widest ModuleList, the live
    generate object and every block -- plus one level of children below each, so
    a sibling subtree such as an image tower gets its own row;
  * that generate object's .generate and .logits, and core's FixedKVBias.decode;
  * anything named by --wrap.

Rows are self time, a region's total minus the regions nested directly inside
it, so the table adds up instead of counting a block's attention twice.

    cd /path/to/ComfyUI
    .venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/chain_profile.py \
        --clip qwen3.5-9b.safetensors --max-length 64

    ... --clip X --image photo.jpg                  # the image tower joins the chain
    ... --clip X --pack off                         # the same run on core alone
    ... --clip X --set text_encoder=int8            # any configure() item
    ... --clip X --set image_tower=off --set textgen=off
    ... --clip X --list-classes                     # the census, to pick --wrap names

The four textgen items have no configure() entry; A/B them with the env
switches that apply() reads: COMFY_GPU_TUNE_DELTANET, _KVDECODE, _LLAMA,
_QWEN35.

ComfyUI flags pass through (--disable-cuda-graphs, --highvram). Library use
parses an empty argv unless comfy.options.args_parsing is set first, so this
script sets it and leaves comfy only the flags it does not claim.

Calls captured into a CUDA graph cannot carry events: they land in the graphed
column and contribute no time. --disable-cuda-graphs makes them measurable, but
that run is not comparable to a graphed one, since the graphs are what hide the
per-block dispatch cost.

The run holds a checkpoint's worth of VRAM. Do not run it while ComfyUI is
generating.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
NODE_DIR = TOOLS_DIR.parent
sys.path.insert(0, str(NODE_DIR))
sys.path.insert(0, str(NODE_DIR.parent.parent))          # ComfyUI root

# Same pin as the pack's __init__.py, which these top-level imports bypass.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import comfy.options                                                      # noqa: E402

# comfy.cli_args parses sys.argv when it is first imported, and a library gets
# parse_args([]) by default: without this the passthrough flags below are silent.
comfy.options.enable_args_parsing(True)

import torch                                                              # noqa: E402

MAX_RECORDS = 400000

_records = {}
_graphed = {}
_stack = []
_dropped = 0


def parse_args():
    parser = argparse.ArgumentParser(description="attribute one text-generation run")
    parser.add_argument("--clip", required=True,
                        help="text encoder file name, as the CLIP loader lists it")
    parser.add_argument("--clip-type", default="qwen_image")
    parser.add_argument("--prompt", default="A fennec fox in the snow")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--image", default="", help="input image file name, as LoadImage lists it")
    parser.add_argument("--max-length", type=int, default=64,
                        help="tokens to generate; per-step numbers converge within a few dozen")
    parser.add_argument("--sampling", choices=("on", "off"), default="on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mtp", default="auto", choices=("auto", "off", "2", "3", "4", "5"))
    parser.add_argument("--pack", choices=("on", "off"), default="on",
                        help="off runs the same chain on core alone")
    parser.add_argument("--set", action="append", default=[], metavar="ITEM=VALUE",
                        help="configure() override, repeatable: --set text_encoder=int8 --set image_tower=off")
    parser.add_argument("--wrap", action="append", default=[],
                        help="attribute one more class, by name; repeatable")
    parser.add_argument("--list-classes", action="store_true",
                        help="load the checkpoint, print the module class census, exit")
    parser.add_argument("--kitchen", action="store_true",
                        help="also install patches/profiler.py's hooks for the kitchen level view under these rows")
    parser.add_argument("--json", action="store_true", help="one machine readable line at the end")
    args, passthrough = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + passthrough
    return args


def timed(cls, attr, key):
    """Record one CUDA event pair per call of cls.attr, tagged with its caller."""
    original = getattr(cls, attr)

    def wrapper(self, *args, **kwargs):
        global _dropped
        if torch.cuda.is_current_stream_capturing():
            _graphed[key] = _graphed.get(key, 0) + 1
            return original(self, *args, **kwargs)
        parent = _stack[-1] if _stack else None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _stack.append(key)
        try:
            return original(self, *args, **kwargs)
        finally:
            _stack.pop()
            end.record()
            rows = _records.setdefault(key, [])
            if len(rows) < MAX_RECORDS:
                rows.append((parent, start, end))
            else:
                _dropped += 1

    setattr(cls, attr, wrapper)


def keep(child):
    """Whether an anchor's child class deserves its own row.

    comfy.ops and torch.nn hold the leaf layers -- Linear, Embedding, Conv, the
    norm containers. Wrapping one of them wraps every instance, which pulls the
    leaf time out of every region above and flattens them to self 0. That view
    is patches/profiler.py's int8_linear hook (--kitchen), or an explicit --wrap Linear.
    """
    return not type(child).__module__.startswith(("comfy.ops", "torch.nn"))


def attribution_targets(root, extra):
    """(key, class, attribute) to instrument, plus the class census and the step key.

    The step key is the module that owns the block list: its forward runs once
    for the prefill and once per generated token, which is what splits the two.
    """
    census = {}
    for module in root.modules():
        census[type(module)] = census.get(type(module), 0) + 1

    import comfy.text_encoders.llama as llama
    # the generate object is what the TextGenerate node calls, and every TE
    # subclass overrides generate(), so wrapping the base class would time
    # nothing: take the class of the live instance instead
    generators = [module for module in root.modules() if isinstance(module, llama.BaseGenerate)]

    blocks, owner = None, root
    for module in root.modules():
        for child in module.children():
            if isinstance(child, torch.nn.ModuleList) and len(child) and (blocks is None or len(child) > len(blocks)):
                blocks, owner = child, module

    # Anchors are the modules whose own forward marks a stage; each contributes
    # its class plus one level of children, so a sibling subtree such as an
    # image tower gets its own row instead of hiding in its parent's self time.
    # Every block is an anchor, not just the first: a hybrid model alternates
    # block contents, and one instance would miss the attention half.
    anchors = [root, owner, *generators] + list(blocks or [])
    classes = []
    for anchor in anchors:
        classes.append(type(anchor))
        classes.extend(type(child) for child in anchor.children() if keep(child))
    classes.extend(cls for cls in census if cls.__name__ in extra)

    seen, targets = set(), []
    for cls in classes:
        if cls not in seen:
            seen.add(cls)
            targets.append((cls.__name__, cls, "forward"))

    for cls in {type(module) for module in generators}:
        targets.append((f"{cls.__name__}.generate", cls, "generate"))
        targets.append((f"{cls.__name__}.logits", cls, "logits"))
    targets.append((f"{llama.FixedKVBias.__name__}.decode", llama.FixedKVBias, "decode"))
    step_key = type(owner).__name__
    return targets, census, step_key


def summarize():
    """Totals, self times and the regions nothing else contains."""
    torch.cuda.synchronize()
    totals, nested = {}, {}
    for key, rows in _records.items():
        totals[key] = sum(start.elapsed_time(end) for _, start, end in rows)
        for parent, _, _ in rows:
            if parent is not None and parent != key:
                nested.setdefault(parent, set()).add(key)
    selfs = {key: max(total - sum(totals.get(child, 0.0) for child in nested.get(key, ())), 0.0)
             for key, total in totals.items()}
    roots = {key for key, rows in _records.items() if any(parent is None for parent, _, _ in rows)}
    return totals, selfs, roots


def main():
    args = parse_args()

    import comfy.sd
    import nodes

    if not hasattr(comfy.sd.CLIPType, args.clip_type.upper()):
        known = ", ".join(name.lower() for name in comfy.sd.CLIPType.__members__)
        print(f"unknown --clip-type {args.clip_type}; ComfyUI knows: {known}")
        return 2
    if not torch.cuda.is_available():
        print("no device: torch.cuda.is_available() is False")
        return 2

    tuning = None
    if args.pack == "on":
        import gpu_tuning as tuning

        tuning.apply()
        overrides = dict(item.partition("=")[::2] for item in args.set)
        tuning.configure(profile="on" if args.kitchen else "off", **overrides)
        state = tuning._state
        print(f"[pack] arch={state.get('arch')} applied={state.get('applied')} "
              f"te_int8={state.get('te_int8')} gemv={state.get('gemv_model')} "
              f"vision={state.get('vision_enabled')} set={','.join(args.set) or '-'}")
        print(f"[pack] textgen {tuning.textgen_summary()}")
    elif args.set or args.kitchen:
        ignored = " ".join(name for name, used in (("--set", args.set), ("--kitchen", args.kitchen)) if used)
        print(f"[pack] --pack off ignores {ignored}: those belong to the pack, "
              f"so this run measures core's own defaults")

    started = time.perf_counter()
    clip = nodes.CLIPLoader().load_clip(args.clip, args.clip_type)[0]
    load_s = time.perf_counter() - started
    image = nodes.LoadImage().load_image(args.image)[0] if args.image else None
    root = clip.cond_stage_model
    print(f"[load] {load_s:.2f} s {type(root).__name__} dtype={getattr(root, 'dtype', None)} "
          f"mtp={getattr(root, 'mtp', None)} image={tuple(image.shape) if image is not None else None}")

    targets, census, step_key = attribution_targets(root, set(args.wrap))
    if args.list_classes:
        wrapped = {cls for _, cls, _ in targets}
        for cls, count in sorted(census.items(), key=lambda kv: -kv[1]):
            print(f"{count:6d}  {cls.__module__}.{cls.__name__}  {'wrapped' if cls in wrapped else ''}")
        return 0
    for key, cls, attr in targets:
        timed(cls, attr, key)
    print(f"[wrap] {len(targets)} regions, step key {step_key}")

    from comfy_extras import nodes_textgen

    sampling = {"sampling_mode": args.sampling, "seed": args.seed}
    started = time.perf_counter()
    out = nodes_textgen.TextGenerate.execute(clip=clip, prompt=args.prompt, max_length=args.max_length,
                                             sampling_mode=sampling, image=image, mtp=args.mtp,
                                             system_prompt=args.system_prompt)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - started
    text = out.result[0]

    if args.kitchen:
        from patches import profiler
        print(f"[kitchen]\n{profiler.summary()}")

    totals, selfs, roots = summarize()
    print(f"\n=== {args.clip} pack={args.pack} mtp={args.mtp} sampling={args.sampling} "
          f"max_length={args.max_length} set={','.join(args.set) or '-'} ===")
    print(f"load {load_s:.2f} s | run {wall_s:.2f} s wall | "
          f"output {len(text)} chars md5={hashlib.md5(text.encode()).hexdigest()[:12]} head={text[:40]!r}")
    print(f"{'region':<30} {'calls':>7} {'graphed':>8} {'total s':>8} {'self s':>8} "
          f"{'median ms':>10} {'of run':>7}")
    for key in sorted(totals, key=lambda name: -selfs[name]):
        rows = _records[key]
        samples = sorted(start.elapsed_time(end) for _, start, end in rows)
        print(f"{key:<30} {len(rows) + _graphed.get(key, 0):7d} {_graphed.get(key, 0):8d} "
              f"{totals[key] / 1000:8.2f} {selfs[key] / 1000:8.2f} "
              f"{(samples[len(samples) // 2] if samples else 0.0):10.2f} {selfs[key] / 1000 / wall_s:7.1%}")
    if _dropped:
        print(f"{_dropped} record(s) dropped past MAX_RECORDS={MAX_RECORDS}: the totals above are low")
    if sum(_graphed.values()):
        print(f"{sum(_graphed.values())} call(s) ran inside a CUDA graph capture, which cannot carry "
              f"events: they are counted above and timed nowhere. Replays do not reach python at all, "
              f"so per-step attribution below {step_key} needs --disable-cuda-graphs")

    # a root's total, not its self: its children are attributed too, they are
    # just attributed to themselves
    attributed = sum(totals[key] for key in roots) / 1000
    print(f"unattributed {wall_s - attributed:.2f} s of {wall_s:.2f} s wall "
          f"(tokenize, python, launch gaps, and anything not wrapped)")
    passes = len(_records.get(step_key, []))
    if passes > 1:
        _, first_start, first_end = _records[step_key][0]
        prefill = first_start.elapsed_time(first_end)
        decode = totals[step_key] - prefill
        print(f"{passes} forward passes: prefill {prefill:.1f} ms + {passes - 1} steps "
              f"{decode / 1000:.2f} s = {decode / (passes - 1):.2f} ms/step, "
              f"{(passes - 1) / (decode / 1000):.2f} tok/s")

    if args.json:
        print(json.dumps({"clip": args.clip, "pack": args.pack, "mtp": args.mtp,
                          "max_length": args.max_length, "load_s": round(load_s, 2),
                          "wall_s": round(wall_s, 2), "passes": passes,
                          "md5": hashlib.md5(text.encode()).hexdigest(),
                          "regions": {key: {"calls": len(_records[key]) + _graphed.get(key, 0),
                                            "graphed": _graphed.get(key, 0),
                                            "total_ms": round(totals[key], 1),
                                            "self_ms": round(selfs[key], 1)}
                                      for key in totals}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
