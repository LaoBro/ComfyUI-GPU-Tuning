"""Locate fp16 overflow inside a DiT forward, with H3 function probes.

Opt-in via the model node's fp16_trace switch. Leaf module outputs are sampled
while the traced forward runs: every sample is a device-side abs-max / isfinite
/ over-limit count, so a forward costs no synchronisation and a prompt pays one
sync in the report. The report names the modules whose output left the target
range (or came close), which is where a targeted down-scale belongs.

For MiniMax H3 the module hooks miss the functional sites that can overflow:
the modulated norm (_mod_scale_shift), the gated residual (_mod_gate), the
attention result and the MLP result. Those are wrapped as well, so a report
tells apart "the norm/modulation already blew up" from "the attention/MLP did".
The trace can also record bf16/fp32 instead of fp16, which is the reference run
used to estimate how much scaling a site needs.

The idea comes from the Lumina workaround in ComfyUI PR #11187: that model
divides the attention output by 4 and the FFN w3 branch by 32 after finding out
which tensors overflowed. This trace reports the same kind of site without
hard-coding a model.
"""

import logging
import math

import torch

# fp16 tops out at 65504; report values above this before they become inf.
FP16_LIMIT = 60000.0
# One prompt of a 32-block DiT is a few thousand calls; cap memory.
MAX_ENTRIES = 20000
DTYPE_MODES = {
    "fp16": (torch.float16,),
    "bf16": (torch.bfloat16,),
    "fp32": (torch.float32,),
    "all": (torch.float16, torch.bfloat16, torch.float32),
}
DTYPE_NAMES = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}

_enabled = False
_trace_dtype = "fp16"
_dtypes = DTYPE_MODES["fp16"]
_entries = []
_step = -1
_in_forward = False
_first_report = False
_saw_forward = False
_reported_sites = set()
_last_peak = 0.0
_hooked = 0
_hooked_model = None
_module_info = {}
_current_block = None
_h3_installed = False


def enabled():
    return _enabled


def set_enabled(flag):
    """Toggle from the tuning node; hooks stay installed but become no-ops."""
    global _enabled, _first_report, _saw_forward
    _enabled = bool(flag)
    if _enabled:
        _first_report = True
    else:
        _entries.clear()
        _reported_sites.clear()
        _last_peak = 0.0
        _saw_forward = False
    return _enabled


def set_dtype(mode):
    """Record fp16 (default), or bf16/fp32/all for a reference run."""
    global _trace_dtype, _dtypes, _first_report
    text = str(mode or "fp16").strip().lower().replace("torch.", "")
    if text not in DTYPE_MODES:
        text = "fp16"
    _trace_dtype = text
    _dtypes = DTYPE_MODES[text]
    _entries.clear()
    _reported_sites.clear()
    _last_peak = 0.0
    _first_report = _enabled
    return _trace_dtype


def dtype_summary():
    return _trace_dtype


def install(model):
    """Hook the leaf modules of one diffusion model; safe to call per forward."""
    global _hooked, _hooked_model
    _install_h3_probes(model)
    if not _enabled or getattr(model, "_gfx906_fp16_trace_hooks", None) is not None:
        return
    model_name = type(model).__name__
    hooked = 0
    for name, module in model.named_modules():
        _module_info[id(module)] = (model_name, name)
        if next(module.children(), None) is not None:
            continue
        module.register_forward_hook(_make_hook(model_name, name, type(module).__name__))
        hooked += 1
    model._gfx906_fp16_trace_hooks = hooked
    _hooked, _hooked_model = hooked, model_name
    if hooked:
        logging.info("gfx906 fp16 trace: %d leaf modules hooked on %s", hooked, model_name)


def begin_forward():
    """Called once before a traced forward, so the report can name the step."""
    global _step, _in_forward, _saw_forward
    if not _enabled:
        return
    _step += 1
    _in_forward = True
    _saw_forward = True


def end_forward():
    global _in_forward, _current_block
    _in_forward = False
    _current_block = None


def hooked_summary():
    return f"{_hooked} leaf modules on {_hooked_model}" if _hooked else ""


def _make_hook(model_name, name, cls_name):
    def hook(module, inputs, output):
        if not _enabled or not _in_forward or len(_entries) >= MAX_ENTRIES:
            return
        outs = _traced_tensors(output)
        if not outs:
            return
        ins = _traced_tensors(inputs)
        # Identity / dropout forward the same tensor; its peak is the producer's.
        if len(ins) == len(outs) and all(a is b for a, b in zip(ins, outs)):
            return
        _record(model_name, name, cls_name, outs, ins)

    return hook


def _install_h3_probes(model):
    """Wrap the H3 functional sites that module hooks cannot see."""
    global _h3_installed
    if _h3_installed:
        return
    try:
        from comfy.ldm.minimax import model as h3
    except ImportError:
        return
    _h3_installed = True
    _wrap_function(h3, "_mod_scale_shift", "_mod_scale_shift")
    _wrap_function(h3, "_mod_gate", "_mod_gate")
    _wrap_method(h3.MLP, "forward")
    _wrap_method(h3.Attention, "forward")
    _wrap_method(h3.DiTBlock, "forward", set_block=True)


def _wrap_function(module, name, label):
    original = getattr(module, name, None)
    if original is None or getattr(original, "_gfx906_trace_wrapped", False):
        return

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        if _enabled and _in_forward:
            _probe_function(label, out)
        return out

    wrapped._gfx906_trace_wrapped = True
    setattr(module, name, wrapped)


def _wrap_method(cls, name, set_block=False):
    original = getattr(cls, name, None)
    if original is None or getattr(original, "_gfx906_trace_wrapped", False):
        return

    def wrapped(self, *args, **kwargs):
        global _current_block
        info = _module_info.get(id(self), (type(self).__name__, type(self).__name__))
        previous = _current_block
        if set_block:
            _current_block = info
        try:
            out = original(self, *args, **kwargs)
        finally:
            if set_block:
                _current_block = previous
        if _enabled and _in_forward:
            outs = _traced_tensors(out)
            if outs:
                _record(info[0], f"{info[1]}.out", type(self).__name__, outs)
        return out

    wrapped._gfx906_trace_wrapped = True
    setattr(cls, name, wrapped)


def _probe_function(label, out):
    if _current_block is not None:
        model_name, block_name = _current_block
        name = f"{block_name}.{label}"
    else:
        model_name, name = "h3", label
    outs = _traced_tensors(out)
    if outs:
        _record(model_name, name, "probe", outs)


def _record(model_name, name, cls_name, outs, ins=None):
    global _entries
    if not _enabled or not _in_forward or len(_entries) >= MAX_ENTRIES:
        return
    ins = ins or []
    try:
        with torch.no_grad():
            out_peak = torch.stack([t.abs().max() for t in outs]).max()
            out_ok = torch.stack([torch.isfinite(t).all() for t in outs]).all()
            out_hi = torch.stack([(t.abs() > FP16_LIMIT).sum() for t in outs]).sum()
            in_peak = (torch.stack([t.abs().max() for t in ins]).max()
                       if ins else out_peak)
    except Exception as exc:
        # a foreign device or a shape the reduction refuses: never break the
        # forward for a diagnostic, turn the trace off and say why
        set_enabled(False)
        logging.warning("gfx906 fp16 trace disabled after %s: %s", name, exc)
        return
    _entries.append({
        "model": model_name,
        "name": name,
        "cls": cls_name,
        "step": _step,
        "shape": "x".join(str(dim) for dim in outs[0].shape),
        "dtype": DTYPE_NAMES.get(outs[0].dtype, str(outs[0].dtype)),
        "in": in_peak.detach(),
        "out": out_peak.detach(),
        "ok": out_ok.detach(),
        "hi": out_hi.detach(),
    })


def _traced_tensors(value):
    found = []
    _collect(value, found)
    return found


def _collect(value, found):
    if isinstance(value, torch.Tensor):
        if value.dtype in _dtypes and value.numel():
            found.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect(item, found)
    elif isinstance(value, dict):
        for item in value.values():
            _collect(item, found)


def _worse(a, b):
    if math.isnan(b):
        return False
    if math.isnan(a):
        return True
    return a > b


def _sort_peak(value):
    return value if math.isfinite(value) else math.inf


def take_report():
    """Format and clear the samples; called when a prompt finishes."""
    global _first_report, _last_peak, _saw_forward
    if not _enabled:
        return ""
    saw = _saw_forward
    _saw_forward = False
    if not _entries:
        if saw:
            return (f"trace on but no {_trace_dtype} tensors were recorded "
                    "(is dit_fp16 enabled, and does the reference dtype match the model?)")
        return ""

    entries = list(_entries)
    _entries.clear()
    try:
        out_peaks = torch.stack([entry["out"] for entry in entries]).float().cpu().tolist()
        in_peaks = torch.stack([entry["in"] for entry in entries]).float().cpu().tolist()
        finite = torch.stack([entry["ok"] for entry in entries]).cpu().tolist()
        over = torch.stack([entry["hi"] for entry in entries]).cpu().tolist()
    except Exception as exc:  # tensors on several devices, or a dead context
        return f"trace could not read the samples: {exc}"

    records = {}
    for index, entry in enumerate(entries):
        key = (entry["model"], entry["name"], entry["dtype"])
        peak = out_peaks[index]
        ok = bool(finite[index])
        record = records.get(key)
        if record is None:
            records[key] = {
                "cls": entry["cls"], "shape": entry["shape"],
                "first": index, "step": entry["step"],
                "in": in_peaks[index], "peak": peak, "peak_step": entry["step"],
                "hi": int(over[index]),
                "bad": not ok,
                "bad_first": index if not ok else -1,
                "bad_step": entry["step"] if not ok else -1,
            }
            continue
        record["hi"] = max(record["hi"], int(over[index]))
        if not ok and not record["bad"]:
            record["bad"] = True
            record["bad_first"] = index
            record["bad_step"] = entry["step"]
        if _worse(peak, record["peak"]):
            record["peak"] = peak
            record["in"] = in_peaks[index]
            record["peak_step"] = entry["step"]

    models = {key[0] for key in records}
    dtypes = {key[2] for key in records}

    def label(key):
        model, name, dtype = key
        text = f"{model}: {name}" if len(models) > 1 else name
        if _trace_dtype == "all" or dtype != "fp16":
            text = f"{text} [{dtype}]"
        return text

    bad = sorted(((key, rec) for key, rec in records.items() if rec["bad"]),
                 key=lambda item: item[1]["bad_first"])
    new_bad = [(key, rec) for key, rec in bad if key not in _reported_sites]
    peaks = sorted(records.items(),
                   key=lambda item: (_sort_peak(item[1]["peak"]), -item[1]["first"]),
                   reverse=True)
    top_peak = peaks[0][1]["peak"] if peaks else 0.0
    new_peak = _worse(top_peak, _last_peak)

    lines = []
    if bad and (new_bad or _first_report):
        show = bad if _first_report else new_bad
        lines.append("overflow sites (call order; later entries are usually downstream victims):")
        for key, record in show[:12]:
            lines.append(
                f"  {label(key):64s} step {record['bad_step']:3d}  [{record['shape']}]  "
                f"in {record['in']:.6g} -> out {record['peak']:.6g}  "
                f"hi {record['hi']}  ({record['cls']})")
        if len(show) > 12:
            lines.append(f"  ... {len(show) - 12} more")
        _reported_sites.update(key for key, _ in bad)

    near = [(key, record) for key, record in peaks
            if math.isfinite(record["peak"])]
    probe_sites = [(key, record) for key, record in near
                   if record["cls"] == "probe"][:8]
    if near and (_first_report or new_peak):
        title = ("top sites by abs-max (reference run; >60000 are scale candidates):"
                 if _trace_dtype != "fp16" else
                 f"top sites by abs-max (>{FP16_LIMIT:g} is the fp16 danger zone):")
        lines.append(title)
        for key, record in near[:8]:
            lines.append(
                f"  {label(key):64s} step {record['peak_step']:3d}  [{record['shape']}]  "
                f"in {record['in']:.6g} -> out {record['peak']:.6g}  "
                f"hi {record['hi']}  ({record['cls']})")
    if probe_sites and _first_report:
        lines.append("H3 modulation/residual probes:")
        for key, record in probe_sites:
            lines.append(
                f"  {label(key):64s} step {record['peak_step']:3d}  [{record['shape']}]  "
                f"in {record['in']:.6g} -> out {record['peak']:.6g}  "
                f"hi {record['hi']}")

    if lines:
        steps = len({entry["step"] for entry in entries})
        lines.insert(0, f"sampled {len(entries)} calls over {steps} steps, "
                        f"{len(records)} sites (dtype {_trace_dtype}, "
                        f"{len(dtypes)} observed)")
        lines.append("hint: compare this run with COMFY_GPU_TUNE_FP16_TRACE_DTYPE=bf16/fp32 and "
                     "dit_fp16 off; scale S = ceil(reference_peak / 60000), ideally a power of two, "
                     "then compensate exactly where the site feeds a linear or a gate. "
                     "See ComfyUI PR #11187 for the attention /4 and FFN w3 /32 shape.")
    _first_report = False
    _last_peak = max(_last_peak, _sort_peak(top_peak))
    return "\n".join(lines)
