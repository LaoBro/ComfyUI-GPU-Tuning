"""Per-call GPU timers for the hot ComfyUI entry points.

Opt-in via the model node's profile switch. Each instrumented call records a
CUDA event pair, and the report is printed when a prompt finishes; there is
no profiler node. Because the model node runs before the sampler in a normal
graph, its one-shot status covers the next prompt.
"""

import logging
import sys

import torch

MAX_RECORDS = 20000

ORDER = ("model_forward", "attention", "int8_linear")
DEPTH = {"model_forward": 0, "attention": 1, "int8_linear": 1}

_records = []
_installed = []
_shapes = {}
_wrapped = {}
_providers = []


def watch(provider):
    """Register a callable returning extra report lines; it is called once per report."""
    _providers.append(provider)


_enabled = None


def enabled():
    if _enabled is not None:
        return _enabled
    return False


def set_enabled(flag):
    """Toggle from the tuning node; installs the hooks on first use."""
    global _enabled
    _enabled = bool(flag)
    if not _enabled:
        _records.clear()
    elif not _installed:
        patch()
        logging.info("gfx906 profiler hooks: %s (report printed when each prompt finishes)", ", ".join(_installed))


def _wrap(label, fn):
    if fn is None:
        return None

    def timed(*args, **kwargs):
        if not enabled() or len(_records) >= MAX_RECORDS:
            return fn(*args, **kwargs)
        if not _records:
            logging.info("gfx906 profiler: recording %s (first call of this prompt)", label)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        _records.append((label, start, end))
        seen = _shapes.setdefault(label, {})
        try:
            first = args[0] if args else next(iter(kwargs.values()))
            key = (tuple(first.shape), str(first.dtype).removeprefix("torch."))
            seen[key] = seen.get(key, 0) + 1
        except Exception:
            pass
        return out

    return timed


def _wrap_once(label, fn):
    """Wrap each distinct function object once, so re-exports do not nest timers."""
    entry = _wrapped.get((label, id(fn)))
    if entry is not None and entry[0] is fn:
        return entry[1]
    wrapper = _wrap(label, fn)
    _wrapped[(label, id(fn))] = (fn, wrapper)
    return wrapper


def _rebind(old, new):
    """Swap every module-level reference to old, so already-imported callers see new."""
    swapped = 0
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            namespace = vars(module)
        except TypeError:
            continue
        for name, value in list(namespace.items()):
            if value is old:
                setattr(module, name, new)
                swapped += 1
    return swapped


def install():
    if _installed or not enabled():
        return _installed
    return patch()


def patch():
    """Instrument the model forward, int8 linear and the attention entry point."""
    import comfy_kitchen
    from comfy.ldm.modules import attention
    from comfy.model_base import BaseModel

    BaseModel.apply_model = _wrap_once("model_forward", BaseModel.apply_model)
    _installed.append("model_forward")

    # comfy_kitchen dispatches through getattr(backend_module, "int8_linear"), and each
    # backend re-exports the function, so the module it is defined in is not the one used.
    from comfy_kitchen.backends.triton import quantization

    quantization.int8_linear = _wrap_once("int8_linear", quantization.int8_linear)
    for module in comfy_kitchen.registry._backends.values():
        if getattr(module, "int8_linear", None) is not None:
            module.int8_linear = _wrap_once("int8_linear", module.int8_linear)
    _installed.append("int8_linear")

    target = getattr(attention, "optimized_attention", None)
    if _rebind(target, _wrap_once("attention", target)):
        _installed.append("attention")
    patch_prompt_end()
    return _installed


def patch_prompt_end():
    """Log the report when a prompt finishes, so nothing depends on graph execution.

    Installed when the tuning patch loads, not when the node first runs: the queue
    thread resolves execute_async when it starts a prompt, so a hook added by a
    node inside that prompt would only take effect on the next one.
    """
    import execution

    original = execution.PromptExecutor.execute_async
    if getattr(original, "_gfx906_prompt_end", False):
        return

    async def execute_async(self, *args, **kwargs):
        before = _clock_reference() if enabled() else None
        try:
            return await original(self, *args, **kwargs)
        finally:
            if enabled() or _records:
                report = summary(before, _clock_reference() if before is not None else None)
                if report:
                    logging.info("gfx906 profiler\n%s", report)

    execute_async._gfx906_prompt_end = True
    execution.PromptExecutor.execute_async = execute_async


def summary(clock_before=None, clock_after=None):
    if not _installed:
        return "profiler not installed, enable the profile switch on the model settings node"
    if not _records:
        return ""

    torch.cuda.synchronize()
    totals = {}
    for label, start, end in _records:
        totals.setdefault(label, []).append(start.elapsed_time(end))
    _records.clear()

    lines = [f"hooks: {', '.join(_installed)} (first call of each includes triton JIT)"]
    for label in ORDER:
        samples = totals.get(label)
        if not samples:
            continue
        samples.sort()
        total = sum(samples)
        lines.append(
            f"{'  ' * DEPTH[label]}{label:13s} {len(samples):5d} calls {total / 1000:7.2f} s   "
            f"median {samples[len(samples) // 2]:7.2f} ms   max {samples[-1]:8.2f} ms"
        )
        shapes = _shapes.get(label)
        if shapes:
            top = sorted(shapes.items(), key=lambda kv: -kv[1])[:3]
            lines.append("  " * (DEPTH[label] + 1) + "shapes " + ", ".join(
                f"{shape} {dtype} x{count}" for (shape, dtype), count in top))
    _shapes.clear()

    forward = sum(totals.get("model_forward", []))
    if forward:
        attention_ms = sum(totals.get("attention", []))
        linear_ms = sum(totals.get("int8_linear", []))
        lines.append(
            f"model_forward {forward / 1000:.2f} s = attention {attention_ms / 1000:.2f} s"
            f" + int8_linear {linear_ms / 1000:.2f} s"
            f" + other {(forward - attention_ms - linear_ms) / 1000:.2f} s"
        )
    # the first prompt after profile is switched on has no clock_before, but its
    # counters still say what ran, so report them either way
    lines.extend(line for provider in _providers for line in provider())
    if clock_before is not None:
        lines.append(
            f"clock ref {clock_before:.2f} -> {clock_after:.2f} ms (4096^3 fp16 matmul, before/after the prompt; "
            "lower is faster, so a rising value means the GPU throttled)"
        )
    return "\n".join(lines)


def _clock_reference():
    """One 4096^3 fp16 matmul, so runs at different clocks can be compared."""
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    samples = []
    for _ in range(4):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        a @ b
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return sorted(samples)[1]
