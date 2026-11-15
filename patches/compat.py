"""Capability probes for the private ComfyUI APIs this pack patches.

The pack sits on model, attention and text-encoder internals that ComfyUI
refactors between releases. A version string does not identify those refactors,
so each probe checks the contract the patch actually needs: the caller then
picks the core path or the node path from that answer.
"""

import inspect


_COMBINED_LINEAR = None
_COMBINED_LINEAR_READY = False


def combined_linear():
    """The core CombinedLinear class, or None on builds that do not have it."""
    global _COMBINED_LINEAR, _COMBINED_LINEAR_READY
    if not _COMBINED_LINEAR_READY:
        _COMBINED_LINEAR_READY = True
        try:
            import comfy.ops
        except ImportError:
            return None
        _COMBINED_LINEAR = getattr(comfy.ops, "CombinedLinear", None)
    return _COMBINED_LINEAR


def is_combined_linear(linear):
    cls = combined_linear()
    return cls is not None and isinstance(linear, cls)


def vision_premerge():
    """Whether Qwen35VisionModel.forward accepts the premerge keyword."""
    try:
        from comfy.text_encoders import qwen35
        params = inspect.signature(qwen35.Qwen35VisionModel.forward).parameters
    except (ImportError, AttributeError, TypeError, ValueError):
        return False
    return "premerge" in params


def qwen35_native():
    """Whether core Qwen3.5 already owns the fused attention and KV-cache contract."""
    try:
        from comfy.text_encoders import qwen35
    except ImportError:
        return False
    # comfy_attention is an instance attribute, so probe the class-level cache
    # method; 0.39 has neither, 0.44+ has both.
    return hasattr(qwen35.GatedAttention, "init_kv_cache")


def fixed_kv_bias():
    """The legacy FixedKVBias class, or None once core moved to FixedKVCache."""
    try:
        import comfy.text_encoders.llama as llama
    except ImportError:
        return None
    return getattr(llama, "FixedKVBias", None)


def fixed_kv_cache():
    try:
        import comfy.text_encoders.llama as llama
    except ImportError:
        return False
    return hasattr(llama, "FixedKVCache")


def qwen21_prefix_cache():
    try:
        from comfy.ldm.qwen_image21 import model
    except ImportError:
        return False
    return hasattr(model, "PrefixCache")


def summary():
    """One line for status(): which contracts core already owns."""
    core = []
    if qwen35_native():
        core.append("qwen35")
    if fixed_kv_cache():
        core.append("kv cache")
    if combined_linear() is not None:
        core.append("combined linear")
    if qwen21_prefix_cache():
        core.append("qwen21 cache")
    return ", ".join(core) if core else "none"
