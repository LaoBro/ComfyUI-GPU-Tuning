"""Version-pinned Qwen3.5 text encoder for the node pack.

``model`` is a byte-copy of ComfyUI 0.39.0's ``comfy/text_encoders/qwen35.py``
(the measured-fast path), with its ``llama.py`` primitives copied into
``llama_base``. Core 0.44+ rewrote these classes around ``CombinedLinear``,
``FixedKVCache`` and the deferred DeltaNet kernels; on gfx906 those fall back to
torch and cost ~3x. This module keeps one compute implementation across core
versions by owning the ``te()``/``tokenizer()`` factories core calls in
``comfy/sd.py``.

The factory contract is the injection boundary: core loads the returned TEModel
by module attribute at workflow-load time, so swapping ``te()``/``tokenizer()``
makes core build the vendored model without touching any core file. The contract
held unmodified from 0.39 through 0.45.4; upstream master later added
``vocab_size``/``tokenizer_json`` kwargs. ``_te`` accepts ``**kwargs`` so a core
update cannot crash a workflow load -- unknown kwargs are dropped with a
one-shot warning and shown as ``DRIFT(...)`` in ``status()``/``summary()``.

See reference/ for pristine upstream copies (0.39.0 = the pinned compute source,
0.45.4 = newest tag kept for forward-diffing upstream changes) and FORK_BASE for
the tag the vendored code came from.
"""

import inspect
import logging
from pathlib import Path

_FORK_BASE = (Path(__file__).parent / "FORK_BASE").read_text().strip()

_gate = None
_installed = False
# Unexpected kwargs core has started passing to te(). Kept so the drift is
# visible in status() after the one-shot log warning scrolls away.
_drift = set()


def _enabled():
    return _gate is None or _gate()


def fork_base():
    return _FORK_BASE


def install(gate=None):
    """Point core's qwen35 factories at the vendored ones.

    The node's kernel patches stay global: the vendored model calls the same
    ``comfy_kitchen.*`` symbols the pack already optimizes (Triton DeltaNet,
    folded KV decode). We only pin the model classes, which is the volatile
    part ComfyUI rewrites between releases.
    """
    global _gate, _installed
    _gate = gate
    if _installed:
        return True
    try:
        import comfy.text_encoders.qwen35 as core
    except ImportError:
        logging.warning("gpu-tune vendored qwen35: core text encoder unavailable")
        return False

    from . import model as vendored

    # Keep a handle on core's originals so a gate flip can restore them.
    install._core_te = getattr(core, "te", None)
    install._core_tokenizer = getattr(core, "tokenizer", None)
    if install._core_te is None or install._core_tokenizer is None:
        logging.warning("gpu-tune vendored qwen35: core factories not found, skip")
        return False

    core.te = _te
    core.tokenizer = vendored.tokenizer
    _installed = True
    logging.info("gpu-tune vendored qwen35: injected te/tokenizer (fork %s)", _FORK_BASE)
    return True


def _te(dtype_llama=None, llama_quantization_metadata=None, model_type="qwen35_2b",
        mtp=False, **kwargs):
    """Gate-aware te() with a defensive boundary.

    A newer core's ``sd.py`` may call ``te()`` with kwargs the pinned fork does
    not know yet (e.g. 0.46 added ``vocab_size``/``tokenizer_json``). Accepting
    ``**kwargs`` keeps a workflow load from crashing on a core update. Any kwarg
    the vendored ``te()`` does not declare is dropped with a one-shot warning,
    and recorded in ``boundary_drift()`` / ``summary()`` so an operator can see
    the fork needs a STRUCT-layer port. Once the vendored ``te()`` is updated,
    those kwargs are forwarded automatically -- no change to this wrapper.

    Gate off delegates to core's original factory, which does understand the new
    kwargs, so the fallback path stays fully functional.
    """
    if not _enabled():
        return install._core_te(dtype_llama=dtype_llama,
                                llama_quantization_metadata=llama_quantization_metadata,
                                model_type=model_type, mtp=mtp, **kwargs)

    from . import model as vendored
    accepted = set(inspect.signature(vendored.te).parameters)
    forward = {k: v for k, v in kwargs.items() if k in accepted}
    unexpected = set(kwargs) - accepted
    if unexpected:
        newly = unexpected - _drift
        _drift.update(unexpected)
        if newly:
            logging.warning(
                "gpu-tune vendored qwen35: core passed new te() kwargs %s that fork %s "
                "does not handle; dropping them (model still builds). Run "
                "tools/vendor_forward_diff.py and port the STRUCT layer, or the "
                "next core update may ship a checkpoint this fork cannot load.",
                sorted(newly), _FORK_BASE)
    return vendored.te(dtype_llama=dtype_llama,
                       llama_quantization_metadata=llama_quantization_metadata,
                       model_type=model_type, mtp=mtp, **forward)


def boundary_drift():
    """Unexpected te() kwargs seen this session (empty = boundary intact)."""
    return sorted(_drift)


def summary():
    drift = (" drift=%s" % ",".join(boundary_drift())) if _drift else ""
    return "vendored@%s%s%s" % (_FORK_BASE, " on" if _installed else " off", drift)
