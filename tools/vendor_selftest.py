"""Self-test for the vendored qwen35 fork: no GPU or checkpoint needed.

    python custom_nodes/ComfyUI-GPU-Tuning/tools/vendor_selftest.py

Checks the properties that must hold for the fork to be a faithful, version-
independent replacement for core's Qwen3.5 classes:
  * FORK_BASE names the tag the vendored code came from
  * the vendored model.py is a byte copy of that tag's core qwen35.py (only the
    llama import path is rewired) -- so weight keys and math stay identical
  * both pristine reference copies are present for forward-diffing upstream
  * te()/tokenizer() match core's factory signatures (the injection boundary)
  * install() swaps core's factories and the gate restores them
"""

import inspect
import sys
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent.parent
COMFY_ROOT = NODE_DIR.parent.parent  # .../ComfyUI-0.x/custom_nodes/<node> -> ComfyUI root
sys.path.insert(0, str(NODE_DIR))
if (COMFY_ROOT / "comfy").is_dir():
    sys.path.insert(0, str(COMFY_ROOT))

from vendor import qwen35 as v  # noqa: E402

FAILURES = []


def check(label, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def norm(source):
    """Normalize the imports rewrite so byte comparison is meaningful."""
    return source.replace("from .llama_base import", "from .llama import").replace(
        "from __future__ import annotations\n", "")


def def_source(source, name):
    """Return the source of one top-level def/class by name (AST-free scan)."""
    lines = source.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("class %s" % name) or line.startswith("def %s" % name):
            start = i
            break
    if start is None:
        return ""
    out = [lines[start]]
    for line in lines[start + 1:]:
        if line and not line[0].isspace() and not line.startswith(")"):
            break
        out.append(line)
    return "\n".join(out)


def main():
    base = v.fork_base()
    check("FORK_BASE", base, "v0.39.0")

    ref_dir = NODE_DIR / "vendor" / "qwen35" / "reference"
    check("reference 0.39 present", (ref_dir / "qwen35_v0.39.0.py").is_file(), True)
    check("reference 0.45.4 present", (ref_dir / "qwen35_v0.45.4.py").is_file(), True)

    vendored_src = (NODE_DIR / "vendor" / "qwen35" / "model.py").read_text()
    ref_src = (ref_dir / "qwen35_v0.39.0.py").read_text()
    llama_src = (NODE_DIR / "vendor" / "qwen35" / "llama_base.py").read_text()
    # The node's gfx906 compute optimizations are baked into the vendored copies
    # (not applied as runtime class patches on core, which would miss the fork).
    # Check each optimized method body is present so the fork keeps the fast path.
    for marker in ("_merged_ab", "unfold(2, self.conv_kernel_size",
                   "torch.stack((query, key))", "apply_rope_split_half_",
                   "baddbmm_", "_freq_table"):
        check("model.py baked opt %s" % marker, marker in vendored_src, True)
    check("llama_base.py baked RMSNorm prepared weight",
          "_prepared_weight" in llama_src, True)
    # int8 lm_head logits: the core patch_logits cannot reach the fork's class
    # copy; without this baked GEMV the int8_convrot 9B dequantizes the head
    # every token (~4x slower: 31 -> 7 it/s on the real checkpoint).
    check("llama_base.py baked int8 logits GEMV",
          "isinstance(weight, QuantizedTensor) and weight.device == input.device" in llama_src, True)
    # The freq-table cache must slice the position axis (dim 2). Slicing the
    # last axis returns an empty tensor once start >= feature dim (~64), which
    # feeds a null pointer to apply_rope_kernel and hangs the GPU on first decode.
    check("freq cache slices position axis",
          "f[:, :, start:start + n]" in vendored_src, True)
    check("freq cache does not slice last axis",
          "f[..., start:start + n]" in vendored_src, False)
    # STRUCT layer was ported from upstream master; weight-key dataclass stays pinned.
    check("Qwen35Config still pinned to fork base",
          norm(def_source(vendored_src, "Qwen35Config")) == norm(def_source(ref_src, "Qwen35Config")), True)

    from vendor.qwen35 import model as vm
    for name in ("Qwen35", "Qwen35Transformer", "GatedAttention", "GatedDeltaNet",
                 "te", "tokenizer", "Qwen35Config", "_make_config"):
        check(f"vendored exports {name}", hasattr(vm, name), True)

    # STRUCT layer: te() carries the upstream vocab_size/tokenizer_json port, so
    # the gate wrapper forwards them with zero drift.
    vp = list(inspect.signature(vm.te).parameters)
    check("te() has ported struct params", vp,
          ["dtype_llama", "llama_quantization_metadata", "model_type", "mtp",
           "vocab_size", "tokenizer_json"])
    check("te() ports embedded-tokenizer config",
          "config[\"stop_tokens\"]" in def_source(vendored_src, "te"), True)

    v.install(gate=lambda: True)
    v._drift.clear()
    try:
        v._te(model_type="qwen35_2b", vocab_size=248320, tokenizer_json=None)
        check("ported te() kwargs do not raise", True, True)
    except TypeError:
        check("ported te() kwargs do not raise", "TypeError", True)
    check("ported kwargs produce no drift", v.boundary_drift(), [])

    # a truly unknown kwarg must still be dropped + warned, not crash
    v._te(model_type="qwen35_2b", future_kwarg=1)
    check("unknown kwarg recorded as drift", v.boundary_drift(), ["future_kwarg"])
    check("summary surfaces drift", "drift=" in v.summary(), True)

    v.install(gate=lambda: False)
    seen = {}
    v.install._core_te = lambda **kw: seen.update(kw) or "core"
    v._te(model_type="qwen35_2b", vocab_size=1, tokenizer_json="x")
    check("gate-off forwards all kwargs to core",
          all(k in seen for k in ("vocab_size", "tokenizer_json")), True)

    # resident graph must wrap the fork's OWN generate methods, not just core's.
    # Without this the non-MTP decode path never drops stale graphs -> HSA fault
    # and a big slowdown (the 9B checkpoint has no MTP weights, so it uses this path).
    rg_src = (NODE_DIR / "patches" / "textgen" / "resident_graph.py").read_text()
    check("resident graph wraps vendored generate methods",
          "_wrap_generate(vendor_llama_base.BaseGenerate)" in rg_src
          and "_wrap_generate(vendor_model.Qwen35)" in rg_src, True)

    if FAILURES:
        print("\nvendor selftest: %d failure(s)" % len(FAILURES))
        return 1
    print("\nvendor selftest: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
