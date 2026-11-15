"""Self-test for the ComfyUI capability probes: no torch and no GPU needed.

    python custom_nodes/ComfyUI-GPU-Tuning/tools/compat_selftest.py

The probes decide which core path the pack delegates to, so they are tested
against the two contract shapes they must tell apart: the 0.39-era symbols and
the 0.44/0.45 replacements.
"""

import sys
import types
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NODE_DIR))

from patches import compat                                          # noqa: E402

FAILURES = []


def check(label, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def fake_modules():
    comfy = types.ModuleType("comfy")
    ops = types.ModuleType("comfy.ops")
    text_encoders = types.ModuleType("comfy.text_encoders")
    llama = types.ModuleType("comfy.text_encoders.llama")
    qwen35 = types.ModuleType("comfy.text_encoders.qwen35")
    ldm = types.ModuleType("comfy.ldm")
    qwen_image21 = types.ModuleType("comfy.ldm.qwen_image21")
    qwen21_model = types.ModuleType("comfy.ldm.qwen_image21.model")

    class CombinedLinear:
        pass

    class FixedKVBias:
        pass

    class FixedKVCache:
        pass

    class GatedAttention:
        def init_kv_cache(self):
            return None

    class Qwen35VisionModel:
        def forward(self, x, grid_thw, premerge=None):
            return None

    class PrefixCache:
        pass

    ops.CombinedLinear = CombinedLinear
    llama.FixedKVBias = FixedKVBias
    llama.FixedKVCache = FixedKVCache
    qwen35.GatedAttention = GatedAttention
    qwen35.Qwen35VisionModel = Qwen35VisionModel
    qwen21_model.PrefixCache = PrefixCache
    text_encoders.llama = llama
    text_encoders.qwen35 = qwen35
    qwen_image21.model = qwen21_model
    comfy.ops = ops
    comfy.text_encoders = text_encoders
    comfy.ldm = ldm
    ldm.qwen_image21 = qwen_image21

    sys.modules.update({
        "comfy": comfy,
        "comfy.ops": ops,
        "comfy.text_encoders": text_encoders,
        "comfy.text_encoders.llama": llama,
        "comfy.text_encoders.qwen35": qwen35,
        "comfy.ldm": ldm,
        "comfy.ldm.qwen_image21": qwen_image21,
        "comfy.ldm.qwen_image21.model": qwen21_model,
    })
    return ops, llama, qwen35, qwen21_model


def main():
    ops, llama, qwen35, qwen21_model = fake_modules()
    check("combined linear", compat.is_combined_linear(ops.CombinedLinear()), True)
    check("plain linear", compat.is_combined_linear(object()), False)
    check("vision premerge", compat.vision_premerge(), True)
    check("qwen35 native", compat.qwen35_native(), True)
    check("fixed kv bias", compat.fixed_kv_bias() is llama.FixedKVBias, True)
    check("fixed kv cache", compat.fixed_kv_cache(), True)
    check("qwen21 prefix cache", compat.qwen21_prefix_cache(), True)

    qwen35.Qwen35VisionModel.forward = lambda self, x, grid_thw: None
    check("vision legacy", compat.vision_premerge(), False)

    delattr(qwen35.GatedAttention, "init_kv_cache")
    check("qwen35 legacy", compat.qwen35_native(), False)

    delattr(llama, "FixedKVBias")
    check("fixed kv bias removed", compat.fixed_kv_bias() is None, True)
    check("fixed kv cache remains", compat.fixed_kv_cache(), True)

    if FAILURES:
        print("\ncompat selftest: %d failure(s)" % len(FAILURES))
        return 1
    print("\ncompat selftest: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
