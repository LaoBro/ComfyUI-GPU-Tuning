"""Runtime tuning patches for gfx906 (MI50 / MI60 / Radeon VII).

Two independent patches, both skipped on other architectures and both opt-out
with COMFY_GFX906_TUNING=off:

* int8 linear: the comfy-kitchen triton autotune list uses CDNA2+/CUDA tile
  shapes (128x256x64, 8 warps). gfx906 has 256 VGPR/thread and no matrix cores,
  so a 128x256x32 tile with 4 warps and 2 stages is ~1.15x faster than every
  config in that list, on every layer shape of a quantized DiT.
* flash attention: Triton's AMD kernel is only fast on fp16 here (bf16 is
  emulated in fp32, fp32 never reaches the packed FMA). Casting at the
  attention boundary and casting the result back is ~2x on head_dim 128.
"""

import logging
import os
import sys

import torch

try:
    from .profiler import patch_prompt_end as _patch_prompt_end
    from .profiler import set_enabled as _set_profiler
    from .profiler import enabled as _profiler_enabled
except ImportError:
    from profiler import patch_prompt_end as _patch_prompt_end
    from profiler import set_enabled as _set_profiler
    from profiler import enabled as _profiler_enabled

INT8_TILE = {"block_m": 128, "block_n": 256, "block_k": 32, "group_size_m": 8}
INT8_WARPS = 4
INT8_STAGES = 2
INT8_KERNELS = ("_int8_matmul_dequant_per_row_kernel", "_int8_matmul_dequant_kernel")
# Text generation decodes one token at a time: m=1 against the whole weight, which
# is a GEMV, not a GEMM. The 128-row tile that wins at DiT token counts computes
# 64x the MACs for one row, so small batches get their own tile. At m=1 the
# narrow N tile is what matters, not the row count: a 256-wide tile leaves a
# 4096-wide weight with only 16 programs for 60 CUs. Measured at m=1 on
# 4096x4096 (bn 256/128/64/32): 0.62 / 0.40 / 0.37 / 0.30 ms; bn32 also wins on
# 12288, 24576 and 4096x12288, and only the 152k lm_head prefers bn128 by 3%.
INT8_TILE_SMALL = {"block_m": 16, "block_n": 32, "block_k": 64, "group_size_m": 8}
INT8_SMALL_WARPS = 4
INT8_SMALL_STAGES = 3
INT8_SMALL_MAX_M = 8
# Decode is m=1: no row tile at all, one program per 16 output rows walking K.
# See int8_gemv.py; it streams the weight once instead of computing a 16x64x32 tile.
INT8_GEMV_MAX_M = 1

# fp16 overflows past 65504; leave headroom for the dot products.
FP16_SAFE_AMAX = 60000.0

# convrot group size the butterfly kernel implements: 256 = 4^4, scaled by 1/16.
CONVROT_GROUP = 256
# Largest K the fused rotate+quantize kernel keeps in registers (8192 = 32 groups).
FUSED_MAX_GROUPS = 32
# convrot modes exposed in the node: fused (butterfly + quantize in one kernel),
# butterfly (two passes), off.
CONVROT_MODES = ("fused", "butterfly", "off")
# Own kernels (exposed in the node). "sdpa" also exists as an internal value:
# the auto-tune fallback when a tile loses to SDPA on an untested card.
ATTENTION_KERNELS = ("hand", "int8", "int8qk", "gemm")

# Node switches are three-state: "default" keeps the startup/env value,
# "on"/"off" are explicit overrides. None is the internal spelling of "default".
TRISTATE = ("default", "on", "off")
# Text-generation group: the CLIP node's textgen switch covers all four.
TEXTGEN_ITEMS = ("deltanet", "kvdecode", "llama", "qwen35")


def stock_backend(name):
    """The ComfyUI attention function behind a backend name, or None for our own modes.

    "basic" is not in REGISTERED_ATTENTION_FUNCTIONS: it is what core falls back
    to when no --use-* flag was passed, so it is added by hand.
    """
    from comfy.ldm.modules import attention

    if name == "basic":
        return attention.attention_basic
    return attention.REGISTERED_ATTENTION_FUNCTIONS.get(name)


def backend_options():
    """Backend names ComfyUI registered at startup, for the settings combos."""
    from comfy.ldm.modules import attention

    return ["basic"] + sorted(attention.REGISTERED_ATTENTION_FUNCTIONS)

_state = {
    "applied": False,
    "arch": None,
    "int8_linear": False,
    "int8_tile_replaced": None,
    "int8_tile_small": None,
    "int8_tile_switch": False,
    "attention": False,
    "convrot": False,
    "convrot_mode": None,
    "attn_cast": None,
    "attn_kernel": None,
    "attn_backend_fn": None,
    "attn_tuned": set(),
    "attn_fp16": True,
    "smooth": True,
    "te_int8": False,
    "te_patched": False,
    "te_attention": "default",
    "triton_backend": None,
    "replaced_backend": None,
    "default_int8_configs": None,
    "w4a8": None,
    "qkv": False,
    "prefix_kv": False,
    "reason": "",
    "vae_enabled": True,
    "vae_patched": False,
    "gemv_module": None,
    "textgen_env": {},
    "textgen_ready": {},
    "textgen_override": None,
    "prefix_kv_enabled": True,
    "prefix_kv_patched": False,
    "text_gemv_enabled": True,
    "te_gemv_calls": 0,
    "te_plain_calls": 0,
    "w4a8_module": None,
    "w4a8_calls": 0,
    "amax_guard": True,
    "int8_tile_enabled": True,
    "small_batch_enabled": True,
    "tile_measured": False,
    "native": None,
}


def enabled():
    return os.getenv("COMFY_GFX906_TUNING", "1").lower() not in ("0", "off", "false", "no")


def env_flag(name, default="1"):
    return os.getenv(name, default).strip().lower() not in ("0", "off", "false", "no")


def flag_value(value):
    """None means "follow the startup/env value"; True/False are explicit."""
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("default", ""):
        return None
    if text in ("on", "1", "true", "yes"):
        return True
    if text in ("off", "0", "false", "no"):
        return False
    return None


def resolve_flag(value, env_value):
    want = flag_value(value)
    return bool(env_value) if want is None else want


def textgen_requested(name):
    override = _state.get("textgen_override")
    if override is not None:
        return override
    return bool(_state.get("textgen_env", {}).get(name, True))


def textgen_active(name):
    return bool(_state.get("textgen_ready", {}).get(name)) and textgen_requested(name)


def textgen_summary():
    ready = _state.get("textgen_ready", {})
    return ", ".join(
        f"{name} {'on' if textgen_active(name) else ('off (installed)' if ready.get(name) else 'n/a')}"
        for name in TEXTGEN_ITEMS)


def _init_env_settings():
    """Diagnostics that have no node switch, read once by apply().

    Everything a node can set lives in the widget; only the benchmark knobs
    (per-item textgen pieces), the manifest force list, and the register-
    dequant gate override stay here.
    """
    _state["textgen_env"] = {
        "deltanet": env_flag("COMFY_GFX906_DELTANET"),
        "kvdecode": env_flag("COMFY_GFX906_KVDECODE"),
        "llama": env_flag("COMFY_GFX906_LLAMA"),
        "qwen35": env_flag("COMFY_GFX906_QWEN35"),
    }
    _state["textgen_ready"] = {name: False for name in TEXTGEN_ITEMS}
    _state["textgen_override"] = None
    _state["forced_items"] = {item.strip().lower()
                              for item in os.getenv("COMFY_GFX906_FORCE", "").replace(",", " ").split()
                              if item.strip()}
    _state["te_dequant_env"] = os.getenv("COMFY_GFX906_TE_DEQUANT", "").strip().lower()


def arch_name():
    """The device identity for this process: a gcn target on AMD, sm<major><minor>
    elsewhere. Mirrors kernels.manifest.device_arch -- duplicated because the
    kernels package is optional and the node must not depend on it for identity.
    """
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    name = str(getattr(props, "gcnArchName", ""))
    if name:
        return name.split(":")[0] or None
    return f"sm{props.major}{props.minor}" if props.major else None


def is_amd():
    return torch.version.hip is not None and torch.cuda.is_available()


def native_backend():
    """The comfy-kitchen backend that already covers this device, or None.

    The pack exists because the HIP backend has no code object for gfx906. On a
    card where kitchen registers its own backend, dispatch prefers it and our
    triton kernel patches only serve the calls its constraints decline.
    """
    try:
        from comfy_kitchen import registry
    except ImportError:
        return None
    name = "hip" if torch.version.hip is not None else "cuda"
    return name if registry.is_available(name) else None


def native_covers(op):
    """Whether the native backend serves the kitchen op `op`.

    Only matters where a patch bypasses dispatch (the VAE rewrite intercepts
    ComfyUI before kitchen is called) or where bench asks whether a site's
    records would ever be read (the int8 tile). The triton kernel patches do
    not gate on this: dispatch itself decides when they run.
    """
    backend = native_backend()
    if backend is None:
        return False
    from comfy_kitchen import registry

    caps = registry.list_backends().get(backend, {}).get("capabilities") or ()
    return op in caps


# The code paths that only exist because of what was measured on a gfx906.
# Splitting them lets a card opt into one of them without claiming to be a
# gfx906: installing the DeltaNet replacement on a card whose comfy-kitchen HIP
# op works fine would replace a working fused kernel with our slower Triton one,
# which is exactly the silent slowdown this pack is supposed to avoid.
def feature_on(item):
    """The tier-3 gate: feature.<item> in the arch manifest, off without data.

    A feature whose fastest implementation differs per card has no default
    without data: an unmeasured card keeps the stock path, and one bench run
    turns on whatever won against the real fallback on this card.
    """
    table = measured_table()
    return table is not None and table.default_impl("feature." + item) == "on"


def resolved_convrot_mode():
    """The manifest's convrot.mode > "fused"."""
    table = measured_table()
    mode = table.default_impl("convrot.mode") if table is not None else None
    return mode if mode in CONVROT_MODES else "fused"


def install_ready(item):
    """Whether an architecture-specific code path should be installed here.

    * `COMFY_GFX906_FORCE`: one path opted in by hand, no questions asked;
    * deltanet also installs wherever comfy-kitchen has no fused gated-delta
      op -- the replacement exists for exactly that case, and the structural
      fact does not go stale the way a timing does;
    * everything else needs `[sites."feature.<item>"] default = "on"`: the
      shipped table for measured archs, or a local tools/bench.py run. No
      data means off -- the stock path is the default.
    """
    if item in _state.get("forced_items", set()):
        return True
    if item == "qwen35":
        # the bench case only covers prefill, so its timing cannot write this
        # feature yet; the env switch is the decision. It is an equivalence
        # rewrite (every method verified against core), hence default on.
        return bool(_state.get("textgen_env", {}).get("qwen35"))
    if item == "deltanet":
        try:
            import comfy_kitchen

            if not comfy_kitchen.gated_delta_decode_is_available():
                return True
        except ImportError:
            return True
    table = measured_table()
    return table is not None and table.default_impl(f"feature.{item}") == "on"


def warn_forced_items():
    """Say that the items being installed here were not measured on this card."""
    forced = sorted(_state.get("forced_items", set()))
    if forced:
        logging.warning("gfx906: installing %s on %s because of COMFY_GFX906_FORCE; these kernels "
                        "were not measured on this card, expect the untested ones to be slower or wrong",
                        ", ".join(forced), _state.get("arch"))


def _triton_available(patch):
    try:
        import triton  # noqa: F401
    except ImportError:
        logging.warning("gfx906: triton is not importable, %s patch skipped", patch)
        return False
    return True


def _parse_tile_spec(raw):
    """'block_m=128,block_n=256,block_k=32,...' -> (kwargs, warps, stages), or None."""
    try:
        values = {}
        for part in raw.split(","):
            key, _, value = part.partition("=")
            values[key.strip()] = int(value)
        spec = {k: values[k] for k in ("block_m", "block_n", "block_k")}
        spec["group_size_m"] = values.get("group_size_m", 8)
        warps = values.get("warps", values.get("num_warps", 4))
        stages = values.get("stages", values.get("num_stages", 2))
        return spec, warps, stages
    except (KeyError, ValueError):
        return None


def _tile_spec(name):
    """Parse e.g. COMFY_GFX906_TILE_CONFIG="block_m=128,block_n=256,block_k=32,..."."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    spec = _parse_tile_spec(raw)
    if spec is None:
        logging.warning("gfx906: %s=%r is not a tile spec, using the built-in tile", name, raw)
    return spec


def _triton_config():
    import triton

    spec = _tile_spec("COMFY_GFX906_TILE_CONFIG")
    if spec is None:
        spec = measured_tile("large")
    if spec is not None:
        return triton.Config(dict(spec[0]), num_warps=spec[1], num_stages=spec[2])
    return triton.Config(dict(INT8_TILE), num_warps=INT8_WARPS, num_stages=INT8_STAGES)


def _int8_kernels():
    from comfy_kitchen.backends.triton import quantization

    for name in INT8_KERNELS:
        kernel = getattr(quantization, name, None)
        if kernel is not None and getattr(kernel, "configs", None):
            yield kernel


def _small_tile_config():
    import triton

    spec = _tile_spec("COMFY_GFX906_SMALL_TILE_CONFIG")
    if spec is None:
        spec = measured_tile("small")
    if spec is None and measured_tile("large") is not None:
        # a table that measured only the large tile must not fall back to the
        # narrow gfx906 constant, which was tuned for 60 CUs
        return _triton_config()
    if spec is not None:
        return triton.Config(dict(spec[0]), num_warps=spec[1], num_stages=spec[2])
    return triton.Config(dict(INT8_TILE_SMALL), num_warps=INT8_SMALL_WARPS, num_stages=INT8_SMALL_STAGES)


def select_int8_tile(m):
    """Offer the tile that matches the batch width this call has.

    Autotune caches its pick per (m, n, k), and a shape never moves between the
    two buckets, so swapping the list only happens when the bucket changes: no
    sweep is paid for the decode shapes, and the DiT keeps the 128-row tile.
    Off while the stock list is restored for a measurement, and on cards whose
    best tile we have not measured: there both tiles stay autotune candidates.
    """
    if not _state.get("int8_tile_switch"):
        return
    if not _state.get("int8_tile_replaced"):
        return
    small = _state.get("small_batch_enabled", True) and m <= INT8_SMALL_MAX_M
    if _state.get("int8_tile_small") == small:
        return
    _state["int8_tile_small"] = small
    for kernel in _int8_kernels():
        kernel.configs = [_small_tile_config()] if small else [_triton_config()]


def load_int8_gemv():
    """Import the decode GEMV kernel module shipped next to this file."""
    try:
        from . import int8_gemv
        return int8_gemv
    except ImportError:
        pass
    try:
        import int8_gemv
        return int8_gemv
    except ImportError:
        return None


def load_w4a8_gemv():
    """Import the 4-bit weight kernel module shipped next to this file."""
    try:
        from . import w4a8_gemv
        return w4a8_gemv
    except ImportError:
        pass
    try:
        import w4a8_gemv
        return w4a8_gemv
    except ImportError:
        return None


def load_manifest():
    """The measurement-manifest module, or None when the package is incomplete."""
    try:
        from .kernels import manifest
        return manifest
    except ImportError:
        pass
    try:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from kernels import manifest
        return manifest
    except ImportError:
        return None


def load_registry():
    """The kernel-registry module, or None when the package is incomplete."""
    try:
        from .kernels import registry
        return registry
    except ImportError:
        pass
    try:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from kernels import registry
        return registry
    except ImportError:
        return None


def _install(label, fn, *args, **kwargs):
    """Run one patch; a device-level failure is logged and remembered, not fatal.

    A kernel that cannot build on this card must not stop ComfyUI from starting,
    but it must not vanish either: the label lands in `_state["install_failures"]`
    and `status()` prints it. Anything that is not a device/build error (a typo,
    a wrong call) still raises -- swallowing those would turn this pack into a
    machine for hiding regressions.
    """
    registry = load_registry()
    if registry is None:                       # kernels package missing: run unguarded
        return fn(*args, **kwargs)
    result = registry.safe_install(label, fn, *args, **kwargs)
    if result is registry.FAILED:
        failures = _state.setdefault("install_failures", [])
        if label not in failures:
            failures.append(label)
        return None
    return result


def measured_table():
    """The measurement manifest for this arch, or None. Loaded once."""
    if "manifest_data" not in _state:
        module = load_manifest()
        arch = _state.get("arch") or ""
        _state["manifest_data"] = module.load(arch) if (module is not None and arch) else None
    return _state["manifest_data"]


def measured_tile(bucket):
    """The int8 tile measured on this card ("large" / "small"), or None.

    Stored as the same "block_m=...,warps=...,stages=..." string the startup
    variables take, so an entry can be pasted into COMFY_GFX906_TILE_CONFIG and
    a card that has run tools/bench.py skips autotune without any env variable.
    """
    table = measured_table()
    raw = table.config("int8_gemm.tile", bucket) if table is not None else None
    if raw is None:
        return None
    spec = _parse_tile_spec(raw)
    if spec is None:
        logging.warning("gfx906: manifest int8_gemm.tile/%s=%r is not a tile spec, ignoring",
                        bucket, raw)
    return spec


def measured_state():
    """How much of the table can be used here: measured / stale / absent / off.

    stale is the one that matters: a file exists and looks authoritative, but it
    was measured under a different ROCm/Torch/Triton build, so every number in
    it is being ignored without anyone noticing unless we say so.
    """
    if "measured_state" not in _state:
        if not env_flag("COMFY_GFX906_MEASURED"):
            _state["measured_state"] = "off"
        else:
            data = measured_table()
            if data is None:
                _state["measured_state"] = "absent"
            else:
                _state["measured_state"] = "measured" if data.matches() else "stale"
    return _state["measured_state"]


def warn_measured_state():
    """Say out loud when the measured layer is not in use.

    Silence here is the failure mode this is about: a card that quietly runs
    probes and autotune looks exactly like a card that is using its numbers.
    """
    state = measured_state()
    data = measured_table()
    arch = _state.get("arch")
    if state == "measured":
        if not _state["native"] and not _state.get("tile_measured"):
            logging.warning("gfx906: measured data for %s has no int8 tile, so the tile list "
                            "still goes to autotune. Run tools/bench.py --all --write on this card.",
                            arch)
        return
    if state == "off":
        logging.info("gfx906: measured table disabled by COMFY_GFX906_MEASURED, "
                     "int8 tiles and GEMV backends fall back to autotune/probes")
    elif state == "stale":
        logging.warning("gfx906: the measured data for %s was taken under %s but this process runs %s: "
                        "ignoring it, so int8 tiles go to autotune and the GEMV backends to probes. "
                        "Re-run tools/bench.py --write to refresh it.",
                        arch, data.fingerprint if data else "?", data.runtime_fp if data else "?")
    elif _state["native"]:
        logging.warning("gfx906: no measured data for %s: the GEMV backends come from probes "
                        "(the first call of each new shape pays the sweep). "
                        "Run tools/bench.py --write once on this card to fix that.", arch)
    else:
        logging.warning("gfx906: no measured data for %s: int8 tiles come from the stock autotune list "
                        "and the GEMV backends from probes (the first call of each new shape pays the "
                        "sweep). Run tools/bench.py --write once on this card to fix that.", arch)


def manifest_status_line():
    """One line about arch/<gcn>/manifest.toml, or "" when there is none.

    Loaded once per process: the file is read from disk and its fingerprint
    compared, which is not something to repeat on every settings refresh.
    """
    if not _state.get("arch"):
        return ""
    if "manifest_line" not in _state:
        data = measured_table()
        line = data.status_line() if data is not None else ""
        if line and measured_state() == "off":
            line = line.replace(", in use", ", layer off (COMFY_GFX906_MEASURED)")
        _state["manifest_line"] = line
    return _state["manifest_line"]


def measured_suffix(module):
    """How many calls were served from the table, e.g. "measured 12/14"."""
    registry = getattr(module, "_REGISTRY", None)
    hits = getattr(registry, "hits", None) or {}
    measured = hits.get("measured", 0)
    if not measured:
        return ""
    total = measured + hits.get("probe", 0)
    return f"measured {measured}/{total}"


def patch_kv_decode(gate=None):
    """4-D GQA matmul in FixedKVBias.decode.

    The 5-D form broadcasts the KV over the GQA groups, and ATen answers that by
    materializing a transposed copy of the whole capacity-sized cache once per
    layer per step: 37 MB at a 2.3k prompt, 0.73 ms per layer here. Folding the
    groups into the query rows removes the copy; the math is unchanged (bmm
    blocking order only, <= 7e-7).

    ``gate`` is read per call, so the node switch can fall back to the shipped
    decode without a reload.
    """
    import comfy_kitchen
    import comfy.text_encoders.llama as llama

    kv = llama.FixedKVBias
    if getattr(kv.decode, "_gfx906_folded", False):
        return True

    original = kv.decode

    def decode(self, xq, xk, xv, num_kv_heads):
        if gate is not None and not gate():
            return original(self, xq, xk, xv, num_kv_heads)
        batch_size, num_heads, seq, head_dim = xq.shape
        self.key.index_copy_(2, self.position[:seq], xk)
        self.value.index_copy_(2, self.position[:seq], xv)
        if seq == 1 and self.seqlen is not None:
            out = comfy_kitchen.flash_attention_decode(xq.transpose(1, 2), self.key.transpose(1, 2), self.value.transpose(1, 2), self.seqlen)
            return out.reshape(batch_size, seq, num_heads * head_dim)
        groups = num_heads // num_kv_heads
        q = xq.reshape(batch_size, num_kv_heads, groups * seq, head_dim) * head_dim ** -0.5
        bias = self.bias[..., self.bias.shape[-2] - seq:, :].unsqueeze(1)
        scores = torch.matmul(q, self.key.transpose(-1, -2)).view(batch_size, num_kv_heads, groups, seq, -1).add_(bias)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)
        out = torch.matmul(probs.reshape(batch_size, num_kv_heads, groups * seq, -1), self.value)
        out = out.view(batch_size, num_kv_heads, groups, seq, head_dim)
        return out.permute(0, 3, 1, 2, 4).reshape(batch_size, seq, num_heads * head_dim)

    decode._gfx906_folded = True
    kv.decode = decode
    return True


def load_deltanet():
    """Import the DeltaNet decode kernel module shipped next to this file."""
    try:
        from . import deltanet
        return deltanet
    except ImportError:
        pass
    try:
        import deltanet
        return deltanet
    except ImportError:
        return None


def load_llama_patches():
    """Import the core llama.py hot-path replacements shipped next to this file."""
    try:
        from . import core_llama_patches
        return core_llama_patches
    except ImportError:
        pass
    try:
        import core_llama_patches
        return core_llama_patches
    except ImportError:
        return None


def load_qwen35_model():
    """Import the Qwen3.5 compute classes shipped next to this file."""
    try:
        from . import qwen35_model
        return qwen35_model
    except ImportError:
        pass
    try:
        import qwen35_model
        return qwen35_model
    except ImportError:
        return None


def _gemv_rotation_ok(groupsize, device, dtype):
    """Whether comfy-kitchen can build a Hadamard matrix this size (power of 4)."""
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    known = _state.setdefault("gemv_rotation", {})
    ok = known.get(groupsize)
    if ok is None:
        try:
            _build_hadamard(groupsize, device=device, dtype=dtype)
            ok = True
        except ValueError:
            ok = False
        known[groupsize] = ok
    return ok


def select_int8_gemv(x, weight, weight_scale, bias, out_dtype, input_act, residual, groupsize):
    """Run a single-token int8 linear through int8_gemv, or return None.

    Only the shapes decode actually has: one row, fp32 activation, per-channel
    or tensorwise weight scale, no fused activation and no residual. Everything
    else keeps the GEMM, which is where prefill belongs anyway.
    """
    gemv = _state.get("gemv_module")
    if gemv is None:
        return None
    if x.dim() != 2 or x.shape[0] != 1 or x.dtype is not torch.float32 or not x.is_contiguous():
        return None
    if weight.dim() != 2 or weight.dtype is not torch.int8 or not weight.is_contiguous():
        return None
    if not isinstance(weight_scale, torch.Tensor) or weight_scale.numel() not in (1, weight.shape[0]):
        return None
    if input_act is not None or residual is not None:
        return None
    if groupsize and not _gemv_rotation_ok(groupsize, x.device, x.dtype):
        return None
    if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None
    return gemv.linear(x, weight, weight_scale, bias, out_dtype, groupsize)


def te_int8_gemv(module, x):
    """Decode path for the text encoder that skips the QuantizedTensor dispatch.

    F.linear on a quantized weight costs 159 us per call: 78 us of subclass
    dispatch plus the custom-op registry lookup, and 81 us that actually reaches
    the kernel. The weight's int8 tensor and scale are readable right here, so a
    single-row call goes straight to the GEMV. Multi-row (prefill) calls and
    anything the GEMV does not support return None and keep the dispatch path.
    """
    if _state.get("gemv_module") is None or not _state.get("text_gemv_enabled", True):
        return None
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    weight = module.weight
    rows = x.reshape(-1, x.shape[-1])
    qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    params = weight._params
    groupsize = getattr(params, "convrot_groupsize", 256) if getattr(params, "convrot", False) else 0
    out = select_int8_gemv(rows, qdata, scale, module.bias, x.dtype, None, None, groupsize)
    if out is None:
        return None
    _state["te_gemv_calls"] = _state.get("te_gemv_calls", 0) + 1
    return out.view(*x.shape[:-1], out.shape[-1])


def _te_w4a8_ready(module, x):
    """True when the packed 4-bit weight can be fed to the w4a8 GEMV."""
    from comfy_kitchen.tensor import QuantizedTensor

    weight = module.weight
    return (
        _state["te_int8"]
        and _state["w4a8"] is not None
        and x.is_cuda
        and isinstance(weight, QuantizedTensor)
        and weight._layout_cls == "AsymW4A8Int8Layout"
        and not getattr(weight._params, "transposed", False)
        and weight.device == x.device
        and (module.bias is None or module.bias.device == x.device)
        and not module.weight_function
        and not module.bias_function
    )


def te_w4a8_gemv(module, x):
    """Decode path for grouped 4-bit text-encoder weights.

    Uniform weights go straight to the register-level nibble unpack. A Lloyd-Max
    codebook (the shipped w4a8 checkpoints) resolves through a 16-entry table:
    eight bytes per `v_perm_b32` and the code's top bit picks which half, so the
    decode stays in the dot. Prefill keeps the int8 route instead, one streaming
    pass per layer, because its GEMM wants a contiguous int8 tile.
    """
    gemv = _state["w4a8"]
    if gemv is None:
        return None
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout

    weight = module.weight
    params = weight._params
    rows = x.reshape(-1, x.shape[-1])
    qdata, s_rel, s_channel, correction, codebook = AsymW4A8Int8Layout.get_plain_tensors(weight)
    if correction is not None:
        return None
    if x.dtype is not torch.float32 or not x.is_contiguous():
        return None
    if rows.shape[0] > gemv.MAX_M:
        w8 = gemv.decode_int8(qdata, s_rel, params.group_size, codebook)
        if w8 is None:
            return None
        out = select_int8_gemv(rows, w8, s_channel, module.bias, x.dtype, None, None,
                               params.convrot_groupsize)
        if out is None:
            from comfy_kitchen.backends.triton import quantization

            out = quantization.int8_linear(rows, w8, s_channel, module.bias, out_dtype=x.dtype,
                                           convrot=True, convrot_groupsize=params.convrot_groupsize)
    else:
        levels = inv_level = None
        if codebook is not None:
            cached = getattr(module, "_gfx906_levels", None)
            if cached is None:
                cached = gemv.codebook_words(codebook)
                module._gfx906_levels = cached
            levels, inv_level = cached
        out = gemv.linear(rows, qdata, s_rel, s_channel, module.bias, x.dtype,
                          params.group_size, params.convrot_groupsize,
                          levels=levels, inv_level=inv_level or 1.0)
    if out is None:
        return None
    _state["w4a8_calls"] = _state.get("w4a8_calls", 0) + 1
    return out.view(*x.shape[:-1], out.shape[-1])


def _same_dtype(out, ref):
    """Keep a fallback linear in the caller's dtype.

    A quantized layout answers F.linear in its own out_dtype (the weight's
    orig_dtype), not in the dtype the text encoder executes in. On gfx906
    should_use_bf16 is False, so generation runs in fp32 while the weights were
    stored with a bf16/fp16 orig_dtype: the projection comes back bf16, the KV
    cache is fp32, and attention fails with "expected scalar type BFloat16 but
    found Float". Every layout that reaches its fallback needs the cast, which is
    why it sits on the wrapper instead of one branch.
    """
    return out if out.dtype == ref.dtype else out.to(ref.dtype)


def te_plain_gemv(module, x, compute_dtype=None, want_requant=False, weight_only_quant=False):
    """Single-row GEMV for unquantized text-encoder weights.

    The checkpoints leave the tiny gate projections (Qwen3.5's in_proj_a/b) in
    bf16, and the fp32 GEMM path costs ~61 us per call at [1, 4096]x[4096, 32]
    even though it is 0.5 MFLOP: dispatch and tile overhead, not math. The same
    dot products through torch.mv are ~14 us including the bf16->fp32 weight
    cast. fp32 accumulation either way; only the sum order differs from the
    GEMM, so results match to rounding.
    """
    if not _state.get("text_gemv_enabled", True):
        return None
    if want_requant or weight_only_quant or compute_dtype not in (None, torch.float32):
        return None
    if x.dtype is not torch.float32 or not x.is_contiguous():
        return None
    rows = x.reshape(-1, x.shape[-1])
    if rows.shape[0] != 1:
        return None
    weight = module.weight
    if weight.dim() != 2 or weight.dtype not in (torch.float16, torch.bfloat16):
        return None
    if module.bias is not None or module.weight_function or module.bias_function:
        return None
    out = torch.mv(weight.to(x.dtype), rows[0])
    _state["te_plain_calls"] = _state.get("te_plain_calls", 0) + 1
    return out.view(*x.shape[:-1], out.shape[-1])


def patch_int8_tile_switch():
    """Pick the int8 tile per call.

    The dispatch resolves the backend function by name on every call, and the
    registered backend is the package module (which re-exports the kernel
    function), not the quantization submodule, so both are wrapped.
    """
    try:
        import comfy_kitchen.backends.triton as triton_backend
        from comfy_kitchen.backends.triton import quantization
    except ImportError:
        return False
    previous = getattr(triton_backend, "int8_linear", None)
    if previous is None:
        return False
    if getattr(previous, "_gfx906_tile_switch", False):
        return True

    _state["gemv_module"] = load_int8_gemv()
    _state["gemv_model"] = feature_on("gemv")

    def int8_linear(x, weight, weight_scale, bias=None, out_dtype=torch.bfloat16, convrot=False,
                    convrot_groupsize=256, input_act=None, input_act_weight=None,
                    input_act_eps=0.0, residual=None, residual_scale=None):
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        if rows <= INT8_GEMV_MAX_M and _state.get("gemv_model"):
            out = select_int8_gemv(x, weight, weight_scale, bias, out_dtype, input_act, residual,
                                   convrot_groupsize if convrot else 0)
            if out is not None:
                return out
        select_int8_tile(rows)
        return previous(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize,
                        input_act, input_act_weight, input_act_eps, residual, residual_scale)

    int8_linear._gfx906_tile_switch = True
    triton_backend.int8_linear = int8_linear
    quantization.int8_linear = int8_linear
    return True


def set_untuned_tiles():
    """Refresh the candidate list after the small_batch switch changes.

    Without a measured tile the list is the stock search: an unmeasured card
    never sees gfx906's tiles, and a measured one has a single config anyway.
    """
    if _state.get("int8_tile_replaced") or not _state.get("int8_linear"):
        return
    if _state.get("default_int8_configs") is None:
        return
    for kernel in _int8_kernels():
        kernel.configs = list(_state["default_int8_configs"])
        kernel.cache.clear()


def patch_int8_linear(replace):
    """Add the gfx906 tile to the autotune sweep (or make it the only candidate).

    The stock list is six CDNA2+/CUDA shapes with 3-4 pipeline stages; none of
    them is 128x256x32 with 2 stages, so autotune picks the best of a list that
    does not contain the winner on this card. Autotune itself is fine - the
    search space is what is wrong. On the measured arch we drop the rest so the
    first prompt does not pay for a sweep whose answer is already known.
    """
    try:
        kernels = list(_int8_kernels())
    except ImportError as exc:
        _state["reason"] = f"comfy-kitchen triton backend unavailable: {exc}"
        return False
    if not kernels:
        _state["reason"] = "comfy-kitchen triton int8 kernels not found"
        return False

    if _state.get("default_int8_configs") is None:
        _state["default_int8_configs"] = list(kernels[0].configs)
    _state["int8_tile_replaced"] = replace
    for kernel in kernels:
        ours = _triton_config()
        if replace:
            kernel.configs = [ours]
        else:
            kernel.configs = list(_state["default_int8_configs"])
        kernel.cache.clear()
    _state["int8_tile_small"] = None
    _state["int8_tile_switch"] = patch_int8_tile_switch()
    _state["int8_linear"] = True
    return True


def restore_int8_defaults():
    """Put the stock autotune list back, for A/B measurements."""
    for kernel in _int8_kernels():
        kernel.configs = list(_state["default_int8_configs"])
        kernel.cache.clear()
    _state["int8_tile_switch"] = False
    _state["int8_tile_small"] = None


def set_int8_tile(enabled):
    """Switch the gfx906 tile list on/off without a reload."""
    if enabled:
        if _state.get("int8_linear") and _state.get("default_int8_configs") is not None:
            patch_int8_linear(replace=bool(_state.get("tile_measured")))
    elif _state.get("default_int8_configs") is not None:
        restore_int8_defaults()


def _sdpa(q, k, v, softmax_scale):
    """Plain SDPA on the [B, S, H, D] layout the flash wrapper is called with.

    contiguous() matters: SDPA on a transposed view picks a 4x slower kernel.
    """
    qh, kh, vh = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
    kwargs = {} if softmax_scale is None or softmax_scale <= 0 else {"scale": softmax_scale}
    if qh.shape[1] != kh.shape[1]:
        kwargs["enable_gqa"] = True
    return torch.nn.functional.scaled_dot_product_attention(
        qh, kh, vh, dropout_p=0.0, is_causal=False, **kwargs).transpose(1, 2)


# Bytes of fp16 scores kept in flight by the gemm mode. The score matrix is the
# only large temporary; on a 32 GB card 512 MiB sits well inside what ComfyUI
# leaves free, and larger groups only buy ~4% (8 heads 52.5 ms, 16 heads 51.1).
GEMM_SCORE_BUDGET = 512 << 20


def _gemm_ok(q, k, v):
    """Plain MHA in any float dtype: bmm cares about shapes only.

    The Triton kernels' head-dim list does not apply here, so attention with a
    head dim we never compiled for (SD1.5's 40, SD2.1's 80) still reaches gemm.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    b, _, h, d = q.shape
    return (k.shape[0] == b and v.shape[0] == b
            and k.shape[1] == v.shape[1] and k.shape[1] > 0
            and k.shape[2] == h and v.shape[2] == h
            and k.shape[3] == d and v.shape[3] == d)


def _gemm_attention(q, k, v, softmax_scale):
    """fp16 attention written as bmm -> softmax -> bmm, no fused kernel.

    rocBLAS beats our fp16 Triton kernel on this card (5.0-8.7 vs 4.6 TMAC/s), so
    the plain three-stage form wins as long as nothing extra is copied: the
    transpose to per-head [BH, S, D] happens once, the scale rides on q instead of
    the score matrix, and softmax runs in place. Heads are grouped to bound the
    score buffer. 52 ms at the Qwen shape vs 76 ms for attention_basic/split and
    77.5 ms for the int8 Triton kernel.

    When the score matrix is small (SDXL's 1280 blocks: 42 MB) the 4D matmul form
    on permuted views wins instead - the strided batched GEMM needs no copies, and
    the copies cost more than they save there (0.81 vs 0.87 ms at H=20 S=1024).
    At the 640 blocks (335 MB) the grouped form is faster (6.78 vs 9.32 ms).
    """
    b, sq, h, d = q.shape
    sk = k.shape[1]
    scale = d ** -0.5 if softmax_scale is None else softmax_scale
    if b * h * sq * sk * 2 <= 64 << 20:
        qs = q.transpose(1, 2) * scale
        s = torch.matmul(qs, k.permute(0, 2, 3, 1))
        torch.softmax(s, -1, out=s)
        return torch.matmul(s, v.transpose(1, 2)).transpose(1, 2)
    bh = b * h
    # reshape keeps the transposed strides, so contiguous() here is not optional:
    # a strided A operand measures 1.9x slower through bmm.
    qh = q.transpose(1, 2).reshape(bh, sq, d).contiguous()
    qh.mul_(scale)
    kh = k.transpose(1, 2).reshape(bh, sk, d).transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).reshape(bh, sk, d).contiguous()
    out = torch.empty_like(qh)
    group = max(1, min(bh, GEMM_SCORE_BUDGET // (sq * sk * 2)))
    scores = torch.empty((group, sq, sk), dtype=q.dtype, device=q.device)
    for h0 in range(0, bh, group):
        n = min(group, bh - h0)
        s = scores[:n]
        torch.bmm(qh[h0:h0 + n], kh[h0:h0 + n], out=s)
        torch.softmax(s, -1, out=s)
        torch.bmm(s, vh[h0:h0 + n], out=out[h0:h0 + n])
    return out.view(b, h, sq, d).transpose(1, 2)


def _swap_optimized_attention(old, new):
    """Repoint modules that did `from ... import optimized_attention` before us.

    Only the two names that mean "the default backend" are touched, so the
    registered backend table (attention_pytorch etc.) stays intact.
    """
    if old is new:
        return
    for name in ("optimized_attention", "optimized_attention_masked"):
        for module in list(sys.modules.values()):
            if getattr(module, name, None) is old:
                setattr(module, name, new)


def patch_attention():
    """Run eligible attention calls on the gfx906 kernel, delegate the rest.

    Installed as optimized_attention, re-wrapped with ComfyUI's own wrap_attn so
    transformer_options["optimized_attention_override"] and ComfyAttention still
    take precedence, and given a container_function - the same seam
    attention_comfy_kitchen_int8 uses. A call that passes AttentionTensorContainer
    reaches us before the containers are taken; every ineligible call goes to the
    backend ComfyUI picked instead of being forced onto one.
    """
    from comfy.ldm.modules import attention

    if getattr(attention, "_gfx906_attention", False):
        return True
    if not _triton_available("attention"):
        return False
    try:
        from .attention_kernel import (SUPPORTED_HEAD_DIMS, attention_fwd, attention_fwd_int8,
                                       supported as kernel_supported)
    except ImportError:
        from attention_kernel import (SUPPORTED_HEAD_DIMS, attention_fwd, attention_fwd_int8,
                                      supported as kernel_supported)

    fallback = attention.optimized_attention
    if fallback in (attention.attention_sage, attention.attention_comfy_kitchen_int8):
        logging.info("gfx906: %s was requested explicitly, leaving attention alone",
                     getattr(fallback, "__name__", "?"))
        return False

    kernel = "int8"
    _state["attn_kernel"] = kernel if kernel in ATTENTION_KERNELS or stock_backend(kernel) is not None else "int8"
    _state["attn_backend_fn"] = stock_backend(_state["attn_kernel"])
    _state["smooth"] = True
    # the tile constants were hand-measured on gfx906; the mode is a quality
    # choice (rel 2e-2 int8 vs 2.5e-4 hand), so it stays a manual node
    # switch instead of a manifest verdict
    _state["attn_tuned"] = set(SUPPORTED_HEAD_DIMS)
    _state["attn_fp16"] = True
    _state["attn_contiguous"] = True
    stats = _state["attn_cast"] = {"fp16": 0, "fp32": 0, "hand": 0, "int8": 0, "int8qk": 0,
                                   "gemm": 0, "sdpa": 0}

    def run(q, k, v, softmax_scale):
        """[B, S, H, D] -> same, or None to delegate. fp16 unless the cast is off."""
        mode = _state["attn_kernel"]
        if mode == "sdpa":
            stats["sdpa"] += 1
            return _sdpa(q, k, v, softmax_scale)
        if mode == "gemm":
            if not _gemm_ok(q, k, v):
                return None
            stats["gemm"] += 1
            return _gemm_attention(q, k, v, softmax_scale)
        if not kernel_supported(q, k, v):
            return None
        if mode == "int8":
            stats["int8"] += 1
            return attention_fwd_int8(q, k, v, softmax_scale)
        if mode == "int8qk":
            stats["int8qk"] += 1
            return attention_fwd_int8(q, k, v, softmax_scale, int8_pv=False, smooth=_state["smooth"])
        stats["hand"] += 1
        return attention_fwd(q, k, v, softmax_scale)

    def to_head_major(q, k, v, heads, skip_reshape):
        """-> [B, S, H, D] contiguous, or None when this call is not ours.

        The result is fp16 unless the fp16 cast is switched off, in which case
        it stays in the dtype it came in and the caller runs SDPA on it.
        """
        dim_head = q.shape[-1] if skip_reshape else q.shape[-1] // heads
        # gemm is plain bmm, so it also takes head dims the Triton kernels were never compiled for
        if dim_head not in SUPPORTED_HEAD_DIMS and _state["attn_kernel"] != "gemm":
            return None
        if q.dtype in (torch.float32, torch.bfloat16):
            if _state["attn_fp16"]:
                if _state.get("amax_guard", True):
                    # vector_norm(inf) is max(abs()) without materializing the 66 MB temporary
                    amax = max(torch.linalg.vector_norm(t, ord=float("inf")).item() for t in (q, k, v))
                    if amax >= FP16_SAFE_AMAX:
                        if not stats["fp32"]:
                            logging.warning("gfx906: attention amax %.0f exceeds %d, leaving the inputs alone",
                                            amax, FP16_SAFE_AMAX)
                        stats["fp32"] += 1
                        return None
                q, k, v = q.half(), k.half(), v.half()
                stats["fp16"] += 1
        elif q.dtype != torch.float16:
            return None
        if skip_reshape:
            return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
        b, s = q.shape[0], q.shape[1]
        # head counts may differ (GQA); kernel_supported() rejects that instead of view() failing
        return (q.reshape(b, s, -1, dim_head),
                k.reshape(k.shape[0], k.shape[1], -1, dim_head),
                v.reshape(v.shape[0], v.shape[1], -1, dim_head))

    def contiguous_head_major(q, k, v, heads, skip_reshape):
        """[B,H,S,D] physically contiguous, cast per the fp16 switch, or None.

        The measurement tool for the transpose-without-contiguous issue: core's
        attention_pytorch/sub_quad hand SDPA/bmm transpose *views*, which on
        gfx906 lands 4-15x off the contiguous path.
        """
        prepared = to_head_major(q, k, v, heads, skip_reshape)
        if prepared is None:
            return None
        return tuple(t.transpose(1, 2).contiguous() for t in prepared)

    def gfx906_attention(q, k, v, heads, mask=None, attn_precision=None,
                         skip_reshape=False, skip_output_reshape=False, **kwargs):
        backend = _state["attn_backend_fn"]
        if backend is not None:
            if _state["attn_contiguous"]:
                prepared = contiguous_head_major(q, k, v, heads, skip_reshape)
                if prepared is not None:
                    out = backend(prepared[0], prepared[1], prepared[2], heads, mask=mask,
                                  attn_precision=attn_precision, skip_reshape=True,
                                  skip_output_reshape=True, **kwargs)
                    if out.dtype != q.dtype:
                        out = out.to(q.dtype)
                    if skip_output_reshape:
                        return out
                    out = out.transpose(1, 2)
                    return out.reshape(out.shape[0], out.shape[1], heads * out.shape[3])
            # masks and GQA are the backend's business
            return backend(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                           skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
        if _state["attn_kernel"] not in ATTENTION_KERNELS and _state["attn_kernel"] != "sdpa":
            # untrusted combo value: whatever core picked at startup
            return fallback(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                            skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
        if (mask is None and not kwargs.get("enable_gqa", False)
                and all(isinstance(t, torch.Tensor) for t in (q, k, v))
                and kwargs.get("low_precision_attention", True) is not False):
            out_dtype = q.dtype
            mode = _state["attn_kernel"]
            prepared = to_head_major(q, k, v, heads, skip_reshape)
            if prepared is not None:
                if prepared[0].dtype == torch.float16:
                    out = run(prepared[0], prepared[1], prepared[2], kwargs.get("scale"))
                elif mode == "sdpa":
                    # fp16 cast switched off: SDPA in whatever dtype the model uses
                    stats["sdpa"] += 1
                    out = _sdpa(prepared[0], prepared[1], prepared[2], kwargs.get("scale"))
                else:
                    # the kernels are fp16-only; run nothing rather than silently
                    # benchmarking SDPA under the name of the selected mode
                    raise RuntimeError(
                        f"gfx906: attention mode '{mode}' only runs on fp16 inputs but the fp16 cast is "
                        "off; turn the fp16 switch back on or pick a backend name")
                if out is not None:
                    if out.dtype != out_dtype:
                        out = out.to(out_dtype)
                    if skip_output_reshape:
                        return out.transpose(1, 2)
                    return out.reshape(out.shape[0], out.shape[1], heads * out.shape[3])
        return fallback(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

    def gfx906_attention_containers(q, k, v, heads, **kwargs):
        return gfx906_attention(q.take(), k.take(), v.take(), heads, **kwargs)

    wrapped = attention.wrap_attn(gfx906_attention)
    wrapped.container_function = gfx906_attention_containers
    _swap_optimized_attention(fallback, wrapped)
    attention.optimized_attention = wrapped
    attention.optimized_attention_masked = wrapped
    attention._gfx906_attention = True
    _state["attention"] = True
    _state["replaced_backend"] = getattr(fallback, "__name__", "?")
    return True


def patch_convrot():
    """Replace the O(n^2) convrot rotation matmul with an O(n log n) butterfly."""
    from comfy_kitchen.backends.triton import quantization

    if not _triton_available("convrot"):
        _state["convrot_mode"] = "off"
        return False
    try:
        from .convrot_kernel import QuantizedRotation, convrot_quantize, hadamard_rotate
    except ImportError:
        from convrot_kernel import QuantizedRotation, convrot_quantize, hadamard_rotate

    _state["convrot_mode"] = resolved_convrot_mode()
    if getattr(quantization, "_gfx906_convrot", False):
        return True
    original = getattr(quantization, "_rotate_activation", None)
    original_quantize = getattr(quantization, "triton_quantize_rowwise", None)
    if original is None or original_quantize is None:
        logging.warning("gfx906: comfy-kitchen has no _rotate_activation/triton_quantize_rowwise, "
                        "convrot patch skipped (the rotation falls back to its matmul)")
        _state["convrot_mode"] = "off"
        return False
    _state["original_rotate"] = original

    def rotate_activation(x, h, group_size):
        mode = _state["convrot_mode"]
        if mode == "off" or group_size != CONVROT_GROUP or x.shape[-1] % CONVROT_GROUP or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return original(x, h, group_size)
        groups = x.shape[-1] // CONVROT_GROUP
        if mode == "fused" and x.dtype == torch.float32 and groups <= FUSED_MAX_GROUPS and not groups & (groups - 1):
            return convrot_quantize(x.reshape(-1, x.shape[-1]))
        return hadamard_rotate(x)

    def triton_quantize_rowwise(x):
        if isinstance(x, QuantizedRotation):
            return x.data, x.scale
        return original_quantize(x)

    quantization._rotate_activation = rotate_activation
    quantization.triton_quantize_rowwise = triton_quantize_rowwise
    quantization._gfx906_convrot = True
    _state["convrot"] = True
    return True


def enable_triton_backend():
    """comfy/quant_ops.py disables the comfy-kitchen triton backend at import
    unless --enable-triton-backend was passed, so turn it back on from here.

    Its import-time log line reports the old state, hence the extra line below.
    """
    from comfy_kitchen import registry

    if registry.is_available("triton"):
        return True
    try:
        import triton  # noqa: F401
    except ImportError:
        logging.warning("gfx906: triton is not importable, comfy-kitchen stays on the eager backend")
        return False
    registry.enable("triton")
    ok = registry.is_available("triton")
    if ok:
        logging.info("gfx906: comfy-kitchen triton backend enabled (same as --enable-triton-backend, "
                     "the backend list printed above was from before this line)")
    return ok


def triton_backend_active():
    from comfy_kitchen import registry

    return registry.is_available("triton")


def configure(attention=None, smooth=None, convrot=None, profile=None, text_encoder=None,
              fp16=None, text_attention=None, contiguous=None, qkv=None,
              vae=None, gemv=None, textgen=None, prefix_kv=None, text_gemv=None, w4a8=None,
              amax_guard=None, int8_tile=None, small_batch=None):
    """Runtime settings, used by the GFX906 nodes.

    Every argument is three-state: None/"default" keeps the built-in default
    (the manifest verdict where one exists), other values are explicit
    overrides. Node parameters therefore always win.
    """
    if attention is not None:
        wanted = str(attention).strip().lower()
        if wanted == "default":
            wanted = "int8"
        # combo values are untrusted: stale workflows may still carry removed names
        _state["attn_kernel"] = (wanted if wanted in ATTENTION_KERNELS
                                 or stock_backend(wanted) is not None else "int8")
        _state["attn_backend_fn"] = stock_backend(_state["attn_kernel"])
    if contiguous is not None:
        _state["attn_contiguous"] = resolve_flag(contiguous, True)
    if smooth is not None:
        _state["smooth"] = resolve_flag(smooth, True)
    if fp16 is not None:
        _state["attn_fp16"] = resolve_flag(fp16, True)
    if convrot is not None:
        wanted = str(convrot).strip().lower()
        if wanted == "default":
            wanted = resolved_convrot_mode()
        _state["convrot_mode"] = wanted if wanted in CONVROT_MODES else resolved_convrot_mode()
    if qkv is not None and _state.get("qkv_patched"):
        _state["qkv"] = resolve_flag(qkv, False)
    if text_encoder is not None and _state["applied"]:
        _state["te_int8"] = str(text_encoder).strip().lower() == "int8" and _state["te_patched"]
    if text_attention is not None:
        wanted = str(text_attention).strip().lower()
        # an unknown combo value leaves whatever the previous selection was
        _state["te_attention"] = (wanted if wanted == "default"
                                  or stock_backend(wanted) is not None else "default")
    if vae is not None:
        _state["vae_enabled"] = resolve_flag(vae, feature_on("vae"))
    if gemv is not None:
        want = flag_value(gemv)
        if want is None:
            want = feature_on("gemv")
        if want and _state.get("gemv_module") is None:
            _state["gemv_module"] = load_int8_gemv()
        _state["gemv_model"] = want
    if textgen is not None:
        _state["textgen_override"] = flag_value(textgen)
    if prefix_kv is not None:
        _state["prefix_kv_enabled"] = resolve_flag(prefix_kv, True)
    if text_gemv is not None:
        _state["text_gemv_enabled"] = resolve_flag(text_gemv, True)
    if w4a8 is not None:
        want = flag_value(w4a8)
        if want is None:
            want = True
        if want and _state.get("w4a8_module") is None:
            _state["w4a8_module"] = load_w4a8_gemv()
        _state["w4a8"] = _state.get("w4a8_module") if want else None
    if amax_guard is not None:
        _state["amax_guard"] = resolve_flag(amax_guard, True)
    if int8_tile is not None:
        want = resolve_flag(int8_tile, True)
        if want != _state.get("int8_tile_enabled"):
            set_int8_tile(want)
        _state["int8_tile_enabled"] = want
    if small_batch is not None:
        want = resolve_flag(small_batch, True)
        if want != _state.get("small_batch_enabled"):
            _state["small_batch_enabled"] = want
            set_untuned_tiles()
    # something may have disabled the backend after apply(); the node runs per
    # workflow, so this is the last chance to put it back before the model does
    # its first int8 matmul
    _state["triton_backend"] = enable_triton_backend()
    if profile is not None and _state["applied"]:
        want = resolve_flag(profile, False)
        was = _profiler_enabled()
        _set_profiler(want)
        if want and not was:
            # the model node absorbed the old Tuning Info node: its one-shot
            # report ships with the first profiled prompt
            logging.info("gfx906 tuning status:\n%s", status())
    return _state


def _te_linear_ready(module, x):
    """True when the int8 weight can be fed to F.linear as-is."""
    from comfy_kitchen.tensor import QuantizedTensor

    weight = module.weight
    return (
        _state["te_int8"]
        and x.is_cuda
        and isinstance(weight, QuantizedTensor)
        and weight._layout_cls == "TensorWiseINT8Layout"
        and not getattr(weight._params, "transposed", False)
        and weight.device == x.device
        and (module.bias is None or module.bias.device == x.device)
        and not module.weight_function
        and not module.bias_function
    )


def _te_dequant_enabled():
    """Per-arch enablement for the register-dequantizing decode GEMV.

    COMFY_GFX906_TE_DEQUANT (on/off) and COMFY_GFX906_FORCE=te_dequant outrank
    the manifest; everything else needs [sites."feature.te_dequant"] default
    = "on" in the arch table (a tools/bench.py run records it). No data means
    off -- the stock dequantize-per-call path is the default, exactly like the
    other tier-3 features.
    """
    if "te_dequant" in _state.get("forced_items", set()):
        return True
    raw = str(_state.get("te_dequant_env", "")).strip().lower()
    if raw in ("1", "on", "true", "yes"):
        return True
    if raw in ("0", "off", "false", "no"):
        return False
    table = measured_table()
    return table is not None and table.default_impl("feature.te_dequant") == "on"


def _te_default_ready(module, x):
    """True when the quantized weight can feed the register-dequantizing decode GEMV."""
    if not _te_dequant_enabled():
        return False
    from comfy_kitchen.tensor import QuantizedTensor

    weight = module.weight
    return (
        x.is_cuda
        and 1 <= x.reshape(-1, x.shape[-1]).shape[0] <= 8
        and isinstance(weight, QuantizedTensor)
        and weight._layout_cls in ("TensorWiseINT8Layout", "AsymW4A8Int8Layout")
        and not getattr(weight._params, "transposed", False)
        and weight.device == x.device
        and (module.bias is None or module.bias.device == x.device)
        and not module.weight_function
        and not module.bias_function
    )


def te_default_gemv(module, x):
    """Register-dequantizing GEMV for the te=default decode path.

    Reads the quantized weight directly and converts in registers with fp32
    accumulation, so no fp16 copy of the model is kept in memory and no
    per-call dequantize runs. Dispatches on the layout: TensorWiseINT8 goes to
    int8_gemv.linear_dequant, AsymW4A8Int8 to w4a8_gemv.linear_dequant (W4A16:
    fp16 activations, no activation quant). Single-row up to MTP/spec-verify
    row counts; prefill and anything the kernels refuse fall through to the
    stock dequantize-per-call path.
    """
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout

    weight = module.weight
    rows = x.reshape(-1, x.shape[-1])
    params = weight._params
    if weight._layout_cls == "TensorWiseINT8Layout":
        from comfy_kitchen.tensor import TensorWiseINT8Layout

        mod = _state.get("gemv_module")
        if mod is None:
            mod = load_int8_gemv()
            if mod is None:
                return None
            _state["gemv_module"] = mod
        qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        groupsize = getattr(params, "convrot_groupsize", 256) if getattr(params, "convrot", False) else 0
        out = mod.linear_dequant(rows, qdata, scale, module.bias, x.dtype, groupsize)
    elif weight._layout_cls == "AsymW4A8Int8Layout":
        mod = _state.get("w4a8_module")
        if mod is None:
            mod = load_w4a8_gemv()
            if mod is None:
                return None
            _state["w4a8_module"] = mod
        qdata, s_rel, s_channel, correction, codebook = AsymW4A8Int8Layout.get_plain_tensors(weight)
        if correction is not None or qdata.shape[1] * 2 != rows.shape[1]:
            return None
        out = mod.linear_dequant(rows, qdata, s_rel, s_channel, module.bias, x.dtype,
                                 params.group_size, params.convrot_groupsize, codebook)
    else:
        return None
    if out is None:
        return None
    return out.view(*x.shape[:-1], out.shape[-1])


def patch_text_encoder():
    """Keep int8 text-encoder weights quantized instead of dequantizing per call.

    MixedPrecisionOps casts every int8 weight to the activation dtype before an
    fp16 matmul, so a Qwen3VL-8B encode spends 0.53 s of 0.75 s re-expanding
    8.9 GB of weights. F.linear on the quantized weight dispatches to the same
    int8 kernel the DiT uses, so skipping the cast is all that is needed.
    """
    import comfy.ops

    if getattr(comfy.ops, "_gfx906_te", False):
        return True
    factory = getattr(comfy.ops, "mixed_precision_ops", None)
    if factory is None:
        logging.warning("gfx906: comfy.ops has no mixed_precision_ops, text-encoder patch skipped")
        return False

    def mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[]):
        cls = factory(quant_config, compute_dtype, full_precision_mm, disabled)
        # full_precision_mm is how the text encoders ask for "dequantize the int8
        # weight, do a fp16 matmul"; leave every other model on its own path
        if not full_precision_mm:
            return cls
        linear = cls.Linear
        if getattr(linear, "_gfx906_te", False):
            return cls
        previous = getattr(linear, "forward_comfy_cast_weights", None)
        if previous is None:
            logging.warning("gfx906: MixedPrecisionOps.Linear has no forward_comfy_cast_weights, "
                            "text-encoder patch skipped")
            return cls

        def forward_comfy_cast_weights(self, input, compute_dtype=None, want_requant=False, weight_only_quant=False):
            if _te_linear_ready(self, input):
                out = te_int8_gemv(self, input)
                if out is not None:
                    return out
                return _same_dtype(torch.nn.functional.linear(input, self.weight, self.bias), input)
            if _te_w4a8_ready(self, input):
                out = te_w4a8_gemv(self, input)
                if out is not None:
                    return out
                return _same_dtype(torch.nn.functional.linear(input, self.weight, self.bias), input)
            if _te_default_ready(self, input):
                out = te_default_gemv(self, input)
                if out is not None:
                    return _same_dtype(out, input)
            out = te_plain_gemv(self, input, compute_dtype, want_requant, weight_only_quant)
            if out is not None:
                return out
            return _same_dtype(previous(self, input, compute_dtype, want_requant, weight_only_quant), input)

        linear.forward_comfy_cast_weights = forward_comfy_cast_weights
        linear._gfx906_te = True
        return cls

    comfy.ops.mixed_precision_ops = mixed_precision_ops
    comfy.ops._gfx906_te = True
    return True


def patch_logits():
    """Keep a quantized lm_head quantized through the decode loop.

    BaseGenerate.logits pushes the head through CastBiasWeightContext, which
    dequantizes an int8 head on every token: 90.7 ms/token on Qwen3.5-9B, far
    more than the GEMV it feeds (6.3 ms through the kernel). Token choice is
    unchanged on real activations: argmax and top-5 identical, max |dlogit|
    0.078 on a 16.4 scale. Follows the CLIP node's text_encoder switch, since it
    is the same int8 approximation the encoder already makes.
    """
    import comfy.text_encoders.llama as llama
    from comfy_kitchen.tensor import QuantizedTensor

    original = llama.BaseGenerate.logits
    if getattr(original, "_gfx906_logits", False):
        return True

    def logits(self, x):
        last = x[:, -1:]
        module = self.model.lm_head if hasattr(self.model, "lm_head") else self.model.embed_tokens
        weight = module.weight
        if _state["te_int8"]:
            if (isinstance(weight, QuantizedTensor) and weight.device == last.device
                    and not module.weight_function and not module.bias_function):
                return torch.nn.functional.linear(last, weight, None)
        else:
            if _te_default_ready(module, last):
                out = te_default_gemv(module, last)
                if out is not None:
                    return out
        return original(self, x)

    logits._gfx906_logits = True
    llama.BaseGenerate.logits = logits
    return True


def patch_generate_te_default():
    """Bracket a generate() call with the te=default execution fixes.

    Two live changes while the call runs, both restored on exit:

    * execution_dtype fp16: llama.py's should_use_bf16() fallback pins the TE
    to fp32 on gfx906; fp16 is the card's fast path and matches the register-
    dequantizing decode GEMV, so no per-call weight cast remains.
    * graph capture stays on for the register-dequant path: there are no
    dequantize kernels to pin (~10.5 GB resident, same as int8), and the
    capture absorbs ~800 eager launches per step (34.6 vs 22.0 it/s on
    Qwen3.5-9B). With the stock dequantize-per-call fallback (unmeasured
    arch, gate off) the graphs are turned off instead, since capture would
    pin the fp16 copies in private pools (~21 GB) and OOM a 32 GB card.
    """
    import comfy.text_encoders.llama as llama

    original = llama.BaseGenerate.generate
    if getattr(original, "_gfx906_gen_te_default", False):
        return True

    def generate(self, *args, **kwargs):
        graphs = None
        if not _state["te_int8"]:
            kwargs.setdefault("execution_dtype", torch.float16)
            # graphs stay on only while the register-dequant path is active:
            # it has no dequantize kernels to pin, so pools stay at int8 size
            # and the capture absorbs ~800 eager launches per step (34.6 vs
            # 22.0 it/s on Qwen3.5-9B). With the stock dequantize-per-call
            # fallback (unmeasured arch, gate off) capture would pin the fp16
            # copies in private pools (~21 GB) and OOM a 32 GB card.
            if not _te_dequant_enabled():
                graphs = getattr(self.model, "graph_dynamic_vbar_blocks", None)
                if graphs:
                    self.model.graph_dynamic_vbar_blocks = False
        try:
            return original(self, *args, **kwargs)
        finally:
            if graphs:
                self.model.graph_dynamic_vbar_blocks = graphs

    generate._gfx906_gen_te_default = True
    llama.BaseGenerate.generate = generate
    return True


def patch_te_attention():
    """Let the CLIP settings node pick the text-encoder attention at runtime.

    Text encoders fetch their attention through optimized_attention_for_device,
    which hard-codes attention_basic/attention_pytorch for small inputs. Wrap it
    so a chosen backend replaces that choice for every caller; "default" keeps
    core's behavior. The wrap is installed once, the selection is live state.
    """
    from comfy.ldm.modules import attention

    if getattr(attention, "_gfx906_te_attn", False):
        return True
    original = attention.optimized_attention_for_device

    def optimized_attention_for_device(device, mask=False, small_input=False):
        backend = stock_backend(_state["te_attention"]) if _state["te_attention"] != "default" else None
        if backend is not None:
            return backend
        return original(device, mask=mask, small_input=small_input)

    attention.optimized_attention_for_device = optimized_attention_for_device
    attention._gfx906_te_attn = True
    for module in list(sys.modules.values()):
        if getattr(module, "optimized_attention_for_device", None) is original:
            module.optimized_attention_for_device = optimized_attention_for_device
    return True


def _merge_qkv(attention, x):
    """[3*inner, K] int8 weight with per-row scales, or False if not mergeable.

    The int8 checkpoints store one scale per output row, so concatenating the
    qdata along N and the scales along rows keeps the exact same math as three
    separate layers. convrot rotates the K side, which all three share.

    The weight is fetched through cast_bias_weight, the same door ComfyUI's own
    linear takes: with dynamic VRAM loading the module attribute is not the
    tensor the kernel actually runs on.
    """
    import comfy.ops
    from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

    rotation = None
    data, scale = [], []
    cast = []
    try:
        for name in ("to_q", "to_k", "to_v"):
            module = getattr(attention, name)
            weight, bias, stream = comfy.ops.cast_bias_weight(
                module, x, offloadable=True, compute_dtype=x.dtype, want_requant=True)
            cast.append((module, weight, bias, stream))
            params = getattr(weight, "_params", None)
            truth = (f"{name}: type={type(weight).__name__} layout={getattr(weight, '_layout_cls', None)} "
                     f"bias={bias is not None} "
                     f"functions={len(getattr(module, 'weight_function', None) or [])}"
                     f"/{len(getattr(module, 'bias_function', None) or [])} "
                     f"transposed={getattr(params, 'transposed', None)} "
                     f"convrot={getattr(params, 'convrot', None)}")
            reason = None
            if bias is not None:
                reason = "bias"
            elif getattr(module, "weight_function", None) or getattr(module, "bias_function", None):
                reason = "weight/bias function (patch or LoRA)"
            elif not isinstance(weight, QuantizedTensor):
                reason = "not a QuantizedTensor"
            elif weight._layout_cls != "TensorWiseINT8Layout":
                reason = f"layout {weight._layout_cls}"
            elif getattr(params, "transposed", False):
                reason = "transposed weight"
            if reason is not None:
                _state["qkv_skip_reason"] = f"{reason} ({truth})"
                return False
            this = (getattr(params, "convrot", False), getattr(params, "convrot_groupsize", 256))
            if rotation is None:
                rotation = this
            elif rotation != this:
                _state["qkv_skip_reason"] = f"mixed rotation {rotation} vs {this} ({truth})"
                return False
            qdata, qscale = TensorWiseINT8Layout.get_plain_tensors(weight)
            data.append(qdata)
            qscale = qscale.reshape(-1)
            if qscale.numel() == 1:
                # a per-tensor scale becomes a per-row one so the three can share a kernel
                qscale = qscale.expand(qdata.shape[0])
            scale.append(qscale)
        return torch.cat(data, 0), torch.cat(scale), rotation
    finally:
        for module, weight, bias, stream in cast:
            comfy.ops.uncast_bias_weight(module, weight, bias, stream)


def _fused_qkv(attention, x):
    merged = getattr(attention, "_gfx906_merged", None)
    if merged is None:
        merged = attention._gfx906_merged = _merge_qkv(attention, x)
        if merged is False:
            # every block can fail for its own reason; report each distinct one once
            reason = _state.get("qkv_skip_reason", "?")
            seen = _state.setdefault("qkv_skip_reasons", set())
            if reason not in seen:
                seen.add(reason)
                logging.info("gfx906: qkv fusion off, %s", reason)
        else:
            if not _state.get("qkv_on_logged"):
                _state["qkv_on_logged"] = True
                logging.info("gfx906: qkv fusion on, one %s int8 gemm, rotation %s",
                             tuple(merged[0].shape), merged[2])
    if merged is False:
        _state["qkv_fallbacks"] = _state.get("qkv_fallbacks", 0) + 1
        return None
    weight, scale, (convrot, groupsize) = merged
    if weight.device != x.device:
        # weights were moved since the merge; rebuild from the live parameters
        attention._gfx906_merged = None
        return None
    import comfy.quant_ops

    _state["qkv_calls"] = _state.get("qkv_calls", 0) + 1
    return comfy.quant_ops.ck.int8_linear(x, weight, scale, None, x.dtype,
                                          convrot=convrot, convrot_groupsize=groupsize)


def patch_qkv():
    """One int8 GEMM for q/k/v instead of three (Qwen-Image 2.1 attention).

    to_q/to_k/to_v share M and K, so the fused call is the same per-row dequant
    kernel with N three times wider: on the production tile at (M=4032, K=4096)
    three calls cost 23.5 ms and one fused call 21.1 ms, i.e. -81 ms per step.

    Off by default: the merged weights stay resident (50 MB per block, 1.6 GB
    for a 32-block DiT) and the step is only 1.5% faster. The patch itself is
    always installed; ``_state["qkv"]`` decides per call whether it is used.
    """
    from comfy.ldm.qwen_image21 import model as qwen
    import comfy.model_management

    attention = qwen.Attention
    if getattr(attention, "_gfx906_qkv", False):
        return True
    original_forward = attention.forward
    original_init = attention.__init__

    def project(self, x):
        fused = getattr(self._gfx906_owner, "_gfx906_fused", None)
        if fused is None:
            return self._gfx906_forward(x)
        inner = fused.shape[-1] // 3
        start = self._gfx906_index * inner
        return fused[..., start:start + inner]

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for index, name in enumerate(("to_q", "to_k", "to_v")):
            module = getattr(self, name)
            module._gfx906_forward = module.forward
            module._gfx906_index = index
            # plain attribute: nn.Module would otherwise register the attention
            # as a child of its own projection and recurse forever in .train()
            object.__setattr__(module, "_gfx906_owner", self)
            module.forward = project.__get__(module)

    def forward(self, x, pe, attn_fn, prefix_len, transformer_options={}):
        if comfy.model_management.in_training or not _state["qkv"]:
            self._gfx906_fused = None
            # turning the switch off mid-run drops the 1.6 GB of merged weights too
            self._gfx906_merged = None
        else:
            self._gfx906_fused = _fused_qkv(self, x)
        try:
            return original_forward(self, x, pe, attn_fn, prefix_len, transformer_options)
        finally:
            self._gfx906_fused = None

    attention.__init__ = __init__
    attention.forward = forward
    attention._gfx906_qkv = True
    _state["qkv_patched"] = True
    return True


def _prefix_merge(prefix, target, buffers, slot):
    """Copy prefix and target rows into one reused buffer, as torch.cat would.

    buffers[slot] is allocated by the first call and reused for the block's
    only attention call, so the merge is two copy_ kernels on a fresh
    allocation of exactly the size torch.cat would have made.
    """
    out = buffers[slot]
    if out is None:
        out = buffers[slot] = torch.empty((prefix.shape[0], prefix.shape[1] + target.shape[1], *prefix.shape[2:]),
                                          dtype=prefix.dtype, device=prefix.device)
    out[:, :prefix.shape[1]].copy_(prefix)
    out[:, prefix.shape[1]:].copy_(target)
    return out


def patch_prefix_kv():
    """Merge the cached prefix K/V with copy_ instead of torch.cat.

    prefix_cached_attention rebuilds [prefix_k; k] with torch.cat on every
    block: 4032 target rows against a 4058 row prefix is two 132 MB copies in
    fp32, and cat pays the same copies plus its own dispatch. A buffer sized
    exactly like the cat result, filled with two copy_ kernels, measures
    0.39 -> 0.33 ms per call; the buffer dies with the per-block closure, the
    same lifetime as the cat tensors it replaces.

    The switch is read per call; off hands the call back to the shipped
    torch.cat version.
    """
    import comfy.ldm.qwen_image21.model as qwen
    if getattr(qwen, "_gfx906_prefixkv", False):
        return True

    original = qwen.prefix_cached_attention
    _state["prefix_kv_patched"] = True

    def prefix_cached_attention(prefix_k, prefix_v, transformer_options={}):
        if not _state.get("prefix_kv_enabled", True):
            return original(prefix_k, prefix_v, transformer_options)
        # target-only queries: block-causal reduces to full attention over [cached prefix, target]
        buffers = [None, None]

        def attn(q, k, v, heads, preferred_attention=None):
            q = qwen.AttentionTensorContainer(q.take().flatten(2))
            k = qwen.AttentionTensorContainer(_prefix_merge(prefix_k, k.take(), buffers, 0).flatten(2))
            v = qwen.AttentionTensorContainer(_prefix_merge(prefix_v, v.take(), buffers, 1).flatten(2))
            return qwen.optimized_attention(q, k, v, heads, transformer_options=transformer_options, preferred_attention=preferred_attention)
        return attn

    qwen.prefix_cached_attention = prefix_cached_attention
    qwen._gfx906_prefixkv = True
    _state["prefix_kv"] = True
    return True


def patch_vae():
    """Run depth-1 Conv3d as Conv2d.

    The Wan-style video VAEs that image models ship with (Qwen-Image 2.1 uses the
    Wan 2.2 decoder) are 76 Conv3d layers with a one-frame kernel, and an image
    gives them a single frame. There is no conv3d kernel for gfx906, so torch
    lowers every one of them to im2col + bmm; the profiler puts that at 47% of
    decode time. Squeezing the depth axis makes them ordinary conv2d calls that
    land on the fp16 MIOpen kernels: 2.31 s -> 1.09 s at 1024x1024.

    The switch is read in the hot path, so the node can turn it off without a
    reload; the shipped Conv3d path is still reachable when the gate is off.
    """
    import comfy.ops

    conv3d = comfy.ops.disable_weight_init.Conv3d
    if getattr(conv3d, "_gfx906_vae", False):
        return True
    original = conv3d._conv_forward
    _state["vae_patched"] = True

    def _conv_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
        if _state.get("vae_enabled", True):
            if autopad == "causal_zero":
                weight = weight[:, :, -input.shape[2]:, :, :]
                autopad = None
            if input.ndim == 5 and input.shape[2] == 1 and weight.shape[2] == 1:
                return torch.nn.functional.conv2d(
                    input[:, :, 0], weight[:, :, 0], bias, self.stride[1:],
                    self.padding[1:], self.dilation[1:], self.groups).unsqueeze(2)
        return original(self, input, weight, bias, autopad=autopad, *args, **kwargs)

    conv3d._conv_forward = _conv_forward
    conv3d._gfx906_vae = True
    return True


def apply():
    if _state["applied"]:
        return _state
    _state["applied"] = True
    _state["arch"] = arch_name()
    # parse the startup defaults before any early return, so status() and the
    # nodes can fall back to them on every platform
    _init_env_settings()
    if not enabled():
        _state["reason"] = "disabled by COMFY_GFX906_TUNING"
        return _state
    if not torch.cuda.is_available():
        _state["reason"] = "no CUDA device"
        logging.info("gfx906 tuning skipped: %s", _state["reason"])
        return _state

    _state["native"] = native_backend()
    on_amd = is_amd()
    if on_amd and _state["native"]:
        logging.info("gfx906: comfy-kitchen %s serves this %s; the triton kernel patches "
                     "below only run where its per-call constraints decline a shape",
                     _state["native"], _state["arch"])
    # a hand-measured tile spec, or one measured by tools/bench.py on this card
    # (or shipped for its arch), replaces the autotune list with a single config
    _state["tile_measured"] = (_tile_spec("COMFY_GFX906_TILE_CONFIG") is not None
                               or measured_tile("large") is not None)
    # before anything is patched: a card running without its numbers says so in
    # the boot log, instead of looking identical to one that has them
    warn_measured_state()
    warn_forced_items()
    # The triton kernel patches install on every AMD and let dispatch decide when
    # they run: the native backend is preferred, so where kitchen serves an op
    # these only serve the calls its per-call constraints decline, and where it
    # does not (gfx906) they are the primary path. The gfx906-specific content
    # inside them is guarded by the manifest fingerprint, not by this gate.
    if on_amd:
        _install("int8 tile", patch_int8_linear, replace=_state["tile_measured"])
        if _state.get("int8_linear") and not _state.get("int8_tile_enabled", True):
            _install("int8 tile reset", restore_int8_defaults)
    # tier-3 features: on only where the manifest says so -- the shipped table
    # for measured archs, or a local bench run. env overrides per value.
    if feature_on("convrot"):
        _install("convrot", patch_convrot)
    if feature_on("vae"):
        _install("vae", patch_vae)
    # opt-in: the fused GEMM is 1.5% faster but holds 1.6 GB of merged weights
    _state["qkv"] = False
    _install("qkv", patch_qkv)
    # the TE GEMV is not a kernel replacement: it skips the QuantizedTensor
    # dispatch, which costs the same on every vendor. Needs int8_gemv loaded
    # even where the tile patch above is skipped.
    # the module backs two consumers: text_gemv (tier-1, its own switch) and
    # the model int8 GEMV path (tier-3 -- on only where the manifest says so)
    if (feature_on("gemv") or _state.get("text_gemv_enabled")) and _state.get("gemv_module") is None:
        _state["gemv_module"] = _install("int8 gemv", load_int8_gemv)
    _state["gemv_model"] = feature_on("gemv")
    _install("prefix kv", patch_prefix_kv)
    # before any prompt runs: the queue thread binds execute_async when a prompt
    # starts, so a hook installed later would only report from the next one
    _install("prompt hook", _patch_prompt_end)
    te_ready = _install("text encoder", patch_text_encoder)
    # last: importing comfy.ops above pulls in comfy.quant_ops, which disables
    # the triton backend again when --enable-triton-backend was not passed
    _state["triton_backend"] = _install("triton backend", enable_triton_backend)
    _state["te_patched"] = te_ready
    _state["logits_patched"] = _install("lm_head", patch_logits)
    _install("generate default", patch_generate_te_default)
    _install("encoder attention", patch_te_attention)
    # text-generation group: installed once with a live gate, so the CLIP node
    # can turn it off without a reload. Modules are loaded on the measured arch
    # even when the env default is off, otherwise the node could not turn them on.
    if install_ready("deltanet"):
        deltanet_kernels = _install("deltanet loader", load_deltanet)
        if deltanet_kernels:
            _state["textgen_ready"]["deltanet"] = bool(
                _install("deltanet", deltanet_kernels.install, lambda: textgen_active("deltanet")))
    _state["textgen_ready"]["kvdecode"] = bool(
        _install("kv decode", patch_kv_decode, lambda: textgen_active("kvdecode")))
    llama_patches = _install("llama loader", load_llama_patches)
    if llama_patches:
        _state["textgen_ready"]["llama"] = bool(
            _install("llama", llama_patches.install, lambda: textgen_active("llama")))
    if install_ready("qwen35"):
        qwen35_model = _install("qwen35 loader", load_qwen35_model)
        if qwen35_model:
            # the freqs table belongs to the llama item, not the qwen35 class swap
            _state["textgen_ready"]["qwen35"] = bool(
                _install("qwen35", qwen35_model.install, lambda: textgen_active("qwen35"),
                         lambda: textgen_active("llama")))
    # 4-bit weights: register-unpacked GEMV for decode, int8 decode for prefill
    _state["w4a8_module"] = (_install("w4a8 module", load_w4a8_gemv)
                             if install_ready("w4a8") else None)
    _state["w4a8"] = _state["w4a8_module"]
    # the per-block graph hooks live in core on purpose; say so when they are absent
    def _probe_resident():
        import comfy.model_prefetch

        return hasattr(comfy.model_prefetch, "RESIDENT_GRAPH_MODULES")

    _state["resident_graph_patch"] = bool(_install("resident graph probe", _probe_resident))
    if not _state["resident_graph_patch"]:
        logging.info("gfx906: resident graph hooks absent, decode runs without per-block CUDA graphs "
                     "(see custom_nodes/ComfyUI-gfx906-Tuning/core-patches/README.md)")
    # attention kernels differ in quality (int8 PV rel ~2e-2 vs hand 2.5e-4), so
    # the mode is a manual node/env choice, not a manifest verdict -- and the TE
    # never routes here anyway (its resolver returns core's pytorch/basic). It
    # installs on every AMD; NVIDIA keeps core's attention.
    if on_amd:
        _install("attention", patch_attention)
    logging.info(
        "gfx906 tuning active: int8 linear %s%s, convrot %s, vae %s, qkv %s, attention %s%s%s%s, "
        "text encoder %s + lm_head, encoder attention %s, textgen %s",
        "measured tile" if _state.get("int8_tile_replaced") else ("on (stock candidates)" if _state["int8_linear"] else "off"),
        " + decode gemv" if _state.get("gemv_model") else "",
        _state["convrot_mode"] if _state["convrot"] else "off",
        "conv3d->conv2d" if _state.get("vae_patched") and _state.get("vae_enabled") else "off",
        "fused" if _state.get("qkv") else "off",
        _state["attn_kernel"] if _state["attention"] else "off",
        f" (other calls go to {_state['replaced_backend']})" if _state.get("replaced_backend") else "",
        ", fp16 cast off" if not _state["attn_fp16"] else "",
        ", contiguous off" if not _state.get("attn_contiguous", True) else "",
        "int8" if _state["te_int8"] else "default",
        _state["te_attention"],
        textgen_summary(),
    )
    return _state


def _bench(fn, iters=5, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def report_lines():
    """One-shot line for the profiler: how many attention calls took the fp16 path."""
    cast = _state["attn_cast"]
    if not cast:
        return []
    line = (f"attention fp16-ok {cast['fp16']} / amax overflow {cast['fp32']}"
            f" (own kernel: hand {cast['hand']}, int8 {cast['int8']}, int8qk {cast['int8qk']},"
            f" gemm {cast['gemm']}, sdpa {cast['sdpa']})")
    for k in cast:
        cast[k] = 0
    return [line]


def _format_tile(config):
    params = ", ".join(f"{k}={v}" for k, v in config.kwargs.items())
    return f"{params}, warps={config.num_warps}, stages={config.num_stages}"


def status():
    tile_state = {True: "measured tile", False: "stock candidates, autotune decides", None: "off"}
    if not _state["int8_linear"]:
        tile_line = "int8 linear off"
    elif _state.get("int8_tile_enabled", True):
        tile_line = f"int8 linear {tile_state[_state['int8_tile_replaced']]}"
    else:
        tile_line = f"int8 linear {tile_state[_state['int8_tile_replaced']]}, tile switch off (stock list)"
    if not _state["int8_linear"] or not _state.get("int8_tile_enabled", True):
        small_line = "small batch tile n/a"
    elif _state.get("int8_tile_replaced"):
        small_line = f"small batch tile {'on' if _state.get('small_batch_enabled', True) else 'off (large tile only)'}"
    else:
        small_line = (f"small batch tile {'candidate' if _state.get('small_batch_enabled', True) else 'off (not a candidate)'}")
    prefix_patched = _state.get("prefix_kv_patched")
    if not prefix_patched:
        prefix_line = "prefix kv n/a"
    elif _state.get("prefix_kv_enabled", True):
        prefix_line = "prefix kv reused buffer"
    else:
        prefix_line = "prefix kv torch.cat (off, installed)"
    gemv_loaded = _state.get("gemv_module") is not None
    if _state.get("gemv_model"):
        gemv_backend = getattr(_state.get("gemv_module"), "backend_name", None)
        parts = [f"backend={gemv_backend()}"] if gemv_backend is not None else []
        suffix = measured_suffix(_state.get("gemv_module"))
        if suffix:
            parts.append(suffix)
        gemv_line = "gemv on" + (f" ({', '.join(parts)})" if parts else "")
    else:
        gemv_line = "gemv off (installed)" if gemv_loaded else "gemv n/a"
    if _state.get("text_gemv_enabled", True):
        text_gemv_line = (f"text gemv on: {_state.get('te_gemv_calls', 0)} direct / "
                          f"{_state.get('te_plain_calls', 0)} mv calls" if _state.get("te_patched")
                          else "text gemv on (TE patch n/a)")
    else:
        text_gemv_line = "text gemv off (fallback)"
    w4a8_loaded = _state.get("w4a8_module") is not None
    if _state.get("w4a8") is not None:
        calls = _state.get("w4a8_calls", 0)
        backend = getattr(_state["w4a8"], "backend_name", None)
        backend_name = backend() if backend is not None else "?"
        measured = measured_suffix(_state["w4a8"])
        if not _state.get("te_int8"):
            w4a8_line = f"w4a8 on: backend={backend_name} (needs text encoder int8)"
        else:
            w4a8_line = (f"w4a8 on: {calls} calls, backend={backend_name}"
                         + (f", {measured}" if measured else "")
                         + (" (no 4-bit weights seen yet)" if not calls else ""))
    elif w4a8_loaded:
        w4a8_line = "w4a8 off (installed)"
    else:
        w4a8_line = "w4a8 n/a"
    lines = [
        f"arch {_state['arch']}",
        f"comfy-kitchen triton backend {triton_backend_active()}",
        tile_line,
        small_line,
        f"convrot {_state['convrot_mode'] or 'off'}",
        f"qkv {'one int8 gemm' if _state.get('qkv') else 'off'}"
        + (f", {_state['qkv_calls']} fused / {_state.get('qkv_fallbacks', 0)} fallback"
           if _state.get("qkv_calls") or _state.get("qkv_fallbacks") else ""),
        prefix_line,
        gemv_line,
        text_gemv_line,
        w4a8_line,
        f"vae {'conv3d as conv2d' if _state.get('vae_patched') and _state.get('vae_enabled') else ('off (patched)' if _state.get('vae_patched') else 'off')}",
        f"textgen {textgen_summary()}",
        f"attention {_state['attn_kernel'] if _state['attention'] else 'off'}"
        + (f", smoothing {'on' if _state['smooth'] else 'off'}" if _state["attn_kernel"] == "int8qk" else "")
        + ((", inputs cast to fp16" if _state["attn_fp16"] else ", inputs left in the model dtype")
           + f", amax guard {'on' if _state.get('amax_guard', True) else 'off'}"
           if _state["attention"] else "")
        + (f", rest to {_state['replaced_backend']}" if _state.get("replaced_backend") else ""),
        f"text encoder {'int8' if _state['te_int8'] else 'default (dequantize per call)'}"
        + f", lm_head {'int8' if _state.get('logits_patched') and _state['te_int8'] else 'default'}"
        + f", encoder attention {_state['te_attention']}",
        f"resident graph hooks {'present' if _state.get('resident_graph_patch') else 'absent'}",
    ]
    manifest_line = manifest_status_line()
    if manifest_line:
        lines.insert(1, manifest_line)
    if _state.get("native") and _state.get("int8_linear"):
        lines.insert(1, f"comfy-kitchen {_state['native']} backend present: "
                        f"triton kernel patches are fallback-only here")
    if _state["int8_linear"] and _state.get("int8_tile_enabled", True):
        if _state.get("tile_measured"):
            if _tile_spec("COMFY_GFX906_TILE_CONFIG") is not None:
                suffix = "env tile"
            elif measured_tile("large") is not None:
                suffix = "measured tile"
            else:
                suffix = "forced"
        else:
            suffix = "autotune candidates"
        lines.append(f"int8 tiles ({suffix}): large {_format_tile(_triton_config())}; "
                     + (f"small {_format_tile(_small_tile_config())}"
                        if _state.get("small_batch_enabled", True) else "small batch off"))
    cast = _state["attn_cast"]
    if cast:
        lines.append(f"attention calls: fp16-ok {cast['fp16']}, amax overflow {cast['fp32']}"
                     f" (own kernel: hand {cast['hand']}, int8 {cast['int8']}, int8qk {cast['int8qk']},"
                     f" gemm {cast['gemm']}, sdpa {cast['sdpa']})")
    failures = _state.get("install_failures") or []
    if failures:
        lines.append(f"patches skipped on this device: {', '.join(failures)} (see the boot log)")
    if _state["reason"]:
        lines.append(_state["reason"])
    forced = sorted(_state.get("forced_items") or ())
    if forced:
        lines.append(f"forced on an unmeasured arch: {', '.join(forced)} (COMFY_GFX906_FORCE)")
    return "\n".join(lines)
