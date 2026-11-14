"""Runtime tuning patches for ComfyUI on AMD GPUs, measured on gfx906 (MI50 / MI60 / Radeon VII).

Two independent patches, both skipped on other architectures and both opt-out
with COMFY_GPU_TUNE=off:

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
    from .patches.profiler import patch_prompt_end as _patch_prompt_end
    from .patches.profiler import set_enabled as _set_profiler
    from .patches.profiler import enabled as _profiler_enabled
except ImportError:
    from patches.profiler import patch_prompt_end as _patch_prompt_end
    from patches.profiler import set_enabled as _set_profiler
    from patches.profiler import enabled as _profiler_enabled

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
# See kernels/triton/int8_gemv.py; it streams the weight once instead of computing a 16x64x32 tile.
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
ATTENTION_KERNELS = ("hand", "gemm", "hip")

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
    "attn_wrapper": None,
    "attn_fp16": True,
    "te_int8": False,
    "te_patched": False,
    "te_attention": "default",
    "triton_backend": None,
    "triton_refused_logged": False,
    "replaced_backend": None,
    "default_int8_configs": None,
    "w4a8": None,
    "qkv": False,
    "prefix_kv": False,
    "reason": "",
    "vae_enabled": True,
    "vae_t_enabled": False,
    "vae_patched": False,
    "vision_enabled": False,
    "vision_patched": False,
    "delta_prefill_enabled": False,
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
    "int8_attn_enabled": False,
    "int8_attn_rotate": 128,
    "int8_attn_patched": False,
    "qwen21_int8_cache": False,
    "qwen21_int8_patched": False,
    "qwen21_fp16": True,
    "qwen21_fp16_patched": False,
    "int8_linear_act": True,
    "int8_linear_act_patched": False,
    "qwen21_residual_fuse": True,
    "qwen21_residual_patched": False,
    "int8_tile_enabled": True,
    "small_batch_enabled": True,
    "tile_measured": False,
    "native": None,
}


def enabled():
    return os.getenv("COMFY_GPU_TUNE", "1").lower() not in ("0", "off", "false", "no")


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
        "deltanet": env_flag("COMFY_GPU_TUNE_DELTANET"),
        "kvdecode": env_flag("COMFY_GPU_TUNE_KVDECODE"),
        "llama": env_flag("COMFY_GPU_TUNE_LLAMA"),
        "qwen35": env_flag("COMFY_GPU_TUNE_QWEN35"),
        "graph": env_flag("COMFY_GPU_TUNE_GRAPH"),
    }
    _state["textgen_ready"] = {name: False for name in TEXTGEN_ITEMS}
    _state["textgen_override"] = None
    _state["forced_items"] = {item.strip().lower()
                              for item in os.getenv("COMFY_GPU_TUNE_FORCE", "").replace(",", " ").split()
                              if item.strip()}
    _state["te_dequant_env"] = os.getenv("COMFY_GPU_TUNE_TE_DEQUANT", "").strip().lower()


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


def bf16_is_emulated():
    """True on AMD cards without native bf16 (gfx906 and other old parts).

    comfy's own gate already answers this per card: should_use_bf16 is False
    for gfx900/906 and RDNA2, and True where bf16 is a hardware path. Only the
    emulated case wants the qwen21 fp16 forward; on a native bf16 card the
    cast would trade precision for nothing.
    """
    if not is_amd():
        return False
    try:
        import comfy.model_management
        return not comfy.model_management.should_use_bf16(comfy.model_management.get_torch_device())
    except Exception:
        return False


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

    * `COMFY_GPU_TUNE_FORCE`: one path opted in by hand, no questions asked;
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
        logging.warning("gpu-tune: installing %s on %s because of COMFY_GPU_TUNE_FORCE; these kernels "
                        "were not measured on this card, expect the untested ones to be slower or wrong",
                        ", ".join(forced), _state.get("arch"))


def _triton_available(patch):
    try:
        import triton  # noqa: F401
    except ImportError:
        logging.warning("gpu-tune: triton is not importable, %s patch skipped", patch)
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
    """Parse e.g. COMFY_GPU_TUNE_TILE_CONFIG="block_m=128,block_n=256,block_k=32,..."."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    spec = _parse_tile_spec(raw)
    if spec is None:
        logging.warning("gpu-tune: %s=%r is not a tile spec, using the built-in tile", name, raw)
    return spec


def _triton_config():
    import triton

    spec = _tile_spec("COMFY_GPU_TUNE_TILE_CONFIG")
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

    spec = _tile_spec("COMFY_GPU_TUNE_SMALL_TILE_CONFIG")
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
    """The decode GEMV kernel module, or None when Triton is unavailable."""
    try:
        from .kernels.triton import int8_gemv
    except ImportError:                                   # imported as a top-level module
        try:
            from kernels.triton import int8_gemv
        except ImportError:
            return None
    return int8_gemv


def load_w4a8_gemv():
    """The 4-bit weight kernel module, or None when Triton is unavailable."""
    try:
        from .kernels.triton import w4a8_gemv
    except ImportError:                                   # imported as a top-level module
        try:
            from kernels.triton import w4a8_gemv
        except ImportError:
            return None
    return w4a8_gemv


def load_hip_w4a16():
    """The W4A16 HIP loader, which owns the VALU measurement gate."""
    try:
        from .kernels.hip import w4a16
    except ImportError:                                   # imported as a top-level module
        try:
            from kernels.hip import w4a16
        except ImportError:
            return None
    return w4a16


def load_manifest():
    """The measurement-manifest module, or None when the package is incomplete."""
    try:
        from .kernels import manifest
        return manifest
    except ImportError:
        pass
    try:
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
    variables take, so an entry can be pasted into COMFY_GPU_TUNE_TILE_CONFIG and
    a card that has run tools/bench.py skips autotune without any env variable.
    """
    table = measured_table()
    raw = table.config("int8_gemm.tile", bucket) if table is not None else None
    if raw is None:
        return None
    spec = _parse_tile_spec(raw)
    if spec is None:
        logging.warning("gpu-tune: manifest int8_gemm.tile/%s=%r is not a tile spec, ignoring",
                        bucket, raw)
    return spec


def measured_state():
    """How much of the table can be used here: measured / stale / absent / off.

    stale is the one that matters: a file exists and looks authoritative, but it
    was measured under a different ROCm/Torch/Triton build, so every number in
    it is being ignored without anyone noticing unless we say so.
    """
    if "measured_state" not in _state:
        if not env_flag("COMFY_GPU_TUNE_MEASURED"):
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
            logging.warning("gpu-tune: measured data for %s has no int8 tile, so the tile list "
                            "still goes to autotune. Run tools/bench.py --all --write on this card.",
                            arch)
        return
    if state == "off":
        logging.info("gpu-tune: measured table disabled by COMFY_GPU_TUNE_MEASURED, "
                     "int8 tiles and GEMV backends fall back to autotune/probes")
    elif state == "stale":
        logging.warning("gpu-tune: the measured data for %s was taken under %s but this process runs %s: "
                        "ignoring it, so int8 tiles go to autotune and the GEMV backends to probes. "
                        "Re-run tools/bench.py --write to refresh it.",
                        arch, data.fingerprint if data else "?", data.runtime_fp if data else "?")
    elif _state["native"]:
        logging.warning("gpu-tune: no measured data for %s: the GEMV backends come from probes "
                        "(the first call of each new shape pays the sweep). "
                        "Run tools/bench.py --write once on this card to fix that.", arch)
    else:
        logging.warning("gpu-tune: no measured data for %s: int8 tiles come from the stock autotune list "
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
            line = line.replace(", in use", ", layer off (COMFY_GPU_TUNE_MEASURED)")
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
    """The DeltaNet decode kernel module, or None when Triton is unavailable."""
    try:
        from .kernels.triton import deltanet
    except ImportError:                                   # imported as a top-level module
        try:
            from kernels.triton import deltanet
        except ImportError:
            return None
    return deltanet


def load_llama_patches():
    """The core llama.py hot-path replacements."""
    try:
        from .patches.textgen import llama
    except ImportError:                                   # imported as a top-level module
        try:
            from patches.textgen import llama
        except ImportError:
            return None
    return llama


def load_qwen35_model():
    """The Qwen3.5 compute classes."""
    try:
        from .patches.textgen import qwen35
    except ImportError:                                   # imported as a top-level module
        try:
            from patches.textgen import qwen35
        except ImportError:
            return None
    return qwen35


def load_resident_graph():
    """The per-block resident CUDA graph hooks."""
    try:
        from .patches.textgen import resident_graph
    except ImportError:                                   # imported as a top-level module
        try:
            from patches.textgen import resident_graph
        except ImportError:
            return None
    return resident_graph


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


def _raw_int8_linear(module):
    """The kitchen int8_linear behind our wrapper, or None when it is absent."""
    fn = getattr(module, "int8_linear", None)
    if fn is None:
        return None
    return getattr(fn, "_gfx906_previous", fn)


def _routed_int8_linear(previous):
    def int8_linear(x, weight, weight_scale, bias=None, out_dtype=torch.bfloat16, convrot=False,
                    convrot_groupsize=256, input_act=None, input_act_weight=None,
                    input_act_eps=0.0, residual=None, residual_scale=None):
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        if rows <= INT8_GEMV_MAX_M and _state.get("gemv_model"):
            out = select_int8_gemv(x, weight, weight_scale, bias, out_dtype, input_act, residual,
                                   convrot_groupsize if convrot else 0)
            if out is not None:
                _state["int8_gemv_prompt"] = _state.get("int8_gemv_prompt", 0) + 1
                return out
        gemm = _state.get("hip_gemm_module")
        if gemm is not None and _state.get("int8_gemm_enabled", True):
            out = gemm.linear(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize,
                              input_act, input_act_weight, input_act_eps, residual, residual_scale)
            if out is not None:
                _state["hip_gemm_calls"] = _state.get("hip_gemm_calls", 0) + 1
                _state["int8_gemm_prompt"] = _state.get("int8_gemm_prompt", 0) + 1
                return out
        _state["int8_tile_prompt"] = _state.get("int8_tile_prompt", 0) + 1
        select_int8_tile(rows)
        return previous(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize,
                        input_act, input_act_weight, input_act_eps, residual, residual_scale)

    int8_linear._gfx906_tile_switch = True
    int8_linear._gfx906_previous = previous
    return int8_linear


def patch_int8_tile_switch():
    """Put this pack's kernels in front of every int8_linear dispatch can pick.

    The dispatch resolves the backend function by name on every call, and the
    registered backend is the package module (which re-exports the kernel
    function), not the quantization submodule, so both are wrapped. The triton
    and the eager backend both get it: COMFY_GPU_TUNE_TRITON_BACKEND=off moves
    dispatch to eager, and int8 linear has to keep serving the same shapes
    either way. The last resort is the triton kernel in both states because
    eager's torch._int_mm has no working hipblasLt path on gfx906, so reaching
    it would only be to fail.
    """
    try:
        import comfy_kitchen.backends.triton as triton_backend
        from comfy_kitchen.backends.triton import quantization as triton_quantization
    except ImportError:
        triton_backend = triton_quantization = None
    try:
        import comfy_kitchen.backends.eager as eager_backend
        from comfy_kitchen.backends.eager import quantization as eager_quantization
    except ImportError:
        eager_backend = eager_quantization = None
    targets = [m for m in (triton_backend, triton_quantization, eager_backend, eager_quantization) if m]
    if not targets:
        return False
    previous = _raw_int8_linear(triton_backend) or _raw_int8_linear(eager_backend)
    if previous is None:
        return False

    _state["gemv_module"] = load_int8_gemv()
    _state["gemv_model"] = feature_on("gemv")
    gemm_module = _hip_int8_gemm()
    _state["hip_gemm_module"] = gemm_module if (gemm_module is not None and gemm_module.available()) else None

    for target in targets:
        current = getattr(target, "int8_linear", None)
        if current is not None and not getattr(current, "_gfx906_tile_switch", False):
            target.int8_linear = _routed_int8_linear(previous)
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


def _hip_int8_gemm():
    """The gfx906 HIP int8 GEMM module, or None when it is not built."""
    try:
        from .kernels.hip import int8_gemm
    except ImportError:
        try:
            from kernels.hip import int8_gemm
        except ImportError:
            return None
    return int8_gemm


def _hip_f16_gemm():
    """The gfx906 HIP fp16 GEMM module, or None when it is not built."""
    try:
        from .kernels.hip import f16_gemm
    except ImportError:
        try:
            from kernels.hip import f16_gemm
        except ImportError:
            return None
    return f16_gemm


def patch_f16_linear():
    """Run big aligned fp16 nn.Linear calls on the gfx906 GEMM, delegate the rest.

    Installed on comfy.ops.disable_weight_init.Linear.forward, which
    manual_cast.Linear inherits, so it covers the non-quantized model layers.
    A call with LoRA weight/bias functions, a non-fp16 or unaligned shape, or a
    decode GEMV falls through to the original rocBLAS path.
    """
    import comfy.ops

    mod = _hip_f16_gemm()
    if mod is None or not mod.available():
        return False
    cls = comfy.ops.disable_weight_init.Linear
    if getattr(cls.forward, "_gfx906_f16", False):
        return True
    orig = cls.forward

    def forward(self, *args, **kwargs):
        comfy.ops.run_every_op()
        if args and not self.weight_function and not self.bias_function:
            out = mod.linear(args[0], self.weight, self.bias)
            if out is not None:
                return out
        return orig(self, *args, **kwargs)

    forward._gfx906_f16 = True
    cls.forward = forward
    return True


def _hip_attention():
    """The gfx906 HIP attention extension module, or None when it is not built."""
    try:
        from .kernels.hip import attention
    except ImportError:
        try:
            from kernels.hip import attention
        except ImportError:
            return None
    return attention


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
        from .kernels.triton.attention import (SUPPORTED_HEAD_DIMS, attention_fwd,
                                               supported as kernel_supported)
    except ImportError:
        from kernels.triton.attention import (SUPPORTED_HEAD_DIMS, attention_fwd,
                                              supported as kernel_supported)
    hip = _hip_attention()
    if hip is not None and hip.available():
        hip_supported, hip_fwd = hip.supported, hip.attention_fwd
    else:
        hip_supported = hip_fwd = None

    fallback = attention.optimized_attention
    if fallback in (attention.attention_sage, attention.attention_comfy_kitchen_int8):
        logging.info("gpu-tune: %s was requested explicitly, leaving attention alone",
                     getattr(fallback, "__name__", "?"))
        return False

    # The fused HIP kernel won every measured shape (tools/bench.py --attention)
    # and keeps fp32 accumulators, so it is the default when it was built. Without
    # the binary the default is gemm: the int8 Triton tiers were measured slower than
    # hip at every image shape (13.4 against 8.5 ms at 4096^2) and 30x less accurate,
    # so they are no longer offered at all.
    _state["attn_kernel"] = "hip" if hip_supported is not None else "gemm"
    _state["attn_backend_fn"] = stock_backend(_state["attn_kernel"])
    # The mode is a quality choice (relL2 1.4e-4 for the fused fp32-accumulate
    # kernels against 3.3e-4 for gemm, which rounds the score matrix to fp16), so it
    # stays a manual node switch instead of a manifest verdict.
    _state["attn_fp16"] = True
    _state["attn_contiguous"] = True
    stats = _state["attn_cast"] = {"fp16": 0, "fp32": 0, "hand": 0,
                                   "gemm": 0, "hip": 0, "sdpa": 0}
    # one line per shape that missed our kernels, not one per denoising step
    declined = set()

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
        if mode == "hip":
            if hip_supported is None or not hip_supported(q, k, v):
                key = ("hip", tuple(q.shape), tuple(k.shape), tuple(v.shape),
                       str(q.dtype), q.is_contiguous(), k.is_contiguous(), v.is_contiguous())
                if key not in declined:
                    declined.add(key)
                    logging.info("gpu-tune: hip attention declined q=%s k=%s v=%s dtype=%s "
                                 "contiguous=(%s,%s,%s), falling back to %s",
                                 tuple(q.shape), tuple(k.shape), tuple(v.shape), q.dtype,
                                 q.is_contiguous(), k.is_contiguous(), v.is_contiguous(),
                                 getattr(fallback, "__name__", "?"))
                return None
            stats["hip"] += 1
            return hip_fwd(q, k, v, softmax_scale)
        if not kernel_supported(q, k, v):
            return None
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
                        key = ("amax", tuple(q.shape), tuple(k.shape), tuple(v.shape),
                               str(q.dtype), q.is_contiguous(), k.is_contiguous(),
                               v.is_contiguous())
                        if key not in declined:
                            declined.add(key)
                            logging.warning("gpu-tune: attention amax %.0f exceeds %d, q=%s k=%s v=%s "
                                            "dtype=%s contiguous=(%s,%s,%s), leaving the inputs alone",
                                            amax, FP16_SAFE_AMAX,
                                            tuple(q.shape), tuple(k.shape), tuple(v.shape), q.dtype,
                                            q.is_contiguous(), k.is_contiguous(), v.is_contiguous())
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
                        f"gpu-tune: attention mode '{mode}' only runs on fp16 inputs but the fp16 cast is "
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
    _state["attn_wrapper"] = wrapped
    _swap_optimized_attention(fallback, wrapped)
    attention.optimized_attention = wrapped
    attention.optimized_attention_masked = wrapped
    attention._gfx906_attention = True
    _state["attention"] = True
    _state["replaced_backend"] = getattr(fallback, "__name__", "?")
    return True


def patch_int8_attention():
    """Serve comfy-kitchen's INT8 attention contract with the gfx906 kernel.

    The contract is "fp16 [B,H,S,D] in, quantize, INT8 dot, fp16 out": core's
    attention_comfy_kitchen_int8 calls prequantize_int8_attention +
    int8_attention_from_prequantized, and minimax's VAE calls int8_attention
    directly. On gfx906 comfy-kitchen answers False for both, so those models
    silently lose the path. Replacing the four comfy_kitchen entry points keeps
    the callers unchanged; the kernel that runs is hip_attn_i8 (int8 Q/K/V and
    scales, 8-bit P, fp32 softmax).

    Accuracy is the reason this is opt-in: the measured diffl2 against the fp16
    inputs is 5.4e-3 on the score/P side and 1.1-1.3e-2 on the output, against
    2.5e-4 for the fused fp16 kernel. `int8_attn_rotate` trades a matmul for
    outlier headroom; the switch is live, and with it off every call goes to the
    regular optimized attention instead (never to SDPA unless nothing else is
    installed).
    """
    import comfy_kitchen

    mod = _hip_int8_attention()
    if mod is None or not mod.available():
        return False
    _state["int8_attn_module"] = mod
    if getattr(comfy_kitchen, "_gfx906_int8_attn", False):
        return True

    class _Packed:
        __slots__ = ("q8", "qs", "k8", "ks", "vq", "vs", "scale", "dtype", "fallback")

    def _ready(q, k, v, mask):
        if mask is not None or q.dim() != 4:
            return False
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        b, hq, sq, d = q.shape
        if d not in (64, 128) or sq <= 0 or k.dim() != 4 or k.shape[2] <= 0:
            return False
        if hq % k.shape[1] or k.shape[3] != d:
            return False
        return (k.shape[0] == b and v.shape[0] == b and v.shape[1] == k.shape[1]
                and v.shape[2] == k.shape[2] and v.shape[3] == d)

    def _delegate(q, k, v, mask, scale):
        """Run the call the way it ran before this patch claimed the contract.

        Masked calls go to attention_pytorch: the contract hands masks over in
        [B, H, S, S] shape, which attention_basic/sub_quad do not accept.
        """
        from comfy.ldm.modules import attention

        heads = q.shape[1]
        kwargs = {"mask": mask, "skip_reshape": True, "skip_output_reshape": True}
        if scale is not None:
            kwargs["scale"] = scale
        wrapper = _state.get("attn_wrapper")
        # never delegate back into the int8 function itself: selecting it as the
        # attention backend would otherwise loop through this same shim
        if (mask is None and wrapper is not None
                and _state.get("attn_backend_fn") is not attention.attention_comfy_kitchen_int8):
            return wrapper(q, k, v, heads, **kwargs)
        return attention.attention_pytorch(q, k, v, heads, **kwargs)

    def prequantize(q, k, v, *, scale=None, attn_mask=None):
        p = _Packed()
        p.scale = scale
        p.dtype = q.dtype
        p.fallback = None
        if not _state.get("int8_attn_enabled", False) or not _ready(q, k, v, attn_mask):
            p.fallback = (q, k, v, attn_mask)
            return p
        # the kernel's layout is [B, S, H, D]; the contract's is [B, H, S, D]
        qt = q.transpose(1, 2).contiguous()
        kt = k.transpose(1, 2).contiguous()
        vt = v.transpose(1, 2).contiguous()
        rot = _state.get("int8_attn_rotate", 128)
        p.q8, p.qs = mod.quantize_qk(qt, rot)
        p.k8, p.ks = mod.quantize_qk(kt, rot)
        p.vq, p.vs = mod.quantize_v(vt)
        return p

    def from_prequantized(p):
        if p.fallback is not None:
            q, k, v, mask = p.fallback
            _state["int8_attn_fallbacks"] = _state.get("int8_attn_fallbacks", 0) + 1
            return _delegate(q, k, v, mask, p.scale)
        _state["int8_attn_calls"] = _state.get("int8_attn_calls", 0) + 1
        out = mod.attention_fwd(p.q8, p.qs, p.k8, p.ks, p.vq, p.vs, p.scale)
        out = out.transpose(1, 2).contiguous()
        return out if out.dtype == p.dtype else out.to(p.dtype)

    def int8_attention(q, k, v, *, scale=None, attn_mask=None):
        return from_prequantized(prequantize(q, k, v, scale=scale, attn_mask=attn_mask))

    def is_available(device=None):
        return _state.get("int8_attn_module") is not None

    comfy_kitchen.prequantize_int8_attention = prequantize
    comfy_kitchen.int8_attention_from_prequantized = from_prequantized
    comfy_kitchen.int8_attention = int8_attention
    comfy_kitchen.int8_attention_is_available = is_available
    comfy_kitchen._gfx906_int8_attn = True

    # core reads this constant when it loads a checkpoint's attention config and
    # when it registers the backend name; minimax's VAE imported a copy by value
    from comfy.ldm.modules import attention as attn

    attn.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = True
    attn.register_attention_function("comfy_kitchen_int8", attn.attention_comfy_kitchen_int8)
    for module in list(sys.modules.values()):
        if getattr(module, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", None) is False:
            module.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = True
    _state["int8_attn_patched"] = True
    return True


def _hip_int8_attention():
    """The INT8-input attention module, or None when it is not built."""
    try:
        from .kernels.hip import attention_i8
    except ImportError:
        try:
            from kernels.hip import attention_i8
        except ImportError:
            return None
    return attention_i8


def patch_convrot():
    """Replace the O(n^2) convrot rotation matmul with an O(n log n) butterfly.

    When K is a power of two of at most 32 groups, the butterfly and the int8
    per-row quantizer are one kernel (convrot_quantize): the row is rotated in
    fp32 registers and written straight out as int8 + scale, instead of the
    butterfly writing an fp16/bf16 intermediate that the quantizer reads back.
    Measured at M=4032 on fp16: K=4096 0.196 -> 0.149 ms, K=2048 0.108 -> 0.084
    ms (1.3x); skipping the fp16 intermediate also makes the quantized value
    slightly closer to an fp32 rotation. Larger K (the MLP down projection at
    12288 = 48 groups) keeps the two-pass butterfly.
    """
    from comfy_kitchen.backends.triton import quantization

    if not _triton_available("convrot"):
        _state["convrot_mode"] = "off"
        return False
    try:
        from .kernels.triton.convrot import QuantizedRotation, convrot_quantize, hadamard_rotate
    except ImportError:
        from kernels.triton.convrot import QuantizedRotation, convrot_quantize, hadamard_rotate

    _state["convrot_mode"] = resolved_convrot_mode()
    if getattr(quantization, "_gfx906_convrot", False):
        return True
    original = getattr(quantization, "_rotate_activation", None)
    original_quantize = getattr(quantization, "triton_quantize_rowwise", None)
    if original is None or original_quantize is None:
        logging.warning("gpu-tune: comfy-kitchen has no _rotate_activation/triton_quantize_rowwise, "
                        "convrot patch skipped (the rotation falls back to its matmul)")
        _state["convrot_mode"] = "off"
        return False
    _state["original_rotate"] = original

    def rotate_activation(x, h, group_size):
        mode = _state["convrot_mode"]
        if mode == "off" or group_size != CONVROT_GROUP or x.shape[-1] % CONVROT_GROUP or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return original(x, h, group_size)
        groups = x.shape[-1] // CONVROT_GROUP
        if mode == "fused" and groups <= FUSED_MAX_GROUPS and not groups & (groups - 1):
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


def triton_backend_requested():
    """Whether the comfy-kitchen triton backend should be selectable.

    Unset env + a plain start enables it, which is what the pack has always
    done; an explicit env value is the A/B knob (eager vs triton for the ops
    this pack does not cover), and it wins over the command line. With nothing
    in the env, --disable-triton-backend is a refusal we no longer override.
    """
    want = flag_value(os.getenv("COMFY_GPU_TUNE_TRITON_BACKEND", ""))
    if want is not None:
        return want
    from comfy.cli_args import args
    return not args.disable_triton_backend


def enable_triton_backend():
    """comfy/quant_ops.py disables the comfy-kitchen triton backend at import
    unless --enable-triton-backend was passed, so turn it back on from here.

    Its import-time log line reports the old state, hence the extra line below.
    """
    from comfy_kitchen import registry

    if registry.is_available("triton"):
        return True
    if not triton_backend_requested():
        if not _state.get("triton_refused_logged"):
            _state["triton_refused_logged"] = True
            logging.info("gpu-tune: comfy-kitchen triton backend left off (COMFY_GPU_TUNE_TRITON_BACKEND "
                         "or --disable-triton-backend); kitchen dispatch uses eager for every op this "
                         "pack does not cover, and eager int8 matmul has no working hipblasLt path on "
                         "gfx906")
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        logging.warning("gpu-tune: triton is not importable, comfy-kitchen stays on the eager backend")
        return False
    registry.enable("triton")
    ok = registry.is_available("triton")
    if ok:
        logging.info("gpu-tune: comfy-kitchen triton backend enabled (same as --enable-triton-backend, "
                     "the backend list printed above was from before this line)")
    return ok


def triton_backend_active():
    from comfy_kitchen import registry

    return registry.is_available("triton")


def configure(attention=None, convrot=None, profile=None, text_encoder=None,
              fp16=None, text_attention=None, contiguous=None, qkv=None,
              vae=None, gemv=None, textgen=None, prefix_kv=None, text_gemv=None, w4a8=None,
              amax_guard=None, int8_tile=None, small_batch=None, image_tower=None,
              int8_gemm=None, int8_attn=None, int8_attn_rotate=None, qwen21_int8_cache=None,
              qwen21_fp16=None, int8_linear_act=None, qwen21_residual_fuse=None):
    """Runtime settings, used by the GPU Tuning nodes.

    Every argument is three-state: None/"default" keeps the built-in default
    (the manifest verdict where one exists), other values are explicit
    overrides. Node parameters therefore always win.
    """
    if attention is not None:
        wanted = str(attention).strip().lower()
        if wanted == "default":
            hip = _hip_attention()
            wanted = "hip" if hip is not None and hip.available() else "gemm"
        if wanted == "hip":
            hip = _hip_attention()
            if hip is None or not hip.available():
                logging.info("gpu-tune: HIP attention extension is not built, keeping mode %s",
                             _state["attn_kernel"])
                wanted = _state["attn_kernel"]
        # combo values are untrusted: stale workflows may still carry removed names
        _state["attn_kernel"] = (wanted if wanted in ATTENTION_KERNELS
                                 or stock_backend(wanted) is not None else "gemm")
        _state["attn_backend_fn"] = stock_backend(_state["attn_kernel"])
    if contiguous is not None:
        _state["attn_contiguous"] = resolve_flag(contiguous, True)
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
    if int8_gemm is not None:
        _state["int8_gemm_enabled"] = resolve_flag(int8_gemm, True)
    if int8_attn is not None:
        _state["int8_attn_enabled"] = resolve_flag(int8_attn, False)
    if int8_attn_rotate is not None:
        raw = str(int8_attn_rotate).strip().lower()
        if raw in ("off", "0", "none"):
            _state["int8_attn_rotate"] = 0
        elif raw in ("32", "64", "128"):
            _state["int8_attn_rotate"] = int(raw)
    if image_tower is not None and _state["vision_patched"]:
        # "default" keeps the boot decision (manifest verdict, or a FORCE opt-in)
        _state["vision_enabled"] = resolve_flag(image_tower, _state["vision_enabled"])
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
    if qwen21_int8_cache is not None:
        _state["qwen21_int8_cache"] = resolve_flag(qwen21_int8_cache, False)
    if qwen21_fp16 is not None:
        _state["qwen21_fp16"] = resolve_flag(qwen21_fp16, True)
    if int8_linear_act is not None:
        _state["int8_linear_act"] = resolve_flag(int8_linear_act, True)
    if qwen21_residual_fuse is not None:
        _state["qwen21_residual_fuse"] = resolve_flag(qwen21_residual_fuse, True)
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
            logging.info("gpu-tuning status:\n%s", status())
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

    COMFY_GPU_TUNE_TE_DEQUANT (on/off) and COMFY_GPU_TUNE_FORCE=te_dequant outrank
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
        logging.warning("gpu-tune: comfy.ops has no mixed_precision_ops, text-encoder patch skipped")
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
            logging.warning("gpu-tune: MixedPrecisionOps.Linear has no forward_comfy_cast_weights, "
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
                logging.info("gpu-tune: qkv fusion off, %s", reason)
        else:
            if not _state.get("qkv_on_logged"):
                _state["qkv_on_logged"] = True
                logging.info("gpu-tune: qkv fusion on, one %s int8 gemm, rotation %s",
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


def patch_qwen21_int8_cache():
    """Run the Qwen-Image 2.1 int8 prefix cache through the int8 attention kernel.

    The cache already stores K/V as convrot int8 with one scale per token, but
    the shipped read path dequantizes both to fp16 before every block. This
    keeps them int8: take() stops dequantizing, the current step's Q/K/V are
    rotated and quantized with the same H64 in one fused pass (attn_quant.cu),
    V is packed four keys per word straight out of both sources, the kernel
    folds the cached per-token V scales into P, and the output is rotated once
    more (H is symmetric, so the same transform undoes it). Nothing pays for
    the extra precision the cache never had.

    Measured at 4058 prefix + 4032 target tokens, 32 heads, d128: 50.6 ms per
    block shipped vs 37.4 ms here (1.35x), output diffl2 1.1e-2 against the
    fp16-on-dequant reference (2.6e-4). Opt-in: the switch is live.
    """
    import comfy.ldm.qwen_image21.model as qwen
    import comfy.model_management
    import comfy.utils
    from comfy.ldm.wan.model_animate2 import PoseBranchCache

    mod = _hip_int8_attention()
    if mod is None or not mod.available() or not mod.rotation_ready():
        return False
    if getattr(PoseBranchCache, "_gfx906_int8_cache", False):
        return True

    original_take = PoseBranchCache.take

    def take(self, i, device, dtype, batch_size):
        params = self.slot["params"][i] if self.slot is not None else None
        shape = self.slot["shape"] if self.slot is not None else None
        usable = (params is not None and shape is not None and len(shape) == 5
                  and shape[4] in (64, 128) and getattr(params, "convrot", False)
                  and getattr(params, "convrot_groupsize", 0) == 64)
        if not (_state.get("qwen21_int8_cache") and self.dtype == "int8" and usable):
            return original_take(self, i, device, dtype, batch_size)
        if i not in self._pending:
            self.prefetch(i, device, dtype)
        t, stream = self._pending.pop(i)
        comfy.model_management.sync_stream(device, stream)
        params = params.to_device(t.device)
        kv8 = t.reshape(shape)
        scale = params.scale.reshape(shape[0], 2, shape[2], shape[3]).float()
        _state["qwen21_take"] = (i, kv8, scale)
        _state["qwen21_take_calls"] = _state.get("qwen21_take_calls", 0) + 1
        _state["qwen21_prompt_takes"] = _state.get("qwen21_prompt_takes", 0) + 1
        return comfy.utils.repeat_to_batch_size(kv8, batch_size)

    PoseBranchCache.take = take
    PoseBranchCache._gfx906_int8_cache = True

    previous_prefix = qwen.prefix_cached_attention

    def prefix_cached_attention(prefix_k, prefix_v, transformer_options={}):
        taken = _state.pop("qwen21_take", None)
        if (taken is not None and _state.get("qwen21_int8_cache")
                and prefix_k.dtype == torch.int8):
            kv8, scale = taken[1], taken[2]
            if (kv8.shape[0] == prefix_k.shape[0] and kv8.shape[2] == prefix_k.shape[1]
                    and kv8.shape[3] == prefix_k.shape[2] and kv8.shape[4] == prefix_k.shape[3]):
                return _int8_cache_attention(kv8, scale)
        return previous_prefix(prefix_k, prefix_v, transformer_options)

    def _int8_cache_attention(kv8, scale):
        def attn(q, k, v, heads, preferred_attention=None):
            q_t, k_t, v_t = q.take(), k.take(), v.take()
            out = mod.cache_attention(q_t.contiguous(), k_t.contiguous(), v_t.contiguous(),
                                      kv8, scale)
            _state["qwen21_int8_calls"] = _state.get("qwen21_int8_calls", 0) + 1
            _state["qwen21_prompt_calls"] = _state.get("qwen21_prompt_calls", 0) + 1
            b, st, h, d = q_t.shape
            if out.dtype != q_t.dtype:
                out = out.to(q_t.dtype)
            return out.reshape(b, st, h * d)
        return attn

    qwen.prefix_cached_attention = prefix_cached_attention
    _state["qwen21_int8_patched"] = True
    return True


def _int8_linear_act(linear, x, input_act, act_weight=None, act_eps=0.0,
                     residual=None, residual_scale=None):
    """``linear(act(x))`` with the int8 weight left quantized, or None.

    comfy.ops.linear_input_act hands cast_bias_weight ``compute_dtype=x.dtype``,
    so when the weight's orig_dtype differs from the activation dtype (the
    gfx906 case: an fp32 manual-cast model with bf16 activations) the cast
    requantizes the int8 weight from float and then dequantizes it. The fused
    int8 path never runs and the projection becomes a full bf16/fp16 GEMM.

    The gate is the mismatch itself: when the two dtypes already agree the
    shipped function keeps the QuantizedTensor, so this returns None and the
    caller delegates unchanged. That makes the patch a no-op on cards/models
    where manual_cast != activation dtype, and only takes over the wasteful
    requantize->dequantize case (Qwen-Image 2.1, LTXV, MiniMaxH3, and any other
    linear_input_act site with that combination).

    Returns None when the weight is not a plain int8 layout or the dtypes
    match.
    """
    import comfy.model_management
    import comfy.ops
    from comfy import quant_ops
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy.quant_ops import TensorWiseINT8Layout

    weight = linear.weight
    if (comfy.model_management.in_training
            or not isinstance(weight, QuantizedTensor)
            or isinstance(x, QuantizedTensor)
            or weight._layout_cls != "TensorWiseINT8Layout"
            or weight.dtype == x.dtype
            or getattr(weight._params, "transposed", False)
            or getattr(linear, "_full_precision_mm", False)
            or len(linear.weight_function) or len(linear.bias_function)):
        return None
    w, bias, stream = comfy.ops.cast_bias_weight(
        linear, x, offloadable=True, dtype=weight.dtype, bias_dtype=x.dtype,
        compute_dtype=x.dtype, want_requant=True)
    try:
        if not isinstance(w, QuantizedTensor):
            return None
        qdata, scale = TensorWiseINT8Layout.get_plain_tensors(w)
        return quant_ops.ck.int8_linear(
            x, qdata, scale, bias, x.dtype,
            convrot=getattr(w._params, "convrot", False),
            convrot_groupsize=getattr(w._params, "convrot_groupsize", 256),
            input_act=input_act, input_act_weight=act_weight,
            input_act_eps=act_eps, residual=residual,
            residual_scale=residual_scale)
    finally:
        comfy.ops.uncast_bias_weight(linear, w, bias, stream)


def patch_int8_linear_act():
    """Keep int8 weights quantized at every ``comfy.ops.linear_input_act`` site.

    One choke point covers the MLP down projections and pre-norm blocks of
    Qwen-Image 2.1, LTXV, MiniMaxH3, Wan and the llama text encoder. On the
    measured gfx906 shape the Qwen-Image 2.1 SwiGLU down projection alone goes
    from a 151 ms/block bf16 rocBLAS GEMM to the int8 kernel: 7.92 -> 3.34 s
    per forward (2.37x) with no change to the activation dtype.

    The wrapper only takes the call when _int8_linear_act sees the dtype
    mismatch that makes the shipped path dequantize; everything else goes to
    the shipped function untouched, so models that already stay int8 (Flux,
    Wan's default fp16, fp32 runs) measure identical. The switch is live.
    """
    import comfy.ops

    if getattr(comfy.ops, "_gfx906_linear_act", False):
        return True
    original = comfy.ops.linear_input_act

    def linear_input_act(linear, x, input_act, act_weight=None, act_eps=0.0,
                         residual=None, residual_scale=None):
        if _state.get("int8_linear_act", True):
            out = _int8_linear_act(linear, x, input_act, act_weight, act_eps,
                                   residual, residual_scale)
            if out is not None:
                _state["int8_linear_act_calls"] = _state.get("int8_linear_act_calls", 0) + 1
                return out
        return original(linear, x, input_act, act_weight, act_eps, residual, residual_scale)

    comfy.ops.linear_input_act = linear_input_act
    comfy.ops._gfx906_linear_act = True
    _state["int8_linear_act_patched"] = True
    return True


def patch_qwen21_fp16():
    """Run the Qwen-Image 2.1 DiT in fp16 when it came in as bf16.

    gfx906 emulates bf16 in fp32 and has no fast bf16 rocBLAS tile, while fp16
    is the card's native path. Casting the whole forward once (hidden states,
    context, reference latents) also makes the int8 linears' out_dtype fp16 and
    their convrot rotation fp16, instead of one cast at the attention boundary.
    Measured at 64x64 latent: 7.92 -> 3.83 s per forward (2.07x); the output is
    closer to an fp32 run than the bf16 baseline is (cosine 0.9945 vs 0.9880).

    Only bf16 inputs are touched (fp16/fp32 models keep their dtype) and the
    patch is installed only where comfy reports bf16 as emulated (gfx906 and
    other old AMD), so a native-bf16 card never pays for the cast. Like the
    attention cast this is guarded: if the fp16 pass produces a non-finite
    output the prefix cache is dropped and the call is rerun in the model dtype.
    """
    import comfy.ldm.qwen_image21.model as qwen

    cls = qwen.QwenImage21Transformer2DModel
    if getattr(cls, "_gfx906_qwen21_fp16", False):
        return True
    original = cls._forward

    def _forward(self, x, timesteps, context, ref_latents=None, image_slots=None,
                 transformer_options={}, **kwargs):
        if not _state.get("qwen21_fp16", True) or x.dtype != torch.bfloat16:
            return original(self, x, timesteps, context, ref_latents, image_slots,
                            transformer_options, **kwargs)
        half_ctx = (context.half() if isinstance(context, torch.Tensor)
                    and context.is_floating_point() else context)
        half_refs = [r.half() for r in ref_latents] if ref_latents is not None else ref_latents
        out = original(self, x.half(), timesteps, half_ctx, half_refs, image_slots,
                       transformer_options, **kwargs)
        if torch.isfinite(out).all():
            _state["qwen21_fp16_calls"] = _state.get("qwen21_fp16_calls", 0) + 1
            return out.to(x.dtype)
        # overflowed past the block clamp: drop the fp16 prefix cache (it was
        # filled from the failed pass) and rerun this call in the model dtype
        if not _state.get("qwen21_fp16_overflow_logged"):
            _state["qwen21_fp16_overflow_logged"] = True
            logging.warning("gpu-tune: qwen21 fp16 forward overflowed; rerunning in %s. "
                            "Set qwen21_fp16=off to skip the retry", x.dtype)
        self.reset_prefix_cache(self.prefix_cache_enabled)
        return original(self, x, timesteps, context, ref_latents, image_slots,
                        transformer_options, **kwargs)

    cls._forward = _forward
    cls._gfx906_qwen21_fp16 = True
    _state["qwen21_fp16_patched"] = True
    return True


def _residual_fusion_ready():
    """True when the loaded int8 GEMM has the v2 residual epilogue."""
    gemm = _state.get("hip_gemm_module")
    return bool(gemm is not None and getattr(gemm, "supports_residual", lambda: False)())


def _uniform_gate(gate, prefix_len, rows):
    """The per-channel gate that covers every row, flattened to [D], or None.

    gate is the (prefix, target) pair from _split_rows. The int8 epilogue takes
    one residual_scale for the whole tile, so only an all-prefix or all-target
    call can use it; the mixed first pass keeps the separate addcmul.
    """
    if prefix_len == 0:
        return gate[1].reshape(-1)
    if prefix_len == rows:
        return gate[0].reshape(-1)
    return None


def _attn_with_residual(attn, residual, scale, *args, **kwargs):
    """Run attention with its to_out GEMM adding the gated residual.

    Returns (attention result, whether the residual was fused). The to_out
    module's bound forward is shadowed for the duration of the call so the
    shipped Attention.forward body is not duplicated.
    """
    to_out = attn.to_out[0]
    plain = to_out.forward
    used = [False]

    def forward(x):
        out = _int8_linear_act(to_out, x, None, residual=residual, residual_scale=scale)
        if out is None:
            return plain(x)
        used[0] = True
        return out

    to_out.forward = forward
    try:
        result = attn(*args, **kwargs)
    finally:
        del to_out.forward
    return result, used[0]


def patch_qwen21_residual():
    """Fuse the Qwen-Image 2.1 gated residuals into the int8 GEMM epilogue.

    The block does x += gate * layer_out twice; both layer outputs come from an
    int8 GEMM whose epilogue can do that. The kernel's v2 export takes
    ``residual + rscale[n] * out`` (the same form as comfy_kitchen's
    _apply_residual), so the separate addcmul kernel and its 66 MB round trip
    per residual go away. Measured on gfx906 at the 4032-token shape: the MLP
    down GEMM 14.66 -> 14.56 ms, to_out 5.02 -> 4.99 ms (~0.1% of the whole
    step, below the pack's measurement noise); the switch is live, off restores
    the shipped block. Only all-prefix or all-target calls fuse (one scale
    covers every row); the mixed first pass keeps _gated_residual.
    """
    import comfy.ldm.qwen_image21.model as qwen
    import comfy.ops

    cls = qwen.QwenImage21TransformerBlock
    if getattr(cls, "_gfx906_residual", False):
        return True
    original = cls.forward
    gated = qwen._gated_residual
    modulated = qwen._modulated_norm

    def forward(self, x, mod, pe, attn_fn, prefix_len, transformer_options={}):
        if not _state.get("qwen21_residual_fuse", True) or not _residual_fusion_ready():
            return original(self, x, mod, pe, attn_fn, prefix_len, transformer_options)
        scale1, gate1, scale2, gate2, zero = mod
        normed = modulated(self.img_norm1, x, scale1, prefix_len, zero)
        g = _uniform_gate(gate1, prefix_len, x.shape[1])
        if g is None:
            x = gated(x, self.attn(normed, pe, attn_fn, prefix_len, transformer_options),
                      gate1, prefix_len)
        else:
            y, used = _attn_with_residual(self.attn, x, g, normed, pe, attn_fn,
                                          prefix_len, transformer_options)
            if used:
                x = y
                _state["qwen21_residual_calls"] = _state.get("qwen21_residual_calls", 0) + 1
            else:
                x = gated(x, y, gate1, prefix_len)
        normed = modulated(self.img_norm2, x, scale2, prefix_len, zero)
        g = _uniform_gate(gate2, prefix_len, x.shape[1])
        fused = None
        if g is not None and getattr(self.img_mlp, "fused", False):
            gate_up = self.img_mlp.gate_up(normed)
            fused = _int8_linear_act(self.img_mlp.out, gate_up, "swiglu",
                                     residual=x, residual_scale=g)
            if fused is None:
                x = gated(x, comfy.ops.linear_input_act(self.img_mlp.out, gate_up, "swiglu"),
                          gate2, prefix_len)
            else:
                _state["qwen21_residual_calls"] = _state.get("qwen21_residual_calls", 0) + 1
        else:
            x = gated(x, self.img_mlp(normed), gate2, prefix_len)
        if fused is not None:
            x = fused
        if x.dtype == torch.float16:
            x.clamp_(-65504, 65504)
        return x

    cls.forward = forward
    cls._gfx906_residual = True
    _state["qwen21_residual_patched"] = True
    return True


def split_t_conv3d(input, weight, bias, stride, padding, dilation, groups):
    """A depth>1 Conv3d as a sum of one Conv2d per depth tap.

    Only a unit depth dilation is expressible; anything else returns None so
    the caller delegates to the shipped Conv3d.
    """
    if dilation[0] != 1 or input.ndim != 5 or weight.ndim != 5:
        return None
    kt = weight.shape[2]
    if kt < 2 or input.shape[2] + 2 * padding[0] < kt:
        return None
    n, c, d, h, w = input.shape
    sd, pad_t = stride[0], padding[0]
    d_out = (d + 2 * pad_t - kt) // sd + 1
    if d_out < 1:
        return None
    x = torch.nn.functional.pad(input, (0, 0, 0, 0, pad_t, pad_t)) if pad_t else input
    out = None
    for t in range(kt):
        frames = x[:, :, t : t + (d_out - 1) * sd + 1 : sd][:, :, :d_out]
        if frames.shape[2] != d_out:
            return None
        flat = frames.permute(0, 2, 1, 3, 4).reshape(n * d_out, c, h, w)
        part = torch.nn.functional.conv2d(
            flat, weight[:, :, t], None, stride[1:], padding[1:], dilation[1:], groups)
        part = part.reshape(n, d_out, *part.shape[1:]).permute(0, 2, 1, 3, 4)
        out = part if out is None else out + part
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1, 1)
    return out


def patch_vae():
    """Run depth-1 Conv3d as Conv2d.

    The Wan-style video VAEs that image models ship with (Qwen-Image 2.1 uses the
    Wan 2.2 decoder) are 76 Conv3d layers with a one-frame kernel, and an image
    gives them a single frame. There is no conv3d kernel for gfx906, so torch
    lowers every one of them to im2col + bmm; the profiler puts that at 47% of
    decode time. Squeezing the depth axis makes them ordinary conv2d calls that
    land on the fp16 MIOpen kernels: 2.31 s -> 1.09 s at 1024x1024.

    Multi-frame (K_T>1) convs with unit depth dilation are split into K_T Conv2d
    taps and summed (split_t_conv3d), gated by feature.vae_t. Core only sends
    autopad="causal_zero" on a single frame (the weight is truncated to one tap
    there); multi-frame causal convs arrive pre-padded as ordinary conv3d and
    take the same split. tools/bench.py --vae-t carries the evidence. Anything
    else still delegates to the shipped Conv3d.

    The depth-1 switch is read in the hot path, so the node can turn it off
    without a reload; the shipped Conv3d path is reachable when a gate is off.
    """
    import comfy.ops

    conv3d = comfy.ops.disable_weight_init.Conv3d
    if getattr(conv3d, "_gfx906_vae", False):
        return True
    original = conv3d._conv_forward
    _state["vae_patched"] = True

    def _conv_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
        depth1 = _state.get("vae_enabled", True)
        temporal = _state.get("vae_t_enabled", False)
        if depth1 or temporal:
            causal = autopad == "causal_zero"
            if causal:
                weight = weight[:, :, -input.shape[2]:, :, :]
                autopad = None
            if (depth1 and input.ndim == 5 and input.shape[2] == 1
                    and weight.shape[2] == 1):
                return torch.nn.functional.conv2d(
                    input[:, :, 0], weight[:, :, 0], bias, self.stride[1:],
                    self.padding[1:], self.dilation[1:], self.groups).unsqueeze(2)
            if (temporal and input.ndim == 5 and weight.shape[2] > 1
                    and self.dilation[0] == 1
                    and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
                    and weight.dtype == input.dtype):
                out = split_t_conv3d(input, weight, bias, self.stride, self.padding,
                                     self.dilation, self.groups)
                if out is not None:
                    return out
        return original(self, input, weight, bias, autopad=autopad, *args, **kwargs)

    conv3d._conv_forward = _conv_forward
    conv3d._gfx906_vae = True
    return True


def patch_vision():
    """Qwen3.5 image tower: keep its own dtype and answer the patch Conv3d with a GEMM.

    Core hands the tower an fp32 input (qwen35.py:735) while its weights are the
    text-encoder dtype, so every vision linear upcasts its weights on every call:
    4.04 -> 2.71 s for a 1160x679 image on this card with the dtype alone, and
    1.5-1.7 s with both parts here. The tower output is bit-identical for the
    dtype change because the result lands in the weight dtype anyway.

    The patch projection is a Conv3d whose kernel equals its stride
    ((temporal, patch, patch)), so each output element is one dot product over a
    whole flattened patch and the conv is exactly a Linear. gfx906 has no conv3d
    kernel, so torch lowers it to im2col + bmm: 2.24 s of the 2.7 s left above,
    unchanged by dtype. Measured against F.conv3d on the same weight the GEMM is
    the same answer to within the accumulation order (4.2e-5 maxabs on a 29.8
    scale in fp32, <= 2 bf16 ulp), and it keeps the same weight tensor, so
    checkpoint loading and state dicts do not change.

    Both parts sit behind one live switch (the CLIP node's `image_tower`) whose
    default is the manifest's `feature.vision`, so an unmeasured card keeps the
    shipped path and can still be switched on by hand. Inside the fast path a
    plain-tensor Conv3d with groups 1, no padding, unit dilation and kernel equal
    to stride takes the GEMM; anything else falls through.
    """
    from comfy.text_encoders import qwen35
    from comfy.quant_ops import QuantizedTensor
    import comfy.ops

    if getattr(qwen35.Qwen35VisionModel, "_gfx906_vision", False):
        return True
    model = qwen35.Qwen35VisionModel
    patch = qwen35.Qwen35VisionPatchEmbed
    original_forward = model.forward
    original_proj = patch.forward

    def forward(self, x, grid_thw):
        if not _state.get("vision_enabled", False):
            return original_forward(self, x, grid_thw)
        dtype = next(self.parameters()).dtype
        return original_forward(self, x.to(dtype), grid_thw)

    def proj_forward(self, x):
        conv = self.proj
        weight = conv.weight
        kernel = tuple(conv.kernel_size)
        if (not _state.get("vision_enabled", False)
                or not isinstance(weight, torch.Tensor) or isinstance(weight, QuantizedTensor)
                or weight.is_meta or conv.groups != 1
                or tuple(weight.shape[2:]) != kernel or tuple(conv.stride) != kernel
                or any(conv.padding) or any(d != 1 for d in conv.dilation)):
            return original_proj(self, x)
        # kernel equals stride: every output element is one dot product over a
        # whole flattened patch, so the conv is exactly a Linear. Cast through the
        # same door the shipped Conv3d uses so offload and compute dtype hold.
        flat = x.reshape(x.shape[0], -1)
        w, b, stream = comfy.ops.cast_bias_weight(conv, flat, offloadable=True,
                                                  compute_dtype=flat.dtype)
        try:
            out = torch.nn.functional.linear(flat, w.reshape(w.shape[0], -1), b)
        finally:
            comfy.ops.uncast_bias_weight(conv, w, b, stream)
        return out.view(-1, self.embed_dim)

    model.forward = forward
    patch.forward = proj_forward
    model._gfx906_vision = True
    _state["vision_patched"] = True
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
        _state["reason"] = "disabled by COMFY_GPU_TUNE"
        return _state
    if not torch.cuda.is_available():
        _state["reason"] = "no CUDA device"
        logging.info("gpu-tuning skipped: %s", _state["reason"])
        return _state

    _state["native"] = native_backend()
    on_amd = is_amd()
    if on_amd and _state["native"]:
        logging.info("gpu-tune: comfy-kitchen %s serves this %s; the triton kernel patches "
                     "below only run where its per-call constraints decline a shape",
                     _state["native"], _state["arch"])
    # a hand-measured tile spec, or one measured by tools/bench.py on this card
    # (or shipped for its arch), replaces the autotune list with a single config
    _state["tile_measured"] = (_tile_spec("COMFY_GPU_TUNE_TILE_CONFIG") is not None
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
    # non-quantized fp16 linear: big aligned GEMMs beat rocBLAS on the measured
    # arch; COMFY_GPU_TUNE_F16_GEMM forces it on/off for A/B.
    if env_flag("COMFY_GPU_TUNE_F16_GEMM", "1" if feature_on("f16_gemm") else "0"):
        _state["f16_gemm_enabled"] = True
        _install("f16 gemm", patch_f16_linear)
    if feature_on("convrot"):
        _install("convrot", patch_convrot)
    _state["vae_enabled"] = feature_on("vae")
    _state["vae_t_enabled"] = feature_on("vae_t")
    if _state["vae_enabled"] or _state["vae_t_enabled"]:
        _install("vae", patch_vae)
    # the image tower belongs to the text encoder, but whether its conv3d and fp32
    # cast cost anything is a per-card fact: the wrappers are always installed so
    # the node switch works everywhere, and the measured verdict decides the default
    _install("vision", patch_vision)
    _state["vision_enabled"] = install_ready("vision")
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
    _install("int8 linear_act", patch_int8_linear_act)
    _install("qwen21 int8 cache", patch_qwen21_int8_cache)
    if bf16_is_emulated():
        _install("qwen21 fp16", patch_qwen21_fp16)
    _install("qwen21 residual", patch_qwen21_residual)
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
    if install_ready("deltanet") or install_ready("deltanet_prefill"):
        deltanet_kernels = _install("deltanet loader", load_deltanet)
        if deltanet_kernels:
            _state["textgen_ready"]["deltanet"] = bool(
                _install("deltanet", deltanet_kernels.install, lambda: textgen_active("deltanet")))
            # the prefill transform is a separate measurement: gfx906 may want
            # one and not the other, and both ride the textgen switch when installed
            _state["delta_prefill_enabled"] = install_ready("deltanet_prefill")
            if _state["delta_prefill_enabled"]:
                _install("deltanet prefill", deltanet_kernels.install_prefill,
                         lambda: textgen_active("deltanet"))
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
    # per-block CUDA graphs for resident weights: installed here, not in core.
    # The gate follows the CLIP node's textgen switch like the other four items.
    resident_graph = _install("resident graph loader", load_resident_graph)
    _state["resident_graph"] = bool(
        _install("resident graph", resident_graph.install,
                 lambda: textgen_requested("graph")) if resident_graph else None)
    # attention kernels differ in quality (int8 PV rel ~2e-2 vs hand 2.5e-4), so
    # the mode is a manual node/env choice, not a manifest verdict -- and the TE
    # never routes here anyway (its resolver returns core's pytorch/basic). It
    # installs on every AMD; NVIDIA keeps core's attention.
    if on_amd:
        _install("attention", patch_attention)
        _install("int8 attention", patch_int8_attention)
    logging.info(
        "gpu-tuning active: int8 linear %s%s%s, convrot %s, vae %s, qkv %s, attention %s%s%s%s, "
        "text encoder %s + lm_head, encoder attention %s, textgen %s",
        "measured tile" if _state.get("int8_tile_replaced") else ("on (stock candidates)" if _state["int8_linear"] else "off"),
        " + decode gemv" if _state.get("gemv_model") else "",
        " + hip gemm" if _state.get("hip_gemm_module") is not None and _state.get("int8_gemm_enabled", True) else "",
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
    """One-shot lines for the profiler: attention casts and kernel routing."""
    lines = []
    cast = _state["attn_cast"]
    if cast:
        lines.append(f"attention fp16-ok {cast['fp16']} / amax overflow {cast['fp32']}"
                     f" (own kernel: hip {cast['hip']}, hand {cast['hand']},"
                     f" gemm {cast['gemm']}, sdpa {cast['sdpa']})")
        for k in cast:
            cast[k] = 0
    if _state.get("qwen21_int8_cache"):
        takes = _state.pop("qwen21_prompt_takes", 0)
        calls = _state.pop("qwen21_prompt_calls", 0)
        lines.append(f"qwen21 int8 cache: {calls} attention calls / {takes} int8 takes this prompt"
                     f" (total {_state.get('qwen21_int8_calls', 0)}/"
                     f"{_state.get('qwen21_take_calls', 0)})")
    if _state.get("int8_linear"):
        gemm = _state.pop("int8_gemm_prompt", 0)
        gemv = _state.pop("int8_gemv_prompt", 0)
        tile = _state.pop("int8_tile_prompt", 0)
        lines.append(f"int8 linear: {gemm} hip gemm / {gemv} gemv / {tile} triton this prompt")
    return lines


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
    hip_gemm = _state.get("hip_gemm_module")
    gemm_line = ("int8 gemm hip " + ("on" if _state.get("int8_gemm_enabled", True) else "off")
                 + f" ({_state.get('hip_gemm_calls', 0)} calls)") if hip_gemm is not None else "int8 gemm hip n/a"
    w4a16 = load_hip_w4a16()
    valu_line = ("w4a16 VALU levels " + ("on" if w4a16.valu_enabled() else "off")
                 + (" (env)" if os.getenv("COMFY_GPU_TUNE_W4A16_VALU") else
                    " (measured)" if measured_state() == "measured" else "")) \
        if w4a16 is not None else "w4a16 VALU levels n/a"
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
        f"comfy-kitchen triton backend "
        + ("on" if triton_backend_active() else
           "off (COMFY_GPU_TUNE_TRITON_BACKEND / --disable-triton-backend; int8 linear keeps this "
           "pack's kernels, every other kitchen op runs eager)"),
        tile_line,
        small_line,
        gemm_line,
        f"convrot {_state['convrot_mode'] or 'off'}",
        valu_line,
        f"qkv {'one int8 gemm' if _state.get('qkv') else 'off'}"
        + (f", {_state['qkv_calls']} fused / {_state.get('qkv_fallbacks', 0)} fallback"
           if _state.get("qkv_calls") or _state.get("qkv_fallbacks") else ""),
        prefix_line,
        (f"qwen21 int8 cache {'on' if _state.get('qwen21_int8_cache') else 'off'}"
         + (f", {_state.get('qwen21_int8_calls', 0)} calls" if _state.get("qwen21_int8_calls") else "")
         + ("" if _state.get("qwen21_int8_cache") else " (fp16 dequant path)"))
        if _state.get("qwen21_int8_patched") else "qwen21 int8 cache n/a",
        (f"qwen21 fp16 {'on' if _state.get('qwen21_fp16') else 'off'}"
         + (f", {_state.get('qwen21_fp16_calls', 0)} calls" if _state.get("qwen21_fp16_calls") else ""))
        if _state.get("qwen21_fp16_patched") else "qwen21 fp16 n/a",
        (f"int8 linear_act {'on' if _state.get('int8_linear_act') else 'off'}"
         + (f", {_state.get('int8_linear_act_calls', 0)} calls"
            if _state.get("int8_linear_act_calls") else ""))
        if _state.get("int8_linear_act_patched") else "int8 linear_act n/a",
        (f"qwen21 residual fused {'on' if _state.get('qwen21_residual_fuse') else 'off'}"
         + (f", {_state.get('qwen21_residual_calls', 0)} gemms"
            if _state.get("qwen21_residual_calls") else "")
         + ("" if _residual_fusion_ready() else ", kernel v2 missing"))
        if _state.get("qwen21_residual_patched") else "qwen21 residual n/a",
        gemv_line,
        text_gemv_line,
        w4a8_line,
        f"vae {'conv3d as conv2d' if _state.get('vae_patched') and _state.get('vae_enabled') else ('off (patched)' if _state.get('vae_patched') else 'off')}"
        + f", t {'split conv2d' if _state.get('vae_t_enabled') else 'off'}",
        f"textgen {textgen_summary()}"
        + f", prefill {'block' if _state['delta_prefill_enabled'] else 'core loop'}",
        f"attention {_state['attn_kernel'] if _state['attention'] else 'off'}"
        + ((", inputs cast to fp16" if _state["attn_fp16"] else ", inputs left in the model dtype")
           + f", amax guard {'on' if _state.get('amax_guard', True) else 'off'}"
           if _state["attention"] else "")
        + (f", rest to {_state['replaced_backend']}" if _state.get("replaced_backend") else ""),
        f"text encoder {'int8' if _state['te_int8'] else 'default (dequantize per call)'}"
        + f", lm_head {'int8' if _state.get('logits_patched') and _state['te_int8'] else 'default'}"
        + f", encoder attention {_state['te_attention']}",
        f"int8 attention "
        + ("on" if _state.get("int8_attn_enabled") else "off")
        + (f", rotate {_state.get('int8_attn_rotate', 128)}" if _state.get("int8_attn_enabled") else "")
        + (f", {_state.get('int8_attn_calls', 0)} calls / {_state.get('int8_attn_fallbacks', 0)} fallback"
           if _state.get("int8_attn_calls") or _state.get("int8_attn_fallbacks") else "")
        if _state.get("int8_attn_patched") else "int8 attention n/a",
        f"image tower {'weight dtype + patch gemm' if _state['vision_enabled'] else 'shipped path'}",
        f"resident graph {'on' if _state.get('resident_graph') else 'off'}",
    ]
    manifest_line = manifest_status_line()
    if manifest_line:
        lines.insert(1, manifest_line)
    if _state.get("native") and _state.get("int8_linear"):
        lines.insert(1, f"comfy-kitchen {_state['native']} backend present: "
                        f"triton kernel patches are fallback-only here")
    if _state["int8_linear"] and _state.get("int8_tile_enabled", True):
        if _state.get("tile_measured"):
            if _tile_spec("COMFY_GPU_TUNE_TILE_CONFIG") is not None:
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
                     f" (own kernel: hip {cast['hip']}, hand {cast['hand']},"
                     f" gemm {cast['gemm']}, sdpa {cast['sdpa']})")
    failures = _state.get("install_failures") or []
    if failures:
        lines.append(f"patches skipped on this device: {', '.join(failures)} (see the boot log)")
    if _state["reason"]:
        lines.append(_state["reason"])
    forced = sorted(_state.get("forced_items") or ())
    if forced:
        lines.append(f"forced on an unmeasured arch: {', '.join(forced)} (COMFY_GPU_TUNE_FORCE)")
    return "\n".join(lines)
