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
# COMFY_GFX906_CONVROT: fused (butterfly + quantize in one kernel), butterfly (two passes), off.
CONVROT_MODES = ("fused", "butterfly", "off")
# Own kernels (exposed in the node). "sdpa" also exists as an internal value:
# the auto-tune fallback when a tile loses to SDPA on an untested card.
ATTENTION_KERNELS = ("hand", "int8", "int8qk", "gemm")


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
    "reason": "",
}


def enabled():
    return os.getenv("COMFY_GFX906_TUNING", "1").lower() not in ("0", "off", "false", "no")


def target_archs():
    return os.getenv("COMFY_GFX906_ARCHS", "gfx906").replace(",", " ").split()


def arch_name():
    """The gfx target of device 0, or None on CUDA/CPU (props have no gcnArchName there)."""
    if torch.version.hip is None or not torch.cuda.is_available():
        return None
    name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return name.split(":")[0] or None


def is_amd():
    return torch.version.hip is not None and torch.cuda.is_available()


def on_target_arch():
    """True on the architecture the attention kernel was measured on."""
    return arch_name() in target_archs()


def _triton_available(patch):
    try:
        import triton  # noqa: F401
    except ImportError:
        logging.warning("gfx906: triton is not importable, %s patch skipped", patch)
        return False
    return True


def _triton_config():
    import triton

    return triton.Config(dict(INT8_TILE), num_warps=INT8_WARPS, num_stages=INT8_STAGES)


def _int8_kernels():
    from comfy_kitchen.backends.triton import quantization

    for name in INT8_KERNELS:
        kernel = getattr(quantization, name, None)
        if kernel is not None and getattr(kernel, "configs", None):
            yield kernel


def _small_tile_config():
    import triton

    return triton.Config(dict(INT8_TILE_SMALL), num_warps=INT8_SMALL_WARPS, num_stages=INT8_SMALL_STAGES)


def select_int8_tile(m):
    """Offer the tile that matches the batch width this call has.

    Autotune caches its pick per (m, n, k), and a shape never moves between the
    two buckets, so swapping the list only happens when the bucket changes: no
    sweep is paid for the decode shapes, and the DiT keeps the 128-row tile.
    Off while the stock list is restored for a measurement.
    """
    if not _state.get("int8_tile_switch"):
        return
    small = m <= INT8_SMALL_MAX_M
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
    """Import the 4-bit decode GEMV kernel module shipped next to this file."""
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


def patch_kv_decode():
    """4-D GQA matmul in FixedKVBias.decode.

    The 5-D form broadcasts the KV over the GQA groups, and ATen answers that by
    materializing a transposed copy of the whole capacity-sized cache once per
    layer per step: 37 MB at a 2.3k prompt, 0.73 ms per layer here. Folding the
    groups into the query rows removes the copy; the math is unchanged (bmm
    blocking order only, <= 7e-7).
    """
    import comfy_kitchen
    import comfy.text_encoders.llama as llama

    kv = llama.FixedKVBias
    if getattr(kv.decode, "_gfx906_folded", False):
        return True

    def decode(self, xq, xk, xv, num_kv_heads):
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
    gemv = _state.get("gemv")
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
    if _state.get("gemv") is None:
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

    The uniform grid is the only fast one: the shipped Lloyd-Max codebook decodes
    through a per-group level table, so every weight would need its own lookup.
    Requantized layers (`tools/requant_uniform4.py`) carry no codebook and go
    straight to the register-level nibble unpack. Prefill and anything else
    returns None and keeps the kitchen dequant + int8 GEMM path.
    """
    gemv = _state["w4a8"]
    if gemv is None:
        return None
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout

    weight = module.weight
    params = weight._params
    rows = x.reshape(-1, x.shape[-1])
    qdata, s_rel, s_channel, correction, codebook = AsymW4A8Int8Layout.get_plain_tensors(weight)
    if correction is not None or codebook is not None:
        return None
    if x.dtype is not torch.float32 or not x.is_contiguous():
        return None
    step = gemv.MAX_M
    args = (qdata, s_rel, s_channel, module.bias, x.dtype,
            params.group_size, params.convrot_groupsize)
    if rows.shape[0] <= step:
        out = gemv.linear(rows, *args)
    else:
        # prefill: the kernel takes at most MAX_M rows, and each row's activation
        # scale is its own, so slicing the batch changes nothing
        parts = [gemv.linear(rows[i:i + step], *args) for i in range(0, rows.shape[0], step)]
        out = None if any(p is None for p in parts) else torch.cat(parts, 0)
    if out is None:
        return None
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

    gemv_on = os.getenv("COMFY_GFX906_GEMV", "1").strip().lower() not in ("0", "off", "false")
    _state["gemv"] = load_int8_gemv() if gemv_on else None

    def int8_linear(x, weight, weight_scale, bias=None, out_dtype=torch.bfloat16, convrot=False,
                    convrot_groupsize=256, input_act=None, input_act_weight=None,
                    input_act_eps=0.0, residual=None, residual_scale=None):
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        if rows <= INT8_GEMV_MAX_M:
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

    _state["default_int8_configs"] = list(kernels[0].configs)
    _state["int8_tile_replaced"] = replace
    for kernel in kernels:
        current = list(kernel.configs)
        ours = _triton_config()
        if replace:
            kernel.configs = [ours]
        elif not any(c.kwargs == ours.kwargs and c.num_warps == ours.num_warps
                     and c.num_stages == ours.num_stages for c in current):
            kernel.configs = current + [ours]
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


def patch_attention(auto_tune=False):
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
                                       benchmark_tiles, benchmark_tiles_int8,
                                       supported as kernel_supported, use_config, use_config_int8)
    except ImportError:
        from attention_kernel import (SUPPORTED_HEAD_DIMS, attention_fwd, attention_fwd_int8,
                                      benchmark_tiles, benchmark_tiles_int8,
                                      supported as kernel_supported, use_config, use_config_int8)

    fallback = attention.optimized_attention
    if fallback in (attention.attention_sage, attention.attention_comfy_kitchen_int8):
        logging.info("gfx906: %s was requested explicitly, leaving attention alone",
                     getattr(fallback, "__name__", "?"))
        return False

    kernel = os.getenv("COMFY_GFX906_ATTN", "int8").strip().lower()
    _state["attn_kernel"] = kernel if kernel in ATTENTION_KERNELS or stock_backend(kernel) is not None else "int8"
    _state["attn_backend_fn"] = stock_backend(_state["attn_kernel"])
    _state["smooth"] = os.getenv("COMFY_GFX906_SMOOTH", "1").lower() not in ("0", "off", "false", "no")
    _state["attn_tuned"] = set(SUPPORTED_HEAD_DIMS) if not auto_tune else set()
    _state["attn_fp16"] = os.getenv("COMFY_GFX906_FP16", "1").lower() not in ("0", "off", "false", "no")
    _state["attn_contiguous"] = os.getenv("COMFY_GFX906_CONTIG", "1").lower() not in ("0", "off", "false", "no")
    stats = _state["attn_cast"] = {"fp16": 0, "fp32": 0, "hand": 0, "int8": 0, "int8qk": 0,
                                   "gemm": 0, "sdpa": 0}

    def pick_kernel(q, k, v, softmax_scale):
        """On a head dim we have no numbers for: time the tiles and SDPA, keep the faster."""
        head_dim = q.shape[-1]
        _state["attn_tuned"].add(head_dim)
        mode = _state["attn_kernel"]
        config = kernel_ms = None

        if mode in ("int8", "int8qk"):
            config, kernel_ms = benchmark_tiles_int8(
                q, k, v, softmax_scale, int8_pv=mode == "int8",
                smooth=_state["smooth"] if mode == "int8qk" else False)
            if config is not None:
                use_config_int8(config, head_dim)

        if config is None:
            # int8 did not compile (no dp4a) or the caller asked for fp16
            config, kernel_ms = benchmark_tiles(q, k, v, softmax_scale)
            if config is not None:
                use_config(config, head_dim)
                _state["attn_kernel"] = "hand"

        sdpa_ms = _bench(lambda: _sdpa(q, k, v, softmax_scale), iters=2, warmup=1)
        if config is None or kernel_ms >= sdpa_ms:
            # falling back to the stock backend would be far slower than this floor
            _state["attn_kernel"] = "sdpa"
            logging.info("gfx906: attention kernel %s ms vs SDPA %.1f ms, keeping SDPA",
                         f"{kernel_ms:.1f}" if kernel_ms else "n/a", sdpa_ms)
        else:
            logging.info("gfx906: attention %s %.1f ms vs SDPA %.1f ms, using BLOCK_M=%d BLOCK_N=%d warps=%d stages=%d",
                         _state["attn_kernel"], kernel_ms, sdpa_ms, *config)
        _state["attn_backend_fn"] = None

    def run(q, k, v, softmax_scale):
        """[B, S, H, D] -> same, or None to delegate. fp16 unless the cast is off."""
        if _state["attn_kernel"] not in ("sdpa", "gemm") and q.shape[-1] not in _state["attn_tuned"]:
            pick_kernel(q, k, v, softmax_scale)
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

    mode = os.getenv("COMFY_GFX906_CONVROT", "fused").strip().lower()
    _state["convrot_mode"] = mode if mode in CONVROT_MODES else "fused"
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
              fp16=None, text_attention=None, contiguous=None, linear8=None):
    """Runtime settings, used by the GFX906 Tuning node."""
    if attention is not None:
        # combo values are untrusted: stale workflows may still carry removed names
        _state["attn_kernel"] = (attention if attention in ATTENTION_KERNELS
                                 or stock_backend(attention) is not None else "int8")
        _state["attn_backend_fn"] = stock_backend(_state["attn_kernel"])
    if contiguous is not None:
        _state["attn_contiguous"] = bool(contiguous)
    if smooth is not None:
        _state["smooth"] = bool(smooth)
    if fp16 is not None:
        _state["attn_fp16"] = bool(fp16)
    if convrot is not None:
        _state["convrot_mode"] = convrot
    if linear8 is not None and _state.get("linear8_enabled") is not None:
        _state["linear8_enabled"] = bool(linear8)
    if text_encoder is not None and _state["applied"]:
        _state["te_int8"] = text_encoder == "int8" and _state["te_patched"]
    if text_attention is not None:
        # an unknown combo value leaves whatever the previous selection was
        _state["te_attention"] = (text_attention if text_attention == "default"
                                  or stock_backend(text_attention) is not None else "default")
    # something may have disabled the backend after apply(); the node runs per
    # workflow, so this is the last chance to put it back before the model does
    # its first int8 matmul
    _state["triton_backend"] = enable_triton_backend()
    if profile is not None and _state["applied"]:
        was = _profiler_enabled()
        _set_profiler(profile)
        if profile and not was:
            # absorbed the Tuning Info node: its one-shot report ships with the
            # first profiled prompt
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
        if _state["te_int8"]:
            last = x[:, -1:]
            module = self.model.lm_head if hasattr(self.model, "lm_head") else self.model.embed_tokens
            weight = module.weight
            if (isinstance(weight, QuantizedTensor) and weight.device == last.device
                    and not module.weight_function and not module.bias_function):
                return torch.nn.functional.linear(last, weight, None)
        return original(self, x)

    logits._gfx906_logits = True
    llama.BaseGenerate.logits = logits
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
    kernel with N three times wider: at the block shape (M=4032, K=4096) three
    calls cost 26.12 ms and one fused call 23.40 ms, i.e. -163 ms per step.
    """
    if os.getenv("COMFY_GFX906_QKV", "1").lower() in ("0", "off", "false", "no"):
        return False
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
        self._gfx906_fused = None if comfy.model_management.in_training else _fused_qkv(self, x)
        try:
            return original_forward(self, x, pe, attn_fn, prefix_len, transformer_options)
        finally:
            self._gfx906_fused = None

    attention.__init__ = __init__
    attention.forward = forward
    attention._gfx906_qkv = True
    _state["qkv_fused"] = True
    return True


def _int8_linear_forward(self, x):
    """Runtime int8 GEMM for plain fp16 Linear weights (SDXL-class checkpoints).

    The int8 weight is quantized once per module and device and cached, so the
    hot path never touches cast_bias_weight: casting per call re-pins the vbar
    pages of the fp16 weight every forward and aimdo spams a WARNING per page
    per call (2700 per 20-step SDXL run). The cast that produced the cache uses
    offloadable=True + uncast_bias_weight, so nothing stays pinned either.
    Linear layers have no softmax to amplify quantization error, unlike
    attention, so per-row int8 weights are safe here quality-wise.

    Both sides get the 256-wide ConvRot Hadamard rotation (the weight at requant
    time, the activation online inside int8_linear), which is what keeps SDXL's
    activation outliers from dominating the per-row x quantization.
    """
    import comfy.quant_ops
    from comfy_kitchen.backends.triton import quantization as ckq

    if (not _state.get("linear8_enabled", True)
            or comfy.model_management.in_training
            or getattr(self, "weight_function", None)
            or getattr(self, "bias_function", None)
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        return self._gfx906_linear_forward(x)
    w8 = getattr(self, "_gfx906_w8", None)
    if w8 is None or w8.device != x.device:
        weight, bias, stream = comfy.ops.cast_bias_weight(self, x, offloadable=True)
        try:
            n, k = weight.shape
            rot = k % 256 == 0
            if rot:
                h = ckq._build_hadamard(256, device=x.device, dtype=weight.dtype)
                weight = ckq._rotate_activation(weight, h, 256)
            scale = (weight.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
            self._gfx906_w8 = (weight / scale).round().to(torch.int8)
            self._gfx906_ws = scale
            # the cast's offload stream reclaims the returned tensors; keep copies
            self._gfx906_b = None if bias is None else bias.clone()
            self._gfx906_rot = rot
        finally:
            comfy.ops.uncast_bias_weight(self, weight, bias, stream)
    _state["linear8_calls"] = _state.get("linear8_calls", 0) + 1
    return comfy.quant_ops.ck.int8_linear(x, self._gfx906_w8, self._gfx906_ws,
                                          self._gfx906_b, x.dtype,
                                          convrot=self._gfx906_rot, convrot_groupsize=256)


def _walk_linear8(root):
    """Swap eligible Linear forwards to the int8 path; returns how many."""
    import torch.nn as nn
    from comfy.quant_ops import QuantizedTensor

    count = 0
    for module in root.modules():
        if (isinstance(module, nn.Linear) and module.in_features >= 1280
                and not isinstance(module.weight, QuantizedTensor)
                and not getattr(module, "_gfx906_l8", False)):
            module._gfx906_l8 = True
            module._gfx906_linear_forward = module.forward
            module.forward = _int8_linear_forward.__get__(module)
            count += 1
    return count


def patch_linear8():
    """int8 GEMM for the fp16 Linear layers of non-quantized checkpoints.

    SDXL spends 40-60% of its forward in rocBLAS fp16 GEMMs that none of the
    int8 machinery reaches (the checkpoint has no QuantizedTensor weights).
    Quantizing weights with in_features >= 1280 once and running int8_linear
    instead wins on every such shape at the SDXL token counts (1.2-2.1x per
    layer); the 640-wide layers at M=4096 lose and stay fp16. Layers grow LoRA
    patches later than load time, so weight_function is re-checked on every
    call. The node's linear8 switch gates the swapped forwards live: the walk
    is installed regardless, so toggling needs no reload.
    """
    import comfy.sd

    if getattr(comfy.sd, "_gfx906_l8", False):
        return True
    _state["linear8_enabled"] = os.getenv("COMFY_GFX906_LINEAR8", "1").lower() not in ("0", "off", "false", "no")

    def wrap_loader(original):
        import functools

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            try:
                patcher = result[0] if isinstance(result, tuple) else result
                count = _walk_linear8(patcher.get_model_object("diffusion_model"))
                if count:
                    _state["linear8_modules"] = _state.get("linear8_modules", 0) + count
                    logging.info("gfx906: %d Linear layers (in_features >= 1280) on the int8 path", count)
            except Exception as exc:
                logging.warning("gfx906: linear8 walk failed: %s", exc)
            return result

        wrapper._gfx906_l8 = True
        return wrapper

    for name in ("load_checkpoint_guess_config", "load_diffusion_model",
                 "load_diffusion_model_state_dict"):
        original = getattr(comfy.sd, name, None)
        if original is not None and not getattr(original, "_gfx906_l8", False):
            wrapped = wrap_loader(original)
            setattr(comfy.sd, name, wrapped)
    comfy.sd._gfx906_l8 = True
    _state["linear8"] = True
    return True


def patch_vae():
    """Run depth-1 Conv3d as Conv2d.

    The Wan-style video VAEs that image models ship with (Qwen-Image 2.1 uses the
    Wan 2.2 decoder) are 76 Conv3d layers with a one-frame kernel, and an image
    gives them a single frame. There is no conv3d kernel for gfx906, so torch
    lowers every one of them to im2col + bmm; the profiler puts that at 47% of
    decode time. Squeezing the depth axis makes them ordinary conv2d calls that
    land on the fp16 MIOpen kernels: 2.31 s -> 1.09 s at 1024x1024.
    """
    if os.getenv("COMFY_GFX906_VAE", "1").lower() in ("0", "off", "false", "no"):
        return False
    import comfy.ops

    conv3d = comfy.ops.disable_weight_init.Conv3d
    if getattr(conv3d, "_gfx906_vae", False):
        return True
    original = conv3d._conv_forward

    def _conv_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
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
    _state["vae_conv2d"] = True
    return True


def apply():
    if _state["applied"]:
        return _state
    _state["applied"] = True
    _state["arch"] = arch_name()
    if not enabled():
        _state["reason"] = "disabled by COMFY_GFX906_TUNING"
        return _state
    if not is_amd():
        _state["reason"] = "not an AMD device (torch.version.hip is None)"
        logging.info("gfx906 tuning skipped: %s", _state["reason"])
        return _state

    tuned = on_target_arch()
    patch_int8_linear(replace=tuned)
    patch_convrot()
    patch_vae()
    patch_qkv()
    patch_linear8()
    # before any prompt runs: the queue thread binds execute_async when a prompt
    # starts, so a hook installed later would only report from the next one
    _patch_prompt_end()
    te_ready = patch_text_encoder()
    # last: importing comfy.ops above pulls in comfy.quant_ops, which disables
    # the triton backend again when --enable-triton-backend was not passed
    _state["triton_backend"] = enable_triton_backend()
    _state["te_patched"] = te_ready
    te = os.getenv("COMFY_GFX906_TE", "int8").strip().lower()
    _state["te_int8"] = te != "default" and te_ready
    te_attn = os.getenv("COMFY_GFX906_TE_ATTN", "default").strip().lower()
    _state["te_attention"] = te_attn if te_attn == "default" or stock_backend(te_attn) is not None else "default"
    _state["logits_patched"] = patch_logits()
    patch_te_attention()
    # the kitchen fused DeltaNet decode has no gfx906 code object; these triton kernels
    # replace it for the text encoder's linear-attention layers
    deltanet_on = os.getenv("COMFY_GFX906_DELTANET", "1").strip().lower() not in ("0", "off", "false", "no")
    deltanet_kernels = load_deltanet() if tuned and deltanet_on else None
    _state["deltanet_fused"] = bool(deltanet_kernels and deltanet_kernels.install())
    kv_decode_on = os.getenv("COMFY_GFX906_KVDECODE", "1").strip().lower() not in ("0", "off", "false", "no")
    _state["kv_decode_folded"] = patch_kv_decode() if kv_decode_on else False
    llama_on = os.getenv("COMFY_GFX906_LLAMA", "1").strip().lower() not in ("0", "off", "false", "no")
    llama_patches = load_llama_patches() if llama_on else None
    _state["llama_patches"] = bool(llama_patches and llama_patches.install())
    # after the base-class patches: the qwen35 class wraps its own compute_freqs_cis
    # only when the shared base carries the position table
    qwen35_on = os.getenv("COMFY_GFX906_QWEN35", "1").strip().lower() not in ("0", "off", "false", "no")
    qwen35_model = load_qwen35_model() if tuned and qwen35_on else None
    _state["qwen35_node_model"] = bool(qwen35_model and qwen35_model.install())
    # decode-only 4-bit GEMV for AsymW4A8 weights; prefill keeps the kitchen path
    w4a8_on = os.getenv("COMFY_GFX906_W4A8", "1").strip().lower() not in ("0", "off", "false", "no")
    _state["w4a8"] = load_w4a8_gemv() if tuned and w4a8_on else None
    # the per-block graph hooks live in core on purpose; say so when they are absent
    import comfy.model_prefetch
    _state["resident_graph_patch"] = hasattr(comfy.model_prefetch, "RESIDENT_GRAPH_MODULES")
    if tuned and not _state["resident_graph_patch"]:
        logging.info("gfx906: resident graph hooks absent, decode runs without per-block CUDA graphs "
                     "(see custom_nodes/ComfyUI-gfx906-Tuning/core-patches/README.md)")
    if tuned:
        patch_attention()
    else:
        logging.info("gfx906: %s has no measured attention tiles; the first attention call will "
                     "benchmark the candidates against SDPA and keep the faster", _state["arch"])
        patch_attention(auto_tune=True)
    logging.info(
        "gfx906 tuning active: int8 linear %s%s, convrot %s, vae %s, qkv %s, attention %s%s%s%s, "
        "text encoder %s + lm_head, encoder attention %s, deltanet %s",
        "tuned" if _state.get("int8_tile_replaced") else ("on (tile appended)" if _state["int8_linear"] else "off"),
        " + decode gemv" if _state.get("gemv") else "",
        _state["convrot_mode"] if _state["convrot"] else "off",
        "conv3d->conv2d" if _state.get("vae_conv2d") else "off",
        "fused" if _state.get("qkv_fused") else "off",
        _state["attn_kernel"] if _state["attention"] else "off",
        f" (other calls go to {_state['replaced_backend']})" if _state.get("replaced_backend") else "",
        ", fp16 cast off" if not _state["attn_fp16"] else "",
        ", contiguous off" if not _state.get("attn_contiguous", True) else "",
        "int8" if _state["te_int8"] else "default",
        _state["te_attention"],
        "fused decode" if _state["deltanet_fused"] else "eager",
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


def benchmark(m=4032, iters=5):
    """Per-layer cost of the three linear shapes in a 4096-token quantized DiT."""
    from comfy_kitchen.backends.triton import quantization

    if not _state["int8_linear"]:
        return ["int8 linear patch not applied, nothing to compare"]

    lines = []
    for n, k in ((4096, 4096), (24576, 4096), (4096, 12288)):
        x = torch.randn(m, k, device="cuda", dtype=torch.float32)
        w = torch.randint(-127, 127, (n, k), device="cuda", dtype=torch.int8)
        ws = torch.rand(n, 1, device="cuda", dtype=torch.float32) * 0.01
        run = lambda: quantization.int8_linear(x, w, ws, None, out_dtype=torch.float32, convrot=True)
        tuned = _bench(run, iters=iters)
        if _state["default_int8_configs"] is not None:
            restore_int8_defaults()
            base = _bench(run, iters=iters)
            patch_int8_linear(replace=True)
            lines.append(f"N={n:5d} K={k:5d}  {tuned:6.2f} ms  vs {base:6.2f} ms stock  ({base/tuned:.2f}x)")
        else:
            lines.append(f"N={n:5d} K={k:5d}  {tuned:6.2f} ms")
    lines.extend(_bench_convrot(m))
    return lines


def _bench_convrot(m):
    original = _state.get("original_rotate")
    if original is None:
        return []
    from comfy_kitchen.backends.triton import quantization
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    x = torch.randn(m, 4096, device="cuda", dtype=torch.float32)
    h = _build_hadamard(CONVROT_GROUP, device=x.device, dtype=x.dtype)
    fast = _bench(lambda: quantization.triton_quantize_rowwise(quantization._rotate_activation(x, h, CONVROT_GROUP)), iters=10)
    slow = _bench(lambda: quantization.triton_quantize_rowwise(original(x, h, CONVROT_GROUP)), iters=10)
    return [f"convrot rotate+quant  {fast:6.2f} ms  vs {slow:6.2f} ms matmul  ({slow/fast:.2f}x)"]


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
    lines = [line]
    if _state.get("linear8_calls"):
        lines.append(f"linear8 {_state['linear8_calls']} int8 linear calls")
        _state["linear8_calls"] = 0
    return lines


def status():
    tile_state = {True: "tuned", False: "config appended, autotune decides", None: "off"}
    lines = [
        f"arch {_state['arch']}",
        f"comfy-kitchen triton backend {triton_backend_active()}",
        f"int8 linear {tile_state[_state['int8_tile_replaced'] if _state['int8_linear'] else None]}",
        f"convrot {_state['convrot_mode'] or 'off'}",
        f"vae {'conv3d as conv2d' if _state.get('vae_conv2d') else 'off'}",
        f"qkv {'one int8 gemm' if _state.get('qkv_fused') else 'off'}"
        + (f", {_state['qkv_calls']} fused / {_state.get('qkv_fallbacks', 0)} fallback"
           if _state.get("qkv_calls") or _state.get("qkv_fallbacks") else ""),
        f"linear8 {'on' if _state.get('linear8_enabled', True) else 'off'}: "
        f"{_state.get('linear8_modules', 0)} layers, {_state.get('linear8_calls', 0)} calls",
        f"attention {_state['attn_kernel'] if _state['attention'] else 'off'}"
        + (f", smoothing {'on' if _state['smooth'] else 'off'}" if _state["attn_kernel"] == "int8qk" else "")
        + (", inputs cast to fp16" if _state["attn_fp16"] else ", inputs left in the model dtype")
        + (f", rest to {_state['replaced_backend']}" if _state.get("replaced_backend") else ""),
        f"text encoder {'int8' if _state['te_int8'] else 'default (dequantize per call)'}"
        + (", 4-bit decode gemv on" if _state["w4a8"] is not None else "")
        + f", encoder attention {_state['te_attention']}",
    ]
    if _state["int8_linear"]:
        tile = ", ".join(f"{k}={v}" for k, v in INT8_TILE.items())
        lines.append(f"int8 tile {tile}, {INT8_WARPS} warps, {INT8_STAGES} stages")
    cast = _state["attn_cast"]
    if cast:
        lines.append(f"attention calls: fp16-ok {cast['fp16']}, amax overflow {cast['fp32']}"
                     f" (own kernel: hand {cast['hand']}, int8 {cast['int8']}, int8qk {cast['int8qk']},"
                     f" gemm {cast['gemm']}, sdpa {cast['sdpa']})")
    if _state["reason"]:
        lines.append(_state["reason"])
    return "\n".join(lines)
