"""gfx906 HIP int8 GEMM for the large-row int8_linear path.

Loads the `int8` library for this target (built from int8_gemm.cu by
tools/build_hip_attn.py). One block per CU with a 128x128 tile and 64 int32
accumulators: 4 waves/CU is the measured peak, see the kernel header.

`linear` has the same contract as comfy_kitchen's int8_linear and returns None
when the call is not one this kernel should take, so the caller falls through to
the Triton path. It reuses comfy_kitchen's rowwise quantizer (including the
node's fused convrot rotation), so only the GEMM itself differs.
"""

import ctypes

import torch

from .. import runtime

_LIB = None
_V2 = None
_PROBED = False

# The 128-row tile wastes rows below this; measured parity up to 32 rows and a
# win from 128 (see gfx906fix/hip_attn_v2/NOTES.md).
MIN_ROWS = 128


def _load():
    global _LIB, _V2, _PROBED
    if _PROBED:
        return _LIB
    _PROBED = True
    lib = runtime.load("int8", "HIP int8 gemm")
    if lib is None:
        return None
    lib.gfx906_int8_gemm.restype = ctypes.c_int
    lib.gfx906_int8_gemm.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p,
                                                             ctypes.c_void_p]
                                     + [ctypes.c_int] * 4 + [ctypes.c_void_p])
    v2 = getattr(lib, "gfx906_int8_gemm_v2", None)
    if v2 is not None:
        v2.restype = ctypes.c_int
        v2.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p,
                                                ctypes.c_void_p]
                       + [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_void_p])
        _V2 = v2
    _LIB = lib
    return lib


def supports_residual():
    """Whether the loaded library has the v2 residual epilogue."""
    _load()
    return _V2 is not None


def available():
    return _load() is not None


def linear(x, weight, weight_scale, bias=None, out_dtype=torch.float16, convrot=False,
           convrot_groupsize=256, input_act=None, input_act_weight=None, input_act_eps=0.0,
           residual=None, residual_scale=None):
    """int8 linear for large row counts, or None to let the caller fall through."""
    lib = _load()
    if lib is None:
        return None
    if residual is not None and _V2 is None:
        return None
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if not isinstance(x, torch.Tensor) or not x.is_contiguous():
        return None
    if weight.dtype != torch.int8 or weight.dim() != 2 or not weight.is_contiguous():
        return None
    if input_act not in (None, "none"):
        # the same eager activation the Triton path applies, so only the GEMM
        # implementation differs; swiglu also halves the row to match K
        from comfy_kitchen.backends._activations import apply_input_act
        x = apply_input_act(x, input_act, input_act_weight, input_act_eps)
    k = weight.shape[1]
    n = weight.shape[0]
    if x.shape[-1] != k:
        return None
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if m < MIN_ROWS:
        return None
    if not isinstance(weight_scale, torch.Tensor):
        weight_scale = torch.tensor([float(weight_scale)], device=x.device, dtype=torch.float32)
    if weight_scale.dtype != torch.float32:
        return None
    scalar = weight_scale.numel() == 1
    if not scalar and (weight_scale.numel() != n or not weight_scale.is_contiguous()):
        return None
    if bias is not None and (bias.dtype != torch.float32 or bias.numel() != n
                             or not bias.is_contiguous()):
        return None

    from comfy_kitchen.backends.triton import quantization

    if convrot:
        h = quantization._build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x_rot = quantization._rotate_activation(x2, h, convrot_groupsize)
        x8, xs = quantization.triton_quantize_rowwise(x_rot)
    else:
        x8, xs = quantization.triton_quantize_rowwise(x2)
    xs = xs.reshape(-1)
    if not x8.is_contiguous() or x8.dtype != torch.int8 or xs.numel() != m:
        return None

    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    stream = torch.cuda.current_stream().cuda_stream
    dtype_code = 2 if out_dtype is torch.float32 else (1 if out_dtype is torch.bfloat16 else 0)
    if residual is None:
        rc = _LIB.gfx906_int8_gemm(x8.data_ptr(), xs.data_ptr(), weight.data_ptr(),
                                   weight_scale.data_ptr(), int(scalar),
                                   bias.data_ptr() if bias is not None else None,
                                   out.data_ptr(), m, n, k, dtype_code,
                                   ctypes.c_void_p(stream))
    else:
        if (not isinstance(residual, torch.Tensor) or residual.dtype != out_dtype
                or not residual.is_contiguous()
                or tuple(residual.shape) != (*x.shape[:-1], n) or residual_scale is None):
            return None
        rs = residual_scale.reshape(-1)
        if rs.numel() != n:
            return None
        if rs.dtype != torch.float32:
            rs = rs.float()
        if not rs.is_contiguous():
            rs = rs.contiguous()
        rc = _V2(x8.data_ptr(), xs.data_ptr(), weight.data_ptr(),
                 weight_scale.data_ptr(), int(scalar),
                 bias.data_ptr() if bias is not None else None,
                 out.data_ptr(), m, n, k, dtype_code,
                 residual.data_ptr(), rs.data_ptr(), ctypes.c_void_p(stream))
    if rc != 0:
        raise RuntimeError(f"gfx906 HIP int8 gemm launch failed ({rc})")
    return out.view(*x.shape[:-1], n)
