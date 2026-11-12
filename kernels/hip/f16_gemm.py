"""gfx906 HIP fp16 GEMM for the non-quantized linear path.

Loads the `fp16` library for this target (built from fp16_gemm.cu by
tools/build_hip_attn.py). 128x128 tile, 64 fp32 accumulators, v_dot2_f32_f16,
2 blocks/CU (8 waves); see the kernel header and docs/HIP_KERNELS.md.

`linear` matches nn.Linear (C = x @ weight^T + bias, weight is [N, K]) and
returns None when the call is not one this kernel should take, so the caller
falls through to rocBLAS. Only big aligned fp16 GEMMs are taken: the 128-row
tile wastes rows on decode GEMVs, and the fast tile needs K % 32 / N % 64.
"""

import ctypes

import torch

from .. import runtime

_LIB = None
_PROBED = False

MIN_ROWS = 128


def _load():
    global _LIB, _PROBED
    if _PROBED:
        return _LIB
    _PROBED = True
    lib = runtime.load("fp16", "HIP fp16 gemm")
    if lib is None:
        return None
    lib.gfx906_f16_gemm.restype = ctypes.c_int
    lib.gfx906_f16_gemm.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
                                    + [ctypes.c_void_p])
    _LIB = lib
    return lib


def available():
    return _load() is not None


def linear(x, weight, bias=None):
    """fp16 linear for big aligned rows, or None to let the caller use rocBLAS."""
    if _load() is None:
        return None
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float16 or not x.is_contiguous():
        return None
    if not x.is_cuda:
        return None
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.float16 \
            or weight.dim() != 2 or not weight.is_contiguous():
        return None
    # bypassing CastBiasWeightContext is only safe when the weight already sits on
    # the input's device; an offloaded (CPU) weight must go through the cast path
    if weight.device != x.device:
        return None
    n, k = weight.shape
    if x.shape[-1] != k or k % 32 != 0 or n % 64 != 0:
        return None
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if m < MIN_ROWS:
        return None
    if bias is not None and (bias.dtype != torch.float16 or bias.numel() != n
                             or not bias.is_contiguous() or bias.device != x.device):
        return None
    out = torch.empty((m, n), device=x.device, dtype=torch.float16)
    stream = torch.cuda.current_stream().cuda_stream
    rc = _LIB.gfx906_f16_gemm(x2.data_ptr(), weight.data_ptr(),
                              bias.data_ptr() if bias is not None else None,
                              out.data_ptr(), m, n, k, ctypes.c_void_p(stream))
    if rc != 0:
        raise RuntimeError(f"gfx906 HIP fp16 gemm launch failed ({rc})")
    return out.view(*x.shape[:-1], n)