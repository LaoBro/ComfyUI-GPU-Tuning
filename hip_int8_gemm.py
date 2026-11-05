"""gfx906 HIP int8 GEMM for the large-row int8_linear path.

Loads arch/gfx906/kernels/libgfx906_int8.so (built from int8_gemm.cu by
tools/build_hip_attn.py). One block per CU with a 128x128 tile and 64 int32
accumulators: 4 waves/CU is the measured peak, see the kernel header.

`linear` has the same contract as comfy_kitchen's int8_linear and returns None
when the call is not one this kernel should take, so the caller falls through to
the Triton path. It reuses comfy_kitchen's rowwise quantizer (including the
node's fused convrot rotation), so only the GEMM itself differs.
"""

import ctypes
import glob
import logging
import os

import torch

_LIB = None
_PROBED = False
_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "arch", "gfx906", "kernels", "libgfx906_int8.so")

# The 128-row tile wastes rows below this; measured parity up to 32 rows and a
# win from 128 (see gfx906fix/hip_attn_v2/NOTES.md).
MIN_ROWS = 128


def _open_hip_runtime():
    try:
        return ctypes.CDLL("libamdhip64.so.7", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    root = os.path.dirname(os.path.dirname(torch.__file__))
    for path in glob.glob(os.path.join(root, "_rocm_sdk_core", "lib", "libamdhip64.so*")):
        try:
            return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
    return None


def _load():
    global _LIB, _PROBED
    if _PROBED:
        return _LIB
    _PROBED = True
    if torch.version.hip is None or not torch.cuda.is_available() or not os.path.exists(_SO):
        return None
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx906"):
        return None
    if _open_hip_runtime() is None:
        logging.info("gfx906: libamdhip64 not found, HIP int8 gemm stays off")
        return None
    try:
        lib = ctypes.CDLL(_SO)
        lib.gfx906_int8_gemm.restype = ctypes.c_int
        lib.gfx906_int8_gemm.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p,
                                                                 ctypes.c_void_p]
                                         + [ctypes.c_int] * 4 + [ctypes.c_void_p])
    except OSError as exc:
        logging.info("gfx906: HIP int8 gemm extension unavailable (%s)", exc)
        return None
    _LIB = lib
    return lib


def available():
    return _load() is not None


def linear(x, weight, weight_scale, bias=None, out_dtype=torch.float16, convrot=False,
           convrot_groupsize=256, input_act=None, input_act_weight=None, input_act_eps=0.0,
           residual=None, residual_scale=None):
    """int8 linear for large row counts, or None to let the caller fall through."""
    if _load() is None:
        return None
    if input_act is not None or residual is not None:
        return None
    if out_dtype not in (torch.float16, torch.bfloat16):
        return None
    if not isinstance(x, torch.Tensor) or not x.is_contiguous():
        return None
    if weight.dtype != torch.int8 or weight.dim() != 2 or not weight.is_contiguous():
        return None
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
    rc = _LIB.gfx906_int8_gemm(x8.data_ptr(), xs.data_ptr(), weight.data_ptr(),
                               weight_scale.data_ptr(), int(scalar),
                               bias.data_ptr() if bias is not None else None,
                               out.data_ptr(), m, n, k,
                               1 if out_dtype is torch.bfloat16 else 0,
                               ctypes.c_void_p(stream))
    if rc != 0:
        raise RuntimeError(f"gfx906 HIP int8 gemm launch failed ({rc})")
    return out.view(*x.shape[:-1], n)
