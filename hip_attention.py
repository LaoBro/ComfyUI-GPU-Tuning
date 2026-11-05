"""gfx906 HIP fused fp16 attention.

Loads arch/gfx906/kernels/libgfx906_attn.so, built from hip_attn.cu by
tools/build_hip_attn.py. The kernel is an online-softmax flash attention with
fp32 accumulators: 128 Q rows x 64 keys per block, one block per CU, dot2 with
a 4x8 per-lane tile (see the header of hip_attn.cu for the measurements).
"""

import ctypes
import glob
import logging
import os

import torch

_LIB = None
_PROBED = False
_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "arch", "gfx906", "kernels", "libgfx906_attn.so")


def _open_hip_runtime():
    """Make libamdhip64's symbols global so the extension resolves at load."""
    try:
        return ctypes.CDLL("libamdhip64.so.7", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    # wheels ship the runtime under the ROCm SDK package instead of a system path
    roots = [os.path.dirname(os.path.dirname(torch.__file__))]
    for root in roots:
        hits = glob.glob(os.path.join(root, "_rocm_sdk_core", "lib", "libamdhip64.so*"))
        for path in hits:
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
    # the shipped .so only holds a gfx906 code object
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx906"):
        return None
    if _open_hip_runtime() is None:
        logging.info("gfx906: libamdhip64 not found, HIP attention stays off")
        return None
    try:
        lib = ctypes.CDLL(_SO)
        lib.gfx906_attn_fwd.restype = ctypes.c_int
        lib.gfx906_attn_fwd.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
                                        + [ctypes.c_float, ctypes.c_void_p])
    except OSError as exc:
        logging.warning("gfx906: %s did not load (%s); rebuild it for this HIP runtime with "
                        "tools/build_hip_attn.py --force", os.path.basename(_SO), exc)
        return None
    _LIB = lib
    return lib


def available():
    return _load() is not None


def supported(q, k, v):
    """fp16, 4D, contiguous [B, S, H, D] with D=64/128 and matching head counts."""
    if _load() is None:
        return False
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        return False
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return False
    b, sq, h, d = q.shape
    return (d in (64, 128) and sq > 0 and k.shape[1] > 0
            and k.shape[0] == b and v.shape[0] == b
            and k.shape[2] == h and v.shape[2] == h
            and k.shape[3] == d and v.shape[3] == d)


def attention_fwd(q, k, v, softmax_scale=None):
    """q, k, v: [B, S, H, D] fp16 contiguous. Returns [B, S, H, D] fp16."""
    out = torch.empty_like(q)
    b, sq, h, d = q.shape
    scale = d ** -0.5 if softmax_scale is None or softmax_scale <= 0 else softmax_scale
    stream = torch.cuda.current_stream().cuda_stream
    rc = _LIB.gfx906_attn_fwd(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
                              b, sq, k.shape[1], h, d, ctypes.c_float(scale),
                              ctypes.c_void_p(stream))
    if rc != 0:
        raise RuntimeError(f"gfx906 HIP attention launch failed ({rc})")
    return out
