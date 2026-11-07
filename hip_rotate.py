"""gfx906 HIP activation rotation for the convrot decode path.

Loads the `rotate` library for this target (built from rotate.cu by
tools/build_hip_attn.py). The kernel replaces int8_gemv.py's Triton rotation
with a radix-4 butterfly that generates the Hadamard signs in registers, so no
256x256 matrix is streamed: ~12x in a CUDA graph on the fp32 decode path, with
bit-identical int8 activations. See gfx906fix/hip_rotate/NOTES.md.

The stream is read at call time: torch.cuda.graph() switches current_stream to
its capture stream, so a cached handle would leave the launch out of the graph.
"""
import ctypes

import torch

try:
    from .kernels import runtime
except ImportError:                                   # imported as a top-level module
    from kernels import runtime

_LIB = None
_PROBED = False


def _load():
    global _LIB, _PROBED
    if _PROBED:
        return _LIB
    _PROBED = True
    lib = runtime.load("rotate", "HIP rotate")
    if lib is None:
        return None
    lib.gfx906_rotate.restype = ctypes.c_int
    lib.gfx906_rotate.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
                                  + [ctypes.c_void_p])
    _LIB = lib
    return lib


def available():
    return _load() is not None


def rotate(x, out, gmax, groupsize=256):
    """Fill out/gmax, or return False for a shape this kernel does not take."""
    lib = _load()
    if (lib is None or groupsize != 256 or not x.is_contiguous()
            or x.dtype not in (torch.float32, torch.float16)
            or x.shape[1] % 256):
        return False
    rc = lib.gfx906_rotate(x.data_ptr(), out.data_ptr(), gmax.data_ptr(),
                           x.shape[0], x.shape[1], groupsize,
                           0 if x.dtype == torch.float32 else 1,
                           ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"gfx906 HIP rotate launch failed ({rc})")
    return True
