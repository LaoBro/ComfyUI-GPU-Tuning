"""gfx906 HIP activation rotation for the convrot decode path.

Loads arch/gfx906/kernels/libgfx906_rotate.so (built from rotate.cu by
tools/build_hip_attn.py). The kernel replaces int8_gemv.py's Triton rotation
with a radix-4 butterfly that generates the Hadamard signs in registers, so no
256x256 matrix is streamed: ~12x in a CUDA graph on the fp32 decode path, with
bit-identical int8 activations. See gfx906fix/hip_rotate/NOTES.md.

The stream is read at call time: torch.cuda.graph() switches current_stream to
its capture stream, so a cached handle would leave the launch out of the graph.
"""
import ctypes
import glob
import logging
import os

import torch

_LIB = None
_PROBED = False
_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "arch", "gfx906", "kernels", "libgfx906_rotate.so")


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
        logging.info("gfx906: libamdhip64 not found, HIP rotate stays off")
        return None
    try:
        lib = ctypes.CDLL(_SO)
        lib.gfx906_rotate.restype = ctypes.c_int
        lib.gfx906_rotate.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
                                      + [ctypes.c_void_p])
    except OSError as exc:
        logging.warning("gfx906: %s did not load (%s); rebuild it for this HIP runtime with "
                        "tools/build_hip_attn.py --force", os.path.basename(_SO), exc)
        return None
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
