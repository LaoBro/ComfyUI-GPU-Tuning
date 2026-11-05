"""gfx906 HIP W4A16 decode GEMV: packed int4 weights, unquantised activations.

Loads arch/gfx906/kernels/libgfx906_w4a16.so (built from w4a16.cu by
tools/build_hip_attn.py) and replaces w4a8_gemv.dequant_w4a16_kernel: the nibbles
stay packed in registers and resolve through a 16-entry fp32 table in LDS. Two
entries, picked by the row count -- one for single-row decode (m=1, 3.6-4.4x the
Triton kernel, 185-274 GB/s against a 631-776 GB/s plane) and one for m=2..8 that
shares each decoded dword across the batch (1.6-5.3x, 18-238 GB/s). m > 8, fp16
activations for the multi-row entry and any other mismatch fall back to Triton.

COMFY_GFX906_W4A16_HIP=off restores the Triton kernel for an A/B run.

The stream is read at call time: torch.cuda.graph() switches current_stream to its
capture stream, so a cached handle would leave the launch out of the graph.
"""
import ctypes
import glob
import logging
import os

import torch

_LIB = None
_PROBED = False
_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "arch", "gfx906", "kernels", "libgfx906_w4a16.so")
BK = 1024                     # k elements per staged chunk, must divide the row


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
        logging.info("gfx906: libamdhip64 not found, HIP W4A16 GEMV stays off")
        return None
    try:
        lib = ctypes.CDLL(_SO)
        lib.gfx906_w4a16.restype = ctypes.c_int
        lib.gfx906_w4a16.argtypes = ([ctypes.c_void_p] * 7 + [ctypes.c_int] * 7
                                     + [ctypes.c_int64, ctypes.c_void_p])
        # An .so built before the multi-row entry simply keeps m>1 on Triton.
        try:
            lib.gfx906_w4a16_m.restype = ctypes.c_int
            lib.gfx906_w4a16_m.argtypes = ([ctypes.c_void_p] * 7 + [ctypes.c_int] * 7
                                           + [ctypes.c_int64, ctypes.c_void_p])
        except AttributeError:
            lib.gfx906_w4a16_m = None
    except OSError as exc:
        logging.warning("gfx906: %s did not load (%s); rebuild it for this HIP runtime with "
                        "tools/build_hip_attn.py --force", os.path.basename(_SO), exc)
        return None
    _LIB = lib
    return lib


def available():
    return _load() is not None


def enabled():
    return os.getenv("COMFY_GFX906_W4A16_HIP", "1").strip().lower() not in ("0", "off", "false", "no")


def _accepted(x, qdata, s_rel, s_channel, bias, out, group_size, codebook):
    """The layout both entry points need; row count and dtypes are checked by the caller."""
    n, k = qdata.shape[0], x.shape[1]
    return not (n < 1 or k % BK or group_size < 32 or group_size % 32
                or BK % group_size or k % group_size
                or x.dtype not in (torch.float16, torch.float32) or not x.is_contiguous()
                or qdata.dtype != torch.uint8 or qdata.shape[1] * 2 != k
                or qdata.stride(0) != qdata.shape[1] or not qdata.is_contiguous()
                or s_rel.dtype != torch.float8_e4m3fn or s_rel.stride(0) != s_rel.shape[1]
                or s_channel.dtype != torch.float32 or s_channel.numel() != n
                or out.dtype not in (torch.float16, torch.float32)
                or out.shape[-1] != n or not out.is_contiguous()
                or (codebook is not None and (codebook.numel() != 16
                                              or codebook.dtype != torch.float32))
                or (bias is not None and bias.dtype != out.dtype))


def linear_dequant(x, qdata, s_rel, s_channel, bias, out, group_size, codebook):
    """Fill out[:, :], or return False for a shape or dtype these kernels do not take.

    x is the activation rows (fp16 for one row, fp32 for more), s_rel the fp8
    per-group scale plane and codebook the 16 fp32 levels, or None for the uniform
    `code - 8` grid. A bias must be in the output dtype; anything else the kernels
    cannot take is left to the Triton path.
    """
    lib = _load()
    rows = x.shape[0]
    if (lib is None or rows > 8
            or not _accepted(x, qdata, s_rel, s_channel, bias, out, group_size, codebook)):
        return False
    n, k = qdata.shape[0], x.shape[1]
    if rows > 1 and (lib.gfx906_w4a16_m is None or x.dtype != torch.float32):
        # The multi-row kernel stages M x BK fp32 activations in LDS; handing it an
        # fp16 row set would read past the plane. m=1 keeps its own fp16 path.
        return False
    cb_ptr = s_channel.data_ptr() if codebook is None else codebook.data_ptr()
    args = [x.data_ptr(), qdata.view(torch.int32).data_ptr(),
            s_rel.view(torch.uint8).data_ptr(), s_channel.data_ptr(),
            out.data_ptr() if bias is None else bias.data_ptr(), out.data_ptr(), cb_ptr]
    if rows > 1:
        ints = [rows, n, k, group_size, 1 if out.dtype == torch.float16 else 0,
                1 if codebook is not None else 0, 1 if bias is not None else 0]
        entry = lib.gfx906_w4a16_m
    else:
        ints = [n, k, group_size, 1 if x.dtype == torch.float16 else 0,
                1 if out.dtype == torch.float16 else 0,
                1 if codebook is not None else 0, 1 if bias is not None else 0]
        entry = lib.gfx906_w4a16
    rc = entry(*args, *ints, k // group_size,
               ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"gfx906 HIP W4A16 launch failed ({rc})")
    return True
