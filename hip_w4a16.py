"""gfx906 HIP W4A16 decode GEMV: packed int4 weights, unquantised activations.

Loads arch/gfx906/kernels/libgfx906_w4a16.so (built from w4a16.cu by
tools/build_hip_attn.py). It replaces w4a8_gemv.dequant_w4a16_kernel for the
single-row te=default decode: the nibbles stay packed in registers and resolve
through a 16-entry fp32 table in LDS, which is 2.0-2.5x the Triton kernel at
110-137 GB/s (that one runs at 48-62 GB/s; the plane itself streams at
631-776 GB/s, so both are still far from the ceiling). Multi-row batches (MTP)
keep the Triton kernel: the same layout measured 0.9-1.1x there, because eight
rows of fp32 accumulators plus their activation reads blow the issue budget.

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


def linear_dequant(x, qdata, s_rel, s_channel, bias, out, group_size, codebook):
    """Fill out[0, :], or return False for a shape or dtype this kernel does not take.

    x is the activation row (fp16 or the fp32 convrot-rotated one), s_rel the fp8
    per-group scale plane and codebook the 16 fp32 levels, or None for the uniform
    `code - 8` grid. A bias must be in the output dtype; anything else the kernel
    cannot take is left to the Triton path.
    """
    lib = _load()
    n, k = qdata.shape[0], x.shape[1]
    if (lib is None or x.shape[0] != 1 or n < 1 or k % BK or group_size < 32
            or group_size % 32 or BK % group_size or k % group_size
            or x.dtype not in (torch.float16, torch.float32) or not x.is_contiguous()
            or qdata.dtype != torch.uint8 or qdata.shape[1] * 2 != k
            or qdata.stride(0) != qdata.shape[1] or not qdata.is_contiguous()
            or s_rel.dtype != torch.float8_e4m3fn or s_rel.stride(0) != s_rel.shape[1]
            or s_channel.dtype != torch.float32 or s_channel.numel() != n
            or out.dtype not in (torch.float16, torch.float32)
            or (codebook is not None and (codebook.numel() != 16
                                          or codebook.dtype != torch.float32))
            or (bias is not None and bias.dtype != out.dtype)):
        return False
    rc = lib.gfx906_w4a16(
        x.data_ptr(), qdata.view(torch.int32).data_ptr(), s_rel.view(torch.uint8).data_ptr(),
        s_channel.data_ptr(), out.data_ptr() if bias is None else bias.data_ptr(),
        out.data_ptr(), s_channel.data_ptr() if codebook is None else codebook.data_ptr(),
        n, k, group_size, 1 if x.dtype == torch.float16 else 0,
        1 if out.dtype == torch.float16 else 0, 1 if codebook is not None else 0,
        1 if bias is not None else 0, k // group_size,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"gfx906 HIP W4A16 launch failed ({rc})")
    return True
