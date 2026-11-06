"""gfx906 HIP W4A16 decode GEMV: packed int4 weights, unquantised activations.

Loads arch/gfx906/kernels/libgfx906_w4a16.so (built from w4a16.cu by
tools/build_hip_attn.py) and replaces w4a8_gemv.dequant_w4a16_kernel: the nibbles
stay packed in registers and resolve through a 16-entry fp32 table in LDS. Three
entries by row count and dtype -- m=1 decode (3.6-4.4x Triton, 185-274 GB/s
against a 631-776 GB/s plane), m=2..8 with fp32 activations sharing one decoded
dword (1.6-5.3x), and a third that takes fp16 activations and does the math with
v_dot2_f32_f16 for any m=1..8 (4.1-8.7x). That feed is the default unless the
checkpoint carries a codebook, because the uniform `code - 8` grid is exact in
fp16; a codebook rounds its 16 levels (~2.5e-4) so it needs
COMFY_GFX906_W4A16_F16ACT=1. m > 8, convrot and any other mismatch fall back to Triton.

COMFY_GFX906_W4A16_HIP=off restores the Triton kernel for an A/B run;
COMFY_GFX906_W4A16_F16ACT=1 also lets a codebook checkpoint use that feed;
COMFY_GFX906_W4A16_VALU=off restores the tab[16] feed for an A/B run; the
VALU-built levels are the default for m = 2..4 (m > 4 keeps the table).

The stream is read at call time: torch.cuda.graph() switches current_stream to its
capture stream, so a cached handle would leave the launch out of the graph.
"""
import ctypes
import glob
import logging
import os
import sys

import torch

try:
    from .kernels import registry as kreg
except ImportError:                                   # imported as a top-level module
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kernels import registry as kreg

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
        # An .so built before one of these entries leaves that attribute None and
        # the caller falls back for the shapes it needs.
        for name in ("gfx906_w4a16", "gfx906_w4a16_m", "gfx906_w4a16_h",
                     "gfx906_w4a16_mu"):
            try:
                fn = getattr(lib, name)
                fn.restype = ctypes.c_int
                fn.argtypes = ([ctypes.c_void_p] * 7 + [ctypes.c_int] * 7
                               + [ctypes.c_int64, ctypes.c_void_p])
            except AttributeError:
                setattr(lib, name, None)
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


def f16_enabled():
    """Whether a checkpoint *with* a codebook may use the fp16 feed. Without one the
    grid is exact in fp16 and the caller uses that feed regardless; with one it rounds
    the 16 levels, a ~2.5e-4 systematic relative error rather than rounding noise."""
    return os.getenv("COMFY_GFX906_W4A16_F16ACT", "0").strip().lower() in ("1", "on", "true", "yes")


def valu_enabled():
    """Explicit env > `feature.w4a16_valu` in the arch manifest > off.

    The measurement gate is tier-3: an unmeasured card keeps the tab[16] feed,
    one bench run turns VALU on where it won. gfx906 ships measured data, so it
    is on there. `linear_dequant` still sends m > 4 to the table entry: that
    shape spills at 256 VGPR and the VALU levels cost more than they save.
    """
    explicit = os.getenv("COMFY_GFX906_W4A16_VALU")
    if explicit is not None and explicit.strip():
        return explicit.strip().lower() not in ("0", "off", "false", "no")
    table = kreg.runtime_manifest()
    return table is not None and table.default_impl("feature.w4a16_valu") == "on"


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


def _launch_multi(entry, x, qdata, s_rel, s_channel, bias, out, group_size, codebook):
    """Both multi-row entries take (ptrs, m, n, k, gs, out_f16, has_cb, has_bias, stride)."""
    if entry is None:
        return False
    m, n, k = x.shape[0], qdata.shape[0], x.shape[1]
    rc = entry(x.data_ptr(), qdata.view(torch.int32).data_ptr(),
               s_rel.view(torch.uint8).data_ptr(), s_channel.data_ptr(),
               out.data_ptr() if bias is None else bias.data_ptr(), out.data_ptr(),
               s_channel.data_ptr() if codebook is None else codebook.data_ptr(),
               m, n, k, group_size, 1 if out.dtype == torch.float16 else 0,
               1 if codebook is not None else 0, 1 if bias is not None else 0,
               k // group_size, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"gfx906 HIP W4A16 launch failed ({rc})")
    return True


def linear_dequant(x, qdata, s_rel, s_channel, bias, out, group_size, codebook):
    """Fill out[:, :], or return False for a shape or dtype these kernels do not take.

    x is the activation rows (fp16 for one row, fp32 for more), s_rel the fp8
    per-group scale plane and codebook the 16 fp32 levels, or None for the uniform
    `code - 8` grid. A bias must be in the output dtype; anything else the kernels
    cannot take is left to the Triton path. Every path here keeps the levels in
    fp32, so the result matches Triton inside rounding.
    """
    lib = _load()
    rows = x.shape[0]
    if (lib is None or rows > 8
            or not _accepted(x, qdata, s_rel, s_channel, bias, out, group_size, codebook)):
        return False
    n, k = qdata.shape[0], x.shape[1]
    if rows > 1:
        # The fp32 multi-row kernel stages M x BK fp32 activations in LDS; handing
        # it an fp16 row set would read past the plane. m=1 keeps its own fp16 path.
        if x.dtype != torch.float32:
            return False
        entry = lib.gfx906_w4a16_m
        if codebook is None and lib.gfx906_w4a16_mu is not None and valu_enabled():
            entry = lib.gfx906_w4a16_mu
        return _launch_multi(entry, x, qdata, s_rel, s_channel, bias, out,
                             group_size, codebook)
    if lib.gfx906_w4a16 is None:
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


def linear_dequant_h(x, qdata, s_rel, s_channel, bias, out, group_size, codebook):
    """The fp16-activation kernel: products resolved by v_dot2_f32_f16 off a packed
    fp16 level table, which needs 29-59 VGPRs instead of 94-240 and therefore runs two
    to four blocks per CU. Takes fp16 x and 1..8 rows. The uniform `code - 8` grid is
    exact in fp16 so it costs nothing numerically; a codebook's 16 levels round to
    fp16 (~2.5e-4 relative), which is why the caller gates that case.
    """
    lib = _load()
    rows = x.shape[0]
    if (lib is None or rows > 8 or x.dtype != torch.float16
            or not _accepted(x, qdata, s_rel, s_channel, bias, out, group_size, codebook)):
        return False
    return _launch_multi(lib.gfx906_w4a16_h, x, qdata, s_rel, s_channel, bias, out,
                         group_size, codebook)
