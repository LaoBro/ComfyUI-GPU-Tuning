"""Split-K int8 GEMV for single-token text decoding on gfx906.

The comfy-kitchen int8 kernel is a tiled GEMM: it holds a block of rows of the
weight for a block of N outputs and runs tl.dot. At m=1 that wastes the whole
row dimension, and the N tile is picked for throughput, not for how many programs
a 60-CU card can keep busy. Decode is a bandwidth problem instead: every step
streams the entire weight once.

So this kernel gives up the row tile entirely. One program owns BLOCK_N output
rows, walks K in BLOCK_K chunks and accumulates tl.int32 per row. Integer sums
are exact, so the result matches the GEMM path bit for bit for the same quantized
activation. Rotation and quantization are folded in as well: the rotate kernel
writes the rotated activation plus one max per 256-group, and each GEMV program
turns that into the row scale and quantizes on the fly. That removes two of the
three launches a decode step used to pay for, which matters more than the kernel
itself at these sizes.
"""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from comfy_kitchen.tensor.int8_utils import _build_hadamard

BLOCK_N = 16
BLOCK_K = 1024
BLOCK_J = 32
WARPS = 4
STAGES = 2
ROTATE_WARPS = 4
ASM_BLOCK_N = 16
ASM_BLOCK_K = 2048

try:
    from .kernels import registry as kreg
except ImportError:                                   # imported as a top-level module
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kernels import registry as kreg


@triton.jit
def rotate_kernel(x_ptr, h_ptr, out_ptr, gmax_ptr, ngroups, k,
                  G: tl.constexpr, BJ: tl.constexpr):
    """Rotate one activation group and record its max.

    out[g, j] = sum_i x[g, i] * h[i, j]; the max over the groups is the row max
    the quantizer needs, so it is collected here instead of in a second pass.
    One program per (row, group), so a multi-row prefetch batch costs the same
    launches as a single row.
    """
    pid = tl.program_id(0)
    row = pid // ngroups
    g = pid - row * ngroups
    base = row * k + g * G
    offs = tl.arange(0, G)
    xg = tl.load(x_ptr + base + offs)
    mx = 0.0
    for j0 in range(0, G, BJ):
        offs_j = j0 + tl.arange(0, BJ)
        h = tl.load(h_ptr + offs_j[:, None] * G + offs[None, :])
        rot = tl.sum(h * xg[None, :], axis=1)
        tl.store(out_ptr + base + offs_j, rot)
        mx = tl.maximum(mx, tl.max(tl.abs(rot)))
    tl.store(gmax_ptr + pid, mx)


@triton.jit
def gemv_kernel(a_ptr, b_ptr, c_ptr, gmax_ptr, b_scale_ptr, bias_ptr,
                n, k, ngroups, stride_bn,
                BN: tl.constexpr, BK: tl.constexpr, NG: tl.constexpr,
                BSCALE_VEC: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    mask_n = offs_n < n

    offs_g = tl.arange(0, NG)
    gmax = tl.load(gmax_ptr + offs_g, mask=offs_g < ngroups, other=0.0)
    a_scale = tl.maximum(tl.max(gmax) / 127.0, 1e-30)

    acc = tl.zeros((BN,), dtype=tl.int32)
    for kk in range(0, k, BK):
        offs_k = kk + tl.arange(0, BK)
        mask_k = offs_k < k
        a = tl.load(a_ptr + offs_k, mask=mask_k, other=0.0)
        q = tl.clamp(libdevice.rint(a / a_scale), -128.0, 127.0).to(tl.int32)
        b = tl.load(b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :],
                    mask=mask_k[None, :] & mask_n[:, None], other=0).to(tl.int32)
        acc += tl.sum(b * q[None, :], axis=1)

    if BSCALE_VEC:
        b_scale = tl.load(b_scale_ptr + offs_n, mask=mask_n, other=0.0)
    else:
        b_scale = tl.load(b_scale_ptr)
    out = acc.to(tl.float32) * (a_scale * b_scale)
    if HAS_BIAS:
        out += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    tl.store(c_ptr + offs_n, out.to(c_ptr.dtype.element_ty), mask=mask_n)


def rotate(x, groupsize):
    """Rotate an [m, k] activation; returns the rotated tensor and group maxima [m, k // groupsize]."""
    rows, k = x.shape
    ngroups = k // groupsize
    h = _build_hadamard(groupsize, device=x.device, dtype=x.dtype)
    rotated = torch.empty_like(x)
    gmax = torch.empty((rows, ngroups), device=x.device, dtype=torch.float32)
    rotate_kernel[(rows * ngroups,)](x, h, rotated, gmax, ngroups, k, G=groupsize,
                                     BJ=BLOCK_J, num_warps=ROTATE_WARPS)
    return rotated, gmax, ngroups


@triton.jit
def _dot4_asm(a, b):
    return tl.inline_asm_elementwise(
        "v_dot4_i32_i8 $0, $1, $2, $3",
        "=v,v,v,v", [a, b, tl.zeros_like(a)], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _pack4(q):
    return tl.sum((q & 0xFF) << (tl.arange(0, 4) * 8)[None, None, :], axis=2)


@triton.jit
def asm_kernel(a_ptr, b_ptr, c_ptr, gmax_ptr, b_scale_ptr, bias_ptr,
               n, k, ngroups, stride_bn_words,
               BN: tl.constexpr, BK: tl.constexpr, NG: tl.constexpr,
               BSCALE_VEC: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    mask_n = offs_n < n
    offs_g = tl.arange(0, NG)
    gmax = tl.load(gmax_ptr + offs_g, mask=offs_g < ngroups, other=0.0)
    a_scale = tl.maximum(tl.max(gmax) / 127.0, 1e-30)

    acc = tl.zeros((BN,), dtype=tl.int32)
    for kk in range(0, k, BK):
        offs_k = kk + tl.arange(0, BK)
        mask_k = offs_k < k
        a = tl.load(a_ptr + offs_k, mask=mask_k, other=0.0)
        q = tl.clamp(libdevice.rint(a / a_scale), -128.0, 127.0).to(tl.int32)
        qp = _pack4(tl.reshape(q, (1, BK // 4, 4)))
        w = tl.load(b_ptr + offs_n[:, None] * stride_bn_words + (kk // 4 + tl.arange(0, BK // 4))[None, :],
                    mask=mask_n[:, None], other=0)
        qrow = tl.broadcast_to(qp, (BN, BK // 4))
        acc += tl.sum(_dot4_asm(w, qrow), axis=1)

    if BSCALE_VEC:
        b_scale = tl.load(b_scale_ptr + offs_n, mask=mask_n, other=0.0)
    else:
        b_scale = tl.load(b_scale_ptr)
    out = acc.to(tl.float32) * (a_scale * b_scale)
    if HAS_BIAS:
        out += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    tl.store(c_ptr + offs_n, out.to(c_ptr.dtype.element_ty), mask=mask_n)



def _row_block(rows):
    """Next power of two up to 8, the only width tl.arange accepts."""
    return 1 if rows <= 1 else 2 if rows <= 2 else 4 if rows <= 4 else 8


@triton.jit
def dequant_gemv_kernel(a_ptr, b_ptr, c_ptr, b_scale_ptr, n, k, m, stride_bn,
                        BN: tl.constexpr, BK: tl.constexpr, M: tl.constexpr,
                        BSCALE_VEC: tl.constexpr, HAS_BIAS: tl.constexpr):
    # out[m, n] = scale[n] * sum_k rot_x[m, k] * q[n,k];q 在寄存器里转 fp32,
    # fp32 累加;每个 K 块的权重读取由 M 行摊薄
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    mask_n = offs_n < n
    offs_m = tl.arange(0, M)
    mask_m = offs_m < m
    acc = tl.zeros((M, BN), dtype=tl.float32)
    for kk in range(0, k, BK):
        offs_k = kk + tl.arange(0, BK)
        mask_k = offs_k < k
        b = tl.load(b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :],
                    mask=mask_k[None, :] & mask_n[:, None], other=0).to(tl.float32)
        for mi in tl.static_range(M):
            a = tl.load(a_ptr + mi * k + offs_k, mask=mask_k & (mi < m), other=0.0)
            part = tl.sum(b * a[None, :], axis=1)
            acc += tl.where((offs_m == mi)[:, None], part[None, :], 0.0)
    if BSCALE_VEC:
        b_scale = tl.load(b_scale_ptr + offs_n, mask=mask_n, other=0.0)
    else:
        b_scale = tl.load(b_scale_ptr)
    acc = acc * b_scale[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


def linear_dequant(x, weight, weight_scale, bias, out_dtype, groupsize):
    """C = x @ dequant(weight).T for up to 8 rows, fp32 accumulate, no act quant.

    The te=default decode path: reads the int8 weight directly and dequantizes
    in registers, so no fp16 copy of the model is ever materialized and no
    per-call dequantize runs. The activation is rotated with the checkpoint's
    convrot H first, upcast to fp32 so the fp16 execution path still rotates in
    fp32. Covers the MTP/spec-verify row counts (m <= 8, padded to a power of
    two); larger M belongs to a GEMM, which would re-read the weight per chunk.
    Returns None for shapes it does not cover; the caller keeps its dispatch
    fallback.
    """
    m, k = x.shape
    if m > 8 or not weight.is_contiguous():
        return None
    n = weight.shape[0]
    if groupsize:
        xr, _, _ = rotate(x.float(), groupsize)
    else:
        xr = x.float()
    rows = _row_block(m)
    bk = 2048 if rows == 1 else 1024 if rows <= 4 else 512
    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    b_scale = weight_scale.reshape(-1)
    dequant_gemv_kernel[(triton.cdiv(n, BLOCK_N),)](
        xr, weight, out, b_scale, n, k, m, weight.stride(0),
        BN=BLOCK_N, BK=bk, M=rows, BSCALE_VEC=b_scale.numel() > 1,
        HAS_BIAS=bias is not None, num_warps=WARPS, num_stages=STAGES)
    return out


@triton.jit
def _probe_dot4(a, b, c):
    x = tl.load(a + tl.arange(0, 1))
    y = tl.load(b + tl.arange(0, 1))
    tl.store(c + tl.arange(0, 1), _dot4_asm(x, y))


def _probe_asm():
    """Whether the v_dot4 inline asm compiles on this device."""
    a = torch.zeros(1, device="cuda", dtype=torch.int32)
    b = torch.zeros(1, device="cuda", dtype=torch.int32)
    c = torch.zeros(1, device="cuda", dtype=torch.int32)
    _probe_dot4[(1,)](a, b, c)
    return True


# One site, two implementations: the asm GEMV when v_dot4 compiles here, the
# int32 tl.sum kernel otherwise. A third one (HIP, or a kernel written for
# another gcn target) is either a new entry here or a file in
# arch/<gcn>/kernels/ that declares SITE = "int8_gemv.m1".
_REGISTRY = kreg.BackendRegistry(
    "int8_gemv.m1",
    kreg.site_candidates("int8_gemv.m1", [
        kreg.Candidate("asm", kind="asm", env="COMFY_GFX906_INT8_GEMV_ASM",
                       probe=_probe_asm, priority=50, requires=("v_dot4_i32_i8",)),
        kreg.Candidate("triton", kind="portable"),
    ]),
)


def backend_name():
    return _REGISTRY.resolve()


def _asm_supported():
    """Whether the asm candidate can run here; kept for existing callers."""
    return _REGISTRY.supported("asm")


def _linear_asm(x, weight, weight_scale, bias, out_dtype, groupsize):
    n, k = weight.shape
    if x.shape[0] != 1 or k % 4 or not weight.is_contiguous():
        return None
    if groupsize:
        x, gmax, ngroups = rotate(x, groupsize)
    else:
        gmax, ngroups = x.abs().amax().reshape(1), 1
    out = torch.empty((1, n), device=x.device, dtype=out_dtype)
    b32 = weight.view(torch.int32)
    b_scale = weight_scale.reshape(-1)
    asm_kernel[(triton.cdiv(n, ASM_BLOCK_N),)](
        x, b32, out, gmax, b_scale, bias if bias is not None else x,
        n, k, ngroups, b32.stride(0),
        BN=ASM_BLOCK_N, BK=ASM_BLOCK_K, NG=triton.next_power_of_2(ngroups),
        BSCALE_VEC=b_scale.numel() > 1, HAS_BIAS=bias is not None,
        num_warps=WARPS, num_stages=STAGES)
    return out


def _linear_triton(x, weight, weight_scale, bias, out_dtype, groupsize):
    n, k = weight.shape
    if groupsize:
        x, gmax, ngroups = rotate(x, groupsize)
    else:
        gmax, ngroups = x.abs().amax().reshape(1), 1

    out = torch.empty((1, n), device=x.device, dtype=out_dtype)
    b_scale = weight_scale.reshape(-1)
    gemv_kernel[(triton.cdiv(n, BLOCK_N),)](
        x, weight, out, gmax, b_scale, bias if bias is not None else x,
        n, k, ngroups, weight.stride(0),
        BN=BLOCK_N, BK=BLOCK_K, NG=triton.next_power_of_2(ngroups),
        BSCALE_VEC=b_scale.numel() > 1, HAS_BIAS=bias is not None,
        num_warps=WARPS, num_stages=STAGES)
    return out


def linear(x, weight, weight_scale, bias, out_dtype, groupsize):
    """C = x @ weight.T for a single row, with convrot rotation when groupsize is set.

    x: [1, k] float32, weight: [n, k] int8, weight_scale: [n] or [1].
    The activation is quantized per row inside the kernel; groupsize 0 skips
    rotation and uses the row max of the raw activation.

    Backend: v_dot4 asm when it compiles here, otherwise the int32 tl.sum kernel.
    COMFY_GFX906_INT8_GEMV_ASM=auto/on/off selects it; a measurement for this
    shape in arch/<gcn>/manifest.toml overrides the probe order.
    """
    try:
        backend = _REGISTRY.choose(n=weight.shape[0], k=weight.shape[1], m=x.shape[0])
    except (AttributeError, IndexError):          # not a plain [n, k] pair
        backend = backend_name()
    if backend == "asm":
        out = _linear_asm(x, weight, weight_scale, bias, out_dtype, groupsize)
        if out is not None:
            return out
    return _linear_triton(x, weight, weight_scale, bias, out_dtype, groupsize)
