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


def linear(x, weight, weight_scale, bias, out_dtype, groupsize):
    """C = x @ weight.T for a single row, with convrot rotation when groupsize is set.

    x: [1, k] float32, weight: [n, k] int8, weight_scale: [n] or [1].
    The activation is quantized per row inside the kernel; groupsize 0 skips
    rotation and uses the row max of the raw activation.
    """
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
