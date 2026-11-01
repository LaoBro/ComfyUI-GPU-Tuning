"""Single-token W4A8 GEMV for gfx906: nibbles unpacked in registers.

Decode is one row, so the INT8 GEMM tile is pure overhead and the step is a
bandwidth problem. Keeping the weight at 4 bits halves what has to be streamed,
but only if the unpacking happens in registers: one int32 word holds 8 nibbles,
`& 0x0F0F0F0F` and `>> 4` turn them into two registers of four int8 lanes, and
gfx906's `v_dot4_i32_i8` multiplies four lanes per instruction (it is how the
MI50 reaches its 53 TOPS INT8). Triton does not emit that instruction on its
own, so it goes through inline asm.

The nibbles are kept as 0..15 and the zero point is applied after the group sum
(`part - 8 * sum(q)`) because a byte-wise `- 0x08080808` borrows across bytes.

Only the uniform grid is handled. The shipped checkpoints carry a per-group
Lloyd-Max codebook, whose decoded levels differ per group, so a fast kernel
would need one table lookup per weight; requantize to uniform first
(`tools/requant_uniform4.py`).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

try:
    from . import int8_gemv
except ImportError:
    import int8_gemv

BN = 16
BK = 2048          # weights per K chunk; must divide every decode K
WARPS = 4
STAGES = 2


@triton.jit
def _dot4(a, b):
    """Four int8 products accumulated into int32, one instruction."""
    return tl.inline_asm_elementwise(
        "v_dot4_i32_i8 $0, $1, $2, $3",
        "=v,v,v,v", [a, b, tl.zeros_like(a)], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _pack4(q):
    """[BW, 4] int32 in [-128, 127] -> [BW] int32, one byte per lane."""
    return tl.sum((q & 0xFF) << (tl.arange(0, 4) * 8)[None, :], axis=1)


@triton.jit
def gemv_kernel(a_ptr, b_ptr, c_ptr, gmax_ptr, srel_ptr, sch_ptr, bias_ptr,
                n, k, ngroups, stride_bn, stride_bs,
                BN: tl.constexpr, BK: tl.constexpr, NG: tl.constexpr,
                GS: tl.constexpr, BW: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    mask_n = offs_n < n

    offs_g = tl.arange(0, NG)
    gmax = tl.load(gmax_ptr + offs_g, mask=offs_g < ngroups, other=0.0)
    a_scale = tl.maximum(tl.max(gmax) / 127.0, 1e-30)
    s_ch = tl.load(sch_ptr + offs_n, mask=mask_n, other=0.0)

    bpgw: tl.constexpr = GS // 8      # int32 weight words per group
    ngpc: tl.constexpr = BK // GS     # groups per K chunk
    epg: tl.constexpr = GS // 2       # even (or odd) activations per group

    acc = tl.zeros((BN,), dtype=tl.float32)
    for kk in range(0, k, BK):
        base = kk // 8
        idx = tl.arange(0, 4 * BW)
        qe = tl.clamp(libdevice.rint(tl.load(a_ptr + kk + 2 * idx) / a_scale), -128.0, 127.0).to(tl.int32)
        qo = tl.clamp(libdevice.rint(tl.load(a_ptr + kk + 2 * idx + 1) / a_scale), -128.0, 127.0).to(tl.int32)
        pe = _pack4(tl.reshape(qe, (BW, 4)))
        po = _pack4(tl.reshape(qo, (BW, 4)))
        qsum = tl.sum(tl.reshape(qe, (ngpc, epg)), axis=1) + tl.sum(tl.reshape(qo, (ngpc, epg)), axis=1)

        w = tl.load(b_ptr + offs_n[:, None] * stride_bn + (base + tl.arange(0, BW))[None, :],
                    mask=mask_n[:, None], other=0)
        pair = _dot4(w & 0x0F0F0F0F, pe[None, :]) + _dot4((w >> 4) & 0x0F0F0F0F, po[None, :])

        part = tl.sum(tl.reshape(pair, (BN, ngpc, bpgw)), axis=2)
        grp = kk // GS + tl.arange(0, ngpc)
        sg = tl.load(srel_ptr + offs_n[:, None] * stride_bs + grp[None, :]).to(tl.float32)
        acc += tl.sum((part - 8 * qsum[None, :]).to(tl.float32) * sg, axis=1)

    acc *= a_scale * s_ch
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(c_ptr + offs_n, acc.to(c_ptr.dtype.element_ty), mask=mask_n)


def linear(x, qdata, s_rel, s_channel, bias, out_dtype, group_size, convrot_groupsize):
    """x @ W.T for one row, or None when the shape is not the decode shape.

    qdata is the packed nibble plane [n, k/2], s_rel the fp8 per-group scale
    [n, k/group_size], s_channel the per-row scale [n]. convrot_groupsize 0
    skips the Hadamard rotation and uses the raw row max, matching int8_gemv.
    """
    n, half = qdata.shape
    k = s_rel.shape[1] * group_size
    if k % BK or half * 2 != k or group_size % 8 or BK % group_size:
        return None
    if qdata.stride(0) != half or not qdata.is_contiguous() or s_rel.stride(0) != s_rel.shape[1]:
        return None
    if x.dim() != 2 or x.shape[0] != 1 or x.shape[1] != k:
        return None
    if s_channel.numel() != n or s_channel.dtype != torch.float32:
        return None
    if convrot_groupsize and k % convrot_groupsize:
        return None

    if convrot_groupsize:
        x, gmax, ngroups = int8_gemv.rotate(x, convrot_groupsize)
    else:
        gmax, ngroups = x.abs().amax().reshape(1), 1

    out = torch.empty((1, n), device=qdata.device, dtype=out_dtype)
    gemv_kernel[(triton.cdiv(n, BN),)](
        x, qdata.view(torch.int32), out, gmax, s_rel, s_channel,
        bias if bias is not None else x,
        n, k, ngroups, half // 4, s_rel.stride(0),
        BN=BN, BK=BK, NG=triton.next_power_of_2(ngroups), GS=group_size, BW=BK // 8,
        HAS_BIAS=bias is not None, num_warps=WARPS, num_stages=STAGES)
    return out
