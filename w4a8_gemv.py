"""Single-token W4A8 GEMV for gfx906: nibbles unpacked in registers.

Decode is one row, so the INT8 GEMM tile is pure overhead and the step is a
bandwidth problem. Keeping the weight at 4 bits halves what has to be streamed,
but only if the unpacking happens in registers: one int32 word holds 8 nibbles,
`& 0x0F0F0F0F` and `>> 4` turn them into two registers of four int8 lanes, and
gfx906's `v_dot4_i32_i8` multiplies four lanes per instruction (it is how the
MI50 reaches its 53 TOPS INT8). Triton does not emit that instruction on its
own, so it goes through inline asm.

`linear` also takes up to 8 rows, which is what speculative decoding (MTP) feeds
the model each step. The weight is still streamed once: every row is a
broadcast of the same `w` tile through the dot, and only the activation
quantization is per row.

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
MAX_M = 8
WARPS = 4
STAGES = 2


def _block_k(m):
    """K chunk per row count: the m-row dot keeps M * BN * BW operands live."""
    if m == 1:
        return 2048
    return 1024 if m <= 4 else 512


def _row_block(rows):
    """Next power of two, the only width tl.arange accepts."""
    return 1 if rows <= 1 else 2 if rows <= 2 else 4 if rows <= 4 else 8


@triton.jit
def _dot4(a, b):
    """Four int8 products accumulated into int32, one instruction."""
    return tl.inline_asm_elementwise(
        "v_dot4_i32_i8 $0, $1, $2, $3",
        "=v,v,v,v", [a, b, tl.zeros_like(a)], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _pack4(q):
    """[M, BW, 4] int32 in [-128, 127] -> [M, BW] int32, one byte per lane."""
    return tl.sum((q & 0xFF) << (tl.arange(0, 4) * 8)[None, None, :], axis=2)


@triton.jit
def gemv_kernel(a_ptr, b_ptr, c_ptr, gmax_ptr, srel_ptr, sch_ptr, bias_ptr,
                m, n, k, ngroups, stride_bn, stride_bs,
                BN: tl.constexpr, BK: tl.constexpr, NG: tl.constexpr,
                GS: tl.constexpr, BW: tl.constexpr, M: tl.constexpr,
                HAS_BIAS: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    mask_n = offs_n < n
    offs_g = tl.arange(0, NG)
    offs_m = tl.arange(0, M)
    mask_m = offs_m < m

    gm = tl.load(gmax_ptr + offs_m[:, None] * ngroups + offs_g[None, :],
                 mask=mask_m[:, None] & (offs_g[None, :] < ngroups), other=0.0)
    a_scale = tl.maximum(tl.max(gm, axis=1) / 127.0, 1e-30)      # [M]
    s_ch = tl.load(sch_ptr + offs_n, mask=mask_n, other=0.0)     # [BN]

    bpgw: tl.constexpr = GS // 8      # int32 weight words per group
    ngpc: tl.constexpr = BK // GS     # groups per K chunk
    epg: tl.constexpr = GS // 2       # even (or odd) activations per group

    acc = tl.zeros((M, BN), dtype=tl.float32)
    for kk in range(0, k, BK):
        base = kk // 8
        idx = tl.arange(0, 4 * BW)
        off = offs_m[:, None] * k + kk + 2 * idx[None, :]
        even = tl.load(a_ptr + off, mask=mask_m[:, None], other=0.0)
        odd = tl.load(a_ptr + off + 1, mask=mask_m[:, None], other=0.0)
        qe = tl.clamp(libdevice.rint(even / a_scale[:, None]), -128.0, 127.0).to(tl.int32)
        qo = tl.clamp(libdevice.rint(odd / a_scale[:, None]), -128.0, 127.0).to(tl.int32)
        pe = _pack4(tl.reshape(qe, (M, BW, 4)))
        po = _pack4(tl.reshape(qo, (M, BW, 4)))
        qsum = (tl.sum(tl.reshape(qe, (M, ngpc, epg)), axis=2)
                + tl.sum(tl.reshape(qo, (M, ngpc, epg)), axis=2))       # [M, ngpc]

        w = tl.load(b_ptr + offs_n[:, None] * stride_bn + (base + tl.arange(0, BW))[None, :],
                    mask=mask_n[:, None], other=0)
        pair = (_dot4(w[None, :, :] & 0x0F0F0F0F, pe[:, None, :])
                + _dot4((w[None, :, :] >> 4) & 0x0F0F0F0F, po[:, None, :]))   # [M, BN, BW]

        part = tl.sum(tl.reshape(pair, (M, BN, ngpc, bpgw)), axis=3)
        grp = kk // GS + tl.arange(0, ngpc)
        sg = tl.load(srel_ptr + offs_n[:, None] * stride_bs + grp[None, :]).to(tl.float32)
        acc += tl.sum((part - 8 * qsum[:, None, :]).to(tl.float32) * sg[None, :, :], axis=2)

    acc *= a_scale[:, None] * s_ch[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


def linear(x, qdata, s_rel, s_channel, bias, out_dtype, group_size, convrot_groupsize):
    """x @ W.T for one to eight rows, or None when the shape is not a decode of that size.

    qdata is the packed nibble plane [n, k/2], s_rel the fp8 per-group scale
    [n, k/group_size], s_channel the per-row scale [n]. convrot_groupsize 0
    skips the Hadamard rotation and uses the raw row max, matching int8_gemv.
    """
    rows, xk = x.shape
    n, half = qdata.shape
    k = s_rel.shape[1] * group_size
    if rows < 1 or rows > MAX_M or xk != k or half * 2 != k or group_size % 8:
        return None
    if qdata.stride(0) != half or not qdata.is_contiguous() or s_rel.stride(0) != s_rel.shape[1]:
        return None
    if s_channel.numel() != n or s_channel.dtype != torch.float32:
        return None
    if convrot_groupsize and k % convrot_groupsize:
        return None
    m = _row_block(rows)
    bk = _block_k(m)
    if k % bk:
        return None

    if convrot_groupsize:
        x, gmax, ngroups = int8_gemv.rotate(x, convrot_groupsize)
    else:
        gmax, ngroups = x.abs().amax(dim=1, keepdim=True), 1

    out = torch.empty((rows, n), device=qdata.device, dtype=out_dtype)
    gemv_kernel[(triton.cdiv(n, BN),)](
        x, qdata.view(torch.int32), out, gmax, s_rel, s_channel,
        bias if bias is not None else x,
        rows, n, k, ngroups, half // 4, s_rel.stride(0),
        BN=BN, BK=bk, NG=triton.next_power_of_2(ngroups), GS=group_size, BW=bk // 8, M=m,
        HAS_BIAS=bias is not None, num_warps=WARPS, num_stages=STAGES)
    return out
