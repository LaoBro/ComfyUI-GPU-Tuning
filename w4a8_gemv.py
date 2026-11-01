"""Packed 4-bit weight kernels for gfx906: a decode-only GEMV and the int8
decode prefill runs before comfy-kitchen's GEMM.

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

Prefill (more rows) runs `decode_int8` first and hands the result to the int8
GEMM: unpacking into the dot stream instead costs more than the streaming pass,
because every decoded tile has to change layout before `tl.dot` accepts it
(1.35x the int8 time on a 12288x4096 layer, against 1.05x for one extra pass).

A per-group Lloyd-Max codebook resolves through a 16-entry table instead of the
`code - 8` ramp: `v_perm_b32` picks one of eight bytes per lane (the second
operand supplies codes 0-3, the first codes 4-7), the code's top bit selects
which half, and the levels land on a per-layer int8 grid whose reciprocal folds
into the epilogue. Dropping the zero-point reduction pays for the lookup: over
the 200 text-encoder layers it is 16.4 ms/token against 17.5 for the uniform
kernel. Prefill still unpacks whole layers to int8 (`decode_int8`), because the
int8 GEMM needs a contiguous tile.
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
DECODE_BN = 2
DECODE_BK = 1024
DECODE_WARPS = 4
CB_BLOCK_K = 1024


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
def _perm(hi, lo, sel):
    """Eight int8 levels selected per byte of `sel` (low 3 bits), four per word.

    Indices 0-3 come from the second source operand and 4-7 from the first, so
    `lo` carries levels 0-3 and `hi` levels 4-7.
    """
    return tl.inline_asm_elementwise(
        "v_perm_b32 $0, $1, $2, $3", "=v,v,v,v", [hi, lo, sel],
        dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _lut_bytes(w, t0, t1, t2, t3):
    """Four int8 codebook levels per packed nibble word, one per byte lane."""
    sel = w & 0x07070707
    low = _perm(t1, t0, sel)
    high = _perm(t3, t2, sel)
    mask = ((w & 0x08080808) >> 3) * 0xFF      # 0xFF for codes 8..15
    return low ^ ((low ^ high) & mask)


@triton.jit
def gemv_kernel(a_ptr, b_ptr, c_ptr, gmax_ptr, srel_ptr, sch_ptr, bias_ptr, cb_ptr,
                m, n, k, ngroups, stride_bn, stride_bs, inv_level,
                BN: tl.constexpr, BK: tl.constexpr, NG: tl.constexpr,
                GS: tl.constexpr, BW: tl.constexpr, M: tl.constexpr,
                HAS_BIAS: tl.constexpr, HAS_CB: tl.constexpr):
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
    t0 = tl.load(cb_ptr + 0)
    t1 = tl.load(cb_ptr + 1)
    t2 = tl.load(cb_ptr + 2)
    t3 = tl.load(cb_ptr + 3)

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

        w = tl.load(b_ptr + offs_n[:, None] * stride_bn + (base + tl.arange(0, BW))[None, :],
                    mask=mask_n[:, None], other=0)
        grp = kk // GS + tl.arange(0, ngpc)
        sg = tl.load(srel_ptr + offs_n[:, None] * stride_bs + grp[None, :]).to(tl.float32)
        if HAS_CB:
            # the table's levels already carry the zero point and the group scale
            pair = (_dot4(_lut_bytes(w, t0, t1, t2, t3)[None, :, :], pe[:, None, :])
                    + _dot4(_lut_bytes(w >> 4, t0, t1, t2, t3)[None, :, :], po[:, None, :]))
            part = tl.sum(tl.reshape(pair, (M, BN, ngpc, bpgw)), axis=3)
            acc += tl.sum(part.to(tl.float32) * sg[None, :, :], axis=2)
        else:
            qsum = (tl.sum(tl.reshape(qe, (M, ngpc, epg)), axis=2)
                    + tl.sum(tl.reshape(qo, (M, ngpc, epg)), axis=2))       # [M, ngpc]
            pair = (_dot4(w[None, :, :] & 0x0F0F0F0F, pe[:, None, :])
                    + _dot4((w[None, :, :] >> 4) & 0x0F0F0F0F, po[:, None, :]))
            part = tl.sum(tl.reshape(pair, (M, BN, ngpc, bpgw)), axis=3)
            acc += tl.sum((part - 8 * qsum[:, None, :]).to(tl.float32) * sg[None, :, :], axis=2)

    acc *= a_scale[:, None] * s_ch[None, :] * inv_level
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


def codebook_words(codebook):
    """Pack the 16 levels into four int8 bytes-per-lane words, plus the grid scale.

    A codebook level is a 16-way table, but `v_perm_b32` only picks one of eight
    bytes, so the codes are resolved per half (`code & 7` plus the sign bit).
    Levels land on a per-layer int8 grid (`127/max|level|`), which costs at most
    half an int8 step against the exact `round(level * s_rel)` grid. Compute this
    once per layer: the caller caches it, it is 16 bytes.
    """
    cb = codebook.float()
    scale = 127.0 / float(cb.abs().max().clamp(min=1e-8))
    g = (cb * scale).round().clamp(-127.0, 127.0).to(torch.int32).tolist()
    words = []
    for i in range(0, 16, 4):
        v = (g[i] & 0xFF) | ((g[i + 1] & 0xFF) << 8) | ((g[i + 2] & 0xFF) << 16) | ((g[i + 3] & 0xFF) << 24)
        words.append(v - (1 << 32) if v >= (1 << 31) else v)
    return torch.tensor(words, dtype=torch.int32, device=codebook.device), 1.0 / scale


def linear(x, qdata, s_rel, s_channel, bias, out_dtype, group_size, convrot_groupsize,
           levels=None, inv_level=1.0):
    """x @ W.T for one to eight rows, or None when the shape is not a decode of that size.

    qdata is the packed nibble plane [n, k/2], s_rel the fp8 per-group scale
    [n, k/group_size], s_channel the per-row scale [n]. convrot_groupsize 0
    skips the Hadamard rotation and uses the raw row max, matching int8_gemv.
    `levels`/`inv_level` are `codebook_words` output: they replace the uniform
    `code - 8` ramp with a table lookup and fold the level grid into the scale.
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
    if levels is not None and levels.numel() != 4:
        return None
    m = _row_block(rows)
    # the LUT path keeps five more word tensors live, so a wide chunk spills
    bk = min(_block_k(m), CB_BLOCK_K) if levels is not None else _block_k(m)
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
        levels if levels is not None else qdata.view(torch.int32).reshape(-1)[:4],
        rows, n, k, ngroups, half // 4, s_rel.stride(0), inv_level,
        BN=BN, BK=bk, NG=triton.next_power_of_2(ngroups), GS=group_size, BW=bk // 8, M=m,
        HAS_BIAS=bias is not None, HAS_CB=levels is not None,
        num_warps=WARPS, num_stages=STAGES)
    return out


@triton.jit
def decode_kernel(b_ptr, srel_ptr, cb_ptr, out_ptr, n, k, stride_b, stride_s,
                  GS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  HAS_CB: tl.constexpr):
    """One program turns BN rows of packed bytes into BK int8 weights each.

    The byte at (row, i) holds k=2i in its low nibble and k=2i+1 in its high
    nibble, so `interleave` restores the row order the GEMM expects.

    A codebook replaces the `code - 8` ramp with one gather into the per-layer
    level table; the gather hits a 64-byte table in L1, so it costs far less
    than the sixteen compares a select tree would need.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    idx = pid_k * BK + tl.arange(0, BK)
    mask_n = offs_n < n
    b = tl.load(b_ptr + offs_n[:, None] * stride_b + idx[None, :], mask=mask_n[:, None], other=0)
    sg = tl.load(srel_ptr + offs_n[:, None] * stride_s + (idx // (GS // 2))[None, :],
                 mask=mask_n[:, None], other=0.0).to(tl.float32)
    if HAS_CB:
        level = tl.load(cb_ptr + (b & 0x0F).to(tl.int32)).to(tl.float32)
        lo = tl.clamp(libdevice.rint(level * sg), -127.0, 127.0).to(tl.int8)
        level = tl.load(cb_ptr + ((b >> 4) & 0x0F).to(tl.int32)).to(tl.float32)
        hi = tl.clamp(libdevice.rint(level * sg), -127.0, 127.0).to(tl.int8)
    else:
        lo = tl.clamp(libdevice.rint(((b & 0x0F).to(tl.float32) - 8.0) * sg), -127.0, 127.0).to(tl.int8)
        hi = tl.clamp(libdevice.rint((((b >> 4) & 0x0F).to(tl.float32) - 8.0) * sg), -127.0, 127.0).to(tl.int8)
    tl.store(out_ptr + offs_n[:, None] * k + (2 * pid_k * BK + tl.arange(0, 2 * BK))[None, :],
             tl.interleave(lo, hi), mask=mask_n[:, None])


def decode_int8(qdata, s_rel, group_size, codebook=None):
    """Unpack a whole layer to the int8 grid the int8 kernels consume, or None.

    Bit-identical to comfy_kitchen's `_dequant_int4_grouped_to_int8`, so the
    result does not depend on which path produced the int8 weight.
    """
    n, half = qdata.shape
    k = half * 2
    if group_size % 2 or k % group_size or qdata.stride(0) != half or not qdata.is_contiguous():
        return None
    if s_rel.stride(0) != s_rel.shape[1]:
        return None
    if codebook is not None and codebook.numel() != 16:
        return None
    bk = DECODE_BK
    while bk > 128 and half % bk:
        bk //= 2
    if half % bk:
        return None
    out = torch.empty((n, k), device=qdata.device, dtype=torch.int8)
    decode_kernel[(triton.cdiv(n, DECODE_BN), half // bk)](
        qdata.view(torch.uint8), s_rel, codebook if codebook is not None else s_rel,
        out, n, k, half, s_rel.stride(0),
        GS=group_size, BN=DECODE_BN, BK=bk, HAS_CB=codebook is not None,
        num_warps=DECODE_WARPS, num_stages=2)
    return out
