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

from . import int8_gemv
from ..hip import w4a16 as hip_w4a16

BN = 16
MAX_M = 8
WARPS = 4
STAGES = 2
DOT_BN = 32
DOT_M = 16
# The portable floor runs one (output block, row) per program: no tl.dot, so no
# minimum K per step, and nothing beyond plain Triton.
SUM_BN = 16
SUM_BK = 256
DECODE_BN = 2
DECODE_BK = 1024
DECODE_WARPS = 4
CB_BLOCK_K = 1024

_hip = None


def _block_k(m):
    """K chunk per row count: the m-row dot keeps M * BN * BW operands live."""
    if m == 1:
        return 2048
    return 1024 if m <= 4 else 512


def _row_block(rows):
    """Next power of two, the only width tl.arange accepts."""
    return 1 if rows <= 1 else 2 if rows <= 2 else 4 if rows <= 4 else 8


@triton.jit
def _dot4_asm(a, b):
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
            pair = (_dot4_asm(_lut_bytes(w, t0, t1, t2, t3)[None, :, :], pe[:, None, :])
                    + _dot4_asm(_lut_bytes(w >> 4, t0, t1, t2, t3)[None, :, :], po[:, None, :]))
            part = tl.sum(tl.reshape(pair, (M, BN, ngpc, bpgw)), axis=3)
            acc += tl.sum(part.to(tl.float32) * sg[None, :, :], axis=2)
        else:
            qsum = (tl.sum(tl.reshape(qe, (M, ngpc, epg)), axis=2)
                    + tl.sum(tl.reshape(qo, (M, ngpc, epg)), axis=2))       # [M, ngpc]
            pair = (_dot4_asm(w[None, :, :] & 0x0F0F0F0F, pe[:, None, :])
                    + _dot4_asm((w[None, :, :] >> 4) & 0x0F0F0F0F, po[:, None, :]))
            part = tl.sum(tl.reshape(pair, (M, BN, ngpc, bpgw)), axis=3)
            acc += tl.sum((part - 8 * qsum[:, None, :]).to(tl.float32) * sg[None, :, :], axis=2)

    acc *= a_scale[:, None] * s_ch[None, :] * inv_level
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, gmax_ptr, srel_ptr, sch_ptr, bias_ptr, cb_ptr,
               m, n, k, ngroups, stride_bn, stride_bs, inv_level,
               GS: tl.constexpr, BN: tl.constexpr, MM: tl.constexpr,
               NG: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_CB: tl.constexpr):
    """Portable fallback: one int8 tl.dot per weight group of GS elements."""
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    mask_n = offs_n < n
    offs_m = tl.arange(0, MM)
    mask_m = offs_m < m

    gm = tl.load(gmax_ptr + offs_m[:, None] * ngroups + tl.arange(0, NG)[None, :],
                 mask=mask_m[:, None] & (tl.arange(0, NG)[None, :] < ngroups), other=0.0)
    a_scale = tl.maximum(tl.max(gm, axis=1) / 127.0, 1e-30)
    sch = tl.load(sch_ptr + offs_n, mask=mask_n, other=0.0)

    acc = tl.zeros((MM, BN), dtype=tl.float32)
    aidx = tl.arange(0, GS)
    cidx = tl.arange(0, GS // 2)
    for g in range(0, k // GS):
        off = g * GS
        a = tl.load(a_ptr + offs_m[:, None] * k + off + aidx[None, :], mask=mask_m[:, None], other=0.0)
        q = tl.clamp(libdevice.rint(a / a_scale[:, None]), -128.0, 127.0).to(tl.int8)
        b = tl.load(b_ptr + offs_n[None, :] * stride_bn + (off // 2 + cidx)[:, None],
                    mask=mask_n[None, :], other=0).to(tl.int32)
        lo = b & 0x0F
        hi = (b >> 4) & 0x0F
        code = tl.reshape(tl.permute(tl.join(lo, hi), (0, 2, 1)), (GS, BN))
        if HAS_CB:
            word = tl.load(cb_ptr + (code >> 2))
            byte = (word >> ((code & 3) * 8)) & 0xFF
            w8 = tl.where(byte >= 128, byte - 256, byte).to(tl.int8)
        else:
            w8 = (code - 8).to(tl.int8)
        sg = tl.load(srel_ptr + offs_n * stride_bs + g, mask=mask_n, other=0.0)
        acc += tl.dot(q, w8, out_dtype=tl.int32).to(tl.float32) \
            * (a_scale[:, None] * (sg * sch)[None, :] * inv_level)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _probe_dot4(a, b, c):
    x = tl.load(a + tl.arange(0, 1))
    y = tl.load(b + tl.arange(0, 1))
    tl.store(c + tl.arange(0, 1), _dot4_asm(x, y))


from .. import registry as kreg


def _load_hip():
    """Optional pure-HIP extension next to this file."""
    global _hip
    if _hip is not None:
        return _hip
    try:
        from ... import w4a8_hip as hip
    except ImportError:
        try:
            import w4a8_hip as hip
        except ImportError:
            return None
    if not hasattr(hip, "linear"):
        return None
    _hip = hip
    return hip


def _probe_asm():
    """Whether the v_dot4 inline asm compiles on this device."""
    if torch.version.hip is None:
        # v_dot4 is a GCN/CDNA instruction. Compiling it on a CUDA device
        # aborts the Triton process instead of raising, so never probe there.
        return False
    a = torch.zeros(1, device="cuda", dtype=torch.int32)
    b = torch.zeros(1, device="cuda", dtype=torch.int32)
    c = torch.zeros(1, device="cuda", dtype=torch.int32)
    _probe_dot4[(1,)](a, b, c)
    return True


# The site's implementation list, in preference order. Adding a kernel means
# adding one entry here -- a HIP module as an "extension", a new asm kernel as
# an "asm" -- plus the module itself, or dropping a file into
# arch/<gcn>/kernels/ so it only exists for that target. The portable Triton
# kernel is the floor and must stay last.
_REGISTRY = kreg.BackendRegistry(
    "w4a8.m1",
    kreg.site_candidates("w4a8.m1", [
        kreg.Candidate("hip", kind="extension", env="COMFY_GPU_TUNE_W4A8_HIP",
                       loader=_load_hip, priority=10, requires=("hip extension",)),
        kreg.Candidate("asm", kind="asm", env="COMFY_GPU_TUNE_W4A8_ASM",
                       probe=_probe_asm, priority=50, requires=("v_dot4_i32_i8", "v_perm_b32")),
        kreg.Candidate("triton", kind="portable"),
        # floor: any group size, no instruction the card may lack
        kreg.Candidate("sum", kind="portable", priority=60),
    ]),
)


def backend_name():
    """hip > asm (when it compiles) > per-group tl.dot > tl.sum floor."""
    return _REGISTRY.resolve()


def _asm_supported():
    """Whether the asm candidate can run here; kept for existing callers."""
    return _REGISTRY.supported("asm")


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

@triton.jit
def dequant_w4a16_kernel(a_ptr, b_ptr, c_ptr, srel_ptr, sch_ptr, bias_ptr, lut_ptr,
                         n, k, m, stride_bw, stride_bs,
                         BN: tl.constexpr, BK: tl.constexpr, M: tl.constexpr,
                         GS: tl.constexpr, BSCALE_VEC: tl.constexpr,
                         HAS_BIAS: tl.constexpr, HAS_LUT: tl.constexpr):
    # out[m, n] = s_ch[n] * sum_g s_rel[n, g] * sum_{k in g} w[n, k] * x[m, k]
    # w 的 nibble 在寄存器里解包到 k 序并转 fp32;fp8 s_rel 位操作软解码
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    mask_n = offs_n < n
    offs_m = tl.arange(0, M)
    mask_m = offs_m < m
    t = tl.arange(0, 8)
    shift = (8 * (t // 2) + 4 * (t % 2))[None, None, :]
    NGPC: tl.constexpr = BK // GS
    acc = tl.zeros((M, BN), dtype=tl.float32)
    for kk in range(0, k, BK):
        w = tl.load(b_ptr + offs_n[:, None] * stride_bw + kk // 8 + tl.arange(0, BK // 8)[None, :],
                    mask=mask_n[:, None], other=0)
        c = ((w[:, :, None] >> shift) & 0xF).to(tl.int32)
        c = tl.reshape(c, (BN, NGPC, GS))
        grp = kk // GS + tl.arange(0, NGPC)
        u8 = tl.load(srel_ptr + offs_n[:, None] * stride_bs + grp[None, :],
                     mask=mask_n[:, None], other=0).to(tl.int32)
        exp = (u8 >> 3) & 0xF
        man = (u8 & 7).to(tl.float32) * 0.125
        sg = tl.where(exp > 0, tl.exp2((exp - 7).to(tl.float32)) * (1.0 + man),
                      man * tl.exp2(-6.0))
        if HAS_LUT:
            dv = tl.load(lut_ptr + c)
        else:
            dv = (c - 8).to(tl.float32)
        dv = dv * sg[:, :, None]
        for mi in tl.static_range(M):
            a = tl.load(a_ptr + mi * k + kk + tl.arange(0, BK),
                        mask=(mi < m), other=0.0).to(tl.float32)
            a3 = tl.reshape(a, (NGPC, GS))
            part = tl.sum(dv * a3[None, :, :], axis=2)
            acc += tl.where((offs_m == mi)[:, None], tl.sum(part, axis=1)[None, :], 0.0)
    if BSCALE_VEC:
        s_ch = tl.load(sch_ptr + offs_n, mask=mask_n, other=0.0)
    else:
        s_ch = tl.load(sch_ptr)
    acc = acc * s_ch[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


def linear_dequant(x, qdata, s_rel, s_channel, bias, out_dtype, group_size,
                   convrot_groupsize, codebook):
    """C = x @ dequant(weight).T for up to 8 rows: int4 weights dequantized in
    registers, fp32 accumulate, fp16/fp32 activations (no activation quant).

    The te=default path for w4a8 checkpoints -- the W4A16 counterpart of
    int8_gemv.linear_dequant. Both encodings resolve in-kernel: the uniform
    grid as (nibble - 8) * s_rel, a codebook as one gather into the 16 float
    levels per nibble. fp8 s_rel decodes via bit manipulation (gfx906 has no
    fp8 ALU). Returns None for shapes it does not cover.
    """
    n, half = qdata.shape
    m, k = x.shape
    if (m > MAX_M or k % group_size or group_size < 4
            or (16 % group_size != 0 and group_size % 16 != 0)
            or half * 2 != k
            or half % 4
            or qdata.stride(0) != half or not qdata.is_contiguous()
            or s_rel.stride(0) != s_rel.shape[1]
            or s_channel.numel() != n or s_channel.dtype != torch.float32
            or s_rel.dtype != torch.float8_e4m3fn):
        return None
    rows = _row_block(m)
    lut = codebook.float().reshape(-1).contiguous() if codebook is not None else None
    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    # The fp16 feed takes x unconverted, so it has to be tried before the cast it
    # skips. Without a codebook it runs by default: the uniform `code - 8` grid is
    # exact in fp16, so that tier gains speed and nothing else changes. A codebook
    # rounds its 16 levels to fp16 (~2.5e-4), which is opt-in through
    # COMFY_GPU_TUNE_W4A16_F16ACT.
    if (hip_w4a16.enabled() and x.dtype == torch.float16 and not convrot_groupsize
            and (codebook is None or hip_w4a16.f16_enabled())):
        if hip_w4a16.linear_dequant_h(x, qdata, s_rel, s_channel, bias, out, group_size, lut):
            return out
    if convrot_groupsize:
        xr, _, _ = int8_gemv.rotate(x.float(), convrot_groupsize)
    else:
        xr = x.float()
    bk = min(4 * group_size, 256)
    # The HIP kernels read the nibbles as nibbles instead of materialising one fp32
    # value per code: one entry per row for the single-row decode, a multi-row one
    # that shares each decoded dword across the batch. The loader declines anything
    # it cannot take (m > 8, fp16 activation for the multi-row entry) back to Triton.
    if hip_w4a16.enabled():
        if hip_w4a16.linear_dequant(xr, qdata, s_rel, s_channel, bias, out, group_size, lut):
            return out
    dequant_w4a16_kernel[(triton.cdiv(n, BN),)](
        xr, qdata.view(torch.int32), out, s_rel.view(torch.uint8), s_channel,
        bias if bias is not None else xr, lut if lut is not None else xr,
        n, k, m, qdata.view(torch.int32).stride(0), s_rel.stride(0),
        BN=BN, BK=bk, M=rows, GS=group_size,
        BSCALE_VEC=s_channel.numel() > 1, HAS_BIAS=bias is not None,
        HAS_LUT=codebook is not None, num_warps=WARPS, num_stages=STAGES)
    return out

def _linear_asm(x, qdata, s_rel, s_channel, bias, out_dtype, group_size, convrot_groupsize,
                levels=None, inv_level=1.0):
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


def _linear_dot(x, qdata, s_rel, s_channel, bias, out_dtype, group_size, convrot_groupsize,
                levels=None, inv_level=1.0):
    """Portable fallback: one tl.dot per group; needs group_size >= 16."""
    rows, xk = x.shape
    n, half = qdata.shape
    k = s_rel.shape[1] * group_size
    if rows < 1 or rows > MAX_M or xk != k or half * 2 != k:
        return None
    if group_size < DOT_M or group_size % DOT_M:
        return None
    if qdata.stride(0) != half or not qdata.is_contiguous() or s_rel.stride(0) != s_rel.shape[1]:
        return None
    if s_channel.numel() != n or s_channel.dtype != torch.float32:
        return None
    if convrot_groupsize and k % convrot_groupsize:
        return None
    if levels is not None and levels.numel() != 4:
        return None

    if convrot_groupsize:
        x, gmax, ngroups = int8_gemv.rotate(x, convrot_groupsize)
    else:
        gmax, ngroups = x.abs().amax(dim=1, keepdim=True), 1

    out = torch.empty((rows, n), device=qdata.device, dtype=out_dtype)
    dot_kernel[(triton.cdiv(n, DOT_BN),)](
        x, qdata, out, gmax, s_rel, s_channel,
        bias if bias is not None else x,
        levels if levels is not None else qdata.view(torch.int32).reshape(-1)[:4],
        rows, n, k, ngroups, half, s_rel.stride(0), inv_level,
        GS=group_size, BN=DOT_BN, MM=DOT_M, NG=triton.next_power_of_2(ngroups),
        HAS_BIAS=bias is not None, HAS_CB=levels is not None,
        num_warps=WARPS, num_stages=STAGES)
    return out


@triton.jit
def sum_kernel(a_ptr, b_ptr, c_ptr, gmax_ptr, srel_ptr, sch_ptr, bias_ptr, cb_ptr,
               n, k, ngroups, stride_bn, stride_bs, inv_level,
               BN: tl.constexpr, BK: tl.constexpr, NG: tl.constexpr, GS: tl.constexpr,
               HAS_BIAS: tl.constexpr, HAS_CB: tl.constexpr):
    """One (output block, row) per program, reduced with tl.sum.

    No tl.dot means no 16-wide K step, which is the only reason the portable
    dot kernel cannot take a group size below 16. The weight is decoded one
    nibble at a time and multiplied by its group scale, so the arithmetic is
    the same as the other two paths up to summation order.
    """
    pid_n = tl.program_id(0)
    row = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < n

    offs_g = tl.arange(0, NG)
    gm = tl.load(gmax_ptr + row * ngroups + offs_g, mask=offs_g < ngroups, other=0.0)
    a_scale = tl.maximum(tl.max(gm) / 127.0, 1e-30)
    s_ch = tl.load(sch_ptr + offs_n, mask=mask_n, other=0.0)

    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, k, BK):
        idx = k0 + tl.arange(0, BK)
        mask_k = idx < k
        byte = tl.load(b_ptr + offs_n[:, None] * stride_bn + (idx // 2)[None, :],
                       mask=mask_n[:, None] & mask_k[None, :], other=0)
        code = tl.where(((idx % 2) == 0)[None, :], byte & 0x0F, (byte >> 4) & 0x0F)
        if HAS_CB:
            level = tl.load(cb_ptr + code, mask=mask_k[None, :], other=0.0)
        else:
            level = code.to(tl.float32) - 8.0
        sg = tl.load(srel_ptr + offs_n[:, None] * stride_bs + (idx // GS)[None, :],
                     mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        a = tl.load(a_ptr + row * k + idx, mask=mask_k, other=0.0)
        q = tl.clamp(libdevice.rint(a / a_scale), -128.0, 127.0)
        acc += tl.sum(level * sg * q[None, :], axis=1)

    out = acc * (a_scale * s_ch * inv_level)
    if HAS_BIAS:
        out += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    tl.store(c_ptr + row * n + offs_n, out.to(c_ptr.dtype.element_ty), mask=mask_n)


def _linear_sum(x, qdata, s_rel, s_channel, bias, out_dtype, group_size, convrot_groupsize,
                levels=None, inv_level=1.0):
    """Floor for a 4-bit checkpoint the tl.dot kernel cannot take.

    group_size below 16 (or not a multiple of it) used to make the whole site
    decline the shape, so the caller fell all the way back to int8 even on a
    card where w4a8 would have won. This path only needs the packed nibbles, a
    per-group scale and eight rows -- no dot, no gfx9 instruction.
    """
    rows, xk = x.shape
    n, half = qdata.shape
    if group_size < 1:
        return None
    k = s_rel.shape[1] * group_size
    if rows < 1 or rows > MAX_M or xk != k or half * 2 != k:
        return None
    if qdata.stride(0) != half or not qdata.is_contiguous() or s_rel.stride(0) != s_rel.shape[1]:
        return None
    if s_channel.numel() != n or s_channel.dtype != torch.float32:
        return None
    if convrot_groupsize and k % convrot_groupsize:
        return None
    if levels is not None and levels.numel() != 4:
        return None

    if levels is not None:
        # the packed words are the codebook on this layer's int8 grid; the
        # sign lives in the byte, and inv_level brings it back to real units
        packed = levels.view(torch.uint8).to(torch.float32)
        table = (packed - 256.0 * (packed > 127.0)).contiguous()
    else:
        table = torch.zeros(1, device=qdata.device, dtype=torch.float32)

    if convrot_groupsize:
        x, gmax, ngroups = int8_gemv.rotate(x, convrot_groupsize)
    else:
        gmax, ngroups = x.abs().amax(dim=1, keepdim=True), 1

    out = torch.empty((rows, n), device=qdata.device, dtype=out_dtype)
    sum_kernel[(triton.cdiv(n, SUM_BN), rows)](
        x, qdata, out, gmax, s_rel, s_channel,
        bias if bias is not None else x,
        table, n, k, ngroups, half, s_rel.stride(0), inv_level,
        BN=SUM_BN, BK=SUM_BK, NG=triton.next_power_of_2(ngroups), GS=group_size,
        HAS_BIAS=bias is not None, HAS_CB=levels is not None,
        num_warps=WARPS, num_stages=STAGES)
    return out


_FLOOR_SEEN = False


def _note_floor(group_size):
    """Say once that this checkpoint is below what tl.dot can take."""
    global _FLOOR_SEEN
    if _FLOOR_SEEN:
        return
    _FLOOR_SEEN = True
    kreg.logger.info("gpu-tune: w4a8 group %d is below the tl.dot floor of 16, using the tl.sum kernel",
                     group_size)


def linear(x, qdata, s_rel, s_channel, bias, out_dtype, group_size, convrot_groupsize,
           levels=None, inv_level=1.0):
    """x @ W.T for one to eight rows, or None when the shape is not a decode of that size.

    qdata is the packed nibble plane [n, k/2], s_rel the fp8 per-group scale
    [n, k/group_size], s_channel the per-row scale [n]. convrot_groupsize 0
    skips the Hadamard rotation and uses the raw row max, matching int8_gemv.
    `levels`/`inv_level` are `codebook_words` output: they replace the uniform
    `code - 8` ramp with a table lookup and fold the level grid into the scale.

    Backend order: an optional w4a8_hip extension, the v_dot4 asm path when it
    compiles here, the portable per-group tl.dot kernel, then the tl.sum floor.
    Each step is the site's registry list; a shape an implementation declines
    falls through to the next one, and a measured entry for this (n, k, rows)
    wins over the order. The floor is what keeps a group size below 16 from
    sending the whole call back to int8.
    """
    try:
        backend = _REGISTRY.choose(n=qdata.shape[0], k=s_rel.shape[1] * group_size, m=x.shape[0])
    except (AttributeError, IndexError):          # not a plain [n, k] pair
        backend = backend_name()
    args = (x, qdata, s_rel, s_channel, bias, out_dtype, group_size, convrot_groupsize)
    while True:
        if backend == "hip":
            out = (_hip.linear(*args, levels=levels, inv_level=inv_level)
                   if _hip is not None else None)
        elif backend == "asm":
            out = _linear_asm(*args, levels=levels, inv_level=inv_level)
        elif backend == "sum":
            out = _linear_sum(*args, levels=levels, inv_level=inv_level)
        else:
            out = _linear_dot(*args, levels=levels, inv_level=inv_level)
        if out is not None:
            if backend == "sum":
                _note_floor(group_size)
            return out
        following = _REGISTRY.next_after(backend)
        if following == backend:
            return None
        backend = following


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
