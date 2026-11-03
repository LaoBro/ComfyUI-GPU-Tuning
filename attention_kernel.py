"""fp16 flash-attention forward kernel tuned for gfx906.

Upstream flash-attention's AMD Triton kernel leaves ~20% on the table here: it
picks its tile for a square sequence, while a prefix-cached model produces
SQ != SK, and gfx906 has 64-lane wavefronts and 256 VGPR/thread. A 64x16 tile
with 4 warps measured 1.19-1.21x faster on every shape tested and gives the
same error against an fp64 reference (relL2 ~2.9e-4 for both). D=64 heads
(SDXL-class UNets) want different tiles; see DEFAULT_CONFIG_D64 below.

Only plain multi-head attention without a mask is handled; everything else
falls back to the upstream kernel.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

LOG2E = tl.constexpr(1.4426950408889634)

BLOCK_M = 64
BLOCK_N = 16
NUM_WARPS = 4
NUM_STAGES = 2
SUPPORTED_HEAD_DIMS = (64, 128)
QUANT_BLOCK_M = 16
QUANT_V_BLOCK_T = 32

# Head-dim defaults. The D=64 sweep (gfx906fix/tune_tiles_d64.py, 2026-10-26)
# put 64x32 fp16 first on every SDXL shape by 1.2-1.6x over the D=128 winners;
# the D=128 tiles were only measured on Qwen-shaped attention. int8 tiles are
# picked per shape by int8_config below.
DEFAULT_CONFIG = (BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES)
DEFAULT_CONFIG_D64 = (64, 32, 4, 2)


def int8_config(head_dim, batch, heads, seq_q, seq_k, int8_pv):
    """(tile, block_d) for a shape: the int8 tiles measured fastest on gfx906.

    The tile follows the grid, not the head dim alone: 64x32 is 4.7% ahead at
    1280 programs (SDXL's 16384-token block), 2% behind at 320, and ahead again
    at 640, so the crossover sits between the two.

    D=128 is the one dim where the scale granularity moves too: 27.7 ms at 64x32
    with one scale per row against 30.0 ms at 64x16 with 32-wide blocks. Both
    knobs cost error - 64x32 normalizes P over twice as many keys (relL2 1.7e-2
    -> 2.1e-2) and the row scale adds 21% once P stays fp16 - which is why only
    the int8-PV path takes them. Cross-attention reads too few keys to pay back
    the K/V quantisation.
    """
    if seq_k <= 256:
        return (32, 16, 4, 2), 32
    if head_dim == 64:
        programs = batch * heads * triton.cdiv(seq_q, 64)
        return ((64, 32, 4, 2) if programs >= 600 else (32, 16, 4, 2)), 32
    return ((64, 32, 4, 2), head_dim) if int8_pv else ((64, 16, 4, 2), 32)


@triton.jit
def _attn_fwd(Q, K, V, Out, sq, sk, sm_scale,
              stride_qb, stride_qm, stride_qh,
              stride_kb, stride_km, stride_kh,
              stride_vb, stride_vm, stride_vh,
              stride_ob, stride_om, stride_oh,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mmask = offs_m < sq

    q = tl.load(Q + pid_b * stride_qb + pid_h * stride_qh
                + offs_m[:, None] * stride_qm + offs_d[None, :],
                mask=mmask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    for start_n in range(0, sk, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < sk
        k = tl.load(k_ptr + offs_n[:, None] * stride_km + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        v = tl.load(v_ptr + offs_n[:, None] * stride_vm + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(nmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2((s - m_new[:, None]) * LOG2E)
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(Out + pid_b * stride_ob + pid_h * stride_oh
             + offs_m[:, None] * stride_om + offs_d[None, :],
             acc.to(tl.float16), mask=mmask[:, None])


@triton.jit
def _attn_fwd_int8(Q, K8, KS, V, VS, CORR, Out, sq, sk, sm_scale,
                   stride_qb, stride_qm, stride_qh,
                   stride_kb, stride_km, stride_kh,
                   stride_sb, stride_sm, stride_sh,
                   stride_vb, stride_vm, stride_vh,
                   stride_cb, stride_cm, stride_ch,
                   stride_ob, stride_om, stride_oh,
                   H: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
                   INT8_PV: tl.constexpr, SMOOTH: tl.constexpr):
    """int8 QK, with PV in int8 or fp16. QK is per (row, head-dim block), P per tile.

    INT8_PV=False keeps P and V in fp16. That is ~10% slower but 2-30x more
    accurate on real q/k/v, because a flat softmax distribution is what the
    int8 P quantization loses the most on.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mmask = offs_m < sq
    offs_c = tl.arange(0, BLOCK_D)
    NB: tl.constexpr = HEAD_DIM // BLOCK_D

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    vs = tl.load(VS + (pid_b * H + pid_h) * HEAD_DIM + offs_d) / 127.0

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm
    k_base = K8 + pid_b * stride_kb + pid_h * stride_kh
    s_base = KS + pid_b * stride_sb + pid_h * stride_sh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh

    for start_n in range(0, sk, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < sk
        s = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for c in tl.static_range(NB):
            dc = c * BLOCK_D + offs_c
            qc = tl.load(q_base + dc[None, :], mask=mmask[:, None], other=0.0).to(tl.float32)
            kc = tl.load(k_base + offs_n[:, None] * stride_km + dc[None, :], mask=nmask[:, None], other=0)
            qs = tl.maximum(tl.max(tl.abs(qc), 1), 1e-8) / 127.0
            ksc = tl.load(s_base + offs_n * stride_sm + c, mask=nmask, other=1.0)
            q8 = (qc / qs[:, None]).to(tl.int8)
            s += tl.dot(q8, tl.trans(kc), out_dtype=tl.int32).to(tl.float32) * (qs[:, None] * ksc[None, :])

        if SMOOTH:
            # b_j = mu_q . kbar[j]; the two row-constant terms are dropped, softmax ignores them
            corr = tl.load(CORR + pid_b * stride_cb + offs_n * stride_cm + pid_h * stride_ch,
                           mask=nmask, other=0.0)
            s = (s + corr) * sm_scale
        else:
            s = s * sm_scale
        s = tl.where(nmask[None, :], s, float("-inf"))
        m_tile = tl.max(s, 1)
        m_new = tl.maximum(m_i, m_tile)
        p = tl.math.exp2((s - m_tile[:, None]) * LOG2E)      # tile max is 1
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)
        beta = tl.math.exp2((m_tile - m_new) * LOG2E)

        if INT8_PV:
            p8 = (p * 127.0).to(tl.int8)
            v8 = tl.load(v_base + offs_n[:, None] * stride_vm + offs_d[None, :], mask=nmask[:, None], other=0)
            pv = tl.dot(p8, v8, out_dtype=tl.int32).to(tl.float32) * vs[None, :]
            l_i = l_i * alpha + (tl.sum(p8.to(tl.float32), 1) / 127.0) * beta
        else:
            v = tl.load(V + pid_b * stride_vb + pid_h * stride_vh
                       + offs_n[:, None] * stride_vm + offs_d[None, :], mask=nmask[:, None], other=0.0)
            pv = tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, 1) * beta
        acc = acc * alpha[:, None] + pv * beta[:, None]
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(Out + pid_b * stride_ob + pid_h * stride_oh
             + offs_m[:, None] * stride_om + offs_d[None, :],
             acc.to(tl.float16), mask=mmask[:, None])


def supported(q, k, v):
    """Plain MHA in fp16 with matching head counts and a head dim we compiled for."""
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        return False
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return False
    b, sq, h, d = q.shape
    return (d in SUPPORTED_HEAD_DIMS and sq > 0 and k.shape[1] > 0
            and k.shape[0] == b and v.shape[0] == b
            and k.shape[2] == h and v.shape[2] == h
            and k.shape[3] == d and v.shape[3] == d)


def _launch(q, k, v, out, scale, config):
    b, sq, h, d = q.shape
    block_m, block_n, warps, stages = config
    grid = (triton.cdiv(sq, block_m), h, b)
    _attn_fwd[grid](
        q, k, v, out, sq, k.shape[1], scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=d,
        num_warps=warps, num_stages=stages,
    )


# Tiles worth trying on a card whose numbers we do not have. All of these were
# measured on gfx906 (none crashed); the list stays short because every entry
# costs a JIT compile the first time it runs.
CANDIDATES = ((64, 16, 4, 2), (64, 32, 4, 2), (32, 16, 4, 2), (64, 8, 4, 2))
DEFAULT_CONFIG = (BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES)


def config_text(config):
    """(block_m, block_n, warps, stages) -> "64x16w4s2", the manifest's key."""
    return f"{config[0]}x{config[1]}w{config[2]}s{config[3]}"


def config_from_text(text):
    """The inverse of config_text, or None when the text is not a usable tile.

    Tiles come from a hand-editable TOML file, so a block of 0 has to be
    rejected here rather than divided by in the grid.
    """
    try:
        tile, _, tail = str(text).partition("w")
        block_m, block_n = (int(value) for value in tile.split("x"))
        warps, _, stages = tail.partition("s")
        config = (block_m, block_n, int(warps), int(stages))
    except (TypeError, ValueError):
        return None
    return config if min(config) > 0 else None


_selected = {}
_selected_int8 = {}
# int8 keeps the same BLOCK_D grouping, so only the tile shape is free to move
INT8_CANDIDATES = ((64, 16, 4, 2), (64, 32, 4, 2), (32, 16, 4, 2))


def _time_call(fn, runs=2):
    fn()                                                  # compile + warmup
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs


def _time(config, q, k, v, out, scale, runs=2):
    return _time_call(lambda: _launch(q, k, v, out, scale, config), runs)


def benchmark_tiles_int8(q, k, v, softmax_scale=None, int8_pv=True, smooth=False):
    """(config, ms) for the best int8 tile on this shape, or (None, None)."""
    best, best_ms = None, None
    for config in INT8_CANDIDATES:
        try:
            ms = _time_call(lambda: attention_fwd_int8(q, k, v, softmax_scale, int8_pv=int8_pv,
                                                       smooth=smooth, config=config))
        except Exception as exc:
            logging.warning("gfx906: int8 attention tile %s failed to compile (%s), skipped", config, exc)
            continue
        if best_ms is None or ms < best_ms:
            best, best_ms = config, ms
    return best, best_ms


def use_config_int8(config, head_dim):
    _selected_int8[head_dim] = config


def benchmark_tiles(q, k, v, softmax_scale=None):
    """(config, ms) for the best candidate tile on this shape, or (None, None).

    Called once per session and head dim by the tuning patch when the device is
    not one the tiles were measured on; keying on the shape would re-sweep
    whenever the prompt changes the KV length, which costs more than the choice
    is worth.
    """
    out = torch.empty_like(q)
    scale = _scale(softmax_scale, q.shape[-1])
    best, best_ms = None, None
    for config in CANDIDATES:
        try:
            ms = _time(config, q, k, v, out, scale)
        except Exception as exc:
            logging.warning("gfx906: attention tile %s failed to compile (%s), skipped", config, exc)
            continue
        if best_ms is None or ms < best_ms:
            best, best_ms = config, ms
    return best, best_ms


def use_config(config, head_dim):
    _selected[head_dim] = config


def attention_fwd(q, k, v, softmax_scale=None):
    """q, k, v: [B, S, H, D] fp16, contiguous. Returns [B, S, H, D] fp16."""
    out = torch.empty_like(q)
    config = _selected.get(q.shape[-1]) or (DEFAULT_CONFIG_D64 if q.shape[-1] == 64 else DEFAULT_CONFIG)
    _launch(q, k, v, out, _scale(softmax_scale, q.shape[-1]), config)
    return out


@triton.jit
def _quant_k_kernel(X, Q8, KS, n_rows,
                    D: tl.constexpr, BLOCK_D: tl.constexpr, BM: tl.constexpr):
    # one row is one (token, head); D fits in registers, so the whole scale is
    # computed from a single read of the row. fp32 throughout: reshaping an fp16
    # tensor makes triton round the quotient to fp16, which costs an ulp.
    NB: tl.constexpr = D // BLOCK_D
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    mask = rows < n_rows
    offs_d = tl.arange(0, D)
    x = tl.load(X + rows[:, None] * D + offs_d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    xb = tl.reshape(x, (BM, NB, BLOCK_D))
    # quantize with the same fp16 scale the attention kernel will dequantize with:
    # an fp32 scale is exact here but costs 4% of the attention kernel (packed
    # fp16 math for the scale outer product) and is 2x the scale traffic
    scale = (tl.max(tl.abs(xb), 2) / 127.0).to(tl.float16)
    q = tl.reshape(xb / tl.reshape(scale.to(tl.float32), (BM, NB, 1)), (BM, D))
    q8 = tl.minimum(tl.maximum(libdevice.rint(q), -127.0), 127.0).to(tl.int8)
    tl.store(Q8 + rows[:, None] * D + offs_d[None, :], q8, mask=mask[:, None])
    offs_n = tl.arange(0, NB)
    tl.store(KS + rows[:, None] * NB + offs_n[None, :], scale, mask=mask[:, None])


def quantize_k(k, block_d=32):
    """k: [B, S, H, D] fp16 -> int8 with a per-(token, dim block) scale."""
    b, s, h, d = k.shape
    if not k.is_contiguous():
        k = k.contiguous()
    nb = d // block_d
    rows = b * s * h
    k8 = torch.empty((rows, d), device=k.device, dtype=torch.int8)
    k_scale = torch.empty((rows, nb), device=k.device, dtype=torch.float16)
    _quant_k_kernel[(triton.cdiv(rows, QUANT_BLOCK_M),)](
        k, k8, k_scale, rows, D=d, BLOCK_D=block_d, BM=QUANT_BLOCK_M,
        num_warps=4,
    )
    return k8.view(b, s, h, d), k_scale.view(b, s, h, nb)


@triton.jit
def _quant_v_kernel(V, V8, VS, sk,
                    stride_vb, stride_vt, stride_vh,
                    stride_sb, stride_sh,
                    D: tl.constexpr, BT: tl.constexpr):
    # V is quantized per channel, so the scale is shared by every token: one
    # program handles a [BT, D] tile and only reads it once.
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)
    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    mask = offs_t < sk
    ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_t[:, None] * stride_vt + offs_d[None, :]
    v = tl.load(ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
    scale = tl.load(VS + pid_b * stride_sb + pid_h * stride_sh + offs_d)
    q = v / scale[None, :]
    q8 = tl.minimum(tl.maximum(libdevice.rint(q), -127.0), 127.0).to(tl.int8)
    tl.store(V8 + pid_b * stride_vb + pid_h * stride_vh + offs_t[:, None] * stride_vt + offs_d[None, :],
             q8, mask=mask[:, None])


def quantize_v(v):
    """v: [B, S, H, D] fp16 -> int8 with one scale per (batch, head, channel)."""
    if not v.is_contiguous():
        v = v.contiguous()
    b, s, h, d = v.shape
    v_scale = (v.abs().amax(dim=1) / 127.0).clamp_min(1e-8)            # [B, H, D]
    v8 = torch.empty_like(v, dtype=torch.int8)
    _quant_v_kernel[(b, h, triton.cdiv(s, QUANT_V_BLOCK_T))](
        v, v8, v_scale, s,
        v.stride(0), v.stride(1), v.stride(2),
        v_scale.stride(0), v_scale.stride(1),
        D=d, BT=QUANT_V_BLOCK_T, num_warps=4,
    )
    return v8, v_scale


def quantize_kv(k, v, block_d=32):
    """k, v: [B, S, H, D] fp16 -> int8 with per-(token, dim block) and per-channel scales."""
    k8, k_scale = quantize_k(k, block_d)
    v8, v_scale = quantize_v(v)
    return k8, k_scale, v8, v_scale


def attention_fwd_int8(q, k, v, softmax_scale=None, block_d=None, int8_pv=True, smooth=False, config=None):
    """Same contract as attention_fwd, on the int8 path.

    int8_pv=True quantizes P as well (fastest, relL2 ~2e-2); False keeps P and V
    in fp16 (~10% slower, relL2 ~1e-2 to 7e-4 depending on how peaked the
    attention is).

    smooth subtracts the per-channel mean of q and k before quantizing, which
    only pays off when PV stays fp16 - with int8 PV the error is dominated by
    the P quantization and smoothing buys nothing.

    block_d and config left as None are picked from the shape by int8_config.
    """
    b, sq, h, d = q.shape
    sk = k.shape[1]
    out = torch.empty_like(q)
    scale = _scale(softmax_scale, d)
    if config is None:
        config = _selected_int8.get(d)
    if config is None:
        config, picked_d = int8_config(d, b, h, sq, sk, int8_pv)
        if block_d is None:
            block_d = picked_d
    if block_d is None:
        block_d = 32
    block_m, block_n, warps, stages = config
    grid = (triton.cdiv(sq, block_m), h, b)

    q_arg, k_arg = q, k
    if smooth:
        mu_q = q.mean(dim=1, keepdim=True)
        mu_k = k.mean(dim=1, keepdim=True)
        q_arg = q - mu_q
        k_arg = k - mu_k
        corr = (mu_q * k_arg).sum(dim=-1)                   # [B, SK, H]
    else:
        corr = torch.empty((1, 1, 1), device=q.device, dtype=torch.float32)

    if int8_pv:
        k8, k_scale, v_arg, v_scale = quantize_kv(k_arg, v, block_d)
    else:
        k8, k_scale = quantize_k(k_arg, block_d)
        v_arg = v
        v_scale = torch.empty(1, device=v.device, dtype=torch.float32)

    _attn_fwd_int8[grid](
        q_arg, k8, k_scale, v_arg, v_scale, corr, out, sq, sk, scale,
        q_arg.stride(0), q_arg.stride(1), q_arg.stride(2),
        k8.stride(0), k8.stride(1), k8.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        v_arg.stride(0), v_arg.stride(1), v_arg.stride(2),
        corr.stride(0), corr.stride(1), corr.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H=h, BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=d, BLOCK_D=block_d,
        INT8_PV=int8_pv, SMOOTH=smooth,
        num_warps=warps, num_stages=stages,
    )
    return out


def _scale(softmax_scale, d):
    return d ** -0.5 if softmax_scale is None or softmax_scale <= 0 else softmax_scale
