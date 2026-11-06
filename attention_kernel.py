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

LOG2E = tl.constexpr(1.4426950408889634)

BLOCK_M = 64
BLOCK_N = 16
NUM_WARPS = 4
NUM_STAGES = 2
SUPPORTED_HEAD_DIMS = (64, 128)
# Head-dim defaults. The D=64 sweep (gfx906fix/tune_tiles_d64.py, 2026-10-26)
# put 64x32 fp16 first on every SDXL shape by 1.2-1.6x over the D=128 winners;
# the D=128 tiles were only measured on Qwen-shaped attention.
DEFAULT_CONFIG = (BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES)
DEFAULT_CONFIG_D64 = (64, 32, 4, 2)


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


def _scale(softmax_scale, d):
    return d ** -0.5 if softmax_scale is None or softmax_scale <= 0 else softmax_scale
