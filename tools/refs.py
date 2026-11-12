"""Reference implementations shared by the benchmark and verification tools.

Whatever a kernel computes, the tool needs an independent, obviously-correct
version of it to compare against: a 20-line torch expression that a reviewer can
check by eye. These are deliberately slow and written straight from the format
description, not from the kernel source -- if they were derived from the kernel
they would agree with its bugs.

The quantisation error of the *activation* is part of the contract (both GEMV
paths quantise x to int8 before the dot), so a correct kernel still lands about
1% away from these references. The gate in bench.py is 5%: wide enough for
quantisation, far too tight for a wrong nibble order or a missing group scale.
"""

import torch


def rel(a, b):
    """Max relative error of `a` against `b`, as one number."""
    den = b.abs().max().clamp(min=1e-30)
    return float((a - b).abs().max() / den)


def delta_decode_reference(mixed_qkv, x, w_a, w_b, dt_bias, g_decay, state, key_dim,
                           num_key_heads, scale, z, norm_weight, eps):
    """The gated delta rule of kernels/triton/deltanet.py, one (batch, head, step) at a time.

    `state` is updated in place, like the kernels, so every implementation gets
    its own copy. Everything is fp32: the kernels run fp32 too, which is why a
    real mismatch lands far above the 5% gate instead of hiding in rounding.
    """
    batch, _, seq = mixed_qkv.shape
    heads, dk, dv = state.shape[1], state.shape[2], state.shape[3]
    rep = heads // num_key_heads
    xf = x.float()
    beta = torch.sigmoid(xf @ w_b.float().T)                    # [B, S, H]
    a = xf @ w_a.float().T + dt_bias.float()
    g = torch.exp(g_decay.float() * torch.where(a > 20, a, torch.log1p(torch.exp(a))))
    q_all = mixed_qkv.float()[:, :key_dim]
    k_all = mixed_qkv.float()[:, key_dim:2 * key_dim]
    v_all = mixed_qkv.float()[:, 2 * key_dim:]
    out = torch.empty((batch, seq, heads, dv), dtype=torch.float32, device=state.device)
    for b in range(batch):
        for h in range(heads):
            lo = (h // rep) * dk
            for s in range(seq):
                q = q_all[b, lo:lo + dk, s]
                q = q * (scale / max(float(q.norm()), 1e-12))
                k = k_all[b, lo:lo + dk, s]
                k = k / max(float(k.norm()), 1e-12)
                st = state[b, h] * g[b, s, h]
                d = (v_all[b, h * dv:(h + 1) * dv, s] - k @ st) * beta[b, s, h]
                st = st + k[:, None] * d[None, :]
                state[b, h] = st
                out[b, s, h] = q @ st
    norm = torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    gate = torch.nn.functional.silu(z.float().reshape(batch, seq, heads, dv))
    return out * norm * norm_weight.float() * gate


def attention_reference(q, k, v, scale=None):
    """Exact fp32 softmax attention on [B, S, H, D], one head at a time.

    Per head because the score matrix at 4608 keys is 85 MB in fp32: doing all
    24 heads at once allocates 2 GB on a card that may be showing a model.
    """
    b, sq, h, d = q.shape
    scale = d ** -0.5 if scale is None else scale
    out = torch.empty((b, sq, h, d), dtype=torch.float32, device=q.device)
    for bi in range(b):
        for hi in range(h):
            scores = torch.matmul(q[bi, :, hi].float(), k[bi, :, hi].float().T) * scale
            out[bi, :, hi] = torch.matmul(torch.softmax(scores, dim=-1), v[bi, :, hi].float())
    return out


def int8_reference(x, weight, scale):
    """x @ (weight * scale), the fp32 answer the int8 kernels approximate."""
    return (x @ weight.to(torch.float32).T) * scale.reshape(1, -1)


def w4a8_weight(qdata, s_rel, s_channel, group_size, codebook=None):
    """Decode packed nibbles the way the kernel does.

    The byte at (row, i) holds k=2i in its low nibble and k=2i+1 in its high
    nibble; `s_rel` is per (row, group) and `s_channel` per row. Without a
    codebook the four-bit code is a ramp around 8 (`code - 8`); with one it is
    an index into 16 levels, which is what the shipped 4-bit checkpoints use.
    """
    n, half = qdata.shape
    k = half * 2
    b = qdata.to(torch.int32)
    code = torch.empty((n, k), dtype=torch.int32, device=qdata.device)
    code[:, 0::2] = b & 0x0F
    code[:, 1::2] = (b >> 4) & 0x0F
    level = codebook.to(torch.float32)[code] if codebook is not None else code.to(torch.float32) - 8.0
    group = torch.arange(k, device=qdata.device) // group_size
    s = s_rel.to(torch.float32)[:, group]
    return (level * s) * s_channel.reshape(-1, 1)
