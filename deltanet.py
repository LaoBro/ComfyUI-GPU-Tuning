"""GatedDeltaNet decode on gfx906: the comfy-kitchen fused op has no code object for
this arch (its HIP extension fails with "invalid device function"), so decode runs the
eager loop: per layer and per step that is a decay multiply, two einsums, a rank-1
baddbmm and a gated norm, all on a 2 MB fp32 state. Measured 125 us of GPU time per
layer at seq=1, 3 ms per decode step over 24 layers.

These three Triton kernels replace it:
  * _gate_kernel: beta/g for every (token, head) in one launch (2 GEMVs over hidden);
  * _delta_kernel: state decay, kv readout, rank-1 update, q readout, all in registers,
    with the q/k L2 norms done in-kernel (the eager path materializes repeat_interleave'd
    copies of q/k first);
  * _norm_kernel: the gated RMS norm (weight has +1 baked by the caller's RMSNorm add).

Measured 11.5 us per layer for the last two, ~2.9 ms per decode step saved at seq=1.

The prefill side is not a Triton kernel: chunk_rule is core's torch chunk rule
with its 63-step UT loop replaced by a doubling Neumann series (see its docstring).
"""

import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

BV = 64
GATE_BLOCK = 256
DELTA_MAX_SEQ = 6


@triton.jit
def _gate_kernel(x_ptr, wa_ptr, wb_ptr, dtb_ptr, gdec_ptr, beta_ptr, g_ptr,
                 HID: tl.constexpr, BH: tl.constexpr, BLOCKS: tl.constexpr, Hv: tl.constexpr):
    # grid = (batch * seq * Hv): one (token, head) gate pair per program
    i = tl.program_id(0)
    h = i % Hv
    row = i // Hv
    offs = tl.arange(0, BH)
    a = 0.0
    b = 0.0
    for j in range(BLOCKS):
        idx = j * BH + offs
        m = idx < HID
        xv = tl.load(x_ptr + row * HID + idx, mask=m, other=0.0)
        a += tl.sum(tl.load(wa_ptr + h * HID + idx, mask=m, other=0.0) * xv)
        b += tl.sum(tl.load(wb_ptr + h * HID + idx, mask=m, other=0.0) * xv)
    a = a + tl.load(dtb_ptr + h)
    # softplus with the same threshold torch uses
    sp = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
    tl.store(beta_ptr + i, tl.sigmoid(b))
    tl.store(g_ptr + i, tl.exp(tl.load(gdec_ptr + h) * sp))


@triton.jit
def _delta_kernel(q_ptr, k_ptr, v_ptr, beta_ptr, g_ptr, state_ptr, out_ptr, snap_ptr,
                  DK: tl.constexpr, DV: tl.constexpr, BVl: tl.constexpr,
                  S: tl.constexpr, Hv: tl.constexpr, REP: tl.constexpr, Bb: tl.constexpr,
                  qs_b, qs_c, qs_s, vs_b, vs_c, vs_s,
                  scale: tl.constexpr, EPS: tl.constexpr, WRITE_SNAP: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % Hv
    b = pid // Hv
    c = tl.program_id(1)
    kh = h // REP
    offs_k = tl.arange(0, DK)
    offs_v = c * BVl + tl.arange(0, BVl)
    st_base = (b * Hv + h) * DK * DV
    for s in range(S):
        q = tl.load(q_ptr + b * qs_b + (kh * DK + offs_k) * qs_c + s * qs_s)
        k = tl.load(k_ptr + b * qs_b + (kh * DK + offs_k) * qs_c + s * qs_s)
        q = q * (scale / tl.maximum(tl.sqrt(tl.sum(q * q)), EPS))
        k = k / tl.maximum(tl.sqrt(tl.sum(k * k)), EPS)
        beta = tl.load(beta_ptr + (b * S + s) * Hv + h)
        g = tl.load(g_ptr + (b * S + s) * Hv + h)
        st = tl.load(state_ptr + st_base + offs_k[:, None] * DV + offs_v[None, :])
        st = st * g
        kv = tl.sum(k[:, None] * st, axis=0)
        v = tl.load(v_ptr + b * vs_b + (h * DV + offs_v) * vs_c + s * vs_s)
        d = (v - kv) * beta
        st = st + k[:, None] * d[None, :]
        tl.store(state_ptr + st_base + offs_k[:, None] * DV + offs_v[None, :], st)
        if WRITE_SNAP:
            if s < S - 1:
                tl.store(snap_ptr + ((s * Bb + b) * Hv + h) * DK * DV
                         + offs_k[:, None] * DV + offs_v[None, :], st)
        o = tl.sum(q[:, None] * st, axis=0)
        tl.store(out_ptr + ((b * S + s) * Hv + h) * DV + offs_v, o)


@triton.jit
def _norm_kernel(out_ptr, z_ptr, w_ptr, DV: tl.constexpr, EPS: tl.constexpr):
    # the weight is applied as given: the caller's norm_weight is already the final scale
    i = tl.program_id(0)
    offs = tl.arange(0, DV)
    x = tl.load(out_ptr + i * DV + offs)
    r = 1.0 / tl.sqrt(tl.sum(x * x) / DV + EPS)
    w = tl.load(w_ptr + offs)
    z = tl.load(z_ptr + i * DV + offs)
    tl.store(out_ptr + i * DV + offs, x * r * w * (z * tl.sigmoid(z)))


def _pow2(n):
    return n in (32, 64, 128, 256)


def is_available(device=None, key_head_dim=128, value_head_dim=128):
    if device is not None and device.type != "cuda":
        return False
    return _pow2(key_head_dim) and _pow2(value_head_dim)


def conv_step(proj, conv_state, conv_w, conv_b, snapshots=None):
    """Depthwise causal conv + silu over proj [B, S, C]; conv_state is updated in place."""
    kernel = conv_w.shape[-1]
    weight = conv_w.reshape(1, proj.shape[2], 1, kernel)
    combined = torch.cat([conv_state, proj.transpose(1, 2)], dim=-1)
    if snapshots is not None:
        for s in range(proj.shape[1] - 1):
            snapshots[s].copy_(combined[:, :, 1 + s:1 + s + kernel - 1])
    conv_state.copy_(combined[:, :, proj.shape[1]:])
    out = combined.unfold(2, kernel, 1) * weight
    out = out.sum(-1)
    if conv_b is not None:
        out = out + conv_b.reshape(1, -1, 1)
    return F.silu(out)


def gated_delta_decode_fused(mixed_qkv, x, w_a, w_b, dt_bias, g_decay, state,
                             key_dim, num_key_heads, scale, z, norm_weight, eps,
                             snapshots=None):
    """One or more GatedDeltaNet decode steps; state is updated in place."""
    batch, _, seq = mixed_qkv.shape
    heads, dk, dv = state.shape[1], state.shape[2], state.shape[3]
    rep = heads // num_key_heads
    out_dtype = x.dtype
    if mixed_qkv.dtype != torch.float32:
        mixed_qkv = mixed_qkv.float()
    if x.dtype != torch.float32:
        x = x.float()
    x = x.contiguous()
    w_a = w_a.float()
    w_b = w_b.float()
    # scalar-gate inputs arrive in the execution dtype (fp16 when the TE runs it)
    dt_bias = dt_bias.float()
    g_decay = g_decay.float()
    z = z.float()

    beta = torch.empty((batch, seq, heads), dtype=torch.float32, device=state.device)
    g = torch.empty_like(beta)
    hidden = x.shape[-1]
    _gate_kernel[(batch * seq * heads,)](
        x, w_a, w_b, dt_bias, g_decay, beta, g,
        HID=hidden, BH=GATE_BLOCK, BLOCKS=triton.cdiv(hidden, GATE_BLOCK), Hv=heads,
        num_warps=4,
    )

    q = mixed_qkv[:, :key_dim]
    k = mixed_qkv[:, key_dim:2 * key_dim]
    v = mixed_qkv[:, 2 * key_dim:]
    out = torch.empty((batch, seq, heads, dv), dtype=torch.float32, device=state.device)
    snap = snapshots if snapshots is not None else state
    _delta_kernel[(batch * heads, triton.cdiv(dv, BV))](
        q, k, v, beta, g, state, out, snap,
        DK=dk, DV=dv, BVl=BV, S=seq, Hv=heads, REP=rep, Bb=batch,
        qs_b=q.stride(0), qs_c=q.stride(1), qs_s=q.stride(2),
        vs_b=v.stride(0), vs_c=v.stride(1), vs_s=v.stride(2),
        scale=scale, EPS=1e-12, WRITE_SNAP=snapshots is not None, num_warps=4,
    )
    _norm_kernel[(batch * seq * heads,)](
        out, z.reshape(-1, dv), norm_weight, DV=dv, EPS=eps, num_warps=1,
    )
    return out.to(out_dtype) if out.dtype != out_dtype else out


def chunk_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
               output_final_state=False):
    """comfy/text_encoders/qwen35.py:116 with the UT transform as a doubling series.

    Everything but the transform is core's code, kept line for line: a new core
    parameter has to be mirrored here. core builds the chunk's UT transform with a
    63-step Python loop of forward substitution (qwen35.py:143-146), which is
    ~315 small kernels per layer per prefill and 9.9 ms of the 32.8 ms this call
    costs on a 4k prefill. Because the starting matrix is strictly lower it is
    nilpotent, so the transform is the finite Neumann series I + A + ... + A^C-1,
    and that series doubles in log2(chunk_size) steps: 2.2 ms, 4.5x, in fp32
    (maxabs 2.2e-8 against core's own loop; bf16 is 1.8x slower than fp32 here
    because gfx906 has no bf16 matrix unit, and fp16 buys nothing at 64x64).
    """
    initial_dtype = query.dtype
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32)
                                  for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
                      diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

    eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    transform = eye + attn
    power = attn
    step = 2
    while step < chunk_size:
        # P covers I + A + ... + A^(step-1) and A^step is the next power to fold in
        power = power @ power
        transform = transform + power @ transform
        step *= 2
    attn = transform

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device,
                    dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
                      diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1,
                                          core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def install_prefill(gate=None):
    """Route Qwen3.5's prefill chunk rule through chunk_rule.

    ``gate`` is read per call like install()'s, so the textgen switch still
    reaches core's loop. The replacement is bound by name in core and in this
    pack's qwen35_model, so already-imported references are repointed too.
    """
    from comfy.text_encoders import qwen35

    if getattr(qwen35, "_gfx906_chunk_rule", False):
        return True
    original = qwen35.torch_chunk_gated_delta_rule

    def routed(*args, **kwargs):
        if gate is not None and not gate():
            return original(*args, **kwargs)
        return chunk_rule(*args, **kwargs)

    routed._gfx906_chunk_rule = True
    qwen35.torch_chunk_gated_delta_rule = routed
    for module in list(sys.modules.values()):
        if getattr(module, "torch_chunk_gated_delta_rule", None) is original:
            module.torch_chunk_gated_delta_rule = routed
    return routed


def install(gate=None):
    """Point comfy_kitchen's DeltaNet decode at these kernels (gfx906 has no HIP build).

    ``gate`` is read per call: when it returns False the shipped
    comfy_kitchen functions run instead, so the CLIP node's textgen switch can
    turn this off without a reload.
    """
    import comfy_kitchen

    if getattr(comfy_kitchen.gated_delta_decode_fused, "_gfx906_triton", False):
        return True

    original_fused = comfy_kitchen.gated_delta_decode_fused
    original_conv = comfy_kitchen.deltanet_conv_step
    original_avail = getattr(comfy_kitchen, "gated_delta_decode_is_available", None)

    def enabled():
        return gate is None or gate()

    def gated_delta_decode_fused_(*args, **kwargs):
        if not enabled():
            return original_fused(*args, **kwargs)
        return gated_delta_decode_fused(*args, **kwargs)

    def gated_conv_step(*args, **kwargs):
        if not enabled():
            return original_conv(*args, **kwargs)
        return conv_step(*args, **kwargs)

    def gated_availability(device=None, key_head_dim=128, value_head_dim=128):
        if not enabled():
            return bool(original_avail(device, key_head_dim, value_head_dim)) if original_avail is not None else False
        return is_available(device, key_head_dim, value_head_dim)

    gated_delta_decode_fused_._gfx906_triton = True
    comfy_kitchen.gated_delta_decode_fused = gated_delta_decode_fused_
    comfy_kitchen.deltanet_conv_step = gated_conv_step
    comfy_kitchen.gated_delta_decode_is_available = gated_availability
    return True
