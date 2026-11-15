"""Qwen3.5 text-encoder compute classes, owned by this node.

comfy/text_encoders/qwen35.py stays as shipped. The wrapper there builds its
backbone by module-global name (`self.model = Qwen35Transformer(...)`), so
replacing that one class swaps in this copy for every nested layer, and the
node's own classes resolve their children inside this module.

The llama.py base classes (FixedKV/FixedKVBias/RMSNorm/BaseLlama) stay shared:
the cache objects this module creates must be the same class objects core
checks with isinstance.

0.44+ core already carries the fused attention and KV-cache contract, so
gpu_tuning.apply skips this class swap there and keeps the core implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy_kitchen
import comfy.model_management
import comfy.ops
import comfy.text_encoders.qwen35 as _core
from comfy.quant_ops import QuantizedTensor
from comfy.text_encoders.llama import Llama2_, MLP, RMSNorm, precompute_freqs_cis, rope_matrix
from comfy.text_encoders.qwen35 import RMSNormGated, torch_chunk_gated_delta_rule

# Set by install(); read per call. When the gate is off these classes delegate to
# the shipped methods, so the CLIP node can switch the qwen35 patch off without
# rebuilding the text encoder.
_gate = None


def _enabled():
    return _gate is None or _gate()


def apply_partial_rope(xq, xk, freqs_cis, rotary_dim):
    """Apply RoPE to only the first rotary_dim dimensions."""
    if not _enabled():
        return _core.apply_partial_rope(xq, xk, freqs_cis, rotary_dim)
    # rotate the rotary slice where it lies: the pass-through dims need no reassembly,
    # which drops two copies and two temporaries per layer
    comfy_kitchen.apply_rope_split_half_(xq[..., :rotary_dim], xk[..., :rotary_dim], rope_matrix(freqs_cis))
    return xq, xk


class GatedAttention(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.inner_size = self.num_heads * self.head_dim
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)

        # q_proj outputs 2x: query + gate
        self.q_proj = ops.Linear(config.hidden_size, self.inner_size * 2, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        # QK norms with (1+weight) scaling
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None):
        if not _enabled():
            return _core.GatedAttention.forward(self, x, attention_mask=attention_mask, freqs_cis=freqs_cis,
                                               optimized_attention=optimized_attention, past_key_value=past_key_value)
        batch_size, seq_length, _ = x.shape

        # Project Q (with gate), K, V
        qg = self.q_proj(x)
        # Split into query and gate: each is [B, seq, inner_size]
        qg = qg.view(batch_size, seq_length, self.num_heads, self.head_dim * 2)
        xq, gate = qg[..., :self.head_dim], qg[..., self.head_dim:]
        gate = gate.reshape(batch_size, seq_length, -1)  # [B, seq, inner_size]

        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = self.q_norm(xq).transpose(1, 2)  # [B, heads, seq, head_dim]
        xk = self.k_norm(xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply partial RoPE
        xq, xk = apply_partial_rope(xq, xk, freqs_cis, self.rotary_dim)

        # KV cache
        present_key_value = past_key_value
        if past_key_value is not None and seq_length <= 6 and attention_mask is None:
            output = past_key_value.decode(xq, xk, xv, self.num_kv_heads)
        else:
            if past_key_value is not None:
                xk, xv = past_key_value.append(xk, xv)
            gqa_kwargs = {"enable_gqa": True} if self.num_heads != self.num_kv_heads else {}
            output = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True, **gqa_kwargs)

        output = output * gate.sigmoid()
        return self.o_proj(output), present_key_value


class GatedDeltaNet(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()

        hidden = config.hidden_size
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.conv_kernel_size

        key_dim = self.num_key_heads * self.key_head_dim
        value_dim = self.num_value_heads * self.value_head_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        conv_dim = key_dim * 2 + value_dim

        self.in_proj_qkv = ops.Linear(hidden, conv_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_z = ops.Linear(hidden, value_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_b = ops.Linear(hidden, self.num_value_heads, bias=False, device=device, dtype=dtype)
        self.in_proj_a = ops.Linear(hidden, self.num_value_heads, bias=False, device=device, dtype=dtype)
        self.out_proj = ops.Linear(value_dim, hidden, bias=False, device=device, dtype=dtype)

        self.dt_bias = nn.Parameter(torch.empty(self.num_value_heads, device=device, dtype=dtype))
        self.A_log = nn.Parameter(torch.empty(self.num_value_heads, device=device, dtype=dtype))

        self.conv1d = ops.Conv1d(in_channels=conv_dim, out_channels=conv_dim, bias=False, kernel_size=self.conv_kernel_size,
            groups=conv_dim, padding=self.conv_kernel_size - 1, device=device, dtype=dtype)

        self.norm = RMSNormGated(self.value_head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self._ab_weight = None

    def _merged_ab(self, rows):
        # in_proj_a/b read the same input rows; one merged weight halves the decode
        # launches and builds the fp32 copy once instead of casting two weights per call
        cached = self._ab_weight
        w_a, w_b = self.in_proj_a.weight, self.in_proj_b.weight
        if cached is None or cached[1] is not w_a or cached[2] is not w_b or cached[0].dtype != rows.dtype or cached[0].device != rows.device:
            merged = torch.cat([w_b, w_a])
            if merged.dtype != rows.dtype or merged.device != rows.device:
                merged = comfy.model_management.cast_to(merged, dtype=rows.dtype, device=rows.device)
            cached = self._ab_weight = (merged, w_a, w_b)
        return cached[0]

    def forward(self, x, past_key_value=None, **kwargs):
        if not _enabled():
            return _core.GatedDeltaNet.forward(self, x, past_key_value=past_key_value, **kwargs)
        batch_size, seq_len, _ = x.shape

        use_recurrent = (
            past_key_value is not None
            and past_key_value.index > 0
            and seq_len <= 6
        )

        fused_available = getattr(comfy_kitchen, "gated_delta_decode_is_available", None)
        use_fused = (use_recurrent and fused_available is not None and fused_available(x.device, self.key_head_dim, self.value_head_dim)
                     and (seq_len == 1 or past_key_value.snap_backing is not None))

        # Projections (shared)
        proj = self.in_proj_qkv(x)  # [B, seq_len, conv_dim]
        z = self.in_proj_z(x)

        if use_fused:
            # decode: kitchen conv step, then gates + delta rule + gated norm in one kernel
            if seq_len > 1:
                past_key_value.last_seq = seq_len
            with comfy.ops.CastBiasWeightContext(self.conv1d, proj, offloadable=True) as (conv_weight, conv_bias):
                conv_out = comfy_kitchen.deltanet_conv_step(proj, past_key_value.conv_state, conv_weight, conv_bias,
                                                            past_key_value.conv_snap_backing[:seq_len - 1] if seq_len > 1 else None)
            with comfy.ops.CastBiasWeightContext(self.in_proj_a, x, offloadable=True) as (w_a, _), \
                 comfy.ops.CastBiasWeightContext(self.in_proj_b, x, offloadable=True) as (w_b, _):
                if isinstance(w_a, QuantizedTensor):
                    w_a = w_a.dequantize()
                if isinstance(w_b, QuantizedTensor):
                    w_b = w_b.dequantize()
                core_attn_out = comfy_kitchen.gated_delta_decode_fused(
                    conv_out, x, w_a, w_b, past_key_value.dt_bias, past_key_value.g_decay, past_key_value.recurrent_state,
                    self.key_dim, self.num_key_heads, self.key_head_dim ** -0.5, z, past_key_value.norm_weight, self.norm.eps,
                    past_key_value.snap_backing[:seq_len - 1] if seq_len > 1 else None)
            return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1)), past_key_value

        mixed_qkv = proj.transpose(1, 2)  # [B, conv_dim, seq_len]
        if use_recurrent and not isinstance(self.in_proj_a.weight, QuantizedTensor):
            # one merged GEMV for the two 32-row gates instead of two launches
            rows = x.reshape(-1, x.shape[-1])
            w_ab = self._merged_ab(rows)
            ab = torch.mv(w_ab, rows[0]).unsqueeze(0) if rows.shape[0] == 1 else F.linear(rows, w_ab)
            b, a = ab[:, :self.num_value_heads], ab[:, self.num_value_heads:]
        else:
            b = self.in_proj_b(x)
            a = self.in_proj_a(x)

        # Conv1d
        if use_recurrent:
            # decode: exact-width causal window, weight resolved via the vbar-aware context
            combined = torch.cat([past_key_value.conv_state, mixed_qkv], dim=-1)
            if seq_len > 1:
                past_key_value.last_seq = seq_len
                for s in range(seq_len - 1):
                    past_key_value.snapshots[s][1].copy_(combined[:, :, 1 + s:1 + s + self.conv_kernel_size - 1])
            past_key_value.conv_state.copy_(combined[:, :, seq_len:])
            with comfy.ops.CastBiasWeightContext(self.conv1d, combined, offloadable=True) as (conv_weight, _):
                # depthwise with K taps: MIOpen ships no CK grouped-conv library for gfx906 and
                # falls back to its direct solver, so sum the taps here instead
                taps = combined.unfold(2, self.conv_kernel_size, 1)
                mixed_qkv = F.silu((taps * conv_weight.reshape(1, combined.shape[1], 1, self.conv_kernel_size)).sum(-1))
        else:
            if past_key_value is not None:
                conv_state = past_key_value.conv_state
                conv_state_init = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                conv_state.copy_(conv_state_init[:, :, -conv_state.shape[-1]:])
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        # Split QKV and compute beta/g
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, seq_len, conv_dim]
        query, key, value = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        if use_recurrent:
            g_decay, dt_bias = past_key_value.g_decay, past_key_value.dt_bias
        else:
            g_decay = -comfy.model_management.cast_to_device(self.A_log, x.device, torch.float32).exp()
            dt_bias = comfy.model_management.cast_to_device(self.dt_bias, x.device, torch.float32)
            if past_key_value is not None:
                past_key_value.g_decay = g_decay
                past_key_value.dt_bias = dt_bias
                past_key_value.norm_weight = comfy.model_management.cast_to(self.norm.weight, dtype=x.dtype, device=x.device)

        beta = b.sigmoid()
        g = g_decay * F.softplus(a.float() + dt_bias)
        query = query.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        if self.num_value_heads != self.num_key_heads:
            rep = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        # Delta rule
        if use_recurrent:
            scale = self.key_head_dim ** -0.5
            # one normalize over the stacked q/k rows instead of two
            qk = F.normalize(torch.stack((query, key)).float(), dim=-1)
            q, k = qk[0] * scale, qk[1]
            v = value.float()
            beta_t = beta.reshape(batch_size, seq_len, -1)
            g_t = g.reshape(batch_size, seq_len, -1).exp()

            # In-place state update: [B, heads, k_dim, v_dim]
            recurrent_state = past_key_value.recurrent_state
            outs = []
            for s in range(seq_len):
                recurrent_state.mul_(g_t[:, s, :, None, None])
                kv_mem = torch.einsum('bhk,bhkv->bhv', k[:, s], recurrent_state)
                delta = (v[:, s] - kv_mem) * beta_t[:, s, :, None]
                # rank-1 update via baddbmm_: no materialized [B, H, D, D] outer-product temp
                recurrent_state.view(-1, self.key_head_dim, self.value_head_dim).baddbmm_(
                    k[:, s].reshape(-1, self.key_head_dim, 1), delta.reshape(-1, 1, self.value_head_dim))
                outs.append(torch.einsum('bhk,bhkv->bhv', q[:, s], recurrent_state))
                if seq_len > 1 and s < seq_len - 1:
                    past_key_value.snapshots[s][0].copy_(recurrent_state)
            core_attn_out = torch.stack(outs, dim=1).to(x.dtype)
            present_key_value = past_key_value
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=None,
                output_final_state=past_key_value is not None,
            )

            present_key_value = None
            if past_key_value is not None:
                if last_recurrent_state is not None:
                    past_key_value.recurrent_state.copy_(last_recurrent_state.to(past_key_value.recurrent_state.dtype))
                present_key_value = past_key_value

        # Gated norm + output projection (shared)
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.value_head_dim), z.reshape(-1, self.value_head_dim))
        output = self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))
        return output, present_key_value


class Qwen35TransformerBlock(nn.Module):
    def __init__(self, config, index, device=None, dtype=None, ops=None):
        super().__init__()
        self.layer_type = config.layer_types[index]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, device=device, dtype=dtype, ops=ops)
        else:
            self.self_attn = GatedAttention(config, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None):
        output = x
        if self.layer_type == "linear_attention":
            h, present_key_value = self.linear_attn(self.input_layernorm(x), attention_mask=attention_mask, past_key_value=past_key_value)
        else:
            h, present_key_value = self.self_attn(self.input_layernorm(x), attention_mask=attention_mask, freqs_cis=freqs_cis, optimized_attention=optimized_attention, past_key_value=past_key_value)

        # in-place into the input buffer so CUDA-graph replays land in the static x
        x = torch.add(x, h, out=output)
        x = torch.add(x, self.mlp(self.post_attention_layernorm(x)), out=output)
        return x, present_key_value


class Qwen35Transformer(Llama2_):
    def __init__(self, config, device=None, dtype=None, ops=None):
        nn.Module.__init__(self)
        self.config = config
        self.prefetch_dynamic_vbars = True
        self.graph_dynamic_vbar_blocks = True
        self.vocab_size = config.vocab_size
        self.embed_tokens = ops.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            Qwen35TransformerBlock(config, index=i, device=device, dtype=dtype, ops=ops)
            for i in range(config.num_hidden_layers)
        ])

        if config.final_norm:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        else:
            self.norm = None

        if config.lm_head:
            self.lm_head = ops.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)

    def compute_freqs_cis(self, position_ids, device):
        rotary_dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        return precompute_freqs_cis(rotary_dim, position_ids, self.config.rope_theta,
                                    rope_dims=self.config.mrope_section, interleaved_mrope=True, device=device)


def install(gate=None, freqs_gate=None):
    """Swap the backbone the core wrapper builds (it looks the name up per load).

    ``gate`` is the qwen35 switch, ``freqs_gate`` the llama one (the position
    table belongs to the llama patch even when this class is in use).
    """
    global _gate
    _gate = gate
    import comfy.text_encoders.qwen35 as core
    from comfy.text_encoders.llama import Llama2_

    if getattr(core.Qwen35Transformer, "_gfx906_node", False):
        return True
    # the shared base class carries the position-table patch; this class overrides
    # compute_freqs_cis, so wrap it too when the base was wrapped
    if getattr(Llama2_.compute_freqs_cis, "_gfx906_table", False) and not getattr(Qwen35Transformer.compute_freqs_cis, "_gfx906_table", False):
        from . import llama as patches
        wrapped = patches.table_compute_freqs_cis(Qwen35Transformer.__dict__["compute_freqs_cis"], freqs_gate)
        wrapped._gfx906_table = True
        Qwen35Transformer.compute_freqs_cis = wrapped
    Qwen35Transformer._gfx906_node = True
    core.Qwen35Transformer = Qwen35Transformer
    return True
