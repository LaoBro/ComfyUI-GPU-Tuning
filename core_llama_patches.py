"""Two hot-path fixes for comfy/text_encoders/llama.py, kept out of core.

Both are value-identical replacements of one function each:

  * RMSNorm.forward: the weight is fixed after load, so the (+1, cast) result is
    built once instead of two kernels per norm per decode step.
  * compute_freqs_cis: positions are the only input, so one table per
    (config, device) is built and sliced per step instead of rebuilding the
    arange/pow/outer/cos/sin chain (~15 kernels per step).

Anything outside the fast path (multi-row MRoPE position ids, non-consecutive
positions, negative starts) falls through to the shipped implementation.
"""

import torch

import comfy.ldm.common_dit
import comfy.model_management


def patch_rmsnorm():
    import comfy.text_encoders.llama as llama

    if getattr(llama.RMSNorm.forward, "_gfx906_prepared", False):
        return True

    def forward(self, x):
        prepared = getattr(self, "_gfx906_prepared", None)
        w = self.weight
        if prepared is None or prepared[1] is not w or prepared[0].dtype != x.dtype or prepared[0].device != x.device:
            p = (w + 1.0) if self.add else w
            prepared = (comfy.model_management.cast_to(p, dtype=x.dtype, device=x.device), w)
            self._gfx906_prepared = prepared
        return comfy.ldm.common_dit.rms_norm(x, prepared[0], self.eps)

    forward._gfx906_prepared = True
    llama.RMSNorm.forward = forward
    return True


def _stable(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _freqs_key(cfg, device):
    return (device, cfg.head_dim, getattr(cfg, "partial_rotary_factor", 1.0),
            _stable(cfg.rope_theta), _stable(getattr(cfg, "rope_scale", None)),
            _stable(getattr(cfg, "rope_dims", None)),
            bool(getattr(cfg, "interleaved_mrope", False)))


def _slice(tensors, start, n):
    if isinstance(tensors, tuple):
        return tuple(t[:, :, start:start + n] for t in tensors)
    return [tuple(t[:, :, start:start + n] for t in group) for group in tensors]


def table_compute_freqs_cis(original):
    """Wrap compute_freqs_cis(self, position_ids, device) with a position table."""
    cache = {}

    def compute_freqs_cis(self, position_ids, device):
        if position_ids.dim() == 2 and position_ids.shape[0] == 1 and position_ids.numel() > 0:
            positions = position_ids.reshape(-1)
            start = int(positions[0])
            n = positions.numel()
            if start >= 0 and torch.equal(positions - positions[0], torch.arange(n, device=positions.device, dtype=positions.dtype)):
                key = _freqs_key(self.config, device)
                entry = cache.get(key)
                if entry is None or entry[0] < start + n:
                    if len(cache) >= 8:
                        cache.clear()
                    cap = max(start + n, 128) * 2
                    entry = (cap, original(self, torch.arange(cap, device=device, dtype=torch.float).unsqueeze(0), device))
                    cache[key] = entry
                return _slice(entry[1], start, n)
        return original(self, position_ids, device)

    return compute_freqs_cis


def patch_freqs_table():
    import comfy.text_encoders.llama as llama

    patched = False
    for cls in (llama.Llama2_, getattr(llama, "BaseLlama", None)):
        if cls is None or "compute_freqs_cis" not in vars(cls):
            continue
        current = vars(cls)["compute_freqs_cis"]
        if getattr(current, "_gfx906_table", False):
            patched = True
            continue
        wrapped = table_compute_freqs_cis(current)
        wrapped._gfx906_table = True
        cls.compute_freqs_cis = wrapped
        patched = True
    return patched


def install():
    return patch_rmsnorm() and patch_freqs_table()
