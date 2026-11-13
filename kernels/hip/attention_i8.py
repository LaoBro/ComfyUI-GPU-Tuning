"""gfx906 INT8-input fused attention.

The kernel takes quantized tensors, not floats: Q and K are int8 [B, S, H, D]
with one fp32 scale per (token, head), V is int8 packed four keys per int32
word, [B, HKV, SKP/4, D], with one fp32 scale per (head, dim). The output is
fp16 [B, SQ, H, D]. See arch/gfx906/kernels/hip_attn_i8.cu for the tile and the
reason 8-bit P is exact against the packed V.

Quantizing here, rather than in the kernel, is the point: the accuracy knobs
are the quantization (rotation group, per-row vs per-channel) and they stay on
this side of the ABI.
"""

import ctypes

import torch

from .. import runtime

_LIB = None
_LIBQ = None
_PROBED = False

# group sizes the quantization helper can rotate by (Hadamard, power of two)
ROTATE_MODES = (0, 32, 64, 128, 256)


def _load():
    global _LIB, _LIBQ, _PROBED
    if _PROBED:
        return _LIB
    _PROBED = True
    lib = runtime.load("attn_i8", "INT8 attention")
    if lib is None:
        return None
    lib.gfx906_attn_i8_fwd.restype = ctypes.c_int
    lib.gfx906_attn_i8_fwd.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_longlong] + [
        ctypes.c_void_p] * 6 + [ctypes.c_int] * 10 + [ctypes.c_float, ctypes.c_void_p]
    _LIB = lib
    q = runtime.load("attn_quant", "the int8 prefix cache path")
    if q is not None:
        q.gfx906_rot_quant.restype = ctypes.c_int
        q.gfx906_rot_quant.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        q.gfx906_rot_pack_v.restype = ctypes.c_int
        q.gfx906_rot_pack_v.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong,
                                        ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong,
                                        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _LIBQ = q
    return lib


def rotation_ready():
    """Whether the fused rotate/quantize helpers are built (prefix cache path)."""
    _load()
    return _LIBQ is not None


def available():
    return _load() is not None


def _hadamard(group, device, dtype):
    """Normalized Sylvester Hadamard: H @ H.T = I, so dot products survive."""
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < group:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) * (2.0 ** -0.5)
    return h


def _rotate(x, group, h):
    d = x.shape[-1]
    ht = h.transpose(-1, -2)
    if group <= 0:
        return x
    if group >= d:
        return torch.matmul(x, ht)
    return torch.matmul(x.reshape(*x.shape[:-1], d // group, group), ht).reshape(*x.shape)


def quantize_qk(x, group=0):
    """[B, S, H, D] float -> int8 rows + fp32 per-row scale.

    group > 0 rotates each `group` slice of the head dim before quantizing;
    Q and K must use the same group for the scores to be preserved.
    """
    x = x.float()
    if group:
        h = _hadamard(min(group, x.shape[-1]), x.device, x.dtype)
        x = _rotate(x, group, h)
    amax = x.abs().amax(-1, keepdim=True)
    scale = (amax / 127.0).clamp_min(1e-8)
    q = torch.round(x / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale.squeeze(-1).contiguous()


def quantize_v(v):
    """[B, SK, HKV, D] float -> packed int8 quads [B, HKV, SKP/4, D] + per-dim scale.

    Padding keys are zeroed; they are masked out of the scores anyway. The
    scale is per (head, dim), which is the column the PV dot accumulates over.
    """
    b, sk, h, d = v.shape
    vf = v.float()
    scale = (vf.abs().amax(1) / 127.0).clamp_min(1e-8)          # [B, H, D]
    q8 = torch.round(vf / scale.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
    skp = ((sk + 63) // 64) * 64
    if skp != sk:
        padded = torch.zeros((b, skp, h, d), dtype=torch.int8, device=v.device)
        padded[:, :sk] = q8
        q8 = padded
    packed = q8.permute(0, 2, 3, 1).contiguous()                # [B, H, D, SKP]
    packed = packed.view(torch.int32).permute(0, 1, 3, 2).contiguous()  # [B, H, SKP/4, D]
    return packed, scale.contiguous()


def _fwd_split(q8, q_scale, k1, ks1, vk1, pitch1, k2, ks2, vk2, vq, v_scale, sq, sp, st,
               skp, h, hkv, d, scale, tq=2, unrotate=0):
    """One launch over up to two K sources; see the kernel header."""
    out = torch.empty((q8.shape[0], sq, h, d), dtype=torch.float16, device=q8.device)
    rc = _LIB.gfx906_attn_i8_fwd(q8.data_ptr(), q_scale.data_ptr(),
                                 k1, ks1, vk1, int(pitch1),
                                 k2, ks2, vk2,
                                 vq.data_ptr(),
                                 0 if v_scale is None else v_scale.data_ptr(),
                                 out.data_ptr(),
                                 q8.shape[0], sq, sp, st, skp, h, hkv, d, tq, int(unrotate),
                                 ctypes.c_float(float(scale)),
                                 ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"gfx906 INT8 attention launch failed ({rc})")
    return out


def attention_fwd(q8, q_scale, k8, k_scale, vq, v_scale, softmax_scale=None, vkey_scale=None,
                  sk=None, tq=2):
    """Quantized inputs -> fp16 [B, SQ, H, D]. Dark launch, no checks on values.

    vkey_scale switches V to one scale per key row (the cached-prefix layout):
    it is folded into P, so v_scale is unused on that path. Pass sk when k8 is
    padded past the real key count (the kernel masks keys >= sk).
    """
    if _load() is None:
        raise RuntimeError("gfx906 INT8 attention kernel is not available")
    b, sq, h, d = q8.shape
    sk = k8.shape[1] if sk is None else sk
    hkv = k8.shape[2]
    skp = vq.shape[2] * 4
    scale = d ** -0.5 if softmax_scale is None or softmax_scale <= 0 else softmax_scale
    return _fwd_split(q8, q_scale,
                      k8.data_ptr(), k_scale.data_ptr(),
                      0 if vkey_scale is None else vkey_scale.data_ptr(), sk * h,
                      0, 0, 0,
                      vq, v_scale, sq, sk, 0, skp, h, hkv, d, scale, tq)


def rotate_quant(x, out=None, scale=None):
    """Rotate (H64) and int8-quantize rows: [.., D] -> int8 rows + per-row scale.

    The rotation is comfy_kitchen's convrot basis, so rows quantized here mix
    with rows the prefix cache already stored.
    """
    if not rotation_ready():
        raise RuntimeError("gfx906 rotate/quantize helpers are not built")
    d = x.shape[-1]
    rows = x.numel() // d
    xc = x.reshape(rows, d)
    xc = xc if xc.is_contiguous() else xc.contiguous()
    if xc.dtype == torch.bfloat16:
        xc = xc.to(torch.float16)
    code = 0 if xc.dtype == torch.float32 else 1
    if out is None:
        out = torch.empty((rows, d), dtype=torch.int8, device=x.device)
    if scale is None:
        scale = torch.empty(rows, dtype=torch.float32, device=x.device)
    rc = _LIBQ.gfx906_rot_quant(xc.data_ptr(), out.data_ptr(), scale.data_ptr(),
                                rows, d, code,
                                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"gfx906 rotate+quantize launch failed ({rc})")
    return out.view(*x.shape), scale

def supported(q, k, v):
    """fp16/bf16/fp32 contiguous [B, S, H, D] with D in (64, 128) and GQA heads."""
    if _load() is None or q.dim() != 4:
        return False
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return False
    b, sq, h, d = q.shape
    return (d in (64, 128) and sq > 0 and k.shape[1] > 0
            and k.shape[0] == b and v.shape[0] == b
            and v.shape[1] == k.shape[1] and v.shape[3] == d
            and k.shape[2] == v.shape[2] and h % k.shape[2] == 0)


def attention(q, k, v, softmax_scale=None, rotate=0, out_dtype=None):
    """fp16 in -> quantize -> int8 kernel -> fp16 out (or out_dtype).

    rotate picks the Hadamard group applied to Q and K before quantization:
    0 is off, 64 matches a convrot-ints cache, the head dim rotates the whole
    head. V is never rotated, so the output is not in a rotated basis.
    """
    q8, qs = quantize_qk(q, rotate)
    k8, ks = quantize_qk(k, rotate)
    vq, vs = quantize_v(v)
    out = attention_fwd(q8, qs, k8, ks, vq, vs, softmax_scale)
    return out if out_dtype in (None, torch.float16) else out.to(out_dtype)


def cache_attention(q, k, v, prefix_kv8, prefix_scale, softmax_scale=None, tq=2):
    """Attention over a Qwen-Image 2.1 int8 prefix cache plus the current step.

    prefix_kv8 is the cache entry as stored: int8 [B, 2, SP, H, D], convrot
    rotated, with one scale per (token, head) in prefix_scale [B, 2, SP, H].
    The kernel reads the cached K and both scale planes straight out of those
    tensors (its K sources are the cached prefix and this step's K), so nothing
    is copied or dequantized per block; only this step's Q/K/V are rotated and
    quantized, and the kernel un-rotates the output in its epilogue (H is
    symmetric, so the same transform undoes it) instead of paying another pass.
    """
    if not rotation_ready():
        raise RuntimeError("gfx906 rotate/quantize helpers are not built")
    b, sq, h, d = q.shape
    hkv = k.shape[2]
    sp = prefix_kv8.shape[2]
    st = k.shape[1]
    sk = sp + st
    skp = ((sk + 63) // 64) * 64
    dev = q.device
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    scale = d ** -0.5 if softmax_scale is None or softmax_scale <= 0 else softmax_scale

    q8, q_scale = rotate_quant(q)
    k8, k_scale_t = rotate_quant(k)
    v8, v_scale_t = rotate_quant(v)

    vq = torch.empty((b, hkv, skp // 4, d), dtype=torch.int32, device=dev)
    rc = _LIBQ.gfx906_rot_pack_v(prefix_kv8.data_ptr(), sp * hkv, 2 * sp * hkv,
                                 v8.data_ptr(), 0, st * hkv,
                                 vq.data_ptr(), b, hkv, d, sp, st, skp // 4, stream)
    if rc != 0:
        raise RuntimeError(f"gfx906 V pack launch failed ({rc})")

    # K sources: cached prefix plane 0 (rows pitch 2*SP*HKV apart per batch) then
    # the current step. Both scale planes use the same pitch; the V plane just
    # starts SP*HKV elements later, which is the view's data pointer.
    return _fwd_split(q8, q_scale,
                     prefix_kv8.data_ptr(), prefix_scale.data_ptr(),
                     prefix_scale[:, 1].data_ptr(), 2 * sp * hkv,
                     k8.data_ptr(), k_scale_t.data_ptr(), v_scale_t.data_ptr(),
                     vq, None, sq, sp, st, skp, h, hkv, d, scale, tq, 1)
