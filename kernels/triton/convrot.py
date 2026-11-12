"""Hadamard rotation kernels for the convrot quantization path.

Kept out of gpu_tuning so that module imports without triton: on a machine
that has no triton the pack still loads and reports why it is inactive.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

GROUP_SIZE = 256


@triton.jit
def _butterfly(x, BM: tl.constexpr, OUTER: tl.constexpr, D: tl.constexpr):
    """One radix-4 stage of the regular Hadamard transform, stride D."""
    t = tl.reshape(x, (BM, OUTER, 2, 2, D))
    t = tl.permute(t, (0, 1, 4, 2, 3))     # [BM, OUTER, D, i, j]
    tj0, tj1 = tl.split(t)
    e0, e2 = tl.split(tj0)                 # squared index 2*i + j
    e1, e3 = tl.split(tj1)
    s = e0 + e1 + e2 + e3
    # h4 rows: (1,1,1,-1) (1,1,-1,1) (1,-1,1,1) (-1,1,1,1), so every output is s - 2*<one input>
    o = tl.join(tl.join(s - 2 * e3, s - 2 * e1), tl.join(s - 2 * e2, s - 2 * e0))
    o = tl.permute(o, (0, 1, 3, 4, 2))     # [BM, OUTER, i, j, D]
    return tl.reshape(o, (BM, 256))


@triton.jit
def _hadamard256_kernel(X, Y, n_rows, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    mask = rows < n_rows
    cols = tl.arange(0, 256)
    offs = rows[:, None] * 256 + cols[None, :]
    x = tl.load(X + offs, mask=mask[:, None], other=0.0).to(tl.float32)
    # 256 = 4^4: four radix-4 stages, then the 1/sqrt(256) normalization
    x = _butterfly(x, BM, 64, 1)
    x = _butterfly(x, BM, 16, 4)
    x = _butterfly(x, BM, 4, 16)
    x = _butterfly(x, BM, 1, 64)
    tl.store(Y + offs, (x * 0.0625).to(Y.dtype.element_ty), mask=mask[:, None])


def hadamard_rotate(x):
    """x @ (h4 kron h4 kron h4 kron h4) / 16, as four butterfly stages."""
    shape = x.shape
    groups = x.reshape(-1, GROUP_SIZE)
    out = torch.empty_like(groups)
    block = 16
    _hadamard256_kernel[(triton.cdiv(groups.shape[0], block),)](groups, out, groups.shape[0], BM=block, num_warps=4)
    return out.reshape(shape)


@triton.jit
def _convrot_quant_kernel(X, Qp, S, K: tl.constexpr, GROUPS: tl.constexpr):
    """Rotate one activation row in registers, then quantize it to int8."""
    row = tl.program_id(0)
    cols = tl.arange(0, K)
    offs = row * K + cols
    x = tl.load(X + offs).to(tl.float32)
    x = tl.reshape(x, (GROUPS, 256))
    x = _butterfly(x, GROUPS, 64, 1)
    x = _butterfly(x, GROUPS, 16, 4)
    x = _butterfly(x, GROUPS, 4, 16)
    x = _butterfly(x, GROUPS, 1, 64)
    x = x * 0.0625
    scale = tl.maximum(tl.max(tl.abs(x)) / 127.0, 1e-30)
    q = tl.clamp(libdevice.rint(x / scale), -128.0, 127.0).to(tl.int8)
    tl.store(Qp + offs, tl.reshape(q, (K,)))
    tl.store(S + row, scale)


class QuantizedRotation:
    """Activation already rotated and quantized by the fused convrot kernel."""

    __slots__ = ("data", "scale")

    def __init__(self, data, scale):
        self.data = data
        self.scale = scale


def convrot_quantize(x):
    rows = x.shape[0]
    k = x.shape[1]
    data = torch.empty((rows, k), device=x.device, dtype=torch.int8)
    scale = torch.empty((rows,), device=x.device, dtype=torch.float32)
    _convrot_quant_kernel[(rows,)](x, data, scale, K=k, GROUPS=k // GROUP_SIZE, num_warps=4)
    return QuantizedRotation(data, scale.reshape(rows, 1))
