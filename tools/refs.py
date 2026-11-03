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


def int8_reference(x, weight, scale):
    """x @ (weight * scale), the fp32 answer the int8 kernels approximate."""
    return (x @ weight.to(torch.float32).T) * scale.reshape(1, -1)


def w4a8_uniform_weight(qdata, s_rel, s_channel, group_size):
    """Decode packed nibbles the way the kernel does: (code - 8) * s * s_ch.

    The byte at (row, i) holds k=2i in its low nibble and k=2i+1 in its high
    nibble; `s_rel` is per (row, group) and `s_channel` per row.
    """
    n, half = qdata.shape
    k = half * 2
    b = qdata.to(torch.int32)
    code = torch.empty((n, k), dtype=torch.int32, device=qdata.device)
    code[:, 0::2] = b & 0x0F
    code[:, 1::2] = (b >> 4) & 0x0F
    group = torch.arange(k, device=qdata.device) // group_size
    s = s_rel.to(torch.float32)[:, group]
    return ((code.to(torch.float32) - 8.0) * s) * s_channel.reshape(-1, 1)
