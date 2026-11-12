"""Kernel registry: implementation contract plus the backend dispatch chain.

Layout this package grows into (see docs/PORTING.md):

    kernels/            # registry, contracts, manifests -- pure bookkeeping
    kernels/triton/     # portable Triton kernels, no arch-specific asm
    kernels/hip/        # ctypes wrappers around arch/<gcn>/kernels/*.so
    arch/<gcn>/         # per-arch measurement manifests and specialised kernels

Nothing here imports torch or triton: the registry is pure bookkeeping so it
can be unit-tested on a machine without ROCm.
"""

from .contract import Candidate, parse_mode, MODE_ON, MODE_OFF, MODE_AUTO
from .registry import BackendRegistry, FAILED, safe_install, site_candidates
from . import loader
from . import manifest

__all__ = [
    "BackendRegistry",
    "Candidate",
    "FAILED",
    "MODE_AUTO",
    "MODE_OFF",
    "MODE_ON",
    "loader",
    "manifest",
    "parse_mode",
    "safe_install",
    "site_candidates",
]
