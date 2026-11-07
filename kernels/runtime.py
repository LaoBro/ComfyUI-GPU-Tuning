"""Locate and open the compiled kernel libraries for this platform.

A build writes one library per target into arch/<target>/kernels/. The logical
name is the same on every platform; only the filename differs:

    Linux    libgfx906_attn.so     (shipped)
    Windows  gfx906_attn.dll
    Linux    libsm89_attn.so       (an NVIDIA target, should one be built)

Libraries export a flat extern "C" ABI and are linked against their GPU
runtime at build time. On Linux the ROCm wheel keeps libamdhip64 outside the
loader path and the libraries carry no rpath for it, so the runtime is opened
RTLD_GLOBAL first. On Windows the import library handles that and only the
wheel's DLL directories need registering.
"""

import ctypes
import glob
import os
import sys
from functools import lru_cache
from pathlib import Path

from .contract import logger
from .loader import current_arch, node_dir

WINDOWS = sys.platform == "win32"

# os.add_dll_directory removes the entry when its cookie is collected.
_dll_dirs = []


def library_name(arch, name):
    if WINDOWS:
        return f"{arch}_{name}.dll"
    return f"lib{arch}_{name}.so"


def kernel_dir(arch):
    return node_dir() / "arch" / arch / "kernels"


def library_path(arch, name):
    return kernel_dir(arch) / library_name(arch, name)


@lru_cache(maxsize=None)
def _runtime_ready(arch):
    import torch

    if WINDOWS:
        root = os.path.dirname(os.path.dirname(torch.__file__))
        for directory in (root, os.path.join(root, "_rocm_sdk_core", "bin")):
            if os.path.isdir(directory):
                _dll_dirs.append(os.add_dll_directory(directory))
        return True
    if not arch.startswith("gfx"):
        return True                       # CUDA runtime is already linked into torch
    try:
        ctypes.CDLL("libamdhip64.so.7", mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        pass
    root = os.path.dirname(os.path.dirname(torch.__file__))
    for path in glob.glob(os.path.join(root, "_rocm_sdk_core", "lib", "libamdhip64.so*")):
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError:
            continue
    return False


def load(name, label=None):
    """Return the ctypes library for a logical kernel name, or None.

    A missing or unloadable library is a normal fallback, not an error: the
    caller keeps its portable implementation. Only a real load failure is
    logged, with the rebuild command.
    """
    arch = current_arch()
    if arch is None:
        return None
    path = library_path(arch, name)
    if not path.exists():
        return None
    if not _runtime_ready(arch):
        logger.info("gfx906: GPU runtime not found, %s stays off", label or name)
        return None
    try:
        return ctypes.CDLL(str(path))
    except OSError as exc:
        logger.warning("gfx906: %s did not load (%s); rebuild it for this runtime with "
                       "tools/build_hip_attn.py --force", path.name, exc)
        return None
