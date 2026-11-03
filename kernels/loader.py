"""Architecture-specific kernel modules: `arch/<gcn>/kernels/*.py`.

A file in that directory declares which site it implements, and the registry
turns it into a candidate:

    SITE = "w4a8.m1"                   # required; the site name it belongs to
    NAME = "hip"                       # optional; defaults to the file name
    PRIORITY = 10                      # optional; lower runs first (default 30)
    ENV = "COMFY_GFX906_W4A8_HIP"      # optional per-implementation switch
    REQUIRES = ("hip extension",)      # optional, documentation
    def probe(): return True           # optional; "imported" is enough otherwise
    def linear(...): ...               # the site's own signature, unchanged

Rules that keep this simple:

* one file is one implementation, no package layout, no relative imports;
* the module is imported once per process, and an import error is a log line,
  not a crash -- a broken kernel for one arch must not stop ComfyUI from
  starting;
* a compiled extension is wrapped by a two-line shim that imports it, so the
  `.so` can live wherever it was built.

`COMFY_GFX906_ARCH` overrides the detected target (useful on a machine without
the GPU you are targeting), and `COMFY_GFX906_ARCH_KERNELS` adds a search
directory, which is how the self-test exercises discovery without touching the
repository.
"""

import importlib.util
import os
import sys
from pathlib import Path

from .contract import Candidate, logger

ARCH_ENV = "COMFY_GFX906_ARCH"
SEARCH_ENV = "COMFY_GFX906_ARCH_KERNELS"
DEFAULT_PRIORITY = 30

_cache = {}
_imported = {}          # path -> module or None; a broken file is imported once


def node_dir():
    return Path(__file__).resolve().parent.parent


def current_arch():
    """The gcn target this process should load kernels for, or None."""
    override = os.getenv(ARCH_ENV, "").strip()
    if override:
        return override
    try:
        import torch

        if torch.version.hip is None or not torch.cuda.is_available():
            return None
        name = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
        return name.split(":")[0] or None
    except Exception:                                   # noqa: BLE001 - no torch, no arch
        return None


def search_dirs(arch):
    """Shipped directory first, then the override directory from the env."""
    dirs = [node_dir() / "arch" / arch / "kernels"]
    extra = os.getenv(SEARCH_ENV, "").strip()
    if extra:
        dirs.append(Path(extra).expanduser())
    return dirs


def _import(path, arch):
    """Import a single file as its own module. Returns None if it has no loader."""
    key = (str(path), path.stat().st_mtime_ns)
    if key in _imported:
        return _imported[key]
    name = f"_gfx906_{arch}_{path.stem}".replace("-", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _imported[key] = None
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module              # dataclasses/typing need the entry
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)         # a half-imported module must not linger
        _imported[key] = None               # and it is only reported once
        raise
    _imported[key] = module
    return module


def candidates_for(site, arch=None):
    """Candidates discovered for `site` on this architecture, in file order."""
    arch = arch or current_arch()
    if not arch:
        return ()
    dirs = tuple(str(path) for path in search_dirs(arch))
    key = (site, arch, dirs)
    if key in _cache:
        return _cache[key]

    found = []
    for directory in dirs:
        if not Path(directory).is_dir():
            continue
        for path in sorted(Path(directory).glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                module = _import(path, arch)
            except Exception as exc:                    # noqa: BLE001 - see module docstring
                logger.warning("gfx906: %s/%s could not be imported (%s: %s), skipped",
                               Path(directory).name, path.name, type(exc).__name__, exc)
                continue
            if module is None:
                continue
            declared = getattr(module, "SITE", None)
            if declared != site:
                continue
            priority = int(getattr(module, "PRIORITY", DEFAULT_PRIORITY))
            candidate = Candidate(
                name=str(getattr(module, "NAME", path.stem)),
                kind="extension",
                env=getattr(module, "ENV", None),
                loader=lambda mod=module: mod,
                probe=getattr(module, "probe", None),
                priority=priority,
                requires=tuple(getattr(module, "REQUIRES", ())),
            )
            found.append(candidate)
            logger.info("gfx906: %s gets %r from arch/%s/kernels/%s (priority %d)",
                        site, candidate.name, arch, path.name, priority)
    _cache[key] = tuple(found)
    return _cache[key]


def reset():
    """Forget the discovery cache (tests, and after dropping a new file in)."""
    _cache.clear()
    _imported.clear()
