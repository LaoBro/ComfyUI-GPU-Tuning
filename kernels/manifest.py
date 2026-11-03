"""Per-architecture measurement manifest: the numbers that make a card "known".

The registry decides *which implementation can run*; the manifest decides *which
one is faster here*. They are separate files on purpose: the kernel list is code
(code review, unit tests), the measurement table is data (a bench run can rewrite
it, and a diff shows exactly which card gained or lost).

Layout::

    arch/<gcn>/manifest.toml                 shipped with the node
    ~/.config/comfy-gfx-tune/arch/<gcn>/...  local override, wins over the above

Trust is a field, not a convention:

    assumed   someone guessed; never used for dispatch
    derived   transcribed from another machine / another ROCm build
    measured  produced by tools/bench.py on a machine whose fingerprint matches

and the whole file carries a `fingerprint` (ROCm + Triton + gcn target). A file
whose fingerprint does not match the running process is *ignored*, not trusted:
that is the difference between a database and a pile of stale numbers. Nothing
here imports torch at module level, so the loader is unit-testable on a worker
without a GPU.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .contract import logger

SCHEMA_VERSION = 1
GRADES = ("assumed", "derived", "measured")
MIN_GRADE_DEFAULT = "measured"
UNMEASURED = "unmeasured"


def _grade_rank(grade):
    return GRADES.index(grade) if grade in GRADES else -1


def _loads(text):
    """TOML text -> dict, or raise ImportError on a Python without a reader."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib                     # noqa: F401 - Python < 3.11
    return tomllib.loads(text)


def device_arch():
    """The device identity manifests and arch kernels are keyed on.

    `gcnArchName` on AMD; `sm<major><minor>` elsewhere, so an NVIDIA sm120 and
    an sm89 do not share one table. None without a GPU.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        name = str(getattr(props, "gcnArchName", ""))
        if name:
            return name.split(":")[0] or None
        return f"sm{props.major}{props.minor}" if props.major else None
    except Exception:                               # noqa: BLE001 - no torch, no arch
        return None


def runtime_fingerprint(arch=None):
    """What this process can trust: ROCM/CUDA + Torch build + Triton + target.

    The Torch build id is part of it because that wheel carries the compiler
    that produces the kernels: rebuilding torch against the same ROCm version
    can still move a tile by a few percent, and without this segment a stale
    table would keep looking valid. Anything measured under a different
    quadruplet is treated as absent.
    """
    rocm = torch_version = "unknown"
    try:
        import torch

        rocm = torch.version.hip or "cuda"
        torch_version = torch.__version__
        if arch is None and torch.cuda.is_available():
            arch = device_arch()
    except Exception:                               # noqa: BLE001 - diagnostics only
        pass
    try:
        import triton

        triton_version = triton.__version__
    except Exception:                               # noqa: BLE001
        triton_version = "unknown"
    return f"rocm{rocm}|torch{torch_version}|triton{triton_version}|{arch or 'unknown'}"


@dataclass(frozen=True)
class Shape:
    """One (n, k, m) measurement: which impl won, and what it cost."""

    impl: str
    ms: Dict[str, float] = field(default_factory=dict)
    rel: Optional[float] = None
    grade: str = "assumed"
    date: str = ""
    note: str = ""

    def trusted(self, min_grade=MIN_GRADE_DEFAULT):
        return _grade_rank(self.grade) >= _grade_rank(min_grade)


@dataclass
class Site:
    name: str
    default: Optional[str] = None
    default_grade: str = "assumed"
    shapes: Dict[str, Shape] = field(default_factory=dict)

    def at(self, n, k, m):
        """Exact shape, then the same NxK at any batch width."""
        for key in (f"{n}x{k}@m{m}", f"{n}x{k}@*"):
            record = self.shapes.get(key)
            if record is not None:
                return record
        return None

    def winner(self, n, k, m, min_grade=MIN_GRADE_DEFAULT):
        """Implementation recorded for this shape, or None."""
        record = self.at(n, k, m)
        if record is None or not record.trusted(min_grade):
            return None
        return record.impl


class Manifest:
    def __init__(self, arch, fingerprint, capabilities, sites, source=None, note="", runtime=None):
        self.arch = arch
        self.fingerprint = fingerprint                     # what the file was measured on
        self.runtime_fp = runtime if runtime else runtime_fingerprint(arch)
        self.capabilities = tuple(capabilities)
        self.sites = sites
        self.source = source
        self.note = note

    # -- trust ------------------------------------------------------------
    def matches(self, fingerprint=None):
        """A file measured on exactly this ROCm/Triton/gcn triplet is trusted.

        A file with no fingerprint is the shipped per-arch table: it was
        measured for this arch and stamps no build, so it matches every build
        of the arch. A local file always stamps one, and a stale one therefore
        shadows the shipped records until it is refreshed or deleted.
        """
        if not self.fingerprint or self.fingerprint == UNMEASURED:
            return True
        return self.fingerprint == (fingerprint or self.runtime_fp)

    def records(self):
        return sum(len(site.shapes) for site in self.sites.values())

    # -- selection --------------------------------------------------------
    def lookup(self, site, n, k, m, min_grade=MIN_GRADE_DEFAULT):
        """Winner for one shape, else the site default, else None.

        Unmeasured or foreign data always answers None, so the caller falls
        back to probing rather than trusting a number from another machine.
        """
        if not self.matches():
            return None
        entry = self.sites.get(site)
        if entry is None:
            return None
        hit = entry.winner(n, k, m, min_grade)
        return hit if hit is not None else self.default_impl(site, min_grade)

    def default_impl(self, site, min_grade=MIN_GRADE_DEFAULT):
        """Site-wide answer for shapes the table does not cover.

        A `default` without a `default_grade` counts as a guess, so it never
        bypasses the grade filter, and it is dropped as soon as one trusted
        shape disagrees with it.
        """
        entry = self.sites.get(site)
        if entry is None or not entry.default or not self.matches():
            return None
        if _grade_rank(entry.default_grade) < _grade_rank(min_grade):
            return None
        for record in entry.shapes.values():
            if record.trusted(min_grade) and record.impl != entry.default:
                return None
        return entry.default

    def config(self, site, key, min_grade=MIN_GRADE_DEFAULT):
        """The `impl` recorded for a non-shape key, or None.

        Tiles and kernel modes are not keyed by (N, K, m), so they store the
        value that would otherwise be an implementation name: a tile spec, or
        "int8"/"hand"/"off". Same trust rules as a shape -- a file from another
        build answers None and the caller keeps its built-in default.
        """
        if not self.matches():
            return None
        entry = self.sites.get(site)
        if entry is None:
            return None
        record = entry.shapes.get(key)
        if record is None or not record.trusted(min_grade):
            return None
        return record.impl

    # -- reporting --------------------------------------------------------
    def status_line(self):
        counts = {grade: 0 for grade in GRADES}
        for site in self.sites.values():
            for record in site.shapes.values():
                counts[record.grade] = counts.get(record.grade, 0) + 1
        detail = ", ".join(f"{n} {g}" for g, n in counts.items() if n)
        if self.matches():
            state = "in use"
        else:
            state = (f"STALE: measured under {self.fingerprint}, this process is {self.runtime_fp}; "
                     f"re-run tools/bench.py --write")
        return f"measured manifest {self.arch}: {self.records()} shapes ({detail or 'empty'}), {state}"

    def describe(self):
        return {
            "arch": self.arch,
            "fingerprint": self.fingerprint,
            "matches": self.matches(),
            "source": str(self.source) if self.source else None,
            "records": self.records(),
            "sites": {
                name: {"default": site.default,
                       "shapes": {key: {"impl": rec.impl, "ms": dict(rec.ms), "grade": rec.grade}
                                  for key, rec in site.shapes.items()}}
                for name, site in self.sites.items()
            },
        }


# -- loading -------------------------------------------------------------

def node_dir():
    return Path(__file__).resolve().parent.parent


def default_paths(arch):
    """Shipped manifest first, local override second (it wins)."""
    override = Path(os.getenv("COMFY_GFX906_MANIFEST_DIR", "~/.config/comfy-gfx-tune")).expanduser()
    return [node_dir() / "arch" / arch / "manifest.toml",
            override / "arch" / arch / "manifest.toml"]


def _parse_shape(key, raw):
    """One inline table -> Shape, or None when it is not shaped like one."""
    if not isinstance(raw, dict):
        logger.warning("gfx906: manifest %s is not a table, skipped", key)
        return None
    impl = raw.get("impl")
    if not isinstance(impl, str) or not impl:
        logger.warning("gfx906: manifest %s has no impl, skipped", key)
        return None
    ms = {}
    for name, value in (raw.get("ms") or {}).items():
        if isinstance(value, (int, float)) and value > 0:
            ms[str(name)] = float(value)
        else:
            logger.warning("gfx906: manifest %s ms[%s]=%r is not a time, skipped", key, name, value)
    grade = raw.get("grade", "assumed")
    if grade not in GRADES:
        logger.warning("gfx906: manifest %s grade %r unknown, treated as assumed", key, grade)
        grade = "assumed"
    rel = raw.get("rel")
    return Shape(impl=impl, ms=ms,
                 rel=float(rel) if isinstance(rel, (int, float)) else None,
                 grade=grade, date=str(raw.get("date", "")), note=str(raw.get("note", "")))


def _parse_sites(data):
    sites = {}
    for name, raw in (data.get("sites") or {}).items():
        if not isinstance(raw, dict):
            logger.warning("gfx906: manifest site %s is not a table, skipped", name)
            continue
        shapes = {}
        for key, value in (raw.get("shapes") or {}).items():
            record = _parse_shape(f"{name}/{key}", value)
            if record is not None:
                shapes[key] = record
        default = raw.get("default")
        default_grade = raw.get("default_grade", "assumed")
        if default_grade not in GRADES:
            logger.warning("gfx906: manifest site %s default_grade %r unknown, treated as assumed",
                           name, default_grade)
            default_grade = "assumed"
        sites[name] = Site(name=name,
                           default=default if isinstance(default, str) else None,
                           default_grade=default_grade,
                           shapes=shapes)
    return sites


def _read(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return _loads(text)
    except ImportError:
        logger.warning("gfx906: no TOML reader on this Python (need 3.11+), manifest %s ignored", path)
    except Exception as exc:                        # noqa: BLE001 - a broken file must not stop a start
        logger.warning("gfx906: manifest %s is not valid TOML: %s", path, exc)
    return None


def load(arch, path=None, runtime=None):
    """Read and merge the manifests for `arch`. None when there is nothing usable.

    Never raises: a syntax error, an unknown schema or a missing file all mean
    "no data", which is the safe default (the registry falls back to its probe).
    `runtime` overrides the fingerprint this process is compared against; leave
    it unset outside tests.
    """
    if not arch:
        return None
    paths = [Path(path)] if path else default_paths(arch)
    merged, sources, arch_block, note = None, [], {}, ""

    for candidate in paths:
        data = _read(candidate)
        if data is None:
            continue
        schema = data.get("schema")
        if schema != SCHEMA_VERSION:
            logger.warning("gfx906: manifest %s has schema %r, expected %d, ignored",
                           candidate, schema, SCHEMA_VERSION)
            continue
        block = data.get("arch") or {}
        if block.get("gcn") not in (None, arch):
            logger.warning("gfx906: manifest %s is for %r, not %r, ignored",
                           candidate, block.get("gcn"), arch)
            continue
        sites = _parse_sites(data)
        if merged is None:
            merged = {}
        for name, site in sites.items():
            existing = merged.get(name)
            if existing is None:
                merged[name] = site
                continue
            # the later file wins per shape, keeping the other file's entries
            existing.shapes.update(site.shapes)
            if site.default:
                existing.default = site.default
                existing.default_grade = site.default_grade
        arch_block.update(block)
        note = str(block.get("note", note) or "")
        sources.append(candidate)

    if not merged and not sources:
        return None
    return Manifest(
        arch=arch,
        fingerprint=str(arch_block.get("fingerprint", UNMEASURED)),
        capabilities=arch_block.get("capabilities") or (),
        sites=merged or {},
        source=sources[-1] if sources else None,
        note=note,
        runtime=runtime,
    )


def load_runtime(arch=None):
    """Convenience for callers that just want the manifest for this device."""
    fingerprint = runtime_fingerprint(arch)
    if arch is None:
        arch = fingerprint.rsplit("|", 1)[-1]
        if arch == "unknown":
            return None
    return load(arch, runtime=fingerprint)
