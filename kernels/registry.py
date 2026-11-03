"""The backend dispatch chain for one kernel site.

Resolution order, first match wins:

    explicit   a candidate whose env variable says "on": no probe is paid, the
               user asked for it (an extension still has to be installed)
    probed     the first candidate, in declaration order, whose probe passes
    portable   the last candidate in the list; it must always be available

`measured` -- a winner recorded in the per-arch manifest -- slots in between
`explicit` and `probed`. The chain is a single function so that addition stays
local; everything else (caching, fallback, status strings) already goes through
it.

The registry only decides *which* implementation to call. It never touches
tensors: the compute signatures stay in the kernel modules, which is what lets
a HIP module, a Triton+asm kernel and a portable Triton kernel share one list.
"""

# `Candidate` is re-exported so a kernel module declares its site with one
# import: from .kernels import registry as kreg
from .contract import Candidate, MODE_OFF, MODE_ON, logger  # noqa: F401

# Sentinel: "look the table up from the device" as opposed to "no table at all".
AUTO_MANIFEST = object()
_runtime_manifest = None
_runtime_loaded = False

# A patch that fails because the *kernel* cannot be built here should downgrade
# to the fallback and say so. Anything else is a bug and must stay loud, or the
# registry turns into a machine for hiding regressions.
_BENIGN_TYPES = (ImportError, AttributeError, OSError)
_BENIGN_NAMES = frozenset({
    "CompilationError", "OutOfResources", "PTXASError", "HIPError",
    "MsgSpecError", "TritonError", "HIPRuntimeError",
})


def is_downgradable(exc):
    """True when `exc` means "this kernel is not for this device", not "we are wrong"."""
    if isinstance(exc, _BENIGN_TYPES):
        return True
    return any(cls.__name__ in _BENIGN_NAMES for cls in type(exc).__mro__)


# Returned by safe_install when a patch was skipped, so a caller can tell
# "failed" from "the patch legitimately returned None".
FAILED = object()


def safe_install(label, fn, *args, **kwargs):
    """Run one patch; downgrade a device-level failure to a log line.

    Returns fn's result, or FAILED when the patch could not be installed here.
    Raises anything that is not a device/kernel error, so a typo or a logic bug
    still fails the start instead of silently running the slow path.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                                    # noqa: BLE001 - filtered below
        if not is_downgradable(exc):
            raise
        logger.warning("gfx906: %s not available on this device (%s: %s), keeping the fallback",
                       label, type(exc).__name__, exc)
        return FAILED


def _chain(candidates):
    """Order by `priority`, keep declaration order on ties, portable always last.

    A module discovered in arch/<gcn>/kernels arrives with priority 30, so it
    lands ahead of the site's own asm without anyone editing the site file; a
    portable candidate cannot be reordered, which keeps `resolve()` total.
    """
    ordered = [cand for _, cand in sorted(enumerate(candidates),
                                          key=lambda pair: (pair[1].priority, pair[0]))]
    portable = [cand for cand in ordered if cand.kind == "portable"]
    return tuple([cand for cand in ordered if cand.kind != "portable"] + portable)


def runtime_manifest():
    """The measurement table for this process, or None. Read once, lazily.

    Imported lazily so a worker without the manifest package still starts, and
    cached because the file does not change while ComfyUI runs.
    COMFY_GFX906_MEASURED=off skips it entirely: that is the escape hatch when a
    table turns out wrong for a driver whose fingerprint we did not catch.
    """
    global _runtime_manifest, _runtime_loaded
    if _runtime_loaded:
        return _runtime_manifest
    _runtime_loaded = True
    try:
        import os

        if os.getenv("COMFY_GFX906_MEASURED", "1").strip().lower() in ("0", "off", "false", "no"):
            logger.info("gfx906: measured table off (COMFY_GFX906_MEASURED), falling back to probes")
            return None
        from . import manifest as manifest_module

        _runtime_manifest = manifest_module.load_runtime()
    except Exception as exc:                                # noqa: BLE001 - never block a start
        logger.warning("gfx906: measurement manifest unavailable: %s: %s", type(exc).__name__, exc)
        _runtime_manifest = None
    return _runtime_manifest


def site_candidates(site, base, arch=None):
    """A site's declared candidates plus whatever arch/<gcn>/kernels adds.

    This is the one call a site makes to opt into architecture-specific kernels:
    the declared list stays readable in the kernel module, discovery supplies
    the extra implementations, and `_chain` decides the order.
    """
    from . import loader

    taken = {cand.name for cand in base}
    extra = []
    for cand in loader.candidates_for(site, arch):
        if cand.name in taken:
            # a discovered file must not shadow one the site already declares:
            # two candidates under one name would share a memoised availability
            logger.warning("gfx906: %s: dropping arch kernel %r, the site already declares "
                           "that name", site, cand.name)
            continue
        taken.add(cand.name)
        extra.append(cand)
    return _chain(tuple(base) + tuple(extra))


class BackendRegistry:
    """Ordered candidate list for one site, with every decision memoised."""

    def __init__(self, site, candidates, arch=None, manifest=AUTO_MANIFEST, min_grade="measured"):
        if not candidates:
            raise ValueError(f"{site}: a site needs at least one candidate")
        self.site = site
        self.candidates = _chain(candidates)
        self.arch = arch
        self.min_grade = min_grade
        self._by_name = {cand.name: cand for cand in self.candidates}
        self._available = {}          # name -> bool, decided at most once
        self._resolved = None
        self._explicit_flags = None
        self._manifest = None if manifest is AUTO_MANIFEST else manifest
        self._auto_manifest = manifest is AUTO_MANIFEST
        self.hits = {"measured": 0, "probe": 0}

    # -- inspection -------------------------------------------------------
    def names(self):
        return [cand.name for cand in self.candidates]

    def candidate(self, name):
        return self._by_name.get(name)

    def state(self, name):
        """True / False once decided, None while unprobed. Never probes."""
        return self._available.get(name)

    def reset(self):
        """Forget every decision, so a new env value takes effect (tests, tools)."""
        self._available.clear()
        self._resolved = None
        self._explicit_flags = None
        self.hits = {"measured": 0, "probe": 0}

    def set_manifest(self, manifest):
        """Attach (or detach, with None) the measurement table."""
        self._manifest = manifest
        self._auto_manifest = False
        return self

    def table(self):
        """The measurement table for this site, loaded on first use."""
        if self._manifest is None and self._auto_manifest:
            self._manifest = runtime_manifest()
        return self._manifest

    # -- availability -----------------------------------------------------
    def available(self, candidate):
        name = candidate.name
        if name in self._available:
            return self._available[name]
        ok = self._decide(candidate)
        self._available[name] = ok
        return ok

    def supported(self, name):
        """Whether a named candidate can run here; unknown names are False."""
        cand = self._by_name.get(name)
        return bool(cand is not None and self.available(cand))

    def _decide(self, candidate):
        mode = candidate.forced
        if mode == MODE_OFF:
            return False
        if self.arch and candidate.archs and self.arch not in candidate.archs:
            return False
        if candidate.loader is not None:
            # Presence is not negotiable: "on" means "prefer it over a passing
            # probe", not "pretend it is installed".
            try:
                return candidate.loader() is not None
            except Exception as exc:                            # noqa: BLE001
                logger.warning("gfx906: %s %s could not be loaded: %s: %s",
                               self.site, candidate.name, type(exc).__name__, exc)
                return False
        if mode == MODE_ON:
            return True
        if candidate.probe is None:
            return True
        try:
            return bool(candidate.probe())
        except Exception as exc:                                # noqa: BLE001
            logger.info("gfx906: %s %s probe failed on this device: %s: %s",
                        self.site, candidate.name, type(exc).__name__, exc)
            return False

    # -- selection --------------------------------------------------------
    def resolve(self):
        """The implementation this device should use. Memoised."""
        if self._resolved is not None:
            return self._resolved
        for cand in self.candidates:
            if self.available(cand):
                self._resolved = cand.name
                return self._resolved
        # Cannot happen while the last candidate is portable, but a site that
        # breaks that rule should run *something* rather than crash a load.
        self._resolved = self.candidates[-1].name
        logger.warning("gfx906: %s has no available candidate, using %s",
                       self.site, self._resolved)
        return self._resolved

    def next_after(self, name):
        """First available candidate declared after `name`.

        Used when an implementation declines a shape at call time: the caller
        falls through the same chain instead of hard-coding "then use asm".
        """
        names = self.names()
        start = names.index(name) + 1 if name in names else 0
        for cand in self.candidates[start:]:
            if self.available(cand):
                return cand.name
        return self.candidates[-1].name

    def _explicit(self):
        """True when some candidate's env switch is set: the user outranks the table."""
        if self._explicit_flags is None:
            self._explicit_flags = any(cand.forced in (MODE_ON, MODE_OFF)
                                       for cand in self.candidates)
        return self._explicit_flags

    def choose(self, n, k, m):
        """Which implementation to call for one shape.

        explicit env > measured table > probe order. The table is consulted per
        call, which is the point: a card that prefers asm at 12288 rows and the
        int32 kernel at 4096 gets both, without a restart and without a rule in
        the kernel.
        """
        if self._explicit():
            self.hits["probe"] += 1
            return self.resolve()
        table = self.table()
        if table is not None:
            try:
                name = table.lookup(self.site, n, k, m, self.min_grade)
            except Exception as exc:                        # noqa: BLE001
                logger.warning("gfx906: %s table lookup failed (%s: %s), ignoring the table",
                               self.site, type(exc).__name__, exc)
                self._auto_manifest = False
                self._manifest = None
                name = None
            candidate = self._by_name.get(name) if name else None
            # a table naming an implementation this site does not have, or one
            # that cannot run here, is a table bug: fall back to the probe order
            if candidate is not None and self.available(candidate):
                self.hits["measured"] += 1
                return name
        self.hits["probe"] += 1
        return self.resolve()

    def describe(self):
        """Side-effect-free snapshot for `status()` and the bench tool.

        Deliberately does not trigger the table load: reporting state must not
        read a file or import torch behind the caller's back.
        """
        states = {True: "available", False: "absent", None: "unprobed"}
        table = self._manifest
        if table is None:
            measured = "unloaded" if self._auto_manifest else None
        else:
            measured = {
                "source": str(table.source) if getattr(table, "source", None) else None,
                "matches": table.matches(),
                "records": table.records(),
            }
        return {
            "site": self.site,
            "arch": self.arch,
            "resolved": self._resolved,
            "min_grade": self.min_grade,
            "measured": measured,
            "hits": dict(self.hits),
            "candidates": [
                {"name": cand.name, "kind": cand.kind, "env": cand.env, "priority": cand.priority,
                 "state": states[self._available.get(cand.name)]}
                for cand in self.candidates
            ],
        }
