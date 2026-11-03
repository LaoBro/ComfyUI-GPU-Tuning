"""The implementation contract every kernel site shares.

A *site* is one place in the graph where we can swap an implementation, named
`<op>.<shape class>` (for example `w4a8.m1`: the 4-bit weight GEMV for one to
eight rows). A site owns an ordered list of :class:`Candidate` implementations
written in different languages -- a HIP extension, a Triton kernel with inline
asm, a portable Triton kernel -- and the registry picks one for the current
device.

`Candidate` deliberately describes *how to find out whether the implementation
works here*, not *what it computes*. The compute signature stays in the kernel
module, so the three languages keep their own argument shapes and no wrapper
has to normalise tensors. That is why adopting the registry costs a module
nothing but four lines at the bottom of the file.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

MODE_AUTO = "auto"
MODE_ON = "on"
MODE_OFF = "off"

# The spellings the kernels accepted before the registry existed. `off` words
# came from the "asm" switches, where the opposite value was named "triton";
# both are still accepted so existing startup scripts keep working.
MODE_ON_WORDS = ("on", "asm", "hip", "1", "true", "yes")
MODE_OFF_WORDS = ("off", "triton", "none", "0", "false", "no")

logger = logging.getLogger("gfx906")


def parse_mode(value, default=MODE_AUTO):
    """Env text -> auto / on / off. Anything unrecognised means "decide by probe"."""
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in MODE_OFF_WORDS:
        return MODE_OFF
    if text in MODE_ON_WORDS:
        return MODE_ON
    return default


def env_mode(name, default=MODE_AUTO):
    return parse_mode(os.getenv(name), default) if name else default


@dataclass(frozen=True)
class Candidate:
    """One implementation of a site.

    name    the value printed by `backend_name()` and by `status()`; keep it
            short and stable, existing tooling matches on it.
    kind    "extension" (imported module, e.g. a HIP build), "asm" (Triton
            kernel carrying inline asm), "portable" (plain Triton/Torch).
    env     optional variable that forces this candidate on or off, so a user
            who suspects a kernel can pin it without a code change.
    probe   callable returning True when the implementation can run here.
            Called at most once; a failure is cached as "not available".
    loader  callable returning the module object for "extension" candidates,
            or None when it is not installed.
    archs   gcn targets this candidate is written for. Only consulted when the
            registry was given an arch, so an unset arch changes nothing.
    requires instructions/traits it needs, for `describe()` and for the
            measurement manifest. Purely documentation today: the probe is
            still the authority.
    priority where it lands in the chain: lower is tried first, and a portable
            candidate is always moved to the end regardless. Kernel modules
            dropped into arch/<gcn>/kernels default to 30, i.e. ahead of the
            site's own asm (50).
    """

    name: str
    kind: str = "portable"
    env: Optional[str] = None
    probe: Optional[Callable[[], bool]] = None
    loader: Optional[Callable[[], object]] = None
    archs: Optional[Tuple[str, ...]] = None
    requires: Tuple[str, ...] = field(default_factory=tuple)
    priority: int = 50

    @property
    def forced(self):
        """The explicit env override for this candidate: auto, on or off."""
        return env_mode(self.env) if self.env else MODE_AUTO
