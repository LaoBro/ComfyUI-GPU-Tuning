# `arch/<gcn>/kernels/`

One file per implementation, imported by `kernels/loader.py` and added to the
site's candidate chain. A module declares what it is:

```python
SITE = "w4a8.m1"                  # required: which site it implements
NAME = "hip"                      # optional: candidate name, defaults to the file name
PRIORITY = 10                     # optional: lower runs first (default 30)
ENV = "COMFY_GFX906_W4A8_HIP"     # optional: its own on/off switch
REQUIRES = ("hip extension",)     # optional: documentation only

def probe():                      # optional: defaults to "imported means available"
    return True

def linear(x, qdata, s_rel, s_channel, bias, out_dtype, group_size,
           convrot_groupsize, levels=None, inv_level=1.0):
    """Same signature and contract as w4a8_gemv.linear; None = not my shape."""
```

Rules:

- the site's own candidates sit at priority 50 (asm) and the portable one is
  always last, so a file here is tried first unless it says otherwise;
- one file is one implementation: no packages, no relative imports;
- an import error is a log line and the file is skipped once -- a kernel that
  only builds for another card must not stop ComfyUI from starting;
- a compiled artifact is wrapped by a two-line shim, so the `.so` can live
  wherever it was built:

```python
from w4a8_hip_ext import linear          # the compiled module

SITE = "w4a8.m1"
NAME = "hip"
```

Sites that exist today: `w4a8.m1` (`w4a8_gemv.linear`), `int8_gemv.m1`
(`int8_gemv.linear`). Adding a new site means declaring it in the kernel module
first; a file whose `SITE` nobody declares is simply never loaded.

Measurements for whatever you add belong in `arch/<gcn>/manifest.toml`, keyed by
the same `NAME` -- see `tools/bench.py` and docs/PORTING.md (manifest appendix).
