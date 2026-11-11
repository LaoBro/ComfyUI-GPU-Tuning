"""Self-test for the kernel registry: it must not change any decision.

Run without a GPU (the registry never imports torch):

    python custom_nodes/ComfyUI-GPU-Tuning/tools/registry_selftest.py

The first half replays the *old* backend_name() logic verbatim for every
combination of (env value, extension present, asm compiles) and compares it
with the registry. That is the acceptance gate for the P0 refactor: same answer,
same probe count, in every combination. The second half covers the pieces that
did not exist before (fallthrough, arch filter, downgrade rules).
"""

import os
import sys
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NODE_DIR))

from kernels import registry as kreg                                   # noqa: E402
from kernels.contract import MODE_AUTO, MODE_OFF, MODE_ON, parse_mode  # noqa: E402

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


# -- the legacy logic, kept here as the reference ------------------------

OLD_ON = ("on", "asm", "1", "true", "yes")
OLD_OFF = ("off", "triton", "0", "false", "no")


def legacy_w4a8(env, hip, asm_ok, probes):
    """backend_name() from w4a8_gemv.py before the registry."""
    probes.append("hip")                       # _load_hip() was always consulted first
    mode = (env if env is not None else "auto").strip().lower()
    if mode in OLD_OFF:
        asm = False
    elif mode in OLD_ON:
        asm = True
    else:
        probes.append("asm")
        asm = asm_ok
    if hip:
        return "hip", asm
    return ("asm" if asm else "triton"), asm


def legacy_int8(env, asm_ok, probes):
    mode = (env if env is not None else "auto").strip().lower()
    if mode in OLD_OFF:
        return "triton", False
    if mode in OLD_ON:
        return "asm", True
    probes.append("asm")
    return ("asm" if asm_ok else "triton"), asm_ok


# -- the same sites, expressed through the registry ----------------------

def build_site(site, env_name, hip_present, asm_ok, probes, arch=None):
    def loader():
        probes.append("hip")
        return object() if hip_present else None

    def probe():
        probes.append("asm")
        if not asm_ok:
            raise RuntimeError("v_dot4_i32_i8 not supported")
        return True

    candidates = []
    if site == "w4a8.m1":
        candidates.append(kreg.Candidate("hip", kind="extension",
                                         env="COMFY_GPU_TUNE_W4A8_HIP", loader=loader))
    candidates.append(kreg.Candidate("asm", kind="asm", env=env_name, probe=probe))
    candidates.append(kreg.Candidate("triton", kind="portable"))
    return kreg.BackendRegistry(site, candidates, arch=arch)


def compare_all():
    os.environ.pop("COMFY_GPU_TUNE_W4A8_HIP", None)
    cases = 0
    envs = [None, "", "auto", "on", "asm", "1", "off", "triton", "0", "nonsense"]
    sites = [("w4a8.m1", "COMFY_GPU_TUNE_W4A8_ASM"), ("int8_gemv.m1", "COMFY_GPU_TUNE_INT8_GEMV_ASM")]
    for site, env_name in sites:
        for hip in (False, True) if site == "w4a8.m1" else (False,):
            for asm_ok in (False, True):
                for env in envs:
                    cases += 1
                    if env is None:
                        os.environ.pop(env_name, None)
                    else:
                        os.environ[env_name] = env
                    legacy_probes, new_probes = [], []
                    if site == "w4a8.m1":
                        want, want_asm = legacy_w4a8(env, hip, asm_ok, legacy_probes)
                    else:
                        want, want_asm = legacy_int8(env, asm_ok, legacy_probes)

                    reg = build_site(site, env_name, hip, asm_ok, new_probes)
                    got = reg.resolve()
                    label = f"{site} env={env!r} hip={hip} asm={asm_ok}"
                    check(label, got, want)
                    check(label + " asm_supported", reg.supported("asm"), want_asm)
                    check(label + " probes", new_probes, legacy_probes)

                    # an extension that declines a shape falls through the list
                    if site == "w4a8.m1" and got == "hip":
                        check(label + " next_after(hip)", reg.next_after("hip"),
                              "asm" if want_asm else "triton")
                    # repeated calls are memoised: no extra probe, same answer
                    again = reg.resolve()
                    check(label + " cached", again, got)
                    check(label + " probe count", len(new_probes), len(legacy_probes))
                    os.environ.pop(env_name, None)
    return cases


# -- pieces the old code had nowhere to put -----------------------------

def probe_order():
    """Every available candidate is probed in declaration order, first wins."""
    seen = []
    reg = kreg.BackendRegistry("demo", [
        kreg.Candidate("hip", kind="extension", loader=lambda: seen.append("hip") or object()),
        kreg.Candidate("asm", kind="asm", probe=lambda: seen.append("asm") or True),
        kreg.Candidate("triton", kind="portable"),
    ])
    check("order: resolve", reg.resolve(), "hip")
    check("order: only hip probed", seen, ["hip"])


def arch_filter():
    """Candidates tied to other gcn targets are skipped, portable still wins."""
    reg = kreg.BackendRegistry("demo", [
        kreg.Candidate("asm", kind="asm", probe=lambda: True, archs=("gfx906",)),
        kreg.Candidate("triton", kind="portable"),
    ], arch="gfx1100")
    check("arch: filtered", reg.resolve(), "triton")
    reg = kreg.BackendRegistry("demo", [
        kreg.Candidate("asm", kind="asm", probe=lambda: True, archs=("gfx906",)),
        kreg.Candidate("triton", kind="portable"),
    ], arch="gfx906")
    check("arch: matched", reg.resolve(), "asm")
    # no arch given: the filter is off, so nothing changes for today's sites
    reg = kreg.BackendRegistry("demo", [
        kreg.Candidate("asm", kind="asm", probe=lambda: True, archs=("gfx906",)),
        kreg.Candidate("triton", kind="portable"),
    ])
    check("arch: unset keeps the old answer", reg.resolve(), "asm")


def describe_states():
    """describe() must not trigger a probe: status() calls it on every refresh."""
    touched = []
    reg = kreg.BackendRegistry("demo", [
        kreg.Candidate("asm", kind="asm", env="X_NOPE", probe=lambda: touched.append(1) or True),
        kreg.Candidate("triton", kind="portable"),
    ])
    rows = reg.describe()["candidates"]
    check("describe: unprobed", [r["state"] for r in rows], ["unprobed", "unprobed"])
    check("describe: no probe", touched, [])
    reg.resolve()
    rows = reg.describe()["candidates"]
    check("describe: available", rows[0]["state"], "available")
    check("describe: resolved", reg.describe()["resolved"], "asm")


def downgrade_rules():
    """safe_install swallows device errors and re-raises real bugs."""
    check("safe: import error", kreg.safe_install("demo", lambda: (_ for _ in ()).throw(ImportError("no module"))), kreg.FAILED)
    check("safe: kernel build error", kreg.safe_install("demo", _raise_compilation_error), kreg.FAILED)
    check("safe: pass-through value", kreg.safe_install("demo", lambda: 7), 7)
    raised = False
    try:
        kreg.safe_install("demo", lambda: (_ for _ in ()).throw(ValueError("logic bug")))
    except ValueError:
        raised = True
    check("safe: value error propagates", raised, True)


class CompilationError(Exception):
    pass


def _raise_compilation_error():
    raise CompilationError("inline asm rejected")


def mode_words():
    for text, want in ((None, MODE_AUTO), ("", MODE_AUTO), ("AUTO", MODE_AUTO),
                       ("on", MODE_ON), ("asm", MODE_ON), ("1", MODE_ON), ("yes", MODE_ON),
                       ("off", MODE_OFF), ("triton", MODE_OFF), ("0", MODE_OFF), ("no", MODE_OFF),
                       ("weird", MODE_AUTO)):
        check(f"mode {text!r}", parse_mode(text), want)


def choose_layer():
    """explicit env > measured table > probe order, decided per shape."""

    class Table:
        def __init__(self, mapping):
            self.mapping = mapping
            self.lookups = 0
            self.source = "fake.toml"

        def lookup(self, site, n, k, m, min_grade=None):
            self.lookups += 1
            return self.mapping.get((n, k, m))

        def matches(self):
            return True

        def records(self):
            return len(self.mapping)

    def site(available=True, env=None, table=None):
        reg = kreg.BackendRegistry("demo", [
            kreg.Candidate("asm", kind="asm", env=env, probe=lambda: available),
            kreg.Candidate("triton", kind="portable"),
        ], manifest=table)
        return reg

    # no table: same answer as resolve(), every call counted as a probe
    reg = site()
    check("choose: no table", reg.choose(4096, 4096, 1), "asm")
    check("choose: no table hits", reg.hits, {"measured": 0, "probe": 1})

    # the table outvotes the probe order for the shape it names, and only there
    table = Table({(4096, 4096, 1): "triton"})
    reg = site(table=table)
    check("choose: table wins", reg.choose(4096, 4096, 1), "triton")
    check("choose: unknown shape falls back", reg.choose(12288, 4096, 1), "asm")
    check("choose: hits", reg.hits, {"measured": 1, "probe": 1})
    check("choose: table consulted per call", table.lookups, 2)

    # an explicit switch means the user outranks the table: not even read
    table = Table({(4096, 4096, 1): "triton"})
    reg = site(table=table, env="COMFY_GPU_TUNE_DEMO_ASM")
    os.environ["COMFY_GPU_TUNE_DEMO_ASM"] = "on"
    try:
        check("choose: explicit env beats the table", reg.choose(4096, 4096, 1), "asm")
        check("choose: table untouched", table.lookups, 0)
    finally:
        os.environ.pop("COMFY_GPU_TUNE_DEMO_ASM", None)

    # a table naming an implementation we do not have is a table bug
    reg = site(table=Table({(4096, 4096, 1): "rocmblas"}))
    check("choose: unknown impl ignored", reg.choose(4096, 4096, 1), "asm")
    check("choose: unknown impl counted as probe", reg.hits, {"measured": 0, "probe": 1})

    # ... and so is naming one that cannot run on this device
    reg = site(available=False, table=Table({(4096, 4096, 1): "asm"}))
    check("choose: unavailable impl ignored", reg.choose(4096, 4096, 1), "triton")

    # describe must not load or query the table
    table = Table({(4096, 4096, 1): "triton"})
    reg = site(table=table)
    check("choose: describe table view", reg.describe()["measured"]["records"], 1)
    check("choose: describe does not query", table.lookups, 0)
    auto = kreg.BackendRegistry("demo", [kreg.Candidate("asm", probe=lambda: True),
                                         kreg.Candidate("triton", kind="portable")])
    check("choose: auto table shown as unloaded", auto.describe()["measured"], "unloaded")


def arch_kernels():
    """Kernels dropped into arch/<gcn>/kernels/ join the chain by priority."""
    import tempfile

    from kernels import loader as kloader

    base = [kreg.Candidate("asm", kind="asm", probe=lambda: True, priority=50),
            kreg.Candidate("triton", kind="portable")]

    os.environ.pop("COMFY_GPU_TUNE_ARCH", None)
    os.environ.pop("COMFY_GPU_TUNE_ARCH_KERNELS", None)
    kloader.reset()
    check("loader: no arch finds nothing", [c.name for c in kreg.site_candidates("demo.m1", base)],
          ["asm", "triton"])

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "fast_demo.py").write_text(
            'SITE = "demo.m1"\nNAME = "fast"\nPRIORITY = 10\n'
            'ENV = "COMFY_GPU_TUNE_DEMO_FAST"\nREQUIRES = ("demo",)\n\n'
            "def linear(x):\n    return x\n", encoding="utf-8")
        Path(tmp, "slow_demo.py").write_text(
            'SITE = "demo.m1"\nNAME = "slow"\nPRIORITY = 90\n\n'
            "def linear(x):\n    return x\n", encoding="utf-8")
        Path(tmp, "other_site.py").write_text(
            'SITE = "other.m1"\n\n'
            "def linear(x):\n    return x\n", encoding="utf-8")
        Path(tmp, "broken.py").write_text(
            'SITE = "demo.m1"\nraise RuntimeError("not for this card")\n', encoding="utf-8")
        Path(tmp, "shadow.py").write_text(
            'SITE = "demo.m1"\nNAME = "asm"\n\n'
            "def linear(x):\n    return x\n", encoding="utf-8")

        os.environ["COMFY_GPU_TUNE_ARCH"] = "gfx000"
        os.environ["COMFY_GPU_TUNE_ARCH_KERNELS"] = tmp
        kloader.reset()
        try:
            found = kreg.site_candidates("demo.m1", base)
            check("loader: chain order", [c.name for c in found], ["fast", "asm", "slow", "triton"])
            check("loader: broken file skipped", "broken" in [c.name for c in found], False)
            check("loader: other site ignored", "other_site" in [c.name for c in found], False)
            # a file must not shadow a name the site already declares
            check("loader: duplicate name dropped", [c.name for c in found].count("asm"), 1)
            check("loader: env carried", found[0].env, "COMFY_GPU_TUNE_DEMO_FAST")
            check("loader: requires carried", found[0].requires, ("demo",))
            check("loader: kind is extension", found[0].kind, "extension")
            check("loader: probe absent", found[0].probe, None)

            reg = kreg.BackendRegistry("demo.m1", found)
            check("loader: discovered kernel wins", reg.resolve(), "fast")

            # a file with no NAME falls back to its stem, and its site list is
            # built independently of the demo one
            other = kreg.site_candidates("other.m1", [])
            check("loader: stem as name", [c.name for c in other], ["other_site"])

            # disabling discovery through the arch env must be enough: an arch
            # with no directory falls back to the declared list
            os.environ.pop("COMFY_GPU_TUNE_ARCH_KERNELS", None)
            os.environ["COMFY_GPU_TUNE_ARCH"] = "gfx001"
            kloader.reset()
            check("loader: other arch", [c.name for c in kreg.site_candidates("demo.m1", base)],
                  ["asm", "triton"])
        finally:
            os.environ.pop("COMFY_GPU_TUNE_ARCH", None)
            os.environ.pop("COMFY_GPU_TUNE_ARCH_KERNELS", None)
            kloader.reset()


def main():
    mode_words()
    cases = compare_all()
    probe_order()
    arch_filter()
    describe_states()
    downgrade_rules()
    choose_layer()
    arch_kernels()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for line in FAILURES:
            print(f"  - {line}")
        return 1
    print(f"registry selftest: all checks passed "
          f"(legacy parity over {cases} env x extension x asm combinations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
