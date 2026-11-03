"""Self-test for the measurement manifest: parsing, trust, merge, fallback.

    python custom_nodes/ComfyUI-gfx906-Tuning/tools/manifest_selftest.py

No GPU and no torch: everything here is file parsing plus the fingerprint rules,
which is exactly the part that decides whether a number is allowed to be used.
"""

import os
import sys
import tempfile
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NODE_DIR))

from kernels import manifest as mf                                  # noqa: E402
from kernels import manifest_write as mw                            # noqa: E402

FAILURES = []
FAKE_RT = "rocmlike|triton9|gfx000"


def check(label, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def write(root, arch, text):
    path = Path(root) / "arch" / arch / "manifest.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


DEMO = """
schema = 1

[arch]
gcn = "gfx000"
fingerprint = "rocmlike|triton9|gfx000"
capabilities = ["demo"]

[sites."demo.m1"]
default = "asm"
default_grade = "measured"

[sites."demo.m1".shapes]
"64x128@m1" = { impl = "triton", ms = { asm = 0.9, triton = 0.5 }, rel = 0.0, grade = "measured", date = "2026-11-03" }
"64x128@*"  = { impl = "asm", grade = "measured" }
"32x32@m1"  = { impl = "asm", grade = "assumed" }

[sites."empty.m1"]
default = "triton"
default_grade = "measured"

[sites."guess.m1"]
default = "asm"
"""


def shipped():
    """The file in the repo must parse and be trusted on every build of the arch.

    It carries grade = "measured" records and no fingerprint: the node reads its
    tier-3 gates and tiles straight from here, so trust is the point. A local
    bench file (with a real fingerprint) wins per shape on top of it.
    """
    # read the shipped file alone: a local override on the test machine must not
    # change what this checks
    path = Path(NODE_DIR) / "arch" / "gfx906" / "manifest.toml"
    data = mf.load("gfx906", path=path)
    check("shipped: loads", data is not None, True)
    check("shipped: has records", data.records() > 0, True)
    check("shipped: fingerprint", data.fingerprint, mf.UNMEASURED)
    check("shipped: trusted on any build", data.matches(), True)
    check("shipped: lookup live", data.lookup("w4a8.m1", 4096, 4096, 1), "asm")
    check("shipped: line says measured", "measured" in data.status_line(), True)
    # the tier-3 gates the node reads must all answer here
    for site in ("feature.gemv", "feature.w4a8", "feature.vae", "feature.convrot",
                 "convrot.mode"):
        check(f"shipped: {site} default",
              data.default_impl(site) in ("on", "off", "fused", "butterfly"), True)
    check("shipped: int8 tile large",
          data.config("int8_gemm.tile", "large") is not None, True)


def trust_and_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        path = write(tmp, "gfx000", DEMO)
        data = mf.load("gfx000", path=path, runtime=FAKE_RT)
        check("demo: trusted", data.matches(), True)
        check("demo: exact shape", data.lookup("demo.m1", 64, 128, 1), "triton")
        check("demo: wildcard for other m", data.lookup("demo.m1", 64, 128, 8), "asm")
        check("demo: assumed is ignored", data.lookup("demo.m1", 32, 32, 1), None)
        check("demo: assumed opt-in", data.lookup("demo.m1", 32, 32, 1, min_grade="assumed"), "asm")
        # a default that contradicts a measured shape must not be applied
        check("demo: contradictory default dropped", data.lookup("demo.m1", 1, 1, 1), None)
        check("demo: consistent default used", data.lookup("empty.m1", 1, 1, 1), "triton")
        # a default with no grade is a guess and cannot bypass the filter
        check("demo: ungraded default ignored", data.lookup("guess.m1", 1, 1, 1), None)
        check("demo: ungraded default opt-in",
              data.lookup("guess.m1", 1, 1, 1, min_grade="assumed"), "asm")
        check("demo: line says in use", "in use" in data.status_line(), True)

        other = mf.load("gfx000", path=path, runtime="rocmother|triton8|gfx000")
        check("mismatch: not trusted", other.matches(), False)
        check("mismatch: lookup none", other.lookup("demo.m1", 64, 128, 1), None)
        check("mismatch: line says STALE", "STALE" in other.status_line(), True)
        check("mismatch: describe", other.describe()["matches"], False)


def merge():
    """Shipped file and local override merge per shape, override wins."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        write(repo, "gfx000", DEMO)
        override = write(tmp, "gfx000", """
schema = 1

[arch]
gcn = "gfx000"
fingerprint = "rocmlike|triton9|gfx000"

[sites."demo.m1".shapes]
"64x128@m1" = { impl = "asm", grade = "measured", date = "2026-11-04" }
"256x256@m1" = { impl = "triton", grade = "measured" }
""")
        original, os.environ["COMFY_GFX906_MANIFEST_DIR"] = (
            mf.node_dir, str(tmp))
        mf.node_dir = lambda: repo
        try:
            data = mf.load("gfx000", runtime=FAKE_RT)
        finally:
            mf.node_dir = original
            os.environ.pop("COMFY_GFX906_MANIFEST_DIR")
        check("merge: loads", data is not None, True)
        check("merge: override wins on the same shape", data.lookup("demo.m1", 64, 128, 1), "asm")
        check("merge: new shape kept", data.lookup("demo.m1", 256, 256, 1), "triton")
        check("merge: untouched shape kept", data.lookup("demo.m1", 64, 128, 8), "asm")
        check("merge: other site kept", data.lookup("empty.m1", 1, 1, 1), "triton")
        check("merge: source is the override", str(override), str(data.source))


def bad_input():
    with tempfile.TemporaryDirectory() as tmp:
        broken = write(tmp, "gfx001", "schema = 1\n[arch\n")
        check("broken toml: none", mf.load("gfx001", path=broken), None)

        future = write(tmp, "gfx002", 'schema = 99\n[arch]\ngcn = "gfx002"\n')
        check("new schema: ignored", mf.load("gfx002", path=future), None)

        wrong = write(tmp, "gfx003", 'schema = 1\n[arch]\ngcn = "gfx906"\n')
        check("other arch: ignored", mf.load("gfx003", path=wrong), None)

        partial = write(tmp, "gfx004", """
schema = 1
[arch]
gcn = "gfx004"
fingerprint = "rocmlike|triton9|gfx004"
[sites."demo.m1".shapes]
"good@m1"   = { impl = "asm", grade = "measured" }
"noimpl@m1" = { grade = "measured" }
"badms@m1"  = { impl = "asm", ms = { asm = "fast" }, grade = "measured" }
"badgrade@m1" = { impl = "asm", grade = "vibes" }
""")
        data = mf.load("gfx004", path=partial, runtime="rocmlike|triton9|gfx004")
        check("partial: record count", data.records(), 3)      # noimpl dropped
        check("partial: bad grade downgraded", data.sites["demo.m1"].shapes["badgrade@m1"].grade, "assumed")
        check("partial: bad ms dropped", data.sites["demo.m1"].shapes["badms@m1"].ms, {})
        check("partial: trusted", data.matches(), True)
        check("partial: unknown shape", data.lookup("demo.m1", 1, 1, 1), None)


def fingerprint_shape():
    fp = mf.runtime_fingerprint("gfx906")
    parts = fp.split("|")
    check("fingerprint: four parts", len(parts), 4)
    check("fingerprint: arch last", parts[-1], "gfx906")
    check("fingerprint: starts with rocm", parts[0].startswith("rocm"), True)
    check("fingerprint: carries the torch build", parts[1].startswith("torch"), True)
    check("fingerprint: carries triton", parts[2].startswith("triton"), True)


def writer():
    """bench.py edits the shipped text; the result must load and answer lookups."""
    fp = "rocmlike|triton9|gfx906"
    with tempfile.TemporaryDirectory() as tmp:
        source = (Path(NODE_DIR) / "arch" / "gfx906" / "manifest.toml").read_text(encoding="utf-8")
        text = mw.set_arch_identity(source, "gfx906", fp)
        text = mw.set_shapes(text, "int8_gemv.m1", {
            "4096x4096@m1": mw.render("4096x4096@m1", "asm", {"asm": 0.07, "triton": 0.09},
                                      rel=1e-3, date="2026-11-03", note="convrot off"),
            "8192x8192@m1": mw.render("8192x8192@m1", "asm", {"asm": 0.5, "triton": 0.6},
                                      rel=1e-3, date="2026-11-03"),
        })
        path = Path(tmp) / "arch" / "gfx906" / "manifest.toml"
        mw.save(path, text)
        raw = path.read_text(encoding="utf-8")
        check("writer: comment kept", "measurement manifest" in raw, True)
        check("writer: other site untouched", '"12288x4096@m8"' in raw, True)
        check("writer: replaced not duplicated", raw.count("asm = 0.07"), 1)

        data = mf.load("gfx906", path=path, runtime=fp)
        check("writer: loads", data is not None, True)
        check("writer: fingerprint stamped", data.fingerprint, fp)
        check("writer: trusted", data.matches(), True)
        check("writer: replaced record wins", data.lookup("int8_gemv.m1", 4096, 4096, 1), "asm")
        check("writer: new record wins", data.lookup("int8_gemv.m1", 8192, 8192, 1), "asm")
        check("writer: shipped measured record trusted", data.lookup("w4a8.m1", 4096, 4096, 1), "asm")
        source_records = mf.load("gfx906", path=Path(NODE_DIR) / "arch" / "gfx906" / "manifest.toml").records()
        check("writer: record count", data.records(), source_records + 1)   # + the new shape

        bare = mw.set_arch_identity("schema = 1\n", "gfx906", fp)
        check("writer: arch block created", "[arch]" in bare, True)
        check("writer: stamp on bare", 'fingerprint = "' + fp + '"' in bare, True)

        # promoting the site default: every measured shape agreed, so the same
        # answer may stand in for the shapes we did not measure
        promoted = mw.set_default(text, "w4a8.m1", "asm")
        path2 = Path(tmp) / "promoted.toml"
        mw.save(path2, promoted)
        data2 = mf.load("gfx906", path=path2, runtime=fp)
        check("writer: default promoted", data2.sites["w4a8.m1"].default_grade, "measured")
        check("writer: default answers unseen shape", data2.lookup("w4a8.m1", 777, 777, 1), "asm")

        # a fresh file: default block has to be created above the shapes table
        skel = mw.skeleton("gfx906", fp)
        skel = mw.set_shapes(skel, "demo.m1", {
            "64x64@m1": mw.render("64x64@m1", "triton", {"triton": 0.5}, rel=1e-3),
        })
        skel = mw.set_default(skel, "demo.m1", "triton")
        path3 = Path(tmp) / "skeleton.toml"
        mw.save(path3, skel)
        data3 = mf.load("gfx906", path=path3, runtime=fp)
        check("writer: skeleton default", data3.lookup("demo.m1", 1, 1, 1), "triton")
        check("writer: skeleton shape", data3.lookup("demo.m1", 64, 64, 1), "triton")

        try:
            mw.save(Path(tmp) / "broken.toml", "schema = 1\n[arch\n")
        except ValueError:
            check("writer: refuses broken text", True, True)
        else:
            check("writer: refuses broken text", False, True)
        check("writer: nothing written on refusal", (Path(tmp) / "broken.toml").exists(), False)


def main():
    fingerprint_shape()
    shipped()
    trust_and_fallback()
    merge()
    bad_input()
    writer()
    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for line in FAILURES:
            print(f"  - {line}")
        return 1
    print("\nmanifest selftest: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
