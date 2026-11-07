"""Build the node's GPU kernels for this platform and target.

One library per (target, logical name) goes into arch/<target>/kernels/. The
loader derives the same filename from the running target, so a checkout can
carry libraries built on several platforms side by side. Linux keeps the
shipped names (libgfx906_attn.so); Windows uses gfx906_attn.dll.

On Linux the HIP toolchain lives in the ComfyUI venv (torch's ROCm wheel), not
in a system ROCm install, so the SDK is staged into a temp dir where clang
expects it. Windows resolves the SDK through HIP_PATH instead. Re-running is
cheap: an up-to-date library is left alone unless --force is given.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kernels import runtime                                          # noqa: E402

TARGETS = [("hip_attn.cu", "attn"), ("hip_attn_i8.cu", "attn_i8"),
           ("attn_quant.cu", "attn_quant"), ("int8_gemm.cu", "int8"),
           ("rotate.cu", "rotate"), ("w4a16.cu", "w4a16")]
STAGE = os.path.join(tempfile.gettempdir(), "hip_rocm")
EXPORT_RE = re.compile(r'extern\s+"C"\s+int\s+(\w+)\s*\(')


def find_hipcc():
    candidates = [os.path.join(sys.prefix, "bin", "hipcc"), "hipcc"]
    for path in candidates:
        hit = shutil.which(path) if os.sep not in path else path
        if hit and os.path.exists(hit):
            return hit
    return None


def stage_sdk():
    """Link the SDK into the layout clang's ROCm driver looks for."""
    if runtime.WINDOWS:
        return None                       # the HIP SDK exposes HIP_PATH instead
    sdk = None
    for path in glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "_rocm_sdk_core")):
        sdk = path
        break
    if sdk is None:
        return None                       # a system ROCm install needs no staging
    os.makedirs(os.path.join(STAGE, "lib"), exist_ok=True)
    for name in ("include", "amdgcn"):
        target = os.path.join(STAGE, name)
        source = os.path.join(sdk, "include" if name == "include" else "lib/llvm/amdgcn")
        if not os.path.islink(target):
            os.symlink(source, target)
    for lib in glob.glob(os.path.join(sdk, "lib", "*")):
        target = os.path.join(STAGE, "lib", os.path.basename(lib))
        if not os.path.lexists(target):
            os.symlink(lib, target)
    so7 = os.path.join(sdk, "lib", "libamdhip64.so.7")
    if os.path.exists(so7) and not os.path.lexists(os.path.join(STAGE, "lib", "libamdhip64.so")):
        os.symlink(so7, os.path.join(STAGE, "lib", "libamdhip64.so"))
    return STAGE


def missing(arch="gfx906"):
    """Libraries not built yet for this target."""
    directory = runtime.kernel_dir(arch)
    return [runtime.library_name(arch, name) for _, name in TARGETS
            if not (directory / runtime.library_name(arch, name)).exists()]


def build(arch="gfx906", force=False):
    if not re.fullmatch(r"[A-Za-z0-9_]+", arch):
        print(f"invalid target: {arch}", file=sys.stderr)
        return 1
    hipcc = find_hipcc()
    if hipcc is None:
        print("hipcc not found; run this with the ComfyUI venv python", file=sys.stderr)
        return 1
    rocm = stage_sdk()
    directory = runtime.kernel_dir(arch)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for source, name in TARGETS:
            src = directory / source
            out = directory / runtime.library_name(arch, name)
            if not src.exists():
                print(f"missing source: {src}", file=sys.stderr)
                return 1
            if not force and out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
                print(f"up to date: {out}")
                continue
            cmd = [hipcc, "-O3", "-std=c++17", f"--offload-arch={arch}"]
            if rocm:
                cmd.append(f"--rocm-path={rocm}")
            cmd.append("-shared")
            if not runtime.WINDOWS:
                cmd.append("-fPIC")
            cmd += [str(src), "-o", str(out)]
            if runtime.WINDOWS:
                symbols = EXPORT_RE.findall(src.read_text())
                if not symbols:
                    print(f'no extern "C" entry points in {src}', file=sys.stderr)
                    return 1
                def_path = Path(tmp) / f"{arch}_{name}.def"
                def_path.write_text("EXPORTS\n" + "\n".join(f"    {s}" for s in symbols) + "\n")
                hip_path = os.getenv("HIP_PATH")
                if hip_path:
                    cmd.append(f"-L{os.path.join(hip_path, 'lib')}")
                cmd += [f"-Wl,/DEF:{def_path}", "-lamdhip64"]
            else:
                cmd.append("-lamdhip64")
            if rocm:
                cmd.append(f"-L{os.path.join(rocm, 'lib')}")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            print(f"built {out}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    parser.add_argument("--arch", default="gfx906")
    args = parser.parse_args()
    return build(args.arch, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
