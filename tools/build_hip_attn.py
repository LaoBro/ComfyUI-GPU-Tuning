"""Build the node's HIP kernels (attention, int8 GEMM, activation rotation, W4A16 GEMV).

The HIP toolchain lives in the ComfyUI venv (torch's ROCm wheel), not in a
system ROCm install, so the SDK is staged into /tmp where clang expects it.
Re-running is cheap: an up-to-date .so is left alone unless --force is given.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS = os.path.join(ROOT, "arch", "gfx906", "kernels")
TARGETS = [("hip_attn.cu", "libgfx906_attn.so"), ("int8_gemm.cu", "libgfx906_int8.so"),
           ("rotate.cu", "libgfx906_rotate.so"), ("w4a16.cu", "libgfx906_w4a16.so")]
STAGE = "/tmp/hip_rocm"


def find_hipcc():
    candidates = [os.path.join(sys.prefix, "bin", "hipcc"), "hipcc"]
    for path in candidates:
        hit = shutil.which(path) if os.sep not in path else path
        if hit and os.path.exists(hit):
            return hit
    return None


def stage_sdk():
    """Link the SDK into the layout clang's ROCm driver looks for."""
    sdk = None
    for path in glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages",
                                       "_rocm_sdk_core")):
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


def missing():
    """Kernel libraries not built yet."""
    return [output for _, output in TARGETS if not os.path.exists(os.path.join(KERNELS, output))]


def build(arch="gfx906", force=False):
    hipcc = find_hipcc()
    if hipcc is None:
        print("hipcc not found; run this with the ComfyUI venv python", file=sys.stderr)
        return 1
    rocm = stage_sdk()
    for source, output in TARGETS:
        src, so = os.path.join(KERNELS, source), os.path.join(KERNELS, output)
        if not os.path.exists(src):
            print(f"missing source: {src}", file=sys.stderr)
            return 1
        if not force and os.path.exists(so) and os.path.getmtime(so) >= os.path.getmtime(src):
            print(f"up to date: {so}")
            continue
        cmd = [hipcc, "-O3", "-std=c++17", f"--offload-arch={arch}",
               "-shared", "-fPIC", src, "-o", so, "-lamdhip64"]
        if rocm:
            cmd[4:4] = [f"--rocm-path={rocm}"]
            cmd += [f"-L{os.path.join(rocm, 'lib')}"]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"built {so}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    parser.add_argument("--arch", default="gfx906")
    args = parser.parse_args()
    return build(args.arch, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
