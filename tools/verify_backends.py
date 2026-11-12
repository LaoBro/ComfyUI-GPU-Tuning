"""Verify the kernel registry on the real device.

Run it with the ComfyUI interpreter, on the machine that has the GPU:

    /home/liu/ComfyUI-0.39.0/.venv/bin/python \
        custom_nodes/ComfyUI-GPU-Tuning/tools/verify_backends.py

Three things are checked, in order of how much they can lie to you:

1. selection  -- does each site pick the backend the pre-registry logic would
                 have picked, for every env value? (parity, computed against
                 the legacy reference kept in registry_selftest.py)
2. numerics   -- do the asm path and the portable path agree with each other
                 and with a plain torch dequantisation of the same weights?
3. summary    -- what `status()` will print, so it can be compared with the
                 output captured before the refactor.

Exit code 0 only when every check passes. Nothing is patched: the script calls
the kernels directly, so it is safe to run next to a live ComfyUI process.
"""

import os
import sys
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NODE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Same pin as the pack's __init__.py, which these top-level imports bypass.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch                                                           # noqa: E402
import refs                                                            # noqa: E402

FAILURES = []


def check(label, got, want, note=""):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got!r}" + (f"   {note}" if note and not ok else ""))
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")
    return ok


def rel(a, b):
    """Max relative error against b, so one number per comparison."""
    return refs.rel(a, b)


# -- reference implementations ------------------------------------------

int8_reference = refs.int8_reference
w4a8_weight = refs.w4a8_weight


# -- checks --------------------------------------------------------------

def environment():
    import torch
    import triton

    props = torch.cuda.get_device_properties(0)
    arch = str(getattr(props, "gcnArchName", "")).split(":")[0] or f"sm{props.major}{props.minor}"
    print(f"torch {torch.__version__} | hip {torch.version.hip} | triton {triton.__version__}")
    print(f"device0 {props.name} | {arch} | {props.multi_processor_count} CU")
    return arch


def selection(w4a8, int8, legacy):
    """Every env value must resolve the same way the old code did."""
    def asm_ok(module):
        try:
            module._probe_asm()
            return True
        except Exception as exc:                                # noqa: BLE001
            print(f"     {module.__name__}: asm probe failed ({type(exc).__name__}: {exc})")
            return False

    hip = w4a8._load_hip() is not None
    w4a8_asm = asm_ok(w4a8)
    int8_asm = asm_ok(int8)
    print(f"     hip extension: {'present' if hip else 'absent'} | "
          f"w4a8 asm: {'compiles' if w4a8_asm else 'no'} | int8 asm: {'compiles' if int8_asm else 'no'}")

    envs = [None, "", "auto", "on", "asm", "1", "off", "triton", "0", "nonsense"]
    for module, env_name, name, want_fn in (
        (w4a8, "COMFY_GPU_TUNE_W4A8_ASM", "w4a8", lambda e: legacy.legacy_w4a8(e, hip, w4a8_asm, [])[0]),
        (int8, "COMFY_GPU_TUNE_INT8_GEMV_ASM", "int8_gemv", lambda e: legacy.legacy_int8(e, int8_asm, [])[0]),
    ):
        for env in envs:
            if env is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = env
            module._REGISTRY.reset()
            check(f"{name} env={env!r}", module.backend_name(), want_fn(env))
        os.environ.pop(env_name, None)


def numeric_int8(int8_gemv):
    import torch

    torch.manual_seed(0)
    n, k = 4096, 4096
    weight = torch.randint(-127, 127, (n, k), device="cuda", dtype=torch.int8)
    scale = torch.rand(n, device="cuda", dtype=torch.float32) * 0.01 + 1e-4
    x = torch.randn(1, k, device="cuda", dtype=torch.float32)

    out_asm = int8_gemv._linear_asm(x, weight, scale, None, torch.float32, 0)
    out_tri = int8_gemv._linear_triton(x, weight, scale, None, torch.float32, 0)
    ref = int8_reference(x, weight, scale)
    if out_asm is None or out_tri is None:
        FAILURES.append("int8_gemv: an implementation declined the shape")
        print("FAIL int8_gemv: implementation declined the shape")
        return
    d_ab, d_ref = rel(out_asm, out_tri), rel(out_asm, ref)
    print(f"     int8 4096x4096 m=1: asm vs triton {d_ab:.2e}, asm vs torch {d_ref:.2e}")
    check("int8 asm == triton", d_ab < 1e-6, True)
    # the gap to torch is the int8 activation quantisation, not a kernel bug
    check("int8 asm ~= torch (3%)", d_ref < 0.03, True)


def numeric_w4a8(w4a8):
    import torch

    torch.manual_seed(1)
    n, k, gs = 4096, 4096, 128
    qdata = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    s_rel = torch.rand(n, k // gs, device="cuda", dtype=torch.float32) * 0.02 + 1e-4
    s_channel = torch.rand(n, device="cuda", dtype=torch.float32) * 0.5 + 0.5
    weight = w4a8_weight(qdata, s_rel, s_channel, gs)

    for m in (1, 8):
        x = torch.randn(m, k, device="cuda", dtype=torch.float32)
        out_asm = w4a8._linear_asm(x, qdata, s_rel, s_channel, None, torch.float32, gs, 0)
        out_tri = w4a8._linear_dot(x, qdata, s_rel, s_channel, None, torch.float32, gs, 0)
        out_sum = w4a8._linear_sum(x, qdata, s_rel, s_channel, None, torch.float32, gs, 0)
        ref = x @ weight.T
        if out_asm is None or out_tri is None or out_sum is None:
            FAILURES.append(f"w4a8: an implementation declined m={m}")
            print(f"FAIL w4a8 m={m}: implementation declined the shape")
            continue
        d_ab, d_ref, d_sum = rel(out_asm, out_tri), rel(out_asm, ref), rel(out_sum, ref)
        print(f"     w4a8 4096x4096 m={m}: asm vs triton {d_ab:.2e}, asm vs torch {d_ref:.2e}, "
              f"sum vs torch {d_sum:.2e}")
        check(f"w4a8 m={m} asm == triton", d_ab < 1e-5, True)
        check(f"w4a8 m={m} asm ~= torch (3%)", d_ref < 0.03, True)
        check(f"w4a8 m={m} sum ~= torch (3%)", d_sum < 0.03, True)

    # group 8: the tl.dot kernel cannot take it, the floor is what keeps the
    # site from declining the shape and sending the call back to int8
    gs8 = 8
    s_rel8 = torch.rand(n, k // gs8, device="cuda", dtype=torch.float32) * 14.5 + 1.0
    weight8 = w4a8_weight(qdata, s_rel8, s_channel, gs8)
    x = torch.randn(1, k, device="cuda", dtype=torch.float32)
    out_dot = w4a8._linear_dot(x, qdata, s_rel8, s_channel, None, torch.float32, gs8, 0)
    out_sum = w4a8._linear_sum(x, qdata, s_rel8, s_channel, None, torch.float32, gs8, 0)
    detail = "declined" if out_sum is None else f"rel {rel(out_sum, x @ weight8.T):.2e}"
    print(f"     w4a8 group 8: dot {'declines' if out_dot is None else 'accepts'}, sum {detail}")
    check("w4a8 group 8 declined by tl.dot", out_dot is None, True)
    check("w4a8 group 8 floor ~= torch (3%)",
          out_sum is not None and rel(out_sum, x @ weight8.T) < 0.03, True)


def summary(w4a8, int8):
    for module, label in ((int8, "gemv"), (w4a8, "w4a8")):
        print(f"     status(): {label} on (backend={module.backend_name()})")
        print(f"     describe(): {module._REGISTRY.describe()}")


def main():
    import registry_selftest as legacy
    import torch

    if not torch.cuda.is_available():
        print("no device: torch.cuda.is_available() is False")
        print("if you are in a container, /dev/kfd must be mapped; on the host check `rocm-smi`")
        return 2

    environment()
    from kernels.triton import int8_gemv
    from kernels.triton import w4a8_gemv

    print("\n[1] selection parity with the pre-registry logic")
    selection(w4a8_gemv, int8_gemv, legacy)
    print("\n[2] numerics")
    numeric_int8(int8_gemv)
    numeric_w4a8(w4a8_gemv)
    print("\n[3] what status() will print")
    summary(w4a8_gemv, int8_gemv)

    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for line in FAILURES:
            print(f"  - {line}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
