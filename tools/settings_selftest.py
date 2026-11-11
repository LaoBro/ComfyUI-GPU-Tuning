"""Self-test for the node switch layer: three-state resolution and routing.

    .venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/settings_selftest.py

Needs the ComfyUI venv (gpu_tuning imports torch) but no GPU: every case drives
configure()/select_* with fake kernels and tensors on the CPU, so no kernel runs
and nothing outside this process changes.
"""

import contextlib
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

NODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NODE_DIR))

import gpu_tuning as tuning  # noqa: E402

FAILURES = []
BASE = copy.deepcopy(tuning._state)


def check(label, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


@contextlib.contextmanager
def fresh():
    """Start from the post-import state, with no COMFY_GPU_TUNE_* in the env."""
    saved = {key: value for key, value in os.environ.items() if key.startswith("COMFY_GPU_TUNE")}
    for key in saved:
        os.environ.pop(key)
    tuning._state = copy.deepcopy(BASE)
    try:
        yield tuning
    finally:
        for key in [key for key in os.environ if key.startswith("COMFY_GPU_TUNE")]:
            os.environ.pop(key)
        os.environ.update(saved)
        tuning._state = copy.deepcopy(BASE)


@contextlib.contextmanager
def patched(**attrs):
    """Replace module attributes for one case (fake kernels, no kitchen state)."""
    old = {name: getattr(tuning, name) for name in attrs}
    for name, value in attrs.items():
        setattr(tuning, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(tuning, name, value)


# configure() re-enables the kitchen backend on every call, which is state the
# switch tests neither exercise nor want to depend on.
NO_KITCHEN = {"enable_triton_backend": lambda: None}


def textgen_group():
    """The env sets the per-item defaults; the node switch overrides all four."""
    items = tuning.TEXTGEN_ITEMS
    with fresh():
        os.environ.update(COMFY_GPU_TUNE_DELTANET="0", COMFY_GPU_TUNE_KVDECODE="1",
                          COMFY_GPU_TUNE_LLAMA="no", COMFY_GPU_TUNE_QWEN35="yes")
        tuning._init_env_settings()
        check("textgen env defaults", [tuning.textgen_requested(name) for name in items],
              [False, True, False, True])

        with patched(**NO_KITCHEN):
            tuning.configure(textgen="off")
            check("textgen off beats every env", [tuning.textgen_requested(name) for name in items],
                  [False, False, False, False])
            tuning.configure(textgen="on")
            check("textgen on beats every env", [tuning.textgen_requested(name) for name in items],
                  [True, True, True, True])
            tuning.configure(textgen="default")
            check("textgen default returns to env", [tuning.textgen_requested(name) for name in items],
                  [False, True, False, True])


def tile_specs():
    """COMFY_GPU_TUNE_TILE_CONFIG: the three block sizes are required."""
    name = "COMFY_GPU_TUNE_TILE_CONFIG"
    with fresh():
        os.environ[name] = "block_m=64,block_n=128,block_k=64,group_size_m=8,warps=4,stages=3"
        check("tile spec full", tuning._tile_spec(name),
              ({"block_m": 64, "block_n": 128, "block_k": 64, "group_size_m": 8}, 4, 3))
        os.environ[name] = "block_m=16,block_n=32,block_k=64"
        check("tile spec defaults", tuning._tile_spec(name),
              ({"block_m": 16, "block_n": 32, "block_k": 64, "group_size_m": 8}, 4, 2))
        os.environ[name] = "block_m=64,block_n=128"
        check("tile spec missing block_k", tuning._tile_spec(name), None)
        os.environ[name] = "bogus"
        check("tile spec garbage", tuning._tile_spec(name), None)
        os.environ.pop(name)
        check("tile spec unset", tuning._tile_spec(name), None)


def small_batch_buckets():
    """m <= 8 offers the narrow tile, larger m the large one, switch off freezes it."""
    kernel = SimpleNamespace(configs=["stock"], cache={})
    with fresh(), patched(_int8_kernels=lambda: [kernel],
                          _small_tile_config=lambda: "small",
                          _triton_config=lambda: "large", **NO_KITCHEN):
        tuning._state.update(int8_tile_switch=True, int8_tile_replaced=True, int8_tile_small=None)
        tuning.select_int8_tile(1)
        check("small bucket at m=1", kernel.configs, ["small"])
        tuning.select_int8_tile(4096)
        check("large bucket at m=4096", kernel.configs, ["large"])

        tuning.configure(small_batch="off")
        tuning.select_int8_tile(1)
        check("small_batch off keeps the large tile", kernel.configs, ["large"])
        check("small_batch off is state", tuning._state["small_batch_enabled"], False)

        tuning.configure(small_batch="default")
        tuning.select_int8_tile(1)
        check("small_batch default re-arms the bucket", kernel.configs, ["small"])
        check("small_batch default is state", tuning._state["small_batch_enabled"], True)


def w4a8_switch():
    """on/off move the module reference; off keeps it cached for the next on."""
    kernel = object()
    with fresh(), patched(load_w4a8_gemv=lambda: kernel, **NO_KITCHEN):
        tuning.configure(w4a8="on")
        check("w4a8 on", tuning._state["w4a8"], kernel)
        tuning.configure(w4a8="off")
        check("w4a8 off", tuning._state["w4a8"], None)
        check("w4a8 module stays cached", tuning._state["w4a8_module"], kernel)
        tuning.configure(w4a8="default")
        check("w4a8 default reuses the cache", tuning._state["w4a8"], kernel)


def plain_gemv():
    """text_gemv off sends single-row text projections back to the dispatched linear."""

    class Linear:
        weight = torch.zeros(4, 4, dtype=torch.float16)
        bias = None
        weight_function = []
        bias_function = []

    x = torch.zeros(1, 4)
    with fresh(), patched(**NO_KITCHEN):
        out = tuning.te_plain_gemv(Linear(), x)
        check("text_gemv returns one row", tuple(out.shape), (1, 4))
        check("text_gemv counted", tuning._state["te_plain_calls"], 1)

        tuning.configure(text_gemv="off")
        check("text_gemv off returns None", tuning.te_plain_gemv(Linear(), x), None)
        check("text_gemv off is state", tuning._state["text_gemv_enabled"], False)

        tuning.configure(text_gemv="default")
        check("text_gemv default is state", tuning._state["text_gemv_enabled"], True)


def guarded_gemv():
    """select_int8_gemv only takes the decode shape; everything else keeps the GEMM."""
    calls = []

    def linear(*args):
        calls.append(args)
        return "gemv"

    weight = torch.zeros(2, 4, dtype=torch.int8)
    scale = torch.ones(2, 1)
    with fresh():
        tuning._state["gemv_module"] = SimpleNamespace(linear=linear)
        tail = (scale, None, torch.float32, None, None, 0)
        check("gemv takes one fp32 row", tuning.select_int8_gemv(torch.zeros(1, 4), weight, *tail), "gemv")
        check("gemv called once", len(calls), 1)
        check("gemv leaves prefill alone",
              tuning.select_int8_gemv(torch.zeros(2, 4), weight, *tail), None)
        check("gemv needs fp32 activations",
              tuning.select_int8_gemv(torch.zeros(1, 4, dtype=torch.float16), weight, *tail), None)
        check("gemv needs an int8 weight",
              tuning.select_int8_gemv(torch.zeros(1, 4), weight.float(), *tail), None)
        tuning._state["gemv_module"] = None
        check("gemv off", tuning.select_int8_gemv(torch.zeros(1, 4), weight, *tail), None)
        check("no extra kernel calls", len(calls), 1)


def switch_states():
    """Every three-state switch resolves default to the built-in, never to a stale on."""
    kernel = object()
    with fresh(), patched(load_int8_gemv=lambda: kernel, **NO_KITCHEN):
        tuning.configure(amax_guard="off")
        check("amax_guard off", tuning._state["amax_guard"], False)
        tuning.configure(amax_guard="default")
        check("amax_guard default", tuning._state["amax_guard"], True)

        with patched(_int8_kernels=lambda: []):
            tuning.configure(int8_tile="off")
            check("int8_tile off", tuning._state["int8_tile_enabled"], False)
            tuning.configure(int8_tile="default")
            check("int8_tile default", tuning._state["int8_tile_enabled"], True)

        tuning.configure(prefix_kv="off")
        check("prefix_kv off", tuning._state["prefix_kv_enabled"], False)
        tuning.configure(prefix_kv="default")
        check("prefix_kv default", tuning._state["prefix_kv_enabled"], True)

        tuning.configure(vae="off")
        check("vae off", tuning._state["vae_enabled"], False)
        tuning.configure(vae="on")
        check("vae on", tuning._state["vae_enabled"], True)

        tuning.configure(gemv="off")
        check("gemv off", tuning._state["gemv_model"], False)
        tuning.configure(gemv="on")
        check("gemv on", tuning._state["gemv_model"], True)
        check("gemv module loaded", tuning._state["gemv_module"], kernel)


def main():
    textgen_group()
    tile_specs()
    small_batch_buckets()
    w4a8_switch()
    plain_gemv()
    guarded_gemv()
    switch_states()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for line in FAILURES:
            print(f"  - {line}")
        return 1
    print("settings selftest: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
