"""Stream safetensors directly into VRAM on the legacy --highvram loader.

With DynamicVRAM disabled (--highvram / --gpu-only), ComfyUI builds the text
encoder on the GPU and then copies every checkpoint tensor through host memory
with `load_state_dict`. For a 9B int8 encoder that leaves several GB of host
staging behind and can push a 16GB machine into swap.

comfy-aimdo already has the primitive this needs: `load_safetensors` exposes
each tensor as a file slice, and `read_tensor_file_slice_into` reads that slice
straight into a device tensor. The patch keeps the legacy ModelPatcher and its
resident decode path; it only replaces the host-side copy for quantized Linear
and Embedding weights, which are ~99% of an int8 checkpoint.

Everything here is gated on --highvram, comfy-aimdo being loadable, and the
target module being on CUDA. Any tensor without a file slice, or any failed
device read, falls back to the shipped `load_state_dict` path.
"""

import importlib
import logging
import os

import torch

import comfy.model_management
import comfy.memory_management
import comfy.ops
import comfy.storage
import comfy.utils
from comfy.cli_args import args


_state = {
    "active": False,
    "reason": "not installed",
    "streamed": 0,
    "bytes": 0,
    "fallbacks": 0,
}

_PATCHED = False


def _env_enabled():
    return os.getenv("COMFY_GPU_TUNE_STREAM_LOAD", "1").strip().lower() not in ("0", "off", "false", "no")


def _init_aimdo():
    """Load comfy-aimdo's mmap/file reader even when DynamicVRAM itself is off.

    main.py only calls control.init() when DynamicVRAM is enabled, so on
    --highvram the native library is still unloaded. The submodules bind
    `lib` at import time, which happens before this patch runs; reload them
    after init or their `lib` stays None.
    """
    import comfy_aimdo.control as control
    import comfy_aimdo.host_buffer as host_buffer
    import comfy_aimdo.model_mmap as model_mmap

    if control.lib is None and not control.init():
        return False, "comfy-aimdo library unavailable"
    if host_buffer.lib is None:
        importlib.reload(host_buffer)
    if model_mmap.lib is None:
        importlib.reload(model_mmap)

    devices = [d.index for d in comfy.model_management.get_all_torch_devices()]
    headroom = int(getattr(args, "vram_headroom", 0) * 1024 ** 3)
    try:
        ok = control.init_devices((index, headroom) for index in devices)
    except TypeError:
        ok = control.init_devices(index for index in devices)
    if not ok:
        return False, "comfy-aimdo device init failed"
    return True, None


def _load_torch_file(original):
    def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
        if not _state["active"] or not str(ckpt).lower().endswith((".safetensors", ".sft")):
            return original(ckpt, safe_load, device, return_metadata)
        try:
            sd, metadata = comfy.utils.load_safetensors(ckpt)
        except Exception as exc:
            # A corrupt file must still surface the shipped error path; only
            # aimdo-level failures are worth falling back from.
            logging.warning("gpu-tune: stream load fell back for %s: %s", ckpt, exc)
            return original(ckpt, safe_load, device, return_metadata)
        comfy.storage.annotate_state_dict(sd, ckpt)
        return (sd, metadata) if return_metadata else sd

    load_torch_file._gfx906_stream = True
    return load_torch_file


def _stage(weight, device):
    """Copy one file-slice tensor straight to `device`; None means fall back."""
    dest = torch.empty_like(weight, device=device)
    if comfy.memory_management.read_tensor_file_slice_into(weight, dest):
        _state["streamed"] += 1
        _state["bytes"] += weight.numel() * weight.element_size()
        return dest
    del dest
    _state["fallbacks"] += 1
    return None


def _stage_state_key(state_dict, key, device):
    weight = state_dict.get(key)
    if weight is None or weight.device.type != "cpu":
        return
    dest = _stage(weight, device)
    if dest is not None:
        state_dict[key] = dest


def _load_quantized_module(original):
    def load_quantized_module(module, super_load, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs,
                              load_extra_params=False):
        device = getattr(module, "factory_kwargs", {}).get("device")
        if device is not None:
            device = torch.device(device)
            if device.type == "cuda":
                _stage_state_key(state_dict, prefix + "weight", device)
        return original(module, super_load, state_dict, prefix, local_metadata,
                        strict, missing_keys, unexpected_keys, error_msgs,
                        load_extra_params)

    load_quantized_module._gfx906_stream = True
    return load_quantized_module


def _embedding_load(original):
    def load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                             missing_keys, unexpected_keys, error_msgs):
        weight = getattr(self, "weight", None)
        device = getattr(weight, "device", None)
        if device is None or device.type != "cuda":
            return original(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs)

        _stage_state_key(state_dict, prefix + "weight", device)
        result = original(self, state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs)

        # Unlike the Linear loader, the Embedding loader builds QuantizedTensor
        # params (the per-row scale) on the CPU. With the qdata already on the
        # GPU, .to(device) is a no-op and would leave the scale behind.
        weight = getattr(self, "weight", None)
        params = getattr(weight, "_params", None)
        if getattr(weight, "device", None) == device and params is not None and hasattr(params, "to_device"):
            weight._params = params.to_device(device)
        return result

    load_from_state_dict._gfx906_stream = True
    return load_from_state_dict


def _mixed_precision_ops(original):
    def mixed_precision_ops(*args, **kwargs):
        ops = original(*args, **kwargs)
        # `ops.Embedding` only exists on the factory's own class; patching an
        # inherited manual_cast.Embedding would change every other model too.
        embedding = ops.__dict__.get("Embedding", None)
        if embedding is not None and not getattr(embedding._load_from_state_dict, "_gfx906_stream", False):
            embedding._load_from_state_dict = _embedding_load(embedding._load_from_state_dict)
        return ops

    mixed_precision_ops._gfx906_stream = True
    return mixed_precision_ops


def install():
    global _PATCHED
    if _PATCHED:
        return True
    if not _env_enabled():
        _state["reason"] = "COMFY_GPU_TUNE_STREAM_LOAD=0"
        return False
    if not getattr(args, "highvram", False):
        # The DynamicVRAM loader already reads file slices itself.
        _state["reason"] = "not --highvram"
        return False
    if comfy.memory_management.aimdo_enabled:
        _state["reason"] = "dynamic vram active"
        return False
    if not hasattr(comfy.ops, "_load_quantized_module"):
        _state["reason"] = "core quantized loader unavailable"
        return False

    ok, reason = _init_aimdo()
    if not ok:
        _state["reason"] = reason
        logging.info("gpu-tune: stream load off (%s); --highvram keeps the legacy loader", reason)
        return False

    _state["active"] = True
    comfy.utils.load_torch_file = _load_torch_file(comfy.utils.load_torch_file)
    comfy.ops._load_quantized_module = _load_quantized_module(comfy.ops._load_quantized_module)
    if hasattr(comfy.ops, "mixed_precision_ops"):
        comfy.ops.mixed_precision_ops = _mixed_precision_ops(comfy.ops.mixed_precision_ops)
    _PATCHED = True
    _state["reason"] = "highvram"
    logging.info("gpu-tune: stream load on (safetensors file slices -> VRAM, legacy ModelPatcher kept)")
    return True


def summary():
    if _state["active"]:
        text = (f"on: {_state['streamed']} quantized tensors / "
                f"{_state['bytes'] / 1024 ** 3:.2f}GB direct")
        if _state["fallbacks"]:
            text += f", {_state['fallbacks']} fallback"
        return text
    return f"off ({_state['reason']})"
