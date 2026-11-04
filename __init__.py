"""ComfyUI node pack: runtime tuning for gfx906 (MI50 / MI60 / Radeon VII).

Applies the patches on startup so an unmodified ComfyUI gets the tuned int8
linear tiles, the butterfly convrot rotation and the fp16 attention kernel.
Two settings nodes expose the runtime switches (one in the model chain, one in
the CLIP chain); the model node's profile switch additionally reports what was
applied and prints per-prompt timings.
"""

import logging
import os

# Triton 3.9 warns once per process when this knob is unset, and 3.10 flips the
# default to disabled. Every measurement in gfx906fix assumed fusion enabled,
# so pin it before the first kernel compiles; an explicit env still wins.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .gfx906_tuning import (
    ATTENTION_KERNELS,
    CONVROT_MODES,
    TRISTATE,
    apply,
    backend_options,
    configure,
    report_lines,
    status,
)
from .profiler import install as install_profiler
from .profiler import watch as watch_profiler


class Gfx906ModelSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gfx906ModelSettings",
            display_name="GFX906 Model Settings",
            category="advanced/debug",
            description=(
                "Runtime settings for the gfx906 tuning pack. Values take effect on the next execution, "
                "no restart needed. Wire it between the model loader and the sampler. Every switch is "
                "three-state: default keeps the built-in default, on/off overrides it."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("attention", options=["default"] + list(ATTENTION_KERNELS) + backend_options(),
                               default="default",
                               tooltip="default keeps the built-in int8 kernel; int8qk keeps PV "
                                       "in fp16 (slower, 4-7x more accurate); gemm is fp16 built from rocBLAS bmm + "
                                       "softmax + bmm, ~1.45x faster than hand at DiT shapes and the fastest fp16 "
                                       "path on gfx906; hand is the fp16 Triton kernel. basic, pytorch, sub_quad, "
                                       "split and any other backend core registered are ComfyUI's own functions."),
                io.Combo.Input("contiguous", options=list(TRISTATE), default="default",
                               tooltip="When a backend name is selected: transpose to [B,H,S,D] and "
                                       ".contiguous() before calling it. Core hands SDPA/bmm transpose views "
                                       "instead, which measures 4-15x slower on gfx906. Turn off to reproduce "
                                       "core's exact call for comparison on other GPUs. No effect on the "
                                       "int8/int8qk/hand kernels (they build their own layouts)."),
                io.Combo.Input("smooth", options=list(TRISTATE), default="default",
                               tooltip="SageAttention2 mean-centering before quantizing; only used by int8qk."),
                io.Combo.Input("fp16", options=list(TRISTATE), default="default",
                               tooltip="Cast fp32/bf16 attention inputs to fp16. The int8/int8qk/hand kernels "
                                       "are fp16-only and error out instead of running when this is off; the "
                                       "backend names then run in the model's own dtype."),
                io.Combo.Input("amax_guard", options=list(TRISTATE), default="default",
                               tooltip="Skip the fp16 cast when the attention input's abs-max would overflow "
                                       "fp16 (>= 60000) and leave it in the model dtype instead. default keeps "
                                       "the built-in on state; off casts unconditionally."),
                io.Combo.Input("convrot", options=["default"] + list(CONVROT_MODES), default="default",
                               tooltip="default keeps the built-in mode. fused = rotation + int8 quantize in "
                                       "one kernel, butterfly = rotation only, off = the original matmul."),
                io.Combo.Input("int8_tile", options=list(TRISTATE), default="default",
                               tooltip="Keep the gfx906 int8 tile (128x256x32, 4 warps, 2 stages) in the "
                                       "kitchen autotune list. default keeps the injected tiles; off "
                                       "restores the stock list. The m<=8 narrow tile has its own switch."),
                io.Combo.Input("small_batch", options=list(TRISTATE), default="default",
                               tooltip="gfx906: switch the int8 kernel to the 16x32x64 narrow tile when m<=8 "
                                       "(text decode / MTP verify), and back to 128x256x32 for larger batches. "
                                       "Other AMD: add the narrow tile as an autotune candidate and let that "
                                       "card decide. default keeps the built-in split; needs int8_tile on. "
                                       "GEMV has its own switch."),
                io.Combo.Input("qkv", options=list(TRISTATE), default="default",
                               tooltip="One int8 GEMM for q/k/v instead of three (Qwen-Image 2.1): -81 ms "
                                       "of a 5.24 s step (1.5%), bit-exact. The merged weights stay resident "
                                       "(1.6 GB on a 32-block DiT), so the env default is off. No restart "
                                       "needed; turning it off also drops the merged copy."),
                io.Combo.Input("vae", options=list(TRISTATE), default="default",
                               tooltip="Run depth-1 Conv3d as Conv2d in the video VAE. default keeps "
                                       "the manifest verdict; off restores the shipped conv3d path."),
                io.Combo.Input("gemv", options=list(TRISTATE), default="default",
                               tooltip="Single-token int8 decode GEMV for the model's int8 linears. default "
                                       "the manifest verdict; off returns every call to the tiled GEMM."),
                io.Combo.Input("prefix_kv", options=list(TRISTATE), default="default",
                               tooltip="Merge the prefix cache K/V into a reused buffer with copy_ instead of "
                                       "torch.cat (Qwen-Image 2.1). default keeps it on; off "
                                       "restores the shipped torch.cat path."),
                io.Combo.Input("profile", options=list(TRISTATE), default="default",
                               tooltip="Print the tuning status once and per-call timings when each prompt finishes."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, attention: str, contiguous: str, smooth: str, fp16: str, amax_guard: str,
                convrot: str, int8_tile: str, small_batch: str, qkv: str, vae: str,
                gemv: str, prefix_kv: str, profile: str) -> io.NodeOutput:
        apply()
        configure(attention=attention, smooth=smooth, fp16=fp16, convrot=convrot,
                  profile=profile, contiguous=contiguous, qkv=qkv,
                  vae=vae, gemv=gemv, prefix_kv=prefix_kv, amax_guard=amax_guard,
                  int8_tile=int8_tile, small_batch=small_batch)
        return io.NodeOutput(model)


class Gfx906ClipSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gfx906ClipSettings",
            display_name="GFX906 CLIP Settings",
            category="advanced/debug",
            description=(
                "Text-encoder settings for the gfx906 tuning pack. Sits in the CLIP chain, between the "
                "loader and the text encode; applies to every text encoder in the process. The on/off "
                "switches are three-state: default keeps the built-in default; text_encoder "
                "and text_attention use default as an explicit choice instead."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input("text_encoder", options=["int8", "default"], default="default",
                               tooltip="int8 keeps the text-encoder weights quantized with int8 "
                                       "activations (fastest on 4-bit checkpoints, embeddings shift ~6%); "
                                       "default keeps fp16 activations and dequantizes in registers on "
                                       "the fly (best quality, same speed on 8-bit checkpoints)."),
                io.Combo.Input("text_attention", options=["default"], default="default",
                               tooltip="default keeps what core picks for small inputs. Text encoders "
                                       "fetch attention with small_input=True and a named backend does "
                                       "replace that pick, but none of them has a measured win on text "
                                       "shapes on this card, so none is offered here; the routing is live, "
                                       "add a name back when a measurement justifies it."),
                io.Combo.Input("text_gemv", options=list(TRISTATE), default="default",
                               tooltip="Single-row text-encoder GEMV: the direct int8 kernel call and the "
                                       "torch.mv path for the unquantized gate projections. default keeps "
                                       "the built-in on state; off sends both back to the dispatched linear."),
                io.Combo.Input("w4a8", options=list(TRISTATE), default="default",
                               tooltip="4-bit packed text-encoder weights: decode rows go to the w4a8 GEMV, "
                                       "prefill unpacks one layer to int8. default keeps it on; "
                                       "off dequantizes through the shipped layout. status shows the call count "
                                       "so a swapped checkpoint can be confirmed."),
                io.Combo.Input("textgen", options=list(TRISTATE), default="default",
                               tooltip="Text-generation patches: DeltaNet decode kernels, folded KV decode, "
                                       "llama RMSNorm/RoPE caching and the Qwen3.5 compute classes. default "
                                       "the four COMFY_GFX906_DELTANET/KVDECODE/LLAMA/QWEN35 knobs "
                                       "individually; on/off "
                                       "switches all four together without a reload."),
            ],
            outputs=[io.Clip.Output(display_name="clip")],
        )

    @classmethod
    def execute(cls, clip, text_encoder: str, text_attention: str, text_gemv: str, w4a8: str,
                textgen: str) -> io.NodeOutput:
        apply()
        configure(text_encoder=text_encoder, text_attention=text_attention, text_gemv=text_gemv,
                  w4a8=w4a8, textgen=textgen)
        return io.NodeOutput(clip)


class Gfx906TuningExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Gfx906ModelSettings, Gfx906ClipSettings]


async def comfy_entrypoint() -> Gfx906TuningExtension:
    apply()
    watch_profiler(report_lines)
    hooks = install_profiler()
    if hooks:
        logging.info("gfx906 profiler hooks: %s (report printed when each prompt finishes)", ", ".join(hooks))
    return Gfx906TuningExtension()
