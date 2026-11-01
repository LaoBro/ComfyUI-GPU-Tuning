"""ComfyUI node pack: runtime tuning for gfx906 (MI50 / MI60 / Radeon VII).

Applies the patches on startup so an unmodified ComfyUI gets the tuned int8
linear tiles, the butterfly convrot rotation and the fp16 attention kernel.
Two settings nodes expose the runtime switches (one in the model chain, one in
the CLIP chain); the model node's profile switch additionally reports what was
applied and prints per-prompt timings.
"""

import logging

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .gfx906_tuning import (
    ATTENTION_KERNELS,
    CONVROT_MODES,
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
                "no restart needed. Wire it between the model loader and the sampler."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("attention", options=list(ATTENTION_KERNELS) + backend_options(), default="int8",
                               tooltip="int8 is the fast default; int8qk keeps PV in fp16 (slower, 4-7x more "
                                       "accurate); gemm is fp16 built from rocBLAS bmm + softmax + bmm, ~1.45x "
                                       "faster than hand at DiT shapes and the fastest fp16 path on gfx906; "
                                       "hand is the fp16 Triton kernel. basic, pytorch, sub_quad, split and any "
                                       "other backend core registered are ComfyUI's own functions, routed to "
                                       "directly."),
                io.Boolean.Input("contiguous", default=True,
                                 tooltip="When a backend name is selected: transpose to [B,H,S,D] and "
                                         ".contiguous() before calling it. Core hands SDPA/bmm transpose views "
                                         "instead, which measures 4-15x slower on gfx906. Turn off to reproduce "
                                         "core's exact call for comparison on other GPUs. No effect on the "
                                         "int8/int8qk/hand kernels (they build their own layouts)."),
                io.Boolean.Input("smooth", default=True,
                                 tooltip="SageAttention2 mean-centering before quantizing; only used by int8qk."),
                io.Boolean.Input("fp16", default=True,
                                 tooltip="Cast fp32/bf16 attention inputs to fp16. The int8/int8qk/hand kernels "
                                         "are fp16-only and error out instead of running when this is off; the "
                                         "backend names then run in the model's own dtype (slower here, useful "
                                         "as a quality baseline)."),
                io.Combo.Input("convrot", options=list(CONVROT_MODES), default="fused",
                               tooltip="fused = rotation + int8 quantize in one kernel, butterfly = rotation only, "
                                       "off = the original matmul."),
                io.Boolean.Input("linear8", default=True,
                                 tooltip="Quantize fp16 Linear weights (in_features >= 1280) to per-row int8 with "
                                         "ConvRot rotation and run the comfy-kitchen int8 kernel: -17% step time on "
                                         "SDXL (the 640-wide layers stay fp16). Off = full fp16 quality baseline, "
                                         "no restart needed."),
                io.Boolean.Input("profile", default=False,
                                 tooltip="Print the tuning status once and per-call timings when each prompt finishes."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, attention: str, smooth: bool, fp16: bool, convrot: str,
                profile: bool, contiguous: bool, linear8: bool) -> io.NodeOutput:
        apply()
        configure(attention=attention, smooth=smooth, fp16=fp16, convrot=convrot,
                  profile=profile, contiguous=contiguous, linear8=linear8)
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
                "loader and the text encode; applies to every text encoder in the process."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input("text_encoder", options=["int8", "default"], default="int8",
                               tooltip="int8 keeps the text-encoder weights quantized (2.4x faster encode, "
                                       "1.5x faster token generation, embeddings shift ~6%); default "
                                       "dequantizes them on every call, including the lm_head per token."),
                io.Combo.Input("text_attention", options=["default"] + backend_options(), default="default",
                               tooltip="Attention backend for the text encoder (and every other "
                                       "optimized_attention_for_device caller). default keeps what core picks "
                                       "for small inputs; the rest route to that backend, no restart needed. "
                                       "The custom int8/hand kernels are not listed: they reject masked calls."),
            ],
            outputs=[io.Clip.Output(display_name="clip")],
        )

    @classmethod
    def execute(cls, clip, text_encoder: str, text_attention: str) -> io.NodeOutput:
        apply()
        configure(text_encoder=text_encoder, text_attention=text_attention)
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
