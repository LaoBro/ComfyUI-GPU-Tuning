"""The two settings nodes: schema, tooltips and the switch calls into gpu_tuning."""

from comfy_api.latest import io

from .gpu_tuning import (
    ATTENTION_KERNELS,
    CONVROT_MODES,
    TRISTATE,
    apply,
    backend_options,
    configure,
)


class GpuTuningModelSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GpuTuningModelSettings",
            display_name="GPU Tuning Model Settings",
            category="advanced/debug",
            description=(
                "Runtime settings for the GPU Tuning pack. Values take effect on the next execution, "
                "no restart needed. Wire it between the model loader and the sampler. Most switches "
                "are plain on/off; vae, gemv and image_tower also offer default to follow the "
                "measured verdict, and a few are mode choices."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("attention", options=["default"] + list(ATTENTION_KERNELS) + backend_options(),
                               default="default",
                               tooltip="default is hip when the shipped kernel loads (fused fp32-accumulate "
                                       "kernel, fastest on every measured shape), else gemm; hip needs D=64/128 "
                                       "and libgfx906_attn.so, which targets HIP 7 - another runtime rebuilds it "
                                       "with tools/build_hip_attn.py --force; gemm is fp16 built from rocBLAS "
                                       "bmm + softmax + bmm (any head dim, but materializes the score matrix, "
                                       "up to ~630 MiB); hand is the fp16 Triton kernel (same 10-27 MiB footprint "
                                       "as hip, but slower). The int8 tiers were dropped: measured slower than "
                                       "hip at every image shape and 30x less accurate. basic, pytorch, sub_quad, "
                                       "split and any other backend core registered are ComfyUI's own functions, "
                                       "kept only for A/B baselines and explicit requests; the fp32 ones are "
                                       "slower here (pytorch/SDPA is math-only on gfx906). default already "
                                       "picks the fastest path (hip when the .so loads, else gemm)."),
                io.Combo.Input("contiguous", options=["on", "off"], default="on",
                               tooltip="When a backend name is selected: transpose to [B,H,S,D] and "
                                       ".contiguous() before calling it. Core hands SDPA/bmm transpose views "
                                       "instead, which measures 4-15x slower on gfx906. Turn off to reproduce "
                                       "core's exact call for comparison on other GPUs. No effect on the "
                                       "hip/hand kernels (they build their own layouts)."),
                io.Combo.Input("fp16_attn", options=["force", "guarded", "off"], default="guarded",
                               tooltip="How Q/K/V reach the attention kernels. guarded (default): cast "
                                       "fp32/bf16 inputs to fp16 unless their abs-max would overflow "
                                       "(>= 60000), in which case the call falls back to core instead. "
                                       "force: cast unconditionally, even if it overflows. off: leave the "
                                       "inputs in the model dtype; hip/gemm/hand are fp16-only and raise, "
                                       "so pick a core backend name (basic/pytorch/sub_quad/split) with off."),
                io.Combo.Input("convrot", options=["default"] + list(CONVROT_MODES), default="default",
                               tooltip="default keeps the built-in mode. fused = rotation + int8 quantize in "
                                       "one kernel, butterfly = rotation only, off = the original matmul. fused "
                                       "covers fp32/fp16/bf16 when K is a power of two with at most 32 groups "
                                       "(K<=8192); the K=12288 MLP down projection keeps butterfly + quantize."),
                io.Combo.Input("int8_tile", options=["on", "off"], default="on",
                               tooltip="Keep the gfx906 int8 tile (128x256x32, 4 warps, 2 stages) in the "
                                       "kitchen autotune list. on keeps the injected tiles; off "
                                       "restores the stock list. The m<=8 narrow tile has its own switch."),
                io.Combo.Input("small_batch", options=["on", "off"], default="on",
                               tooltip="On gfx906: switch the int8 kernel to the 16x32x64 narrow tile when m<=8 "
                                       "(text decode / MTP verify), and back to 128x256x32 for larger batches. "
                                       "Other AMD: add the narrow tile as an autotune candidate and let that "
                                       "card decide. on keeps the built-in split; needs int8_tile on. "
                                       "GEMV has its own switch."),
                io.Combo.Input("int8_gemm", options=["on", "off"], default="on",
                               tooltip="Run int8 linears with at least 128 rows on the HIP dot4 GEMM "
                                       "(128x128 tile, 4 waves/CU) instead of the Triton kernel: 1.10-1.21x "
                                       "on the measured prefill shapes. on uses it when "
                                       "the shipped libgfx906_int8.so loads; off forces the Triton path. Decode "
                                       "(m<=1) still uses the GEMV."),
                io.Combo.Input("int8_attn", options=["on", "off"], default="off",
                               tooltip="Run comfy_kitchen's INT8 attention contract (models whose checkpoint marks "
                                       "attention as comfy_kitchen_int8, and the int8 minimax VAE) on the HIP "
                                       "int8-input kernel: Q/K int8 with per-token scales, V int8 4-keys-per-word "
                                       "with per-dim scales, 8-bit P, fp32 softmax. Measured output diffl2 1.1-1.3e-2 "
                                       "(fp16 kernel: 2.5e-4), so it is off by default; on is opt-in, off runs the "
                                       "regular optimized attention for those calls."),
                io.Combo.Input("int8_attn_rotate", options=["off", "64", "128"], default="128",
                               tooltip="Hadamard rotation applied to Q and K before INT8 quantization on the int8 "
                                       "attention path. It costs one matmul and bounds per-row outliers without "
                                       "changing dot products; off is the same math with no rotation. 128 is the built-in default."),
                io.Combo.Input("qkv", options=["on", "off"], default="off",
                               tooltip="One int8 GEMM for q/k/v instead of three (Qwen-Image 2.1): -81 ms "
                                       "of a 5.24 s step (1.5%), bit-exact. The merged weights stay resident "
                                       "(1.6 GB on a 32-block DiT), so it is off by default. No restart "
                                       "needed; turning it off also drops the merged copy."),
                io.Combo.Input("vae", options=list(TRISTATE), default="default",
                               tooltip="Run depth-1 Conv3d as Conv2d in the video VAE. default keeps "
                                       "the manifest verdict; off restores the shipped conv3d path."),
                io.Combo.Input("gemv", options=list(TRISTATE), default="default",
                               tooltip="Single-token int8 decode GEMV for the model's int8 linears. default "
                                       "the manifest verdict; off returns every call to the tiled GEMM."),
                io.Combo.Input("prefix_kv", options=["on", "off"], default="on",
                               tooltip="Merge the prefix cache K/V into a reused buffer with copy_ instead of "
                                       "torch.cat (Qwen-Image 2.1). on keeps it on; off "
                                       "restores the shipped torch.cat path."),
                io.Combo.Input("qwen21_int8_cache", options=["on", "off"], default="off",
                               tooltip="Keep the Qwen-Image 2.1 int8 prefix cache's K/V in int8 and run the "
                                       "attention on the int8 kernel: the shipped path dequantizes both to fp16 "
                                       "once per block. Measured 51.2 -> 34.2 ms per block at "
                                       "4058+4032 tokens, 32 heads, d128 (1.50x), output diffl2 1.1e-2 "
                                       "against the fp16-on-dequant reference (2.6e-4). off "
                                       "keeps the fp16 path."),
                io.Combo.Input("profile", options=["on", "off"], default="off",
                               tooltip="Print the tuning status once and per-call timings when each prompt finishes."),
                io.Combo.Input("qwen21_fp16", options=["on", "off"], default="on",
                               tooltip="Run the Qwen-Image 2.1 DiT forward in fp16 when the loaded model computes "
                                       "in bf16: the hidden states, context and reference latents are cast once "
                                       "for the whole forward and the output is cast back. gfx906 emulates bf16 "
                                       "in fp32; measured 7.92 -> 3.83 s per forward at a 64x64 latent "
                                       "(2.07x), and the fp16 output is closer to fp32 than the bf16 baseline. "
                                       "Guarded: a non-finite fp16 pass drops the prefix cache and reruns the "
                                       "call in the model dtype. fp16/fp32 models are left alone, and the "
                                       "patch only installs where the card has no native bf16 (n/a elsewhere)."),
                io.Combo.Input("int8_linear_act", options=["on", "off"], default="on",
                               tooltip="Keep int8 weights quantized at every comfy.ops.linear_input_act site "
                                       "(MLP down projections and pre-norm blocks). When the weight's orig_dtype "
                                       "differs from the activation dtype (fp32 manual-cast models running bf16: "
                                       "Qwen-Image 2.1, LTXV, MiniMaxH3), the shipped path requantizes and then "
                                       "dequantizes the int8 weight, falling back to a full bf16 GEMM; on takes "
                                       "the int8 kernel instead (Qwen-Image 2.1: 7.92 -> 3.34 s per forward at a "
                                       "64x64 latent, 2.37x). Calls whose dtypes already match are left "
                                       "untouched, so Flux / Wan's default fp16 / fp32 runs measure identical. "
                                       "off restores the shipped call."),
                io.Combo.Input("qwen21_residual_fuse", options=["on", "off"], default="on",
                               tooltip="Fuse the Qwen-Image 2.1 gated residual (x += gate * layer_out) into the "
                                       "int8 GEMM epilogue, for both the attention to_out and the SwiGLU down "
                                       "projection. Removes the separate addcmul kernel and its ~66 MB round trip "
                                       "per residual; measured 14.66 -> 14.56 ms on the MLP down GEMM and "
                                       "5.02 -> 4.99 ms on to_out, about 0.1% of a step (below the pack's "
                                       "measurement noise). Needs the v2 residual epilogue in libgfx906_int8.so; "
                                       "the mixed first pass (prefix + target rows) keeps the shipped addcmul."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, attention: str, contiguous: str, fp16_attn: str,
                convrot: str, int8_tile: str, small_batch: str, int8_gemm: str, int8_attn: str,
                int8_attn_rotate: str, qkv: str, vae: str,
                gemv: str, prefix_kv: str, qwen21_int8_cache: str, profile: str,
                qwen21_fp16: str, int8_linear_act: str, qwen21_residual_fuse: str) -> io.NodeOutput:
        apply()
        if fp16_attn == "force":
            fp16, amax_guard = "on", "off"
        elif fp16_attn == "off":
            fp16, amax_guard = "off", "on"
        else:
            fp16, amax_guard = "on", "on"
        configure(attention=attention, fp16=fp16, convrot=convrot,
                  profile=profile, contiguous=contiguous, qkv=qkv,
                  vae=vae, gemv=gemv, prefix_kv=prefix_kv, amax_guard=amax_guard,
                  int8_tile=int8_tile, small_batch=small_batch, int8_gemm=int8_gemm,
                  int8_attn=int8_attn, int8_attn_rotate=int8_attn_rotate,
                  qwen21_int8_cache=qwen21_int8_cache,
                  qwen21_fp16=qwen21_fp16, int8_linear_act=int8_linear_act,
                  qwen21_residual_fuse=qwen21_residual_fuse)
        return io.NodeOutput(model)


class GpuTuningClipSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GpuTuningClipSettings",
            display_name="GPU Tuning CLIP Settings",
            category="advanced/debug",
            description=(
                "Text-encoder settings for the GPU Tuning pack. Sits in the CLIP chain, between the "
                "loader and the text encode; applies to every text encoder in the process. Most "
                "switches are plain on/off; image_tower also offers default to follow the measured "
                "verdict, and text_encoder / text_attention are explicit choices."
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
                io.Combo.Input("text_gemv", options=["on", "off"], default="on",
                               tooltip="Single-row text-encoder GEMV: the direct int8 kernel call and the "
                                       "torch.mv path for the unquantized gate projections. on keeps "
                                       "the built-in on state; off sends both back to the dispatched linear."),
                io.Combo.Input("w4a8", options=["on", "off"], default="on",
                               tooltip="4-bit packed text-encoder weights: decode rows go to the w4a8 GEMV, "
                                       "prefill unpacks one layer to int8. on keeps it on; "
                                       "off dequantizes through the shipped layout. status shows the call count "
                                       "so a swapped checkpoint can be confirmed."),
                io.Combo.Input("image_tower", options=list(TRISTATE), default="default",
                               tooltip="Qwen3.5 image tower: keep the input in the weight dtype instead of "
                                       "the fp32 core hands it, and answer the patch Conv3d with the "
                                       "equivalent GEMM (its kernel equals its stride, and gfx906 has no "
                                       "conv3d kernel). Measured 3.83 -> 1.6 s for a 1160x679 image on this "
                                       "card. default keeps the manifest verdict: unmeasured cards stay on "
                                       "the shipped path, tools/bench.py --vision writes the verdict."),
                io.Combo.Input("textgen", options=["on", "off"], default="on",
                               tooltip="Text-generation patches: DeltaNet decode kernels, folded KV decode, "
                                       "llama RMSNorm/RoPE caching and the Qwen3.5 compute classes. on "
                                       "(default) enables all four, off disables them all without a reload. "
                                       "Without this node the COMFY_GPU_TUNE_DELTANET/KVDECODE/LLAMA/ "
                                       "QWEN35/GRAPH knobs decide each item."),
            ],
            outputs=[io.Clip.Output(display_name="clip")],
        )

    @classmethod
    def execute(cls, clip, text_encoder: str, text_attention: str, text_gemv: str, w4a8: str,
                textgen: str, image_tower: str) -> io.NodeOutput:
        apply()
        configure(text_encoder=text_encoder, text_attention=text_attention, text_gemv=text_gemv,
                  w4a8=w4a8, textgen=textgen, image_tower=image_tower)
        return io.NodeOutput(clip)
