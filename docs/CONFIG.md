# 节点开关与注意力模式

[README](../README.md) · [PATCHES](PATCHES.md) · [GFX906](GFX906.md) · [PORTING](PORTING.md) · [HIP_KERNELS](HIP_KERNELS.md)

这一层回答“怎么调”：两个节点、每个开关的含义、注意力模式选择、int8 大 m 路径，
以及仍然保留的环境变量。

## 两个节点

| 节点 | 位置 | 作用 |
|---|---|---|
| **GPU Tuning Model Settings** | 模型链（加载器 → 采样器之间） | 注意力模式、连续布局、fp16 输入策略、convrot、int8 tile、`m≤8` 窄 tile、int8 大 m GEMM、int8 注意力与旋转、QKV、VAE、GEMV、prefix KV、Qwen2.1 int8 缓存 / fp16 全前向、linear_input_act int8、profiler |
| **GPU Tuning CLIP Settings** | CLIP 链（加载器 → 编码） | 文本编码器 int8、编码器注意力后端、TE GEMV、4-bit w4a8、文本生成补丁组 |

设置节点改动**下一次执行就生效**，不用重启。命令行复测 `benchmark()` 已删除：它没有节点注册、也没有任何调用者，长期是死代码；需要复测时直接用 `gfx906fix/` 下的独立脚本，避免再维护第三个节点。

多数开关是 `on` / `off` 二态，combo 上显示的就是当前状态，默认值即内置默认。测量门/跟随 env
的开关 `vae` / `gemv` / `image_tower` / `dit_fp16` / `fp16_trace` 多一个 `default`：
按 `arch/<gcn>/manifest.toml` 的实测判定或启动 env，没跑过 bench 的卡保持原路径。其余是模式或数值选择：`attention`
（`default` + `hip`/`gemm`/`hand` + core 后端名）、`convrot`（`default` + `fused`/`butterfly`/`off`）、
`fp16_attn`（`force`/`guarded`/`off`）、`int8_attn_rotate`（`off`/`64`/`128`）、
`dit_fp16`（`default` + `off`/`qwen21`/`all`）、
`text_encoder`（`int8`/`default`）、`text_attention`（只有 `default`）。

节点里所有值都是显式的，没有“沿用启动 env”的隐式状态。文本生成的分项 env
（`DELTANET` / `LLAMA` / `QWEN35` / `GRAPH`）只在**没连 CLIP 节点**时生效，
供 `tools/chain_profile.py` 逐项归因；连上节点后由 `textgen` 的 `on`/`off` 完全接管。

### 节点开关清单

Model Settings：`attention`、`contiguous`、`fp16_attn`、`convrot`、`int8_tile`、`small_batch`、`int8_gemm`、`int8_attn`、`int8_attn_rotate`、`qkv`、`vae`、`gemv`、`prefix_kv`、`qwen21_int8_cache`、`profile`、`dit_fp16`、`int8_linear_act`、`qwen21_residual_fuse`、`fp16_trace`。

CLIP Settings：`text_encoder`、`text_attention`、`text_gemv`、`w4a8`、`image_tower`、`textgen`。

补充说明：

- `contiguous`：选中 `basic`/`pytorch`/`sub_quad`/`split` 等 core 后端时，调用前把输入转成连续的 `[B,H,S,D]`；关闭可复现 core 的原始转置视图调用。
- `fp16_attn`：三选一。`guarded`（默认）把 fp32/bf16 转 fp16，但 Q/K/V 绝对值 ≥60000 时放弃本次调用、
  回落 core（fp16 上限 65504，留余量）；`force` 无条件转，可能溢出；`off` 保持模型 dtype，自带
  hip/gemm/hand 是 fp16-only 会报错，需配 core 后端名使用。旧版 `fp16` / `amax_guard` 已并入此开关，
  旧工作流需重新连接。
- `convrot`：`fused` 把蝶形旋转和 int8 量化合成一个 kernel。fp16/bf16 也会走融合路径，
  条件是 K 为 2 的幂且组数 ≤32（K≤8192）；K=12288 的 MLP 下投影仍走 `butterfly`+量化。
  `butterfly` 保留旧的两次读写路径，`off` 回到原始 matmul；M=4032 fp16 实测 K=4096
  0.196 → 0.149 ms（1.3x），且少了中间 fp16 舍入、结果更接近 fp32 旋转。
- `int8_tile`：关闭后恢复上游 autotune 的 tile 列表，不再注入 gfx906 的 `128×256×32`；GEMV 仍由 `gemv` 单独控制。
- `small_batch`：gfx906 上把“`m≤8` 切到 `16×32×64` 窄 tile、`m>8` 切回 `128×256×32`”单独控制；其它 AMD 上表示“把窄 tile 追加为 autotune 候选”（是否真的用它由该卡实测决定）。需要 `int8_tile` 为 on，否则显示 n/a。GEMV 仍由 `gemv` 单独控制。
- `text_gemv`：单行文本编码器解码时，int8 权重直调 GEMV、未量化的小投影走 `torch.mv`；关闭后两者都回到分派层。
- `w4a8`：4-bit 权重解码；`status()` 会打印实际调用次数和当前后端（`hip` / `asm` / `triton`），换 checkpoint 后可据此确认是否真的走到了 w4a8 内核（`on: 0 calls ...` 表示权重布局没被识别）。能否用 `tl.dot` 替代 `v_dot4` 的实测结论见 [GFX906_EXPERIMENTS.md](GFX906_EXPERIMENTS.md) 的「试过但没采用的路径」。
- `textgen`：默认 on，一次控制 DeltaNet 内核、KV decode 折叠、llama RMSNorm/RoPE 缓存、Qwen3.5 计算类；关闭时逐调用委托回 core 实现（Qwen3.5 类符号本身不换回）。不连该节点时由分项 env 决定。
- `int8_gemm`：行数 ≥ 128 的 int8 线性交给自带 HIP dot4 GEMM，`off` 强制回 Triton；详见下节。
  开 `profile` 后每次 prompt 的 profiler 报告会打印本次 hip gemm / gemv / triton 三路的次数；
  带 `input_act`（SwiGLU / gelu / rms_norm）的调用也走 HIP GEMM；`residual` 走 v2 epilogue
  （需要新版 `libgfx906_int8.so`，旧库回落 triton）。
- `int8_attn` / `int8_attn_rotate`：comfy-kitchen 的 int8 注意力契约（默认 off）及其 Q/K 量化前的
  Hadamard 旋转（`off` / `64` / `128`，默认 128）；数值代价与实测见 [PATCHES.md](PATCHES.md)、
  [GFX906_EXPERIMENTS.md](GFX906_EXPERIMENTS.md)。
- `qwen21_int8_cache`：Qwen-Image 2.1 的 int8 prefix cache K/V 保持 int8、直接在 int8 内核上算
  注意力（默认关）；实测见 [GFX906_EXPERIMENTS.md](GFX906_EXPERIMENTS.md)。开 `profile` 后每次 prompt 的
  profiler 报告会打印本次的 int8 注意力调用数与 int8 `take()` 次数（括号内为进程累计），
  `0 calls` 表示本次回落 fp16 反量化路径；`status()` 里显示的是累计值。
- `dit_fp16`（默认 `qwen21`）：bf16 权重 DiT 的激活 fp16 路径，四选一。`qwen21` 只对 Qwen-Image 2.1
  生效，`all` 推广到其它扩散模型（Flux、Wan、LTXV、MiniMax H3 等，未实测所以 opt-in），`off`
  关闭，`default` 跟随 `COMFY_GPU_TUNE_DIT_FP16`（未设等于 `qwen21`）。
  触发依据是**模型权重是 bf16**，不是入口张量的 dtype：gfx906 没有原生 bf16，而
  `[bf16, fp32]` 的 dtype 表在 ComfyUI 里会落到 `manual_cast=fp32`，所以 H3 / Qwen-Image 2.1
  的 DiT 入口实际是 fp32；bf16 和这个 fp32 fallback 都进入同一条路径。A/V 双流模型的 list
  latent 也支持。转换只发生在**激活**上：latent（含 list）、context、extra conds 一次性转 fp16，
  出口还原成入口 dtype；`transformer_options` 和 mask 保持原 dtype/对象。
  权重不转 fp16：int8/convrot 权重保持 int8，内核按激活 dtype 构造旋转和 `out_dtype`；
  非 int8 的 Linear 层通过 hook 用模型原来的计算 dtype（bf16 或 fp32）跑，输出再回到 fp16，
  所以 MiniMax H3 的 fp32 岛（`video_out`/`audio_out`）仍是 fp32。Qwen-Image 2.1 64×64 latent
  实测 7.92 → 3.83 s（2.07x，边界 cast 的原始测量；非 int8 层保持原计算 dtype 会拿回一点精度、
  略掉一点速度）。带守卫：fp16 结果出现非有限值就用原 dtype 重跑本次调用
  （Qwen-Image 2.1 还会清掉 prefix cache）；原生 fp16/fp32 模型不动；只在没有原生 bf16 的卡
  （gfx906 等，`should_use_bf16` 为 False）安装，其它卡显示 n/a。`status()` 会分别打印
  qwen21 / all 的调用次数和其中 fp32 入口的次数。旧工作流的 `qwen21_fp16=on/off` 仍分别按
  `qwen21`/`off` 解释。
- `fp16_trace`（默认 off）：激活 fp16 路径的溢出测量。on 时给扩散模型的每个叶子模块挂
  forward hook，记录输入/输出的 abs-max、`isfinite` 和超过 60000 的元素数；另外对
  MiniMax H3 额外探测模块 hook 看不到的函数式位置：`_mod_scale_shift`（调制后的 norm）、
  `_mod_gate`（gated residual）、`Attention.forward` 和 `MLP.forward`。检查都是 device 端
  归约，前向本身不 sync，prompt 结束时一次性读回并打印：
  - `overflow sites`：输出出现 inf/NaN 的模块，按调用顺序排列（后面的通常是被污染的
    下游），带 step、shape、`in -> out` 峰值、超阈值元素数和模块类名；
  - `top sites by abs-max`：按输出峰值排序的站点；`hi N` 是超过 60000 的元素数；
  - `H3 function probes`：H3 的调制/residual/Attention/MLP 结果。
  参考测量：把 `COMFY_GPU_TUNE_FP16_TRACE_DTYPE` 设为 `bf16` 或 `fp32`，并用
  `dit_fp16=off` 跑同一 prompt，就能拿到同一批站点的真实量级；缩放系数按
  `S = ceil(reference_peak / 60000)` 估算（取 2 的幂并留余量），再按站点选择可精确补偿的
  插入点（attention 输出接线性层、MLP 的 up 分支接 `fc2`、`_mod_gate` 的 `other/gate` 等），
  思路同 ComfyUI PR #11187 的 attention /4、FFN w3 /32。打开后每个站点每次调用都多做几次
  归约，有明显开销，定位完就关掉。`default` 跟随 `COMFY_GPU_TUNE_FP16_TRACE`，未设等于 off。
- `int8_linear_act`（默认 on）：所有 `comfy.ops.linear_input_act` 调用点（SwiGLU / GeGLU 下投影、
  预归一化块）在 int8 权重与激活 dtype 不一致时保持权重量化。shipped 路径用
  `compute_dtype=x.dtype` 走 `cast_bias_weight`，gfx906 上“dtype 表不含 fp16 → bf16 激活 +
  fp32 manual cast”的模型（Qwen-Image 2.1、LTXV、MiniMaxH3 等）因此每块先 requantize 再
  dequantize，退化成 bf16 rocBLAS GEMM。on 直接调 int8 内核（`swiglu`/`gelu_tanh`/`rms_norm`
  和 `residual` 都支持）；dtype 本来就一致的调用原样委托，所以 Flux、Wan 默认 fp16、fp32
  运行测量不变。Qwen-Image 2.1 64×64 latent 实测 7.92 → 3.34 s（2.37x）。
- `qwen21_residual_fuse`（默认 on）：把 Qwen-Image 2.1 的 gated residual（`x += gate * layer_out`）
  融进 int8 GEMM 的 epilogue（attention 的 `to_out` 和 SwiGLU 下投影各一次），省掉单独的
  `addcmul` 和每次约 66 MB 的往返。需要 `libgfx906_int8.so` 里的 v2 residual epilogue；
  4032 token 实测 MLP down 14.66 → 14.56 ms、to_out 5.02 → 4.99 ms，整步约 0.1%
  （低于本包的测量噪声）。只有全 prefix 或全 target（`prefix_len == 0` 或等于行数）的调用能融合，
  prefix+target 混合的首次 pass 仍走 `_gated_residual`。

### 在其它显卡上使用 / 扫描 int8 tile

其它显卡上的功能分级（常开 / 手工 / 测量门）、扫描工具 `tools/tile_sweep.py` 与 `COMFY_GPU_TUNE_TILE_CONFIG` / `SMALL_TILE_CONFIG` 的用法，以及跑通之后的进一步调优流程，见：

**[PORTING.md — 移植与进一步调优](PORTING.md)**

给其它架构写 HIP 内核见 [docs/HIP_KERNELS.md](HIP_KERNELS.md)。

最短路径：先用节点默认值跑通（没跑过 bench 的卡上测量门功能是关的），再运行一次 `python custom_nodes/ComfyUI-GPU-Tuning/tools/bench.py --all --write`。它测出这张卡的 int8 tile，以及 GEMV / w4a8 / deltanet（含 prefill）/ vae / convrot / vision 的开或关，写进 `~/.config/comfy-gpu-tune/arch/<gcn>/manifest.toml`，之后启动不再 autotune。attention 只打印证据不写记录：`hip` 是默认模式（下面有实测，内核随包分发），本机 HIP 装不上它时回落到 `gemm` 并在启动日志提示重编，其余模式仍可在节点里手选。想知道时间落在哪一段（而不是哪个内核快）用 `tools/chain_profile.py`，见 [docs/PORTING.md](PORTING.md)。

### 注意力模式怎么选

| 模式 | 精度 | 适用 |
|---|---|---|
| `hip`（默认） | fp32 累加，rel ≈ 2.5–3.3e-4 | 每个实测形状都最快（对 `gemm` 1.3–3.5x，对本机默认 `sub_quad` 1.5–5.5x），且只占输出显存 10–27 MiB。只支持 D=64/128 |
| `gemm` | fp16 score 物化，rel ≈ 5.6e-4 | 任意 head_dim 都能用（含 SD1.5 的 40、SD2.1 的 80），`hip` 装不上时的默认；也会拷 58–630 MiB 的 score/连续副本 |
| `hand` | fp16 累加，rel ≈ 2.8e-4 | Triton 内核，时间与 `gemm` 同级但只占 1–27 MiB；`hip` 不可用时的低显存选择 |

`default` 已经是本机最快路径（`.so` 装得上走 `hip`，否则 `gemm`），日常不用手选。下拉里的
`basic` / `pytorch` / `sub_quad` / `split` 以及 core 注册的其它后端名（`comfy_kitchen_int8`、
`sage` 等）是 ComfyUI 自带实现，选中后原样转发，**只作为 A/B 基线和特殊契约/显式请求的入口**：
其中 fp32 的几个在本卡明显更慢（`pytorch` 的 SDPA 只有 math 后端），`comfy_kitchen_int8`
另由 `int8_attn` 控制（默认关）。保留它们也是 `contiguous` 开关和 `fp16_attn=off` 的落点。

`int8` 与 `int8qk` 两档已删除：真实 SDXL 形状上它们比 `hip` 慢（4096²：13.4 / 15.3 对 8.5 ms）
且 relL2 差 30 倍（7.7e-3 对 2.1e-4），实测没有任何一个形状赢过 `hip`。原先"`hip` 装不上时
默认走 `int8`"改成默认走 `gemm`。

`hip` 是本包自带的融合注意力内核（`arch/gfx906/kernels/hip_attn.cu`，ctypes 加载，不依赖 torch 扩展）。
同一个脚本也编出 int8 大 m 路径要用的 `libgfx906_int8.so` 和解码激活旋转的 `libgfx906_rotate.so`（见下节）。三个 `.so` 随包分发，按 HIP 7.18（ROCm 10.3）的运行时构建，
动态依赖里写死了 `libamdhip64.so.7`：本机是别的大版本时加载会失败，节点照常跑 Triton 路径，并在启动日志给一行
`... did not load ...; rebuild it for this HIP runtime with tools/build_hip_attn.py --force`。重编命令：

```bash
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/build_hip_attn.py
```

`tools/bench.py --attention` / `--tile` 在 `.so` 缺失时自己跑这个构建（约 4 秒），因为少了它们那两个站点只是静默少一行最快的数据。节点启动不构建任何东西。

产物按目标/平台命名，放在 `arch/<目标>/kernels/`：Linux 是 `lib<目标>_<名字>.so`（gfx906 的即上列三个），Windows 是 `<目标>_<名字>.dll`。代码对象按目标绑定（gfx906 那份的标志是 `gfx906, xnack any, sramecc any`）；加载器按当前 `gcnArchName` 拼出对应名字，找不到就跳过，不会把别的卡带崩。

### int8 线性的大 m 路径

`int8_gemm` 开关（Model Settings）把「行数 ≥ 128」的 int8 线性交给自带的 HIP dot4 GEMM：
128×128 tile、256 线程、每线程 64 个 int32 累加器、**one block/CU（4 波）**。4 波看着反直觉，
但它才是峰值：单 block 才有 256 个寄存器放 64 累加器，LDS 读取比因此从 0.125 降到 0.0625；
实测 2 波 5.5–5.8 TMAC/s、8 波（只能用 16 累加器）9.1–9.6、4 波 **10.0**。

节点内实测（同一进程、convrot 开，含节点自己的旋转+量化）：

| 形状 (m×k×n) | Triton | HIP GEMM | 倍数 |
|---|---:|---:|---:|
| 4032×4096×4096 | 8.26 ms | 7.42 | 1.11x |
| 4032×12288×4096 | 24.05 | 20.65 | 1.16x |
| 4032×4096×24576 | 45.63 | 41.31 | 1.10x |

数值与 Triton 路径的差是 fp16 输出舍入（最大约 1 ulp）；int8 累加本身是精确 int32。
m ≤ 1 的解码仍走 GEMV；m < 128 直接回落 Triton（128 行 tile 在更小 m 上会浪费行）。
`int8_gemm=off` 可强制回到 Triton；`.so` 在本机装不上时该开关自动无效。

## 开关

开关都在节点里（GPU Tuning Model Settings / GPU Tuning CLIP Settings），下一次执行生效。
绝大多数是 `on`/`off` 二态，combo 上显示的就是当前状态；`vae` / `gemv` / `image_tower` 另有
`default` 按 manifest 实测判定。节点开关没有“沿用启动 env”的隐式状态。

环境变量只剩两类：整包总闸，以及没有节点入口的诊断/基准/移植旋钮（含文本生成的分项旋钮，
只在没连 CLIP 节点时生效）。下表是全集；移植与内核细节另见 [PORTING.md](PORTING.md) 和
[HIP_KERNELS.md](HIP_KERNELS.md)。
`__init__.py` 导入时还 `setdefault("TRITON_DEFAULT_FP_FUSION", "1")`：Triton 3.10 会把默认改成关，
本包所有实测都假设 fusion 开；显式设置该变量仍优先，它不是本包的开关。

| 变量 | 默认 | 作用 |
|---|---|---|
| `COMFY_GPU_TUNE` | `1` | `off` 整包关闭；补丁在节点执行前安装，这是唯一总闸 |
| `COMFY_GPU_TUNE_FORCE` | 空 | 逗号分隔的 feature 名，跳过实测判定直接安装：`vision` / `deltanet` / `deltanet_prefill` / `qwen35` / `w4a8` / `te_dequant`（启动会 WARNING：这些内核不是在当前卡上测的） |
| `COMFY_GPU_TUNE_TE_DEQUANT` | 空 | te=default 的寄存器内反量化门：`on`/`off` 覆盖 manifest 的 `feature.te_dequant`；关闭时 te=default 回到逐调用反量化，并自动关掉 TE 图捕获（反量化副本会被图钉 ~21 GB） |
| `COMFY_GPU_TUNE_TILE_CONFIG` | 空 | 手工实测的大 tile，设置后该卡视为已实测（跳过 autotune）；不设时用 manifest 的 `int8_gemm.tile/large` |
| `COMFY_GPU_TUNE_SMALL_TILE_CONFIG` | 空 | 手工实测的窄 tile，替换候选/强制值 |
| `COMFY_GPU_TUNE_INT8_GEMV_ASM` | `auto` | int8 解码 GEMV 后端：`auto` 编译探测 / `on` 强制 asm / `off` 强制 int32 tl.sum |
| `COMFY_GPU_TUNE_W4A8_ASM` | `auto` | 4-bit GEMV 后端：`auto` 编译探测 / `on` 强制 asm / `off` 跳过 asm（还有 HIP 扩展时仍用 HIP，见下一行） |
| `COMFY_GPU_TUNE_W4A8_HIP` | `auto` | 节点目录下的 `w4a8_hip` 扩展：`auto` 有就用 / `off` 忽略它，让 `W4A8_ASM` 的 off 真正落到 per-group tl.dot |
| `COMFY_GPU_TUNE_MEASURED` | `1` | 用 `arch/<gcn>/manifest.toml` 的实测记录按形状选后端；`0` 关掉这一层，回到纯探测 |
| `COMFY_GPU_TUNE_MANIFEST_DIR` | `~/.config/comfy-gpu-tune` | manifest 根目录；本机实测表放这里的 `arch/<gcn>/manifest.toml`。见 [PORTING.md](PORTING.md) |
| `COMFY_GPU_TUNE_ARCH` | 空 | 强制指定目标架构目录（没插卡开发用）。见 [PORTING.md](PORTING.md) |
| `COMFY_GPU_TUNE_ARCH_KERNELS` | 空 | 追加内核 `.so` 搜索目录。见 [PORTING.md](PORTING.md) |
| `COMFY_GPU_TUNE_DELTANET` | `1` | textgen 分项基准旋钮：DeltaNet 解码 Triton 内核 |
| `COMFY_GPU_TUNE_LLAMA` | `1` | 同上：llama RMSNorm 预计算 + RoPE 频率表 |
| `COMFY_GPU_TUNE_QWEN35` | `1` | 同上：vendored Qwen3.5 计算类（含烘焙的 KV decode 折叠、split-half RoPE 等 gfx906 优化） |
| `COMFY_GPU_TUNE_GRAPH` | `1` | textgen 分项基准旋钮：权重常驻时的逐块 CUDA graph；连上节点时由 `textgen` 的 `on`/`off` 覆盖 |
| `COMFY_GPU_TUNE_F16_GEMM` | `1`/`0` | 非量化 fp16 linear 的 HIP GEMM；`off` 强制关掉做 A/B。缺省值由 manifest 的 `feature.f16_gemm` 决定，未实测卡为 `0`。见 [GFX906_EXPERIMENTS.md](GFX906_EXPERIMENTS.md) |
| `COMFY_GPU_TUNE_DIT_FP16` | 空 | `dit_fp16` 的 `default` 值：`off` / `qwen21` / `all`；未设时等于 `qwen21` |
| `COMFY_GPU_TUNE_FP16_TRACE` | `0` | `fp16_trace` 的启动默认值；`1` 时即使不连节点也在前向里记录溢出点 |
| `COMFY_GPU_TUNE_FP16_TRACE_DTYPE` | `fp16` | trace 记录的 dtype：`fp16` / `bf16` / `fp32` / `all`；参考测量用 `bf16`/`fp32` 配 `dit_fp16=off` |
| `COMFY_GPU_TUNE_ROTATE` | `1` | 解码激活旋转的 HIP butterfly 内核；`off` 回落到 Triton 做 A/B。fp16 输入下 HIP 以 fp32 累加，量化激活更准但与 Triton 不等价 |
| `COMFY_GPU_TUNE_W4A16_HIP` | `1` | te=default 4-bit 反量化 GEMV 的 HIP 内核（`arch/gfx906/kernels/w4a16.cu`，m=1 与 m=2..8 两个入口）；`off` 回落到 Triton 的 `dequant_w4a16_kernel` 做 A/B。m>8 或不合形状的批仍走 Triton |
| `COMFY_GPU_TUNE_W4A16_F16ACT` | `0` | 只管**带 codebook** 的检查点是否用 fp16 激活 + `v_dot2_f32_f16` 的 feed（`w4a16_gemv_h`，m=1..8，比无损 feed 快 1.05-4.89x）；代价是 16 个格点退到 fp16（fp32 输出归一化 1.6-2.6e-4、fp16 输出约 1.5 ulp），开启前按 `--sampling off` 重验生文 md5。无 codebook 的检查点不看这个开关，默认就走该 feed（均匀格点在 fp16 里精确） |
| `COMFY_GPU_TUNE_W4A16_VALU` | 空 | **测量门**：缺省时由 `feature.w4a16_valu`（本机/出厂 manifest）决定，出厂表为 `on`；显式 `on`/`off` 覆盖它。均匀格点（无 codebook）的 m=2..4 GEMV 用 `v_cvt_f32_ubyteN` 在 VALU 里构造 16 个格点，替掉 `tab[16]` 的 LDS gather；逐位一致，实测快 3-7%（数字见上文与 [GFX906_EXPERIMENTS.md](GFX906_EXPERIMENTS.md)）。M=8 自动回落表格路径（该形状已 spill），带 codebook 的检查点不看这个开关。`bench.py --w4a16-valu --write` 可重新测并改写判定 |
| `COMFY_GPU_TUNE_TRITON_BACKEND` | 空 | comfy-kitchen 的 triton 后端是否让 dispatch 选：空（默认）= **开**——core 默认禁用 triton，本包在启动时主动启用它（等价于 `--enable-triton-backend`，启动日志那条 `triton backend enabled` 即由此而来），只有显式 `--disable-triton-backend` 才保持关；`on`/`off` 显式覆盖且优先于启动参数。`off` 让本包没覆盖的 kitchen 操作回落 eager，便于 A/B（实测 4096x4096 上 `quantize_int8_rowwise` 113.7→381.3 us、`dequantize_per_tensor_fp8` 666.9→863.4 us）；int8 linear 不受影响——两个后端的 `int8_linear` 都包了本包的路由，兜底始终是 triton 内核，因为 eager 的 `torch._int_mm` 在本卡每个 m 都抛 `HIPBLAS_STATUS_INVALID_VALUE` |

