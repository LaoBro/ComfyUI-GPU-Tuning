# ComfyUI-GPU-Tuning

给 ComfyUI 的 DiT 模型和 LLM 文本生成换成本机实测更快的实现：int8 linear、注意力、VAE、
文本编码器与单 token 解码。全部以运行时包装安装，**不改 ComfyUI 源码**，
`COMFY_GPU_TUNE=off` 可整体关闭；内核按 GPU 架构的实测表分派，gfx906（Vega 56/64、MI50/60、
Radeon VII）是第一个带完整实测表的架构。

实测（gfx906 / MI50 32 GB，ComfyUI 0.39.0 + torch 2.15 ROCm + triton 3.9）：

- 图像：Qwen-Image 2.1 turbo（int8 convrot，1024²，8 步）**18.66 → 5.25 s/it**
- 文本：Qwen3.5-9B int8 convrot 解码 **1.27 → 35–37 it/s**（791 → ~28 ms/token）

> 文本生成要权重常驻才到这个数字：启动加 `--disable-dynamic-vram`（`--highvram` 同样有效），
> 避开 ComfyUI 的动态显存分块（默认档每步 GPU 时间 ~37 → ~135 ms）。这不是补丁的功能；
> 逐块 CUDA graph 由本包默认开启。细节见 [docs/PATCHES.md](docs/PATCHES.md)。

## 效果分解（gfx906 / MI50 32 GB）

| 部分 | 改动前 | 改动后 | 说明 |
|---|---|---|---|
| 注意力 | 641 ms（sub_quad，fp32） | **57 ms**（gemm，fp16） | 布局 × dtype × 内核，合计 11x |
| int8 linear | 默认 tile 列表 | **8.3–10.2 TMAC/s** | 本包追加的 tile 比上游 autotune 可选的最优配置快 ~26% |
| VAE 解码（1024²） | 2016 ms | **787 ms** | conv3d → conv2d，2.6x |
| QKV 投影 | 3 次 GEMM | 1 次（默认关） | -0.08 s/步 |
| 文本编码 | 反量化每次调用 | int8 常驻 | 编码快 2.4x |
| 文本生成 | 791 ms/token（1.27 it/s） | **~28 ms/token（35–37 it/s）** | tile + lm_head + 单 token GEMV + 直调 + 逐块 graph，~28x |

> 阅读路径：安装与开关在本文；八个补丁和文本生成推导见 [docs/PATCHES.md](docs/PATCHES.md)，
> 开关细节与注意力模式见 [docs/CONFIG.md](docs/CONFIG.md)，移植与内核开发见
> [docs/PORTING.md](docs/PORTING.md) / [docs/HIP_KERNELS.md](docs/HIP_KERNELS.md)；
> gfx906 的实测边界见 [docs/GFX906.md](docs/GFX906.md)。

## 安装

```bash
# git clone 到 custom_nodes/
git clone https://github.com/LaoBro/ComfyUI-GPU-Tuning custom_nodes/ComfyUI-GPU-Tuning
```

不需要改动 ComfyUI 源码，也没有安装脚本；逐块 CUDA graph 的 hook 随节点在启动时安装。

重启 ComfyUI，启动日志里会出现：

```
gpu-tuning active: int8 linear measured tile + decode gemv, convrot fused, vae conv3d->conv2d,
qkv off, attention hip, text encoder int8 + lm_head, encoder attention default,
textgen deltanet on, kvdecode on, llama on, qwen35 on
```

## 两个节点

| 节点 | 位置 | 作用 |
|---|---|---|
| **GPU Tuning Model Settings** | 模型链（加载器 → 采样器之间） | 注意力模式、连续布局、fp16 输入策略、convrot、int8 tile、`m≤8` 窄 tile、int8 大 m GEMM、int8 注意力与旋转、QKV、VAE、GEMV、prefix KV、Qwen2.1 int8 缓存、profiler |
| **GPU Tuning CLIP Settings** | CLIP 链（加载器 → 编码） | 文本编码器 int8、编码器注意力后端、TE GEMV、4-bit w4a8、文本生成补丁组 |

设置节点改动**下一次执行就生效**，不用重启。多数开关是 `on`/`off` 二态，combo 显示的
就是当前状态；`vae` / `gemv` / `image_tower` 多一个 `default`，按 manifest 实测判定
（没跑过 bench 的卡保持原路径）。`attention` / `convrot` / `fp16_attn` /
`int8_attn_rotate` / `text_encoder` / `text_attention` 是模式或数值选择。文本生成的分项
`COMFY_GPU_TUNE_*` 只在没连节点时生效，连上节点后完全由节点控制。

Model Settings：`attention`、`contiguous`、`fp16_attn`、`convrot`、`int8_tile`、`small_batch`、`int8_gemm`、`int8_attn`、`int8_attn_rotate`、`qkv`、`vae`、`gemv`、`prefix_kv`、`qwen21_int8_cache`、`profile`。

CLIP Settings：`text_encoder`、`text_attention`、`text_gemv`、`w4a8`、`image_tower`、`textgen`。

每个开关的含义、注意力模式（`hip` / `gemm` / `hand`）怎么选、int8 大 m 路径和环境变量全表，
见 **[docs/CONFIG.md](docs/CONFIG.md)**；八个补丁与文本生成的实现见 **[docs/PATCHES.md](docs/PATCHES.md)**。

## 兼容性

- **gfx906（Vega 56/64、MI50/60、Radeon VII）**：开箱即用，随包带实测表（`arch/gfx906/manifest.toml`），启动不跑 autotune。
- **其它架构**：装好后可直接用，测量门功能默认关、走通用路径。跑一次 `tools/bench.py --all --write` 会按本机实测写出 manifest，之后启动即免 autotune；功能分级、需要移植的项和扫描工具见 [docs/PORTING.md](docs/PORTING.md)。把仓库交给 agent 按 PORTING 执行是最省事的方式，多数情况不需要手写内核。
- **NVIDIA**：内核补丁与注意力不安装，文本生成的改写组照常生效；本包未在 N 卡实测，性能结论由有卡的人跑。
- **CPU**：`torch.cuda.is_available()` 为假，整包不安装。

## 已知边界

- QKV 三合一额外常驻一份合并权重、只省约 1.5%，默认关。
- 文本编码器 int8 会让 embedding 有约 6% 的偏移；追求完全一致时选 `text_encoder=default`。
- 4-bit 权重需要先用 `tools/requant_uniform4.py` 转换，且只在解码（单行）生效。
- 只验证过 fp16/fp32 计算路径；bf16 在目标卡上是模拟的，本包会把注意力转 fp16。
- gfx906 上的实测上限、已删除的路径和“不要再做”的结论见 [docs/GFX906.md](docs/GFX906.md)。

## 验证与细节

所有数字都来自本机实测。测量工具随包分发：

- `tools/bench.py` — 站点微基准（数值门先于计时），`--all --write` 写本机 manifest
- `tools/chain_profile.py` — **生文链路归因**：一次 TextGenerate 的嵌套 CUDA event 分解（图像塔 / block / MLP / lm_head / prefill vs 每步），按结构发现归因点，跨架构可用，见 [docs/PORTING.md](docs/PORTING.md)
- `tools/status.py` / `tools/verify_backends.py` — 不开 ComfyUI 看当前判定 / 改内核后的选择 parity 与数值 A/B
- `tools/registry_selftest.py` / `tools/manifest_selftest.py` / `tools/settings_selftest.py` — 无 GPU 自检：后端链决策、manifest 解析与信任、节点三态开关（settings 需要 venv 里的 torch）

补丁层面的推导（文本生成、注意力、int8 上限、第一版内核）见
[docs/PATCHES.md](docs/PATCHES.md) 末尾的“实验记录”一节。

## 许可

MIT。补丁只改动运行时行为，不附带任何模型权重。
