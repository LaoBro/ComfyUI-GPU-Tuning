# ComfyUI-gfx906-Tuning

给 AMD gfx906（Vega 56/64、MI50、MI60、Radeon VII）上跑 DiT 模型和 LLM 文本生成的加速补丁包。

这张卡没有矩阵核心，`bf16` 是软件模拟，ComfyUI 的默认实现在它上面处处踩慢路径。
本包在启动时给 int8 权重、注意力、VAE 换成本机实测更快的实现，**不改 ComfyUI 源码**，
`COMFY_GFX906_TUNING=off` 可整体关闭。

实测（ComfyUI 0.39.0 + torch 2.15 ROCm + triton 3.9，MI50 32 GB）：

- 图像：Qwen-Image 2.1 turbo（int8 convrot，1024²，8 步）**18.66 → 5.25 s/it**
- 文本：Qwen3.5-9B int8 convrot 文本编码器做 128 token 生成，**4.11 → 6.40 it/s**（243 → 157 ms/token）

## 效果分解

| 部分 | 改动前 | 改动后 | 说明 |
|---|---|---|---|
| 注意力 | 641 ms（sub_quad，fp32） | **57 ms**（gemm，fp16） | 布局 × dtype × 内核，合计 11x |
| int8 linear | 默认 tile 列表 | **8.3–10.2 TMAC/s** | 本包追加的 tile 比上游 autotune 可选的最优配置快 ~26% |
| VAE 解码（1024²） | 2016 ms | **787 ms** | conv3d → conv2d，2.6x |
| QKV 投影 | 3 次 GEMM | 1 次 | -0.08 s/步 |
| 文本编码 | 反量化每次调用 | int8 常驻 | 编码快 2.4x |
| 文本生成 | 786 ms/token | **58 ms/token** | tile + lm_head 量化 + 单 token GEMV + 直调内核，13.6x |

## 八个补丁

1. **int8 linear tile**：上游 autotune 列表是给 CDNA/CUDA 的（128×256×64、8 warps），
   在 gfx906 上换成 128×256×32、4 warps、2 stages。单 token 解码（m ≤ 8）另走一套窄 tile，
   见下面「文本生成」。
2. **convrot 融合**：蝶形 Hadamard 旋转与 int8 量化合成一个 kernel（原本两次全局读写）。
3. **注意力入口**：把 `[B,S,H,D]` 统一转 fp16 + 连续布局，再分发给选中的后端。
   可选 `int8` / `int8qk` / `hand` / `gemm`，以及 ComfyUI 自带的 `basic` / `pytorch` /
   `sub_quad` / `split` 等全部后端。
4. **文本编码器 int8**：权重保持量化（每次调用不再反量化）。
5. **QKV 三合一**：q/k/v 三个投影合成一次 int8 GEMM（逐行 scale 拼接，bit-exact）。
6. **VAE conv3d → conv2d**：图像模型带的视频 VAE 有 76 个 depth=1 的 Conv3d，
   gfx906 没有 conv3d 内核，torch 会降级成 im2col + bmm（占解码时间 47%）。
7. **lm_head 保持量化**：逐 token 生成时 ComfyUI 的 `BaseGenerate.logits` 会先走
   `CastBiasWeightContext`，把 int8 词表头反量化成 fp32 再跑 GEMV——单次 90.7 ms，
   比它喂的那次 GEMV（6.3 ms）还贵。补丁把量化权重直接交给内核；真实激活上
   argmax 与 top-5 完全一致，|Δlogit| 最大 0.078（logit 量级 16.4）。
8. **单 token int8 GEMV**：m=1 时不走分块 GEMM，一程序 16 行输出、沿 K 累加 int32，
   旋转 + 量化 + 点积压成两次启动。文本编码器的 Linear 在解码时直接调这个内核，
   跳过 `QuantizedTensor` 的分派层（159 → 81 µs/次，详见下面「文本生成」）。

## 安装

```bash
# 1. ComfyUI Manager / Registry 搜 "GFX906 Tuning"
# 2. 或 git clone 到 custom_nodes/
git clone https://github.com/<your>/ComfyUI-gfx906-Tuning custom_nodes/ComfyUI-gfx906-Tuning
```

重启 ComfyUI，启动日志里会出现：

```
gfx906 tuning active: int8 linear tuned + decode gemv, convrot fused, vae conv3d->conv2d,
qkv fused, attention int8, text encoder int8 + lm_head, encoder attention default
```

## 文本生成（单 token 解码）

LLM 逐 token 解码时每次乘的是 **m=1** 的矩阵，和 DiT 的 M≈4096 是两回事，
DiT 的最优 tile 在这里反而更慢：128 行的 tile 对一行输入要算 64 倍的 MAC，
256 宽的 N tile 又让 4096 宽的权重只剩 16 个 program 铺在 60 个 CU 上。

所以 m ≤ 8 时改用 `16×32×64`（4 warps / 3 stages），大 batch 仍是 `128×256×32`，
按次切换、不触发 autotune 扫参。m=1 实测（4096×4096 / 4096×12288 / 152k lm_head）：
`0.30 / 0.48 / 2.66 ms`，比 256 宽的 tile 快 2.1x / 1.5x / 1.6x。

| 配置（Qwen3.5-9B int8，128 token 生成） | it/s | ms/token |
|---|---|---|
| 不装本包 | 4.11 | 243 |
| 装包但用 DiT tile（旧行为） | 3.12 | 320 |
| 只有 tile + lm_head + GEMV（上一版） | 6.40 | 157 |
| 现在，默认启动 | 6.3 | 155 |
| 上一版 + `--highvram` | 12.9 | 78 |
| **现在 + `--highvram`** | **16.8–17.4** | **58** |

`--highvram` 现在快一倍，原因是它避开了 ComfyUI 的动态显存分块：默认启动时 CLIP 会被
`load_models_gpu` 判成 "dynamic VRAM loading"（日志 `9031MB Staged`），9 GB 权重按块换入，
GPU 时间从 37.5 涨到 ~135 ms/step，wall 155 ms 里 87% 是 GPU 在等权重。`--highvram` 让权重
全量常驻（`current: cuda:0`），GPU 回到 37.5 ms，瓶颈变成 python 调度（见 `gfx906fix/textgen-tuning.md`）。
它以前更慢是因为那时 TE 还会逐 token 反量化，本包的文本编码器补丁已经绕开那条路。

### m=1 走专用 GEMV

tile 只能把分块 GEMM 调到 55 GB/s；解码的权重是"读一遍、每个元素用一次"，正确形状是不分块。
`int8_gemv.py` 里一程序负责 16 行输出，沿 K 以 1024 为步长累加 `int32`（整数求和可结合，
所以结果与 GEMM 路径同值）。旋转与量化也折进来：旋转内核顺手写出每个 256-group 的最大值，
GEMV 内核自己合成行 scale 并在线量化，一条 linear 从 3 次启动降到 2 次。

单条 linear 实测（合成权重，m=1，深队列下的平均墙钟）：

| 形状 | 分块 GEMM | GEMV | 带宽 |
|---|---|---|---|
| 4096×4096 | 164 µs | **63 µs** | 317 GB/s |
| 12288×4096 | 386 | **99** | 508 |
| 8192×4096 | 288 | **80** | 419 |
| 4096×12288 | 384 | **118** | 458 |
| 152k lm_head | 2695 | **924** | 674 |

解码逐 token 的 GPU 时间从 ~55 ms 降到 **17.4 ms**（旋转另占 3.3 ms）。

### 直调内核，跳过分派层

`F.linear(x, QuantizedTensor)` 一次 159 µs，其中 78 µs 花在 Python 侧：
`__torch_dispatch__` 的子类分派、自定义算子的 registry 查表、`input.contiguous()`，
真正到内核只有 81 µs。文本编码器的 Linear 补丁在解码时（单行输入）自己从权重取出
int8 张量与 scale，直接调 `int8_gemv.linear`；prefill（多行）和任何不满足条件的形状
照旧走 `F.linear`，行为不变。实测单次 159 → 81 µs，每步省 16 ms。

### 4-bit 权重（可选，需先转换）

解码是带宽问题，权重从 1 字节降到 4 bit 就能少流一半显存。gfx906 有
`v_dot4_i32_i8`（MI50 的 53 TOPS INT8 就是靠它），Triton 本身不会发这条指令，
`w4a8_gemv.py` 用 inline asm 直接发；nibble 在寄存器里展开
（`w & 0x0F0F0F0F` / `w >> 4`），零点用「组内和 − 8·Σq」修正（逐字节减 8 会借位）。

**优先从 convrot int8 转**：只量化一次，误差最小。已打包的 `asym_w4a8_int8`
checkpoint 也能转，但它带 16 级 Lloyd-Max codebook，等于再量化一次，误差会多出来。

```bash
python custom_nodes/ComfyUI-gfx906-Tuning/tools/requant_uniform4.py \
  --src models/text_encoders/xxx_int8_convrot.safetensors \
  --dst models/text_encoders/xxx_int8_convrot_uniform4.safetensors \
  --merge models/text_encoders/xxx_w4a8.safetensors   # 补 int8 源里没有的 mtp 层
```

`--merge` 只拷贝源文件缺失的键（通常是 `mtp.*`）。MTP 存在与否会改变生成的解码
路径，做 A/B 时两边必须一致。词表头和视觉塔不转换，保持 int8。

质量（相对 int8 权重的 relL2，越小越好）：

| 权重 | 大小 | relL2 |
|---|---|---|
| int8 convrot（源） | 9.47 GB | 基准 |
| 原 w4a8（codebook） | 6.72 GB | 0.0736 |
| **int8 → uniform4** | 6.72 GB | **0.0784** |
| w4a8 → uniform4（二次量化） | 6.72 GB | 0.1042 |

转换只做一次（本机 200 层约 13 s）。实测 LM 的 linear 从 20.9 → **15.9 ms/step**。

只在 decode（单行）生效，prefill 仍走 kitchen 的 dequant + int8 GEMM；权重形状
不被 2048 整除时同样回退。换好之后在 CLIP 加载器里选转换后的文件。

### 现在的时间分解

用真实 2226 token 系统提示词做差分 profile（两个长度相减，扣掉 prefill），每个 decode step：

| | ms/step |
|---|---|
| 全部 GPU 内核 | **37.5** |
| 其中 int8 GEMV | 17.4 |
| 其中旋转 | 3.3 |
| CPU 自计时（直调前 / 后） | 65 / **49** |
| 稳态墙钟，权重常驻（前 / 后） | 73–78 / **56** |

GPU 37 ms 已经和 CPU 打平，剩下的开销是每步约 7000 个零碎算子（33 ms）加上每次调用里
两次 triton 启动的固定成本。默认启动时这 37 ms 还会被动态显存分块放大到 ~135 ms，
所以那一档的墙是 GPU 而不是 CPU——直调内核只对权重常驻（`--highvram`）有效，
默认启动仍是 6.3 it/s。

再往下只剩 CUDA graph：它把整步录成一张图，回放一次就是一次提交，理论上限就是 37 ms/token。
三个已知前置障碍：KV 位置在 python 侧、prefetch 每层一次 CPU→CUDA 拷贝、delta-rule 首个
decode step 会在 CPU 上开内存。`COMFY_GFX906_GEMV=0` 可关掉 GEMV 与直调做对比。

## 三个节点

| 节点 | 位置 | 作用 |
|---|---|---|
| **GFX906 Model Settings** | 模型链（加载器 → 采样器之间） | 选注意力模式、fp16 开关、convrot 模式、profiler 开关 |
| **GFX906 CLIP Settings** | CLIP 链（加载器 → 编码） | 文本编码器 int8 开关、编码器注意力后端 |
| **GFX906 Tuning Info** | 任意位置（可选） | 打印当前生效的配置、实时计数，`run_benchmark` 可现场复测 int8 linear |

设置节点改动**下一次执行就生效**，不用重启。

### 注意力模式怎么选

| 模式 | 精度 | 适用 |
|---|---|---|
| `int8`（默认） | 有 int8 量化误差 | 最快，SDXL 与 Qwen 都适用 |
| `gemm` | fp16 | **Qwen 这类 head_dim=128、长 K 的 DiT 上最快**，比 `hand` 快 1.5–2x；纯 torch 实现，任意 head_dim 都能用（含 SD1.5 的 40、SD2.1 的 80） |
| `hand` | fp16 | 自研 Triton 内核，D=64 时有优势 |
| `int8qk` | 介于两者 | PV 保留 fp16，慢约 10%，精度好 4-7 倍 |

`basic` / `pytorch` / `sub_quad` / `split` 是 ComfyUI 自带实现，选中后原样转发，
用于对比。注意 `pytorch`（SDPA）在 gfx906 上只有 math 后端，通常不是最快。

## 兼容性

| 设备 | 生效范围 |
|---|---|
| gfx906（Vega 56/64、MI50/60、Radeon VII） | 全部补丁，用实测 tile，不跑 autotune |
| 其它 AMD（gfx900、gfx90a/94x、RDNA1 后期、RDNA2） | int8 tile 追加进 autotune 候选、convrot、VAE、文本编码器全部生效；注意力首次调用现场扫 tile 并与 SDPA 比快慢 |
| NVIDIA / CPU | 整包跳过（Triton 的 NVIDIA 后端最低 Volta，Pascal 及更老无解） |

## 开关

全部可用环境变量覆盖（重启生效）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `COMFY_GFX906_TUNING` | `1` | `off` 整包关闭 |
| `COMFY_GFX906_ATTN` | `int8` | 注意力模式 |
| `COMFY_GFX906_FP16` | `1` | 注意力输入转 fp16 |
| `COMFY_GFX906_CONTIG` | `1` | 后端调用前转连续布局 |
| `COMFY_GFX906_SMOOTH` | `1` | int8qk 的均值中心化 |
| `COMFY_GFX906_CONVROT` | `fused` | `fused` / `butterfly` / `off` |
| `COMFY_GFX906_TE` | `int8` | 文本编码器 int8（含生成时的 lm_head） |
| `COMFY_GFX906_TE_ATTN` | `default` | 文本编码器注意力后端 |
| `COMFY_GFX906_GEMV` | `1` | 单 token 解码走专用 int8 GEMV |
| `COMFY_GFX906_W4A8` | `1` | 单 token 解码走 4-bit GEMV（仅无 codebook 的 uniform 权重） |
| `COMFY_GFX906_VAE` | `1` | VAE conv3d → conv2d |
| `COMFY_GFX906_QKV` | `1` | QKV 三合一 |
| `COMFY_GFX906_PROFILE` | `0` | 每次 prompt 结束打印耗时分解 |
| `COMFY_GFX906_ARCHS` | `gfx906` | 视为"已实测"的架构名单 |

## 已知边界

- **QKV 三合一额外占 1.6 GB 显存**（50 MB × 32 层）。1024² 无压力；1328² 以上若显存吃紧，
  先试 `COMFY_GFX906_QKV=0` 对比。
- **int8 注意力在 D=128/长 K 上不再是首选**（77.7 ms vs gemm 57 ms），只在 D=64 的 SDXL 形状占优。
- **不要试图用 fp16 模拟 int8 linear**：实测慢 1.16–1.85 倍（形状越大差得越少）。int8 只赢
  1.3–1.8 倍而不是规格上的 2 倍，因为瓶颈是每线程的独立累加链数，不是精度格式；换 fp16 一样
  填不满发射槽，还多占一倍显存。社区对 `torch._int_mm` 的回退补丁是 fp32 精确回退，定位是
  正确性而非性能。
- **注意力占 37%、int8 linear 占 57%**，两者都已接近该形状的硬件上限；剩余可动空间在 0.5% 量级。
- 文本编码器 int8 会让 embedding 有约 6% 的偏移，肉眼无差别，但追求完全一致时关掉
  （`COMFY_GFX906_TE=default` 同时关掉 lm_head 的 int8，回到逐 token 反量化的慢路径）。
- **文本生成已经不是 GPU 受限**（权重常驻时）：decode 的 GPU 只要 37 ms/step，墙钟 56 ms，
  差额是 python 侧的调用开销（201 次 linear + 约 7000 个算子）。换更快的 kernel 没有意义，
  下一步只能是 CUDA graph 或合并算子。默认启动时另有一层动态显存分块把 GPU 抬到 ~135 ms，
  那属于加载策略问题，见上面的 it/s 表。
- 只针对 fp16/fp32 计算路径做了验证，bf16 在这张卡上是模拟的，本包会主动把注意力转 fp16。

## 验证与细节

所有数字都来自本机实测，复现脚本和逐项推导在仓库外的 `gfx906fix/` 目录：

- `gfx906fix/textgen-tuning.md` — **文本生成调优记录**：现状、时间分解、五个坑、后续方向
- `gfx906fix/qwen-image21-remaining-ops.md` — 各补丁的收益与测量方法
- `gfx906fix/gfx906-triton-attention-tuning.md` — Triton 注意力调优全过程
- `gfx906fix/gfx906-int8-ceiling.md` — int8 在这张卡上的理论上限
- `gfx906fix/hip_attn/NOTES_F16.md` — HIP fp16 内核（D=128 可用，未进入本包）

## 许可

MIT。补丁只改动运行时行为，不附带任何模型权重。
