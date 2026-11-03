# ComfyUI-gfx906-Tuning

给 AMD gfx906（Vega 56/64、MI50、MI60、Radeon VII）上跑 DiT 模型和 LLM 文本生成的加速补丁包。

这张卡没有矩阵核心，`bf16` 是软件模拟，ComfyUI 的默认实现在它上面处处踩慢路径。
本包在启动时给 int8 权重、注意力、VAE 换成本机实测更快的实现，**不改 ComfyUI 源码**，
`COMFY_GFX906_TUNING=off` 可整体关闭。

内核加速是 gfx906 专属的；**文本生成的改写不是**——m=1 走 GEMM、每层分派开销、
KV cache 容量级拷贝、launch 开销，这些在别的卡上一样存在。所以有 kitchen 原生后端
的设备（RDNA2+、NVIDIA）装文本生成改写组；triton 内核补丁只在 AMD 上装、由 dispatch 决定何时运行（见“兼容性”）。

实测（ComfyUI 0.39.0 + torch 2.15 ROCm + triton 3.9，MI50 32 GB）：

- 图像：Qwen-Image 2.1 turbo（int8 convrot，1024²，8 步）**18.66 → 5.25 s/it**
- 文本：Qwen3.5-9B int8 convrot 文本编码器做 128 token 生成，**4.11 → 6.40 it/s**（243 → 157 ms/token）

## 效果分解

| 部分 | 改动前 | 改动后 | 说明 |
|---|---|---|---|
| 注意力 | 641 ms（sub_quad，fp32） | **57 ms**（gemm，fp16） | 布局 × dtype × 内核，合计 11x |
| int8 linear | 默认 tile 列表 | **8.3–10.2 TMAC/s** | 本包追加的 tile 比上游 autotune 可选的最优配置快 ~26% |
| VAE 解码（1024²） | 2016 ms | **787 ms** | conv3d → conv2d，2.6x |
| QKV 投影 | 3 次 GEMM | 1 次（默认关） | -0.08 s/步 |
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
   只省 1.5%（5.24 s 步里 -81 ms），却常驻 1.6 GB 合并权重，所以**默认关**，
   Model Settings 节点的 `qkv` 开关或 `COMFY_GFX906_QKV=1` 打开。
6. **VAE conv3d → conv2d**：图像模型带的视频 VAE 有 76 个 depth=1 的 Conv3d，
   gfx906 没有 conv3d 内核，torch 会降级成 im2col + bmm（占解码时间 47%）。
7. **lm_head 保持量化**：逐 token 生成时 ComfyUI 的 `BaseGenerate.logits` 会先走
   `CastBiasWeightContext`，把 int8 词表头反量化成 fp32 再跑 GEMV——单次 90.7 ms，
   比它喂的那次 GEMV（6.3 ms）还贵。补丁把量化权重直接交给内核；真实激活上
   argmax 与 top-5 完全一致，|Δlogit| 最大 0.078（logit 量级 16.4）。
8. **单 token int8 GEMV**：m=1 时不走分块 GEMM，一程序 16 行输出、沿 K 累加 int32，
   旋转 + 量化 + 点积压成两次启动。文本编码器的 Linear 在解码时直接调这个内核，
   跳过 `QuantizedTensor` 的分派层（159 → 81 µs/次，详见下面「文本生成」）。

除这 8 项外，包内还有几组独立补丁，各自有节点开关：

- **int8 tile 注入**：`int8_tile` 关闭时恢复上游 autotune 列表，不再注入 gfx906 tile；
  `small_batch` 单独控制 `m≤8` 的窄 tile 切换。
- **prefix KV 复用**：`prefix_kv` 关闭时 `prefix_cached_attention` 回到 `torch.cat`。
- **TE GEMV**：`text_gemv` 控制文本编码器的直调 int8 GEMV 与未量化小投影的 `torch.mv`。
- **4-bit w4a8**：`w4a8` 控制 4-bit 权重的 decode GEMV 与 prefill 现解 int8。
- **文本生成组**：`textgen` 一次控制 `deltanet`、`kvdecode`、`llama`、`qwen35` 四项；
  `default` 时仍按各自环境变量独立生效。
- **amax 保护**：`amax_guard` 控制注意力输入溢出时是否放弃 fp16 转换。

## 安装

```bash
# 1. ComfyUI Manager / Registry 搜 "GFX906 Tuning"
# 2. 或 git clone 到 custom_nodes/
git clone https://github.com/<your>/ComfyUI-gfx906-Tuning custom_nodes/ComfyUI-gfx906-Tuning
```

重启 ComfyUI，启动日志里会出现：

```
gfx906 tuning active: int8 linear measured tile + decode gemv, convrot fused, vae conv3d->conv2d,
qkv off, attention int8, text encoder int8 + lm_head, encoder attention default,
textgen deltanet on, kvdecode on, llama on, qwen35 on
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
  --dst models/text_encoders/xxx_int8_convrot_uniform4.safetensors
```

`--merge 另一个checkpoint` 会把源里缺失的键拷过来（通常是 `mtp.*`）。带 MTP 时需要它
才能和原 w4a8 的键集合一致；不带时 `mtp="auto"` 自动关闭，走普通单 token 解码。

内核接受 **1–8 行**（MTP 的草拟/校验就是这个宽度），权重仍然只流一遍。prefill（更多行）
先用 `decode_int8` 把这一层解成 int8，再交给 kitchen 的 int8 GEMM：多一遍纯带宽的流式
解码（12288×4096 一层 0.19 ms，全模型约 26 ms），临时显存只多一层 int8（本模型最大
50 MB）。实测同一 prompt 的编码（1666 → 1785 ms，+7%，其中 GEMM 部分 +8.5%），
换掉原来常驻 6.8 GB 的 int8 副本。不要把解 nibble 塞进 dot 流里做融合 GEMM：每个解出的
tile 都要换布局才能进 `tl.dot`，实测比 int8 GEMM 慢 35%。

不要按 8 行把 prefill 切块交给 4-bit 内核：每个块都要重读一遍完整权重。

**原版 `w4a8`（带 16 级 codebook）同样可用**：codebook 在 GEMV 里用 `v_perm_b32` 查表
（一条指令从 8 字节里按 lane 选一个，码的最高位决定选上半还是下半），级别落到每层一张
int8 网格、倒数并进 epilogue；省掉的零点归约正好抵掉查表开销，200 层合计 16.4 ms/token
（uniform 内核 17.5）。12288×4096 一层 0.13 ms。精度上表把码表量化到 1/127 网格，
权重相对误差约 0.4%（codebook 本身是 7.4%）。prefill 仍先 `decode_int8`（一层 0.24 ms，
与 `_dequant_int4_grouped_to_int8` 逐位相同）再走 int8 GEMM。

实测（1425 token prompt、开 graph）：原版 w4a8 **1–2 tok/s → 44.6 tok/s**（mtp 关，
22.4 ms/token，7.38 GB）；mtp 开是 13.5–14.4 tok/s（16.3 GB）。同条件 mtp 关下
`int8_convrot_uniform4` 是 50.3 tok/s。这份文件带 `mtp.*` 键，`mtp="auto"` 会开启 MTP，
而 MTP 本身更慢（草拟要跑 1+depth 行），想要速度就显式关掉。

词表头和视觉塔不转换，保持 int8。编码时 `s_rel` 被约束在 `[1, 15.5]`：低于 1 会丢失
4-bit 的级数、高于 127/8 时最高码被截断。

质量（相对 int8 权重的 relL2，越小越好）：

| 权重 | 大小 | relL2 |
|---|---|---|
| int8 convrot（源） | 9.47 GB | 基准 |
| 原 w4a8（codebook） | 6.72 GB | 0.0736 |
| **int8 → uniform4** | 6.45 GB | **0.0786** |
| w4a8 → uniform4（二次量化） | 6.63 GB | 0.1042 |

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

## 两个节点

| 节点 | 位置 | 作用 |
|---|---|---|
| **GFX906 Model Settings** | 模型链（加载器 → 采样器之间） | 注意力模式、连续布局、int8qk 中心化、fp16、amax 保护、convrot、int8 tile、`m≤8` 窄 tile、QKV、VAE、GEMV、prefix KV、profiler |
| **GFX906 CLIP Settings** | CLIP 链（加载器 → 编码） | 文本编码器 int8、编码器注意力后端、TE GEMV、4-bit w4a8、文本生成补丁组 |

设置节点改动**下一次执行就生效**，不用重启。命令行复测 `benchmark()` 已删除：它没有节点注册、也没有任何调用者，长期是死代码；需要复测时直接用 `gfx906fix/` 下的独立脚本，避免再维护第三个节点。

所有开关都是三态 combo：

- `default`：沿用该功能的 `COMFY_GFX906_*` 启动值（完全不改变现有默认行为）；
- `on` / `off`：显式覆盖 env；
- `text_encoder` / `text_attention` 用 `auto` 表示“跟随 env”，因为这两个参数里的 `default` 已有既定含义。

优先级：**节点显式 on/off > 环境变量 > 内置默认**。以前“节点默认值会覆盖 env”的问题已修掉。

### 节点开关清单

Model Settings：`attention`、`contiguous`、`smooth`、`fp16`、`amax_guard`、`convrot`、`int8_tile`、`small_batch`、`qkv`、`vae`、`gemv`、`prefix_kv`、`profile`。

CLIP Settings：`text_encoder`、`text_attention`、`text_gemv`、`w4a8`、`textgen`。

补充说明：

- `contiguous`：选中 `basic`/`pytorch`/`sub_quad`/`split` 等 core 后端时，调用前把输入转成连续的 `[B,H,S,D]`；关闭可复现 core 的原始转置视图调用。
- `smooth`：SageAttention2 的均值中心化，只在 `int8qk` 模式下生效。
- `amax_guard`：注意力输入绝对值超过 60000 时放弃 fp16 转换，留在模型 dtype，避免 fp16 溢出。
- `int8_tile`：关闭后恢复上游 autotune 的 tile 列表，不再注入 gfx906 的 `128×256×32`；GEMV 仍由 `gemv` 单独控制。
- `small_batch`：gfx906 上把“`m≤8` 切到 `16×32×64` 窄 tile、`m>8` 切回 `128×256×32`”单独控制；其它 AMD 上表示“把窄 tile 追加为 autotune 候选”（是否真的用它由该卡实测决定）。需要 `int8_tile` 为 on，否则显示 n/a。GEMV 仍由 `gemv` 单独控制。
- `text_gemv`：单行文本编码器解码时，int8 权重直调 GEMV、未量化的小投影走 `torch.mv`；关闭后两者都回到分派层。
- `w4a8`：4-bit 权重解码；`status()` 会打印实际调用次数和当前后端（`hip` / `asm` / `triton`），换 checkpoint 后可据此确认是否真的走到了 w4a8 内核（`on: 0 calls ...` 表示权重布局没被识别）。能否用 `tl.dot` 替代 `v_dot4` 的实测结论见 [PORTING.md](PORTING.md#w4a8能否用-tldot-替代-v_dot4)。
- `textgen`：一次控制 DeltaNet 内核、KV decode 折叠、llama RMSNorm/RoPE 缓存、Qwen3.5 计算类；关闭时逐调用委托回 core 实现（Qwen3.5 类符号本身不换回）。

### 在其它显卡上使用 / 扫描 int8 tile

其它 AMD 显卡上的功能分级（常开 / 可选开关 / 测量门 / 不可用）、门控与 dispatch 的分工、扫描工具 `tools/tile_sweep.py` 的完整用法，以及 `COMFY_GFX906_TILE_CONFIG` / `SMALL_TILE_CONFIG` 的使用方式，见：

**[PORTING.md — 其它显卡移植与二次开发指南](PORTING.md)**

最短路径：先用节点默认值跑通（没跑过 bench 的卡上测量门功能是关的），再运行一次 `python custom_nodes/ComfyUI-gfx906-Tuning/tools/bench.py --all --write`。它测出这张卡的 int8 tile、attention tile/mode、GEMV/w4a8/vae/convrot 的开或关，写进 `~/.config/comfy-gfx-tune/arch/<gcn>/manifest.toml`，之后启动不再 autotune。

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
功能分三类：**常开**（prefix KV、KV decode、llama、TE GEMV、TE int8）、**可选开关**（qkv、qwen35）、**测量门**（int8 tile、attention、w4a8、deltanet、vae、convrot、gemv——没跑过 bench 就是关，跑过之后由本机/出厂 manifest 决定开或关）。

| 设备 | 生效范围 |
|---|---|
| gfx906（Vega 56/64、MI50/60、Radeon VII） | 全部生效：出厂自带实测表（`arch/gfx906/manifest.toml`），测量门功能开箱即得，不跑 autotune |
| 其它 AMD，kitchen 无后端（gfx900、gfx90a/94x、RDNA1 后期） | 测量门功能默认关（tile 走 stock autotune、attention 首调现场扫参）；跑一次 `bench.py --all --write` 后按本机实测自动开关，启动即免 autotune |
| 其它 AMD，kitchen 已有 hip 后端（RDNA2/3/4） | 同上；tile 补丁照装但退为兜底（dispatch 优先 hip）。attention 照装并自测与 SDPA 的快慢。常开组始终生效 |
| NVIDIA（有 GPU） | triton 内核补丁与 attention 不装（`is_amd()` 为假；cuda 的 capability 集没有 `int8_linear`，N 卡的 int8 线性走 triton，但本包没在 N 卡测过 tile，保持 stock autotune）。常开组生效：prefix KV、KV decode 折叠、llama 缓存、TE GEMV、TE int8；`w4a8`/`deltanet` 由本机 bench 的 `feature.*` 决定（manifest key 是 `sm<major><minor>`）。**未实测**，性能结论要由有卡的人跑；强烈建议加 core-patches（per-block CUDA graph，与 GPU 无关） |
| CPU | `torch.cuda.is_available()` 为假，整包不安装 |

其它 AMD 的功能分级、需要测量/移植的项和扫描工具用法见 [PORTING.md](PORTING.md)。

## 开关

环境变量在 `apply()`（进程启动）时读一次，作为**启动默认值**；节点里显式选 `on/off` 时覆盖它，`default/auto` 时回落到它。所以：

**节点显式 on/off > 环境变量 > 内置默认**

`COMFY_GFX906_TUNING` 是唯一无法被节点替代的开关：补丁在节点执行前就已安装，且没有卸载路径；不接节点的用户也靠它关闭整包。`COMFY_GFX906_PROFILE` 保留为无节点/headless 启动时的入口。

| 变量 | 默认 | 作用 |
|---|---|---|
| `COMFY_GFX906_TUNING` | `1` | `off` 整包关闭 |
| `COMFY_GFX906_ATTN` | `int8` | 注意力模式（int8 / int8qk / hand / gemm / sdpa）：内核质量不同（int8 PV rel ~2% vs hand ~0.03%），是手动质量取舍，不进测量门 |
| `COMFY_GFX906_FP16` | `1` | 注意力输入转 fp16 |
| `COMFY_GFX906_CONTIG` | `1` | 后端调用前转连续布局 |
| `COMFY_GFX906_SMOOTH` | `1` | int8qk 的均值中心化 |
| `COMFY_GFX906_AMAX_GUARD` | `1` | 输入 abs-max ≥ 60000 时跳过 fp16 转换 |
| `COMFY_GFX906_CONVROT | 空 | `fused` / `butterfly` / `off`；空表示用 manifest 的 `convrot.mode`，且 `feature.convrot` 决定装不装 |
| `COMFY_GFX906_INT8_TILE` | `1` | 注入 gfx906 的 `128×256×32` int8 tile |
| `COMFY_GFX906_SMALL_BATCH` | `1` | `m≤8` 切 `16×32×64` 窄 tile（其它 AMD 作为候选） |
| `COMFY_GFX906_TILE_CONFIG` | 空 | 手工实测的大 tile，设置后该卡视为已实测（跳过 autotune）；不设时用 manifest 的 `int8_gemm.tile/large` |
| `COMFY_GFX906_SMALL_TILE_CONFIG` | 空 | 手工实测的窄 tile，替换候选/强制值 |
| `COMFY_GFX906_TE` | `int8` | 文本编码器 int8（含生成时的 lm_head） |
| `COMFY_GFX906_TE_ATTN` | `default` | 文本编码器注意力后端 |
| `COMFY_GFX906_TE_GEMV` | `1` | TE 单行直调 GEMV + 未量化小投影 `torch.mv` |
| `COMFY_GFX906_GEMV` | 空 | 模型 int8 linear 的单 token 解码 GEMV；空表示用 manifest 的 `feature.gemv` |
| `COMFY_GFX906_INT8_GEMV_ASM` | `auto` | int8 解码 GEMV 后端：`auto` 编译探测 / `on` 强制 asm / `off` 强制 int32 tl.sum |
| `COMFY_GFX906_W4A8` | `1` | 4-bit 权重：单 token 走 GEMV（uniform 或 16 级 codebook），prefill 现解 int8；安装判定见 `feature.w4a8`，`FORCE` 可强开 |
| `COMFY_GFX906_W4A8_ASM` | `auto` | 4-bit GEMV 后端：`auto` 编译探测 / `on` 强制 asm / `off` 跳过 asm（还有 HIP 扩展时仍用 HIP，见下一行） |
| `COMFY_GFX906_W4A8_HIP` | `auto` | 节点目录下的 `w4a8_hip` 扩展：`auto` 有就用 / `off` 忽略它，让 `W4A8_ASM` 的 off 真正落到 per-group tl.dot |
| `COMFY_GFX906_MEASURED` | `1` | 用 `arch/<gcn>/manifest.toml` 的实测记录按形状选后端；`0` 关掉这一层，回到纯探测 |
| `COMFY_GFX906_FORCE` | 空 | 逗号分隔，在非目标架构上单独放行 `deltanet` / `qwen35` / `w4a8` / `attention`（启动会 WARNING：这些内核不是在当前卡上测的） |
| `COMFY_GFX906_DELTANET` | `1` | DeltaNet 解码 Triton 内核 |
| `COMFY_GFX906_KVDECODE` | `1` | 折叠 GQA 的 KV decode |
| `COMFY_GFX906_LLAMA` | `1` | llama RMSNorm 预计算 + RoPE 频率表 |
| `COMFY_GFX906_QWEN35` | `1` | Qwen3.5 计算类替换（等价改写，逐方法对过 core） |
| `COMFY_GFX906_VAE` | 空 | VAE conv3d → conv2d；空表示用 manifest 的 `feature.vae` |
| `COMFY_GFX906_QKV` | `0` | QKV 三合一（打开需 1.6 GB 常驻） |
| `COMFY_GFX906_PREFIXKV` | `1` | prefix cache 的 K/V 合并用复用 buffer + `copy_` 代替 `torch.cat` |
| `COMFY_GFX906_PROFILE` | `0` | 每次 prompt 结束打印耗时分解 |

## 已知边界

- **QKV 三合一额外占 1.6 GB 显存**（50 MB × 32 层）且只快 1.5%，因此默认关；
  显存富余时再用节点开关（或 `COMFY_GFX906_QKV=1`）打开。
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
