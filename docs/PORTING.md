# 移植到其它显卡

这个包默认只在 **AMD + ROCm/HIP** 上生效，并且只有 `gfx906` 随包带了实测数据。
NVIDIA 和 CPU 会整包跳过（`apply()` 里的 `is_amd()` 判断）。

好消息是：大部分功能跟显卡型号无关，开箱就能用。只有少数功能需要这张卡自己
测一次才知道该开还是该关。本文讲清楚哪些能直接用、哪些要测、测完写到哪、
出问题怎么查。给别的架构写 HIP 内核请看 [HIP_KERNELS.md](HIP_KERNELS.md)。

## 三步跑通

1. **确认前提**：AMD 卡、PyTorch 是 ROCm/HIP 构建、`comfy-kitchen` 的 triton 后端
   可用、ComfyUI 0.39.0 或更新。逐块 CUDA graph 由节点的 `resident_graph.py`
   在启动时以运行时包装装上，不需要改任何 core 文件。
2. **用节点默认值先跑一遍**。没跑过 bench 的卡上「测量门」功能是关的 ——
   这是预期行为，不是坏了。
3. **跑一次** `bench.py --all --write`（命令见附录 A）。它测出 int8 tile 和各项
   开关，写进本机 manifest，之后启动不再 autotune。用 `tools/status.py` 确认
   `measured ... in use`，日志里不该再有 `no measured data`。

## 三类功能

所有功能按「答案会不会随卡变」分三类，这是整个包的门控总纲：

| 类 | 判据 | 怎么处理 |
|---|---|---|
| **A 常开** | 行为不随卡变、没有取舍 | 直接装，发现更慢再单独关 |
| **B 可选开关** | 代价不是速度，或精度有分层 | 默认给一个合理值，用户可手选 |
| **C 测量门** | 最优解随卡变 | **没数据就是关**，跑一次 bench 由数据决定 |

C 类最关键：**没跑过测量就别猜**。数据有两个来源，本地覆盖出厂：

- `arch/<gcn>/manifest.toml` 随包分发，只对已实测架构生效，开箱即得；
- `~/.config/comfy-gpu-tune/arch/<gcn>/manifest.toml` 由 `bench.py --write` 写，
  带本机指纹，逐形状覆盖出厂表。

指纹包含 ROCm 版本、Torch 构建号、Triton 版本和 gcn 目标。**换一次 torch 就要
重测**：内核是那个编译器产出的，同版本 ROCm 的另一次构建也可能挪动几个百分点。
指纹不符时整张表被忽略（不是逐条赌），启动会 WARNING。

`COMFY_GPU_TUNE_FORCE=deltanet,w4a8,...` 是手动逃生口：跳过一切判定直接装，
启动会 WARNING 说明这些内核没在当前卡上测过。

还有一点容易误解：这里的「安装」是**替换模块级符号或类**，不是设一个开关位。
没装上的项 `status()` 显示 `n/a`，节点里把 `textgen` 选成 `on` 也打不开它。

## 门控放在哪一层

原则：**dispatch 能回答的交给 dispatch，管不到的才用门或开关**。

`comfy-kitchen` 的 dispatch 按每次调用校验 constraints，原生后端拒绝的形状会
自动落到 triton。所以 int8 tile 补丁在所有 AMD 上都无条件安装，但取值由 manifest
指纹自守 —— 没数据的卡拿到的就是 stock 候选列表，不会把 60 CU 调出来的参数
塞给一张不认识的卡（LDS/寄存器预算不同，可能编不过）。

dispatch 管不到的几类：

- `vae` / `convrot` 的改写发生在调用路径里，在 dispatch 之前 → `feature.*` 管；
- `attention` 替换的是 `optimized_attention` 本身 → mode 由节点/env 手动选，
  带 mask / GQA 的调用整段绕行到 core。**文本生成根本不经过它**：TE 用
  `small_input=True` 解析到 core 的原始后端，所以 attention 的开关只影响 DiT；
- `qkv` 是 1.6 GB 常驻的纯权衡 → 默认关；
- 其余文本生成改写与后端无关，全部安装；`deltanet` / `w4a8` 按 C 类判定。

`deltanet` 是半例外：kitchen 没有 fused op 的卡上它结构性开启（这个事实不随
测量过期），有 fused op 的卡上由 `feature.deltanet` 决定。`bench` 也不写
`feature.qwen35`（用例只覆盖 prefill，收益主要在 decode，测量覆盖不了决策面），
所以 qwen35 留在 B 类。

优先级一句话：**节点显式 on/off > `COMFY_GPU_TUNE_*` > 内置默认**。节点里的
`default` / `auto` 表示沿用 env，不会拿硬编码值盖掉它。

## 加一个内核或后端

简短版，完整契约见 `arch/gfx906/kernels/README.md` 和 [HIP_KERNELS.md](HIP_KERNELS.md)：

- 同一算子的多个实现用 `BackendRegistry` 声明，列表顺序即优先级，**最后一项
  必须是 portable**，保证 `resolve()` 总能给出结果；
- 实现遇到自己不支持的形状返回 `None`，调用方沿同一条链往下走，实现之间不互相 import；
- 架构专用实现放 `arch/<gcn>/kernels/`，一个文件一个实现；导入失败只记一行日志
  并跳过，不该让 ComfyUI 起不来；
- 判定结果进程内缓存（含失败），`reset()` 只给工具/测试用；
- 测量值写 `arch/<gcn>/manifest.toml`。**实现列表是代码，测量结果是数据**，
  分开放：前者可 review、可单测，后者一次 bench 就能重写。
- 调度时按形状问注册表，优先级是
  **显式 env > manifest 的 measured 记录 > 探测顺序**。

「节点跑了但没在用实测数据」是最糟的状态：表现和正常一模一样，只是慢。
所以 `measured` / `stale` / `absent` / `off` 四种状态各有对应的 WARNING 或
`status()` 行，`COMFY_GPU_TUNE_MEASURED=0` 可以整体关掉这一层救急。

## 归因与 A/B 纪律

- **找时间落在哪一段**：生文链路用 `tools/chain_profile.py`，DiT 侧用节点的
  profile 开关。微基准只回答「这段里谁快」，不回答「时间在哪」。
- **一次只动一个变量**。同时改 pack 和 CUDA graph 的对比表没有归属（附录 E 里
  有这个反例）。
- **md5 只在 `--sampling off` 下才是等价判据**。采样开启时微小的数值差会逐
  token 放大成完全不同的文本；反过来，更快但生文从第一个 token 就退化，
  也是真失败。
- **热点在 bench 里没有对应站点，先加站点再优化**。`vision.patch_embed` 和
  `deltanet.prefill` 都是这么来的。
- 改动内核后跑 `tools/verify_backends.py`（选择 parity + 数值 A/B，需要 GPU）。
- 数值门槛先于计时：`bench.py` 每个形状先和 `tools/refs.py` 的 torch 参考比对，
  超 `--max-rel` 的实现不参与计时，「更快但算错了」不可能赢。`--min-gain`
  （默认 3%）是噪声下限，差距小于它就不宣布赢家。

## 启动鲁棒性

`apply()` 里每个 patch 都经过 `_install(label, fn)`：只有**设备/编译级**异常
（`ImportError` / `AttributeError` / `OSError` 和各家编译错误）会降级并记进
`install_failures`；`ValueError` / `TypeError` 之类照常抛出 —— 那是我们的 bug，
吞掉只会变成更难查的「某些功能莫名不生效」。降级项会出现在 `status()` 末尾。

`AttributeError` 在白名单里是刻意的：ComfyUI 版本漂移最常见的表现就是属性没了。
代价是可能掩盖我们写错的名字，所以失败必须同时出现在 `status()` 里，不能只留日志。

## 已知边界

- `kernel.configs` 是进程级共享，当前假设模型串行执行；并发要改成按调用传参。
- 未实测卡走候选模式时，每个新 `(m,n,k)` 第一次调用会现测候选，首个 shape 慢。
- 静态 tile 表按卡型固化：驱动、编译器或实际 shape 变了要重测。
- `feature.*` 只有 `default`、没有 per-shape 记录，回答的是「这张卡整体划不划算」，
  不是「这个形状划不划算」。所以 `--shapes` 要覆盖你实际在跑的尺寸。
- `deltanet` 只比了单次调用后的输出和最终 state，`snapshots` 中间态没逐个比。
- `qwen35` 的 A/B 只覆盖 prefill 和合成权重，不含 recurrent 解码和真实 checkpoint。
- `group_size < 16` 时 w4a8 落到 `tl.sum` 保底：保证能跑、结果正确，但通常比 int8 慢。
- attention 的 tile 用内置常量、mode 手动选：同一 head dim 在 prefill 和 decode
  下最优 tile 不同，单一常量两头兼顾不了（好在只影响 DiT）。

---

# 附录

需要查数字、命令或字段时看这里。正文只讲决策，数据都放在后面。

## A. 命令速查

从 ComfyUI 根目录运行：

```
# 只打印，不写文件
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/bench.py

# 一张卡需要的一切：tile + 各 feature 的 on/off，写入本机覆盖
... --all --write

# 只测一个站点 / 指定形状
... --site w4a8.m1 --shapes 4096x4096@m1,4096x4096@m8 --write

# 直接改随包分发的表（准备 PR 时用）
... --write --out custom_nodes/ComfyUI-GPU-Tuning/arch/gfx906/manifest.toml

# 扫单个 tile 候选、打印可直接用的环境变量
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/tile_sweep.py --emit

# 生文链路归因
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/chain_profile.py --clip <文件名> --max-length 32

# 不开 ComfyUI 看判定 / 改内核后的 parity 与数值 A/B
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/status.py
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/verify_backends.py
```

手工指定 tile：`COMFY_GPU_TUNE_TILE_CONFIG` 让该卡视为「已实测」、列表只放这一个
配置、不跑 autotune；`COMFY_GPU_TUNE_SMALL_TILE_CONFIG` 只换小 tile 候选。
两个变量都是启动期读取，manifest 里的 `int8_gemm.tile` 与 env 同权（env 优先）。
manifest 根目录可用 `COMFY_GPU_TUNE_MANIFEST_DIR` 改，`COMFY_GPU_TUNE_ARCH` 可强制
指定目标目录（没插卡时开发用），`COMFY_GPU_TUNE_ARCH_KERNELS` 可追加搜索目录。

## B. 功能全表

**A 常开 + B 可选开关**：TE int8 权重常驻、lm_head 保持量化、TE 注意力后端、
`text_gemv`、`amax_guard`、KV decode 折叠、llama RMSNorm / RoPE 频率表、
prefix KV 复用、profiler（常开）；qkv（1.6 GB 常驻，默认关）、qwen35（等价类
替换，env 默认开）、attention mode（随包 `hip` 内核能加载时默认 `hip`，否则
`int8`，也可手选）。

**C 测量门**：

| 功能 | 没数据时的行为 | 怎么测 |
|---|---|---|
| int8 linear tile | 只用 stock 6 个候选 | `bench.py --all --write` 扫赢家写 `int8_gemm.tile` |
| `small_batch` 窄 tile | 候选模式，autotune 决定 | 测 `m<=8` |
| `gemv`（模型 int8 解码 GEMV） | 不安装 | 与 stock int8 线性基线比，赢则 `feature.gemv=on` |
| 大 m `int8_gemm` | 回落 Triton | `.so` 能本机加载才生效；`bench.py --tile` 出证据 |
| `attention` | mode 手动选 | `--attention` 只打印证据，不写记录 |
| `w4a8` | 不安装 | 与 int8 基线比，赢则 `feature.w4a8=on` |
| DeltaNet | kitchen 无 fused op 时结构性开启 | 与 kitchen fused op 比输出、state、耗时 |
| `deltanet_prefill` | `feature` 决定 | 与 core 的逐行 UT 变换比，mild/coherent 两档取最差 rel |
| `qwen35` | B 类，不由 bench 写 | `--qwen35` 只出证据 |
| `convrot` | 不安装 | kitchen matmul 旋转 vs butterfly vs fused，写 `feature.convrot` + `convrot.mode` |
| `w4a16_valu` | 走 `tab[16]` 表格 | 均匀格点 m=2..4 的表格 gather vs VALU 格点，深队列计时，写 `feature.w4a16_valu` + `w4a16.valu` |
| `vae` | 不安装 | depth-1 Conv3d 三方计时，写 `feature.vae` |
| `image_tower` | 包装器总装，默认看 `feature.vision` | patch 投影三种走法，写 `vision.patch_embed` |
| `qkv` | 默认关 | 纯权衡，不归测量门 |

易踩的坑：`configure(w4a8=on)` 会在节点执行时直接把模块装上，绕过
`feature.w4a8` 的安装判定。非 gfx906 上 asm 探测会自动失败、落到 per-group
`tl.dot`；但显式 `COMFY_GPU_TUNE_W4A8_ASM=on` 会跳过探测，实际调用时报编译错误。

`feature.*` 由 bench 写七项：`gemv`、`w4a8`、`deltanet`、`deltanet_prefill`、
`vae`、`vision`、`convrot`。不写三项：`attention`（质量取舍）、`qwen35`、
`te_dequant`（表里那行手写，bench 无对应站点）。

## C. manifest 结构

```toml
schema = 1

[arch]
gcn = "gfx906"
fingerprint = "rocm7.18.26434|torch2.15.0+rocm10.3.0a20261102|triton3.9.0|gfx906"
capabilities = ["v_dot4_i32_i8", "v_perm_b32"]

[sites."int8_gemv.m1"]
default = "asm"
default_grade = "derived"

[sites."int8_gemv.m1".shapes]
"4096x4096@m1" = { impl = "triton", ms = { asm = 0.0851, triton = 0.0833 }, rel = 0.0, grade = "measured" }

[sites."int8_gemm.tile".shapes]
"large" = { impl = "block_m=128,block_n=256,block_k=32,group_size_m=8,warps=4,stages=2", grade = "measured" }
"small" = { impl = "block_m=16,block_n=32,block_k=64,group_size_m=8,warps=4,stages=3", grade = "measured" }
```

站点形态：

| 站点 | key | impl | 谁读 |
|---|---|---|---|
| `<op>.<形状类>` | `NxK@mM` | 实现名 | 注册表按形状选后端 |
| `int8_gemm.tile` | `large` / `small` | tile 文本 | `_triton_config()` / `_small_tile_config()` |
| `convrot.mode` | `mMxK` | `fused` / `butterfly` | `resolved_convrot_mode()` |
| `deltanet.decode` / `.prefill`、`vae.decode`、`vision.patch_embed`、`qwen35.block` | 各自形状 | 实现名 | 只作证据 |
| `feature.<item>` | 只有 `default` | `on` / `off` | `install_ready()` |

只有前三类加 `feature.*` 会被运行时读到，其余形状记录删掉不影响调度。

| grade | 含义 | 参与调度 |
|---|---|---|
| `measured` | 与 fingerprint 相同的机器上产出 | 是 |
| `derived` | 从别的机器/版本转录 | 否 |
| `assumed` | 猜的 | 否 |

`default` 没有 `default_grade` 就算 `assumed`；只要有一条可信 shape 与它冲突，
`default` 作废。shape key 是 `NxK@mM`，`NxK@*` 表示该 NxK 的所有 batch 宽度。
schema 不符、TOML 语法错、单条记录类型不对都只影响那一条，不拦启动：
`kernels/manifest.py` 的 `load()` 从不抛异常，也不在模块层 import torch。

`--write` 逐行改文本，注释、别的站点、未见过的 shape 原样保留；落盘前用 TOML
解析器验一遍。若某站点所有实测形状赢家一致，顺带把 `default` 提升为 `measured`。

## D. bench 站点明细

`--all` 追加六组测量：

- **int8 tile**：3 个 NxK 乘 `m=1,8,128`，`m<=8` 与 `m>8` 两桶各取中位数赢家。
  这层不用 `--min-gain`，跨形状中位数已压掉单次噪声。
- **attention**：只出证据不写记录。默认 d128 长/短 self+cross 各一形状、d64 一个
  SDXL 形状；每个 config 先比数值再计时，最后打各内核相对 SDPA 的中位比例。
- **deltanet**：默认 `1x16x128x128x1` 与 `...xs4`；先比输出**和** state，
  再和 kitchen fused op 同时计时。对端不可用时只记自己的耗时。
- **vae**：depth-1 Conv3d 三方计时（torch conv3d / conv2d squeeze / kitchen
  `fp16_conv3d`），padding 0 以便可比。
- **convrot**：kitchen matmul 旋转+量化 vs butterfly 两步 vs fused 单内核。
- **qwen35**：合成 4 层小模型（约 240M 参数），同一套权重跑两遍，gate 关逐字调用
  core 的方法。fp32 下 1e-3 门槛是真正的等价性检查。只出证据。

`w4a8.m1` 还多计一条 int8 基线（先 `decode_int8` 解一次、再走 int8 GEMV/GEMM），
那才是关掉 w4a8 时真正会跑的路径。`--w4a8-uniform` 测均匀格点，
`--w4a8-group` 默认 128、改成 8 会跑 `tl.sum` 保底。

`w8a16.m1` / `w4a16.m1` 量 te=default 的寄存器内反量化（m≤8），同形状的
其它内核并排进表，用来判断反量化离上限多远。它们只出证据、不写 `feature.*`：
`feature.te_dequant` 的对端是逐调用反量化，不是另一个 GEMV。读表前先按
HIP_KERNELS.md 基本流程第 1 条分清两种计时——本站点是单次计时，比内核本身多 20-30 us。
站点喂的是 fp32 激活（参考值也在 fp32 算），所以 `dequant` 那一列永远是 fp32 feed 的
内核：fp16 激活的 `w4a16_gemv_h` 不会在这里出现，它的数字见 HIP_KERNELS.md 的深队列表。

各站点形状可用 `--tile-shapes` / `--tile-m` / `--attention-shapes` /
`--deltanet-shapes` / `--vae-shapes` / `--convrot-shapes` / `--qwen35-*` 覆盖。
表里没有的形状回落 `default`，再没有就回到 autotune/探测。

计时默认关掉 convrot（旋转是两种实现共用的独立 kernel，去掉它比较的才是 GEMV
本身）。bench 会自己分配约 1 GB 权重，不要和出图同时跑。

## E. 实测数据

**gfx906 链路归因**（`chain_profile.py`，两次运行差了两个变量，不是净收益）：

| | pack on（图开、te=int8、32 token、有图） | pack off（图关、te=default、16 token、无图） |
|---|---|---|
| 图像塔 | 34.9% | — |
| GatedDeltaNet | 19.9% | 20.6% |
| MLP | 12.6% | 51.9% |
| 每步 forward 自身 | 16.3% | 6.1% |
| `logits` | 1.28 ms/次 | 89.3 ms/次 |
| prefill / 每步 / tok/s | 1511 ms / 25.8 ms / 38.7 | 2208 ms / 670 ms / 1.49 |

固定单一变量的例子（`text_gemv`，两边都 te=int8、图关、同 seed、同 prompt）：

| | `text_gemv=on` | `text_gemv=off` |
|---|---|---|
| 每步（sampling / greedy） | 48.2 / 49.8 ms | 62.6 / 64.4 ms |
| MLP 中位 | 0.45 ms | 0.64 ms |
| `logits` 中位 | 1.45 ms | 1.42 ms |
| tok/s | 20.8 | 16.0 |

读数注意：`graphed` 列的调用不进 python，图开着时 block 以下只包含 prefill 和
捕获那一遍；`unattributed` 大说明瓶颈不在 GPU 内核上；`--pack off` 时
`--set text_encoder=int8` 不生效（工具会打印说明）。

**w4a8：`tl.dot` 能否替代 `v_dot4`**：数学等价可以，指令等价和性能等价不行。
Triton 3.9 没有 `tl.dp4a`，整型 dot 只有 `tl.dot(a_int8, b_int8, out_dtype=tl.int32)`，
要求 M/N/K >= 16；而 w4a8 GEMV 的 m=1..8 要补齐到 16，且每个 16-wide group
有独立 `s_rel`，一次覆盖整个 K 的 dot 无法按 group 乘 scale。纯 Triton 只剩
「每 group 一次 K=16 小 dot」或「折进 fp16 再大 dot」两条路：

| 形状 | m | asm GEMV | int8 dot per group | fp16 dot + scale |
|---|---|---|---|---|
| 4096x4096 | 1 | 104 us | 263 us | 416 us |
| 4096x4096 | 8 | 175 us | 282 us | 482 us |
| 12288x4096 | 1 | 110 us | 775 us | 978 us |
| 12288x4096 | 8 | 381 us | 833 us | 1124 us |

数值正确（rel diff <= 0.0003）但慢 1.6-7 倍。其它架构可以把纯 Triton 版本作为
fallback，别指望在 gfx906 上追平 asm。原型脚本（不随节点分发）：
`gfx906fix/w4a8_dot_proto.py`、`gfx906fix/w4a8_dot_group.py`。

**`int8_gemv` asm 后端**（`BN=16, BK=2048`）：

| 形状 | int32 tl.sum | asm dot4 | 提升 |
|---|---|---|---|
| 4096x4096 | 83.3 us | 85.1 us | 约 -2% |
| 4096x12288 | 127.6 us | 101.5 us | 1.26x |
| 12288x4096 | 117.9 us | 93.0 us | 1.27x |
| 152064x4096（lm_head） | 924 us | 823 us | 1.12x |

提升有限且随形状变：m=1 是带宽问题，`v_dot4` 只减少乘加指令数。其余 int8 内核
（GEMM tile、attention QK/PV）的 `tl.dot` 已在合适形状上走整型 dot 指令，autotune
也在选 tile，不建议手写 asm；解包/量化/旋转类内核没有 dot 可换。

后端链（`w4a8.m1`）：`hip` 扩展 > `asm`（编译探测 `v_dot4_i32_i8`）>
`triton`（per-group `tl.dot`，要求 group_size >= 16 且是 16 的倍数）>
`sum`（`tl.sum` 保底，无条件接受所有合法形状）。`int8_gemv.m1` 用同一套，
候选是 `asm` / `triton`。`COMFY_GPU_TUNE_W4A8_ASM` / `_HIP` /
`COMFY_GPU_TUNE_INT8_GEMV_ASM` 都是 `auto` / `on` / `off` 三态。
选择 parity 用 `tools/registry_selftest.py`（无 GPU，60 种情形），
真机 parity 与数值 A/B 用 `tools/verify_backends.py`。
节点三态开关的语义用 `tools/settings_selftest.py`（需要 venv 里的 torch，无 GPU）。

## F. 非目标设备

| 场景 | 原因 |
|---|---|
| NVIDIA（有 GPU） | triton 内核补丁与 attention 不装（cuda capability 集没有 `int8_linear`，本包也没在 N 卡测过 tile，保持 stock autotune）。文本生成重写组照常生效；`w4a8`/`deltanet` 由本机 bench 决定；qwen35 随 env 默认开。manifest key 是 `sm<major><minor>`。**未在 N 卡实测** |
| NVIDIA，权重非全量常驻 | 逐块 CUDA graph 不会捕获（权重按块换入走 core 的 vbar 路径），只影响 launch 开销 |
| CPU | `torch.cuda.is_available()` 为假，整包不安装 |
| ComfyUI 升级后 | 所有 core 接触点都是包装原函数，最坏降级为 eager 解码，不需要重新生成任何文件 |

AMD 上 kitchen 已有后端的卡走上一节的 dispatch 逻辑：只装与后端无关的改写，
没有需要按卡验证的内核。

## G. 开发流程建议

1. 用默认值跑通（测量门默认关是预期）。
2. `bench.py --all --write`，再用 `tools/status.py` 确认 `measured ... in use`。
3. 归因，别猜；热点没有站点就先加站点。
4. 逐项 A/B，一次只动一个变量。
5. `gfx906` 上 `textgen` 四项都是 on；别的卡上 `status()` 打 `n/a` 是正常的。
6. 改内核后跑 `verify_backends.py` 再记录性能。
7. 换了 torch/驱动就重跑第 2 步；表写错时 `COMFY_GPU_TUNE_MEASURED=0` 整体关掉。
