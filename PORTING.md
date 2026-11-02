# 其它显卡移植与二次开发指南

本节点默认只对 **AMD + HIP** 生效，并且把 `gfx906` 当作“已实测架构”。本文说明在其它 AMD 显卡上：

- 哪些功能可以直接用；
- 哪些需要先测量/调参；
- 哪些需要改代码或移植；
- 哪些完全不能用；
- 如何扫描 int8 tile、如何让同型号显卡免 autotune。

NVIDIA / CPU 上整个包会跳过（[gfx906_tuning.py](gfx906_tuning.py#L251)、`apply()` 的 `is_amd()` 判断），不在本文支持范围内。

## 1. 前提条件

- AMD 显卡，PyTorch 为 ROCm/HIP 构建；
- `comfy-kitchen` 的 triton backend 可用（否则 int8/convrot/attention 补丁会跳过并只打日志）；
- 可选：ComfyUI 0.39.0 + `core-patches` overlay（resident CUDA graph 用，非必须，版本不匹配时不要装）。

## 2. `tuned` 是什么

`tuned` 不是运行时自动调优，而是**静态的“这张卡我们实测过”判定**：

```python
def target_archs():          # COMFY_GFX906_ARCHS，默认 "gfx906"
    return os.getenv("COMFY_GFX906_ARCHS", "gfx906").replace(",", " ").split()

def arch_name():             # 显卡 0 的 gcnArchName，去掉 :sramecc+ 后缀
    ...

def on_target_arch():        # tuned 的来源
    return arch_name() in target_archs()
```

`apply()` 中：

- `tuned=True`：直接替换 int8 tile、使用实测 attention tile，并安装 DeltaNet / Qwen3.5 / w4a8；
- `tuned=False`：int8 tile 作为 autotune 候选、attention 首次调用现场扫参，DeltaNet/Qwen3.5/w4a8 **不安装**。

这里的“安装”是替换模块级符号或类，而不是设置一个关闭状态：

- `deltanet.install()` 替换 `comfy_kitchen.gated_delta_decode_fused` 等三个符号；
- `qwen35_model.install()` 替换 `comfy.text_encoders.qwen35.Qwen35Transformer`；
- w4a8 把模块指针放进 `_state["w4a8_module"]`。

所以未安装时 `status()` 显示 `n/a`，不是 `off (installed)`；节点再选 `textgen=on` 也打不开。

**注意：`COMFY_GFX906_ARCHS` 是“全家桶”开关。** 把新卡加进去会同时强制 int8 tile、安装 DeltaNet/Qwen3.5/w4a8。只是想要 tile 时，请用 `COMFY_GFX906_TILE_CONFIG`，不要改 `ARCHS`。

## 3. 功能分级

### A. 可直接使用（无需改代码）

| 功能 | 说明 |
|---|---|
| TE int8 权重常驻（`text_encoder=int8`） | kitchen int8 内核，无需架构实测；有约 6% embedding 偏移，可关 |
| lm_head 保持量化 | 与 TE int8 同开关 |
| TE 注意力后端（`text_attention`） | core 后端路由，通用 |
| `text_gemv` | TE 直调 GEMV + 未量化小投影 `torch.mv`，数值等价 |
| `amax_guard` | 注意力 fp16 溢出保护，纯通用逻辑 |
| KV decode 折叠（`textgen` 的 `kvdecode`） | 数学等价（仅 bmm 分块顺序差异） |
| llama RMSNorm / RoPE 频率表（`textgen` 的 `llama`） | 值等价缓存，通用 |
| prefix KV 复用（`prefix_kv`） | `copy_` 代替 `torch.cat`；只对 Qwen-Image 2.1 有实际调用 |
| profiler | 纯计时 |

这些功能默认开启后可直接跑；发现更慢时用对应开关关掉即可。

### B. 能用，但建议先测量/调参

| 功能 | 默认行为 | 需要做什么 |
|---|---|---|
| int8 linear tile | 未实测卡上 stock 6 个 + gfx906 大/小 tile 都作为 autotune 候选；每个新 shape 首次调用会扫参 | 用 `tools/tile_sweep.py` 测出 winner，设置 `COMFY_GFX906_TILE_CONFIG` / `SMALL_TILE_CONFIG`；列表只放单一配置，免 autotune |
| `small_batch` 窄 tile | 候选模式，由 autotune 决定 | 测 `m≤8`；更慢或不想付首次扫参成本时设 `small_batch=off` |
| `gemv`（模型 int8 解码 GEMV） | 其它 AMD 也会安装并用于 m=1 | 数值等价；不同卡带宽/占用不同，建议 A/B |
| 注意力入口（`attention`） | 未实测卡首次调用比较 int8/fp16 tile 与 SDPA | 可先测 head_dim/seq；`gemm` 是纯 torch，任意 head_dim 可试 |
| `convrot` | fused/butterfly 可能比原 matmul 快，依赖 Triton 编译 | 在目标卡 A/B `fused` / `butterfly` / `off` |
| `vae` | conv3d→conv2d 默认开 | 其它卡可能本来就有高效 conv3d；建议 A/B |
| `linear8` | 未量化 fp16 Linear→int8 默认开 | 有强 fp16/矩阵核心的卡上可能变慢；建议 A/B |
| `qkv` | 默认关 | 想用时测 1.5% 收益是否值得 1.6 GB 常驻 |

### C. 需要改代码/移植

| 功能 | 现状与原因 | 修改入口 |
|---|---|---|
| `w4a8`（4-bit 权重 GEMV） | 只在 `tuned` 时加载（[gfx906_tuning.py](gfx906_tuning.py#L1772)）；内核用 gfx9 inline asm `v_dot4_i32_i8`（[w4a8_gemv.py](w4a8_gemv.py#L68-L73)） | 换指令或改用 `tl.dot`，按目标 ISA 重新编译验证；未移植前不要打开 |
| DeltaNet Triton 内核 | 只在 `tuned` 时安装（[gfx906_tuning.py](gfx906_tuning.py#L1753)）；原因是 gfx906 上 kitchen HIP fused op 没有 code object | 去掉 `tuned` 门槛并验证；若目标卡 kitchen HIP 路径本来可用，可不做 |
| Qwen3.5 计算类替换 | 只在 `tuned` 时安装（[gfx906_tuning.py](gfx906_tuning.py#L1764)） | 去掉门槛 + 逐层等价性验证；主要收益是 RoPE slice、合并 a/b GEMV、conv tap |
| 按架构静态 tile 表 | 目前只有 gfx906 默认常量；其它卡靠 env/autotune | 可选增强：把 `--emit` 的 `INT8_TILE_TABLE` 条目接进代码，同型号卡默认免 autotune |

**易踩的坑**：`configure(w4a8="on")` 会在节点执行时调用 `load_w4a8_gemv()`，即使 `tuned=False` 也会把模块装上。非 gfx906 用户手动打开 `w4a8` 可能触发 inline asm 编译失败，因此文档默认建议保持 off。

### D. 完全不能使用

| 场景 | 原因 |
|---|---|
| NVIDIA / CPU | `is_amd()` 返回 False，整个包不安装；要支持需改设备判断并重新验证内核 |
| 非 gfx906 上开 `w4a8` | inline asm 是 gfx906 实测指令；未移植前视为不可用 |
| ComfyUI 不是 0.39.0 时使用 core overlay / resident CUDA graph | overlay 是版本固定的三个核心文件（[core-patches/README.md](core-patches/README.md#L43-L53)），换版本需重新生成；节点本身仍可不用 overlay 运行 |

## 4. 模型特定功能只是 no-op

以下功能在非目标模型上是“什么都不做”，不是故障：

- `prefix_kv`、`qkv`：只对 Qwen-Image 2.1 有意义；
- `deltanet`、`qwen35`：只对 Qwen3.5 文本编码器有意义；
- `prefix_kv` 关闭时回到 `torch.cat`；`qkv` 关闭时回到三次 GEMM。

## 5. 扫描 int8 tile（`tools/tile_sweep.py`）

脚本逐个实测候选 tile，打印每个 `(shape, m)` 的前几名，并给出大/小 batch 的 winner：

```bash
# 默认：stock 6 个 + gfx906 大/小 tile，5 个 m，3 个 NxK
python custom_nodes/ComfyUI-gfx906-Tuning/tools/tile_sweep.py

# 只测自己的候选（可重复 --candidate）
python .../tools/tile_sweep.py \
  --candidate block_m=128,block_n=128,block_k=64,group_size_m=8,warps=8,stages=3 \
  --candidate block_m=16,block_n=64,block_k=64,group_size_m=8,warps=4,stages=3 \
  --m 1,2,8,128,4032 --shapes 4096x4096,4096x12288

# 打印可直接用的启动变量和上游表条目
python .../tools/tile_sweep.py --emit
```

脚本不调用 `apply()`；它把 kitchen 的 `configs` 临时设成单个候选再调 `int8_linear`，所以每次只编译/运行这一个 tile，不触发 autotune 扫参。

输出示例：

```text
4096x4096 m=1: 16x32x64 w4 s3 0.30 ms, 128x32x32 w4 s4 0.37 ms, ...
large tile (m>8): block_m=128,...  median 1.24 ms
small tile (m<=8): block_m=16,...  median 0.31 ms
```

`--emit` 给出的启动变量可直接使用，无需改代码：

```bash
COMFY_GFX906_TILE_CONFIG="block_m=128,block_n=256,block_k=32,group_size_m=8,warps=4,stages=2" \
COMFY_GFX906_SMALL_TILE_CONFIG="block_m=16,block_n=32,block_k=64,group_size_m=8,warps=4,stages=3" \
python main.py
```

语义：

- 设置 `COMFY_GFX906_TILE_CONFIG` 后，该卡视为“tile 已实测”：列表只放这一个配置，**不跑 autotune**；
- 只设置 `COMFY_GFX906_SMALL_TILE_CONFIG` 不改变“已实测”判定，只替换小 tile 候选/强制值；
- 两个变量都是启动期读取；节点 `int8_tile` / `small_batch` 的 `default` 会沿用它们。

扫描结果也可以整理成 `INT8_TILE_TABLE` 条目提交上游，让同型号显卡默认免 autotune。

## w4a8：能否用 `tl.dot` 替代 `v_dot4`

结论：**数学等价可以，指令等价和性能等价不行。gfx906 上保留 inline asm；其它卡可以把纯 Triton 版本作为 fallback。**

### Triton 里的整型 dot

Triton 3.9 没有单独的 `tl.dp4a`，整型 dot 的唯一入口是：

```python
tl.dot(a_int8, b_int8, out_dtype=tl.int32)
```

int8 attention 就是这么用的（[attention_kernel.py](attention_kernel.py#L157)、[attention_kernel.py](attention_kernel.py#L176)），AMD 后端会在合适形状下降到整型 dot 指令族。

但 `tl.dot` 是 tile 级操作，要求 M/N/K ≥ 16，而 w4a8 的 GEMV 是：

- m = 1–8，必须补齐到 16；
- 每个 16-wide group 有独立的 `s_rel`，一次覆盖整个 K 的 dot 无法再按 group 乘 scale。

所以只有两条纯 Triton 路径：

1. **每个 group 单独一次 int8 dot**（K=16，K=4096 时 256 次 dot）；
2. **把 s_rel 折进权重转 fp16，再做一次大 dot**。

### gfx906 实测（uniform 4-bit，group=16）

| 形状 | m | 现有 asm GEMV | int8 dot per group | fp16 dot + scale |
|---|---|---|---|---|
| 4096×4096 | 1 | 104 µs | 263 µs | 416 µs |
| 4096×4096 | 8 | 175 µs | 282 µs | 482 µs |
| 12288×4096 | 1 | 110 µs | 775 µs | 978 µs |
| 12288×4096 | 8 | 381 µs | 833 µs | 1124 µs |

- 纯 Triton 版本数值正确（rel diff ≤ 0.0003）；
- 但仍然慢 1.6–7 倍，原因：M 补齐到 16 的浪费、256 次 K=16 小 dot 的启动/布局开销、没有矩阵核心。
- 现有 `w4a8_gemv` 其实是 Triton 内核 + 一条 `v_dot4_i32_i8` inline asm；解包、scale、epilogue 都是 Triton。`tl.inline_asm_elementwise` 是 Triton 官方为这类语言层缺失 intrinsic 预留的入口。

### 移植建议

1. 把 dot 计算做成双实现：
   - gfx906 / 支持 `v_dot4_i32_i8` 的卡：保留 inline asm；
   - 其它卡：用 per-group `tl.dot` 或 fp16 dot fallback。
2. 用架构判断或首次调用编译探测选择实现；可先提供 opt-in 环境变量（例如 `COMFY_GFX906_W4A8_DOT=1`），稳定后再自动选择。
3. 纯 Triton 版本的目标是“正确、可移植、能跑”，不要期望在 gfx906 上追平 asm。
4. 可继续试 3D batched `tl.dot`（batch=group 数）来减少 256 次循环开销，但 gfx906 上没有矩阵核心，提升空间有限。

开发树里的原型脚本（不随节点分发）：

- `gfx906fix/w4a8_dot_proto.py`：fp16 dot + scale 折叠；
- `gfx906fix/w4a8_dot_group.py`：int8 dot per group，BN=32 为实测最佳。

### 后端选择（已实现）

`w4a8_gemv.linear` 按以下顺序选择实现：

1. **`w4a8_hip` 扩展**：如果节点目录下存在可导入的 `w4a8_hip` 且提供 `linear()`，优先使用；
2. **asm**：首次调用时编译探测 `v_dot4_i32_i8` 是否可用（`COMFY_GFX906_W4A8_ASM=auto`），成功则用现有 Triton+asm 内核；
3. **triton**：否则用 per-group `tl.dot` 回退。

环境变量：

- `COMFY_GFX906_W4A8_ASM=auto`（默认）：编译探测；
- `=on` / `asm`：强制 asm（探测失败会在实际调用时报编译错误）；
- `=off` / `triton`：强制 per-group `tl.dot`。

per-group `tl.dot` 需要 `group_size >= 16` 且是 16 的倍数；不满足时 `linear()` 返回 `None`，调用方会回退到 int8/`F.linear` 路径。

`status()` 会显示 `w4a8 on: N calls, backend=hip/asm/triton`。

### HIP 扩展接口

如果以后写出更快的 HIP 内核，把编译产物放到节点目录（模块名 `w4a8_hip`），实现：

```python
def linear(x, qdata, s_rel, s_channel, bias, out_dtype,
           group_size, convrot_groupsize, levels=None, inv_level=1.0):
    """Same contract as w4a8_gemv.linear; return None when unsupported."""
```

加载失败或返回 `None` 时自动回退到 asm / Triton，不需要改节点代码。

## 其它 int8 内核能否改成 asm

结论：**只有 `int8_gemv.py` 已接入 asm 可选路径；int8 GEMM 和 int8 attention 不建议；其余内核没有 dot 可换。**

| int8 内核 | 现状 | 能否 asm 化 | 判断 |
|---|---|---|---|
| int8 GEMM tile（kitchen `_int8_matmul_*`） | `tl.dot(int8, int8, int32)` + autotune | 技术上能手写 dot4 GEMM | `tl.dot` 已在合适 tile 上走整型 dot 指令，autotune 也在选 tile；手写 asm GEMM 维护成本高、难更快，**不建议** |
| int8 attention QK/PV | `tl.dot` int8（[attention_kernel.py](attention_kernel.py#L157)、[attention_kernel.py](attention_kernel.py#L176)） | 同上 | 已经是 dot 指令路径，**不建议** |
| `int8_gemv`（m=1 单行 GEMM） | 默认 asm（编译探测），否则 int32 `tl.sum` | **已接入** | 唯一值得试的候选 |
| `w4a8_gemv` | hip > asm > per-group tl.dot | — | 已做 |
| `decode_int8`、`quantize_k/v`、`convrot`、`rotate` | 解包/量化/旋转，不是 dot | 不适用 | asm 帮不上 |
| `linear8`、QKV 融合 | 走 kitchen int8 GEMM | 同第一行 | 不建议 |

### `int8_gemv` 后端与实测

`int8_gemv.linear()` 的选择顺序：

1. `COMFY_GFX906_INT8_GEMV_ASM=on/asm` → asm；
2. `=off/triton` → int32 tl.sum；
3. `auto`（默认）→ 首次调用编译探测 `v_dot4_i32_i8`，成功用 asm，否则纯 Triton。

asm 版本把量化后的 activation 每 4 个 int8 打包成一个 int32，weight 直接按 int32 视图加载，用 `v_dot4_i32_i8` 沿 K 累加。数学与原实现一致（rel diff 0）。

最好配置 `BN=16, BK=2048`：

| 形状 | 现有 int32 | asm dot4 | 提升 |
|---|---|---|---|
| 4096×4096 | 83.3 µs | 85.1 µs | 约 -2%（持平/略慢） |
| 4096×12288 | 127.6 µs | 101.5 µs | 1.26x |
| 12288×4096 | 117.9 µs | 93.0 µs | 1.27x |
| 152064×4096（lm_head） | 924 µs | 823 µs | 1.12x |

提升有限且与形状相关：m=1 的 GEMV 是带宽问题，`v_dot4` 减少的是乘加指令数，权重流式读取和旋转/量化开销不变。4096×4096 已到带宽/开销平衡点，asm 没有优势。

`status()` 会显示 `gemv on (backend=asm/triton)`。如果目标卡不支持 asm，自动回退，不需要改代码。

### 建议

1. 默认 `auto`：有 asm 就用，没有就回退纯 Triton。
2. 如果发现某个形状 asm 更慢，可用 `COMFY_GFX906_INT8_GEMV_ASM=off` 全局关闭。
3. 不要 asm 化 GEMM/attention：`tl.dot` 已是这些形状的正确入口，autotune 也在工作。

开发树里的原型脚本（不随节点分发）：`gfx906fix/int8_gemv_asm_proto.py`。

## 6. 开关优先级

**节点显式 on/off > `COMFY_GFX906_*` > 内置默认**。节点里的 `default` / `auto` 表示沿用 env，不会用硬编码值覆盖它。完整开关列表见 [README.md](README.md#L223-L240)。

## 7. 二次开发流程建议

1. 确认：AMD + HIP、`comfy-kitchen`、triton 可用；overlay 仅 0.39.0 可选。
2. **不要**先把新卡加进 `COMFY_GFX906_ARCHS`；先用节点默认值跑通。
3. 跑 `tools/tile_sweep.py`，用 `--emit` 设置两个 tile 变量。
4. 逐项 A/B：`vae`、`linear8`、`gemv`、`convrot`、`attention`、`qkv`。
5. `w4a8` 保持 off，除非完成指令移植与验证。
6. `textgen` 默认只会启用 `kvdecode` + `llama`；`deltanet`/`qwen35` 显示 `n/a` 是预期行为。
7. 需要 resident CUDA graph 时再装 overlay；不装不影响正确性。
8. 改动内核后，用本机 process 内 A/B（同状态、同输入）验证数值等价，再记录性能。

## 8. 已知边界

- `kernel.configs` 是进程级共享；当前假设 ComfyUI 模型串行执行。并发执行时需要改成按调用传参。
- 未实测卡走候选模式时，每个新 `(m,n,k)` 第一次调用会编译/实测 7–8 个候选，首个 shape 会慢；之后按 `(m,n,k)` 缓存稳定。
- 静态 tile 表是“按卡型固化”的：驱动、编译器或实际 shape 变化后需要重新测。
- `COMFY_GFX906_TILE_CONFIG` 只影响 int8 tile，不会启用 DeltaNet/Qwen3.5/w4a8。
