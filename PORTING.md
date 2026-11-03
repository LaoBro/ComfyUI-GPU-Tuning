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

后三项和 attention 现在是**逐项**判定的（`install_ready(item)`）：

```python
def install_ready(item):          # item ∈ deltanet / qwen35 / w4a8 / attention
    return _state["tuned"] or item in _state["forced_items"]
```

`COMFY_GFX906_FORCE=deltanet,qwen35` 可以在**不加进 `ARCHS`** 的前提下，只启用其中一项——`ARCHS` 会顺带换掉 int8 tile 和 attention 内核，而这些是 gfx906 的常量，不该被一张新卡继承。强制启用时启动日志会有一条 WARNING，说明这些内核不是在当前卡上测的。

int8 tile 的强制替换仍然只看 `tuned` 或 `COMFY_GFX906_TILE_CONFIG`：那个 tile 是一组具体常量（128x256x32 / 4 warps / 2 stages），在没有当前卡数据的情况下强行指定它，正是我们想避免的“静默变慢”。

这里的“安装”是替换模块级符号或类，而不是设置一个关闭状态：

- `deltanet.install()` 替换 `comfy_kitchen.gated_delta_decode_fused` 等三个符号；
- `qwen35_model.install()` 替换 `comfy.text_encoders.qwen35.Qwen35Transformer`；
- w4a8 把模块指针放进 `_state["w4a8_module"]`。

所以未安装时 `status()` 显示 `n/a`，不是 `off (installed)`；节点再选 `textgen=on` 也打不开。

**注意：`COMFY_GFX906_ARCHS` 仍然是“全家桶”开关。** 把新卡加进去会同时强制 int8 tile、安装 DeltaNet/Qwen3.5/w4a8。只要 tile 时用 `COMFY_GFX906_TILE_CONFIG`；只要某个替换时用 `COMFY_GFX906_FORCE`；不想要全家桶就别改 `ARCHS`。

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
| int8 linear tile | 未实测卡**只用 stock 6 个**候选；gfx906 的大/小 tile 只在 gfx906 上追加（别把 60 CU 的参数塞进不认识的卡，LDS/寄存器预算不同，可能编不过） | 用 `tools/tile_sweep.py` 测出 winner，设置 `COMFY_GFX906_TILE_CONFIG` / `SMALL_TILE_CONFIG`；列表只放单一配置，免 autotune |
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
| `w4a8`（4-bit 权重 GEMV） | 只在 `tuned` 时加载（[gfx906_tuning.py](gfx906_tuning.py#L1772)）。asm 内核用 gfx9 inline asm `v_dot4_i32_i8`；**纯 Triton 的 per-group `tl.dot` 回退已实现**，由 `kernels` 注册表按 `hip > asm > triton` 选（见 §5 后端选择） | 只需去掉 `tuned` 门槛并验证；`group_size` 必须 ≥16 且是 16 的倍数，否则 `linear()` 返回 `None` 回退上游 int8 |
| DeltaNet Triton 内核 | 默认只在 `tuned` 时安装（原因是 gfx906 上 kitchen HIP fused op 没有 code object） | 先试 `COMFY_GFX906_FORCE=deltanet` 单独放行并验证；确认目标卡的 kitchen HIP 路径本来可用时，这个替换反而可能更慢 |
| Qwen3.5 计算类替换 | 默认只在 `tuned` 时安装 | 同上，`COMFY_GFX906_FORCE=qwen35` 单独放行 + 逐层等价性验证；主要收益是 RoPE slice、合并 a/b GEMV、conv tap |
| 按架构静态 tile 表 | 目前只有 gfx906 默认常量；其它卡靠 env/autotune | 可选增强：把 `--emit` 的 `INT8_TILE_TABLE` 条目接进代码，同型号卡默认免 autotune |

**易踩的坑**：`configure(w4a8="on")` 会在节点执行时调用 `load_w4a8_gemv()`，即使 `tuned=False` 也会把模块装上。非 gfx906 上开着 `w4a8` 时 asm 探测会自动失败、落到 per-group `tl.dot`；但如果显式设 `COMFY_GFX906_W4A8_ASM=on`，探测被跳过，asm 会在实际调用时报编译错误。因此文档默认建议保持 off。

### D. 完全不能使用

| 场景 | 原因 |
|---|---|
| NVIDIA / CPU | `is_amd()` 返回 False，整个包不安装；要支持需改设备判断并重新验证内核 |
| 非 gfx906 上开 `w4a8` 且 `group_size < 16` | asm 探测失败会落到 per-group `tl.dot`，但它要求 `group_size >= 16` 且为 16 的倍数；不满足时整条 w4a8 路径返回 `None`，退回 int8 / `F.linear`（是回退，不是崩溃） |
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

### 后端选择（已实现，走统一注册表）

`w4a8_gemv.py` 底部用一个 `kernels.BackendRegistry` 声明本站点的实现列表，顺序即优先级：

```python
_REGISTRY = kreg.BackendRegistry(
    "w4a8.m1",
    [
        kreg.Candidate("hip", kind="extension", env="COMFY_GFX906_W4A8_HIP", loader=_load_hip),
        kreg.Candidate("asm", kind="asm", env="COMFY_GFX906_W4A8_ASM", probe=_probe_asm),
        kreg.Candidate("triton", kind="portable"),
    ],
)
```

`linear()` 不再写死 `hip > asm > triton`：它取 `backend_name()`，某个实现当场返回 `None` 时用 `_REGISTRY.next_after("hip")` 沿同一条链往下走。新增一个内核＝在列表里加一条：

1. **`w4a8_hip` 扩展**（`extension`）：有可导入的模块且提供 `linear()` 时优先；
2. **asm**（`asm`）：首次判定时编译探测 `v_dot4_i32_i8`；
3. **triton**（`portable`）：per-group `tl.dot`，必须是列表最后一项。

环境变量：

- `COMFY_GFX906_W4A8_ASM=auto`（默认）：编译探测；
- `=on` / `asm`：强制 asm（探测失败会在实际调用时报编译错误）；
- `=off` / `triton`：跳过 asm（这是"asm 关掉"，不是"只用 triton"）；
- `COMFY_GFX906_W4A8_HIP=auto`（默认）：有扩展就用；`=off` 忽略扩展，配合 `W4A8_ASM=off` 才真正落到 per-group `tl.dot`。

`int8_gemv.py` 用同一套机制，站点名 `int8_gemv.m1`，候选是 `asm` / `triton`，开关 `COMFY_GFX906_INT8_GEMV_ASM`。

环境变量只读一次、判定结果在进程内缓存（含"探测失败"），与重构前一致；`_REGISTRY.reset()` 用于工具和测试。

per-group `tl.dot` 需要 `group_size >= 16` 且是 16 的倍数；不满足时 `linear()` 返回 `None`，调用方会回退到 int8/`F.linear` 路径。

`status()` 会显示 `w4a8 on: N calls, backend=hip/asm/triton`。

改动后用 `tools/registry_selftest.py` 验证：它把重构前的 `backend_name()` 逻辑原样留作参考实现，逐组合对比 60 种（env × 扩展存在 × asm 可编译）情形，无需 GPU。真机上的选择 parity 与数值 A/B 用 `tools/verify_backends.py`。

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

## 5.1 注册一个新内核

同一个算子的多个实现（HIP 扩展、asm 内核、纯 Triton）都在内核模块底部用一个 `BackendRegistry` 声明，节点侧不再出现 `if backend == "..."`：

```python
from .kernels import registry as kreg

_REGISTRY = kreg.BackendRegistry(
    "<op>.<形状类别>",                    # 站点名，例如 "w4a8.m1"；status() 和测量数据都用它做主键
    [
        kreg.Candidate("hip", kind="extension", env="COMFY_GFX906_..._HIP", loader=_load_hip),
        kreg.Candidate("asm", kind="asm", env="COMFY_GFX906_..._ASM", probe=_probe_asm,
                       requires=("v_dot4_i32_i8",)),
        kreg.Candidate("triton", kind="portable"),
    ],
)
```

| 字段 | 作用 |
|---|---|
| `env` | 该实现自己的开关：`on` 直接选中（不付探测成本）、`off` 直接跳过、`auto` 探测；`on` 也要求扩展确实存在 |
| `probe` | 一次性探测（编译一次或导入一次），失败会被缓存为不可用并打日志 |
| `loader` | `extension` 用：返回模块或 `None`，存在性不由 `on` 伪造 |
| `archs` | 只在这些 gcn 目标上参与选择；注册表没给 `arch` 时不过滤（即今天的默认行为） |
| `requires` | 文档性字段，记录需要的指令/特性；判定仍以 `probe` 为准 |

三条约定：

1. 列表最后一项必须是 `portable`，`resolve()` 才保证总能给出结果；
2. 实现遇到自己不支持的形状就返回 `None`，调用方用 `next_after(name)` 沿同一条链下降——实现之间不要互相 import；
3. 判定结果在进程内缓存（含失败），与重构前一致；`reset()` 只给测试和工具用。

### 架构专用内核：`arch/<gcn>/kernels/`

只为某张卡写的内核放这里，不用改站点文件。一个文件就是一个实现：

```python
SITE = "w4a8.m1"                  # 必填：实现哪个站点
NAME = "hip"                      # 选填：候选名，默认取文件名
PRIORITY = 10                     # 选填：越小越先试，默认 30
ENV = "COMFY_GFX906_W4A8_HIP"     # 选填：它自己的开关
REQUIRES = ("hip extension",)     # 选填：纯文档

def probe(): ...                  # 选填：默认"导入成功即可用"
def linear(...): ...              # 站点的原始签名，`None` 表示这个形状不支持
```

加载由 `kernels/loader.py` 完成：按 `arch/<gcn>/kernels/*.py` 顺序导入、按 `SITE` 归位，编译产物用两行 shim 包一层即可（`.so` 放哪都行）。

优先级：站点自带的 asm 是 50，`portable` 恒定最后；这里新加的文件默认 30，也就是插在 asm 之前——想让某个实现更靠后再写 `PRIORITY`。**表里的 `measured` 记录仍然可以点名任何一个实现**，优先级只决定"没有测量数据时先试谁"。

一个文件导入失败只记一行日志并跳过（只报一次），不会拦住启动——只有显卡不对的内核不该让 ComfyUI 起不来。`COMFY_GFX906_ARCH` 可强制指定目标目录（在没插卡时开发用），`COMFY_GFX906_ARCH_KERNELS` 可追加一个搜索目录（自检就是这么测的）。

还没做（下一步）：`tuned` 仍然以 `COMFY_GFX906_ARCHS` 名单为默认值——**不改成纯 manifest 判定是刻意的**：仓库自带的 gfx906 表是 `derived`、指纹 `unmeasured`，按数据判定会让没跑过 bench 的新用户什么都拿不到；需要的是逐项放行（`COMFY_GFX906_FORCE`），已经做了。

## 5.2 测量数据：`arch/<gcn>/manifest.toml`

注册表回答“这张卡能不能跑”，manifest 回答“在这里哪个更快”。两者分开放是有意的：内核列表是代码（可 review、可单测），测量表是数据（一次 bench 就能重写，diff 里能看清哪张卡赢了）。

读取顺序，后者按 key 覆盖前者、逐条合并：

1. `arch/<gcn>/manifest.toml`（随节点分发）
2. `~/.config/comfy-gfx-tune/arch/<gcn>/manifest.toml`（本地覆盖；根目录可用 `COMFY_GFX906_MANIFEST_DIR` 改）

```toml
schema = 1

[arch]
gcn = "gfx906"
fingerprint = "rocm7.18.26434|torch2.15.0+rocm10.3.0a20261102|triton3.9.0|gfx906"
#             ROCm 版本   | Torch 构建号（含编译日期）              | Triton | 目标
capabilities = ["v_dot4_i32_i8", "v_perm_b32"]

[sites."int8_gemv.m1"]
default = "asm"
default_grade = "derived"

[sites."int8_gemv.m1".shapes]
"4096x4096@m1" = { impl = "triton", ms = { asm = 0.0851, triton = 0.0833 }, rel = 0.0, grade = "measured", date = "2026-11-03" }
```

信任靠字段而不是靠约定：

| grade | 含义 | 参与调度 |
|---|---|---|
| `measured` | `tools/bench.py --write` 在与 fingerprint 相同的机器上产出 | 是（默认门槛） |
| `derived` | 从别的机器 / 别的 ROCm 版本转录 | 否 |
| `assumed` | 猜的 | 否 |

- `fingerprint` 不匹配 → **整个文件被忽略**（`status()` 显示 `ignored (...)`），而不是一个数字一个数字地赌。指纹含 ROCm 版本、Torch 构建号（`+rocm...aYYYYMMDD` 里带编译日期）、Triton 版本和 gcn 目标：**换一次 torch 就要重测**，因为 kernel 是那个编译器产出的，同版本 ROCm 下的另一次构建也可能挪动几个百分点；
- `default` 没有 `default_grade` 就算 `assumed`，绕不过门槛；且只要有一条可信 shape 与 `default` 冲突，`default` 就作废；
- shape key 是 `NxK@mM`，`NxK@*` 表示该 NxK 的所有 batch 宽度；
- schema 不匹配、TOML 语法错、单条记录类型不对（缺 `impl`、`ms` 不是正数、grade 不认识）都只影响那一条，不拦启动：`kernels/manifest.py` 的 `load()` 从不抛异常。

`kernels/manifest.py` 不在模块层 import torch（`runtime_fingerprint()` 内部才延迟 import），所以 `tools/manifest_selftest.py` 无 GPU 也能跑。

### 调度链里的位置

站点不再用进程级的 `backend_name()`，而是按形状问注册表：

```python
backend = _REGISTRY.choose(n=..., k=..., m=...)
```

优先级：**显式 env > manifest 的 measured 记录 > 探测顺序**。

- 只要该站点有任一候选的 env 被显式设置（on/off），整张表就不参与比较——用户的开关压过数据；
- 表里没有这个形状 → 看站点 `default`（必须 `default_grade` 达到门槛，且不被任何可信 shape 反驳）→ 仍然没有就回到探测；
- 表里给的名字在本站点不存在、或该实现在这台机器上跑不了 → 视为表错了，回落到探测顺序。

`status()` 显示真实命中数：`gemv on (backend=asm, measured 12/14)`，即 14 次调用里 12 次由表决定。不想开 ComfyUI 时，`tools/status.py` 会在终端打出同一份内容（每次都是新进程，计数器从零开始）。

`COMFY_GFX906_MEASURED=off` 把整层关掉（表读到了但不用），用于驱动比 fingerprint 漂移时救急。

**现状**：随节点分发的 `arch/gfx906/manifest.toml` 仍是 `derived` 数据、`fingerprint = "unmeasured"`，所以默认显示 `STALE`；跑过 `tools/bench.py --write` 之后本地覆盖会补上 fingerprint 并显示 `in use`。

### 降级必须看得见

"节点跑了但没在用实测数据"是最糟的状态：表现和正常一模一样，只是慢。所以三种状态各有对应动作：

| 状态 | 含义 | 动作 |
|---|---|---|
| `measured` | 有文件、fingerprint 匹配 | 全量，无提示 |
| `stale` | 有文件、fingerprint 不匹配 | 启动 **WARNING**（打印双方指纹 + 修复命令）；`status()` 显示 `STALE: ...` |
| `absent` | 没有该架构的文件/数据 | 启动 **WARNING**（说明 tile 走 autotune、GEMV 走探测）；`status()` 不显示 manifest 行 |
| `off` | `COMFY_GFX906_MEASURED=0` | INFO 一行 |

未知架构上还额外收紧了一条：`_untuned_tile_configs()` 只在 `on_target_arch()` 成立时才把 gfx906 的两个 tile 追加进 autotune 候选——在没测过的卡上注入 60 CU 调出来的 tile 是在赌它能编译，而不是在帮它变快。想让新卡也用这两个 tile，用 `COMFY_GFX906_TILE_CONFIG` 显式指定。

### 跑一次 bench

```bash
# 只打印，不写文件
.venv/bin/python custom_nodes/ComfyUI-gfx906-Tuning/tools/bench.py

# 只测一个站点 / 指定形状，写入本地覆盖
... --site w4a8.m1 --shapes 4096x4096@m1,4096x4096@m8 --write

# 直接改随节点分发的表（准备 PR 时用）
... --write --out custom_nodes/ComfyUI-gfx906-Tuning/arch/gfx906/manifest.toml
```

每个形状的处理顺序是固定的：先用 `tools/refs.py` 的 torch 参考实现比对数值 → 超过 `--max-rel`（默认 5%）的实现被报错并且**不参与计时** → 其余的取 median，记录所有实现的耗时。"更快但算错了"的内核因此不可能赢。`arch/<gcn>/kernels/` 里本站点的实现会被自动纳入计时，它的 `NAME` 就是 manifest 里的 `impl` 键。

`--min-gain`（默认 3%）是噪声下限：第一名和第二名差距小于它时，不宣布赢家，而是沿用探测顺序。实测同一张卡同一个形状在两次运行之间会移动约 10%，0.4% 的"胜出"是时钟不是内核；有了这条规则，表不会在两次 bench 之间来回翻，站点 `default` 也不会因为一个噪声形状被卡住。

写入是逐行改文本：注释、别的站点、未见过的 shape 全部原样保留；落盘前会用 TOML 解析器验一遍，解析不过就拒绝写。默认目标是本地覆盖 `~/.config/comfy-gfx-tune/...`，`--out` 才指向仓库文件。

若某站点所有实测形状的 winner 一致，`--write` 会顺带把站点 `default` 提升为 `measured`，让**未测**形状也走表；winner 不一致时不动 `default`，并打印一行说明。

计时默认关掉 convrot（`group 0`）：旋转是两种实现共用的独立 kernel，去掉它比较的才是 GEMV 本身。bench 会自己分配权重，大形状下约 1 GB 显存，不要和出图同时跑。

## 5.3 启动鲁棒性：单点失败不炸节点

`apply()` 里每个 patch 都经过 `_install(label, fn)`，语义是"能装就装，装不上就记一笔继续"：

- 只有**设备/编译级**异常会降级：`ImportError`、`AttributeError`、`OSError`，以及类名属于 `CompilationError` / `OutOfResources` / `PTXASError` / `HIPError` / `TritonError` 的那一类；
- 其它异常（`ValueError`、`TypeError`、`KeyError`…）照常抛出——那是我们的 bug，吞掉只会变成更难查的"某些功能莫名不生效"；
- 降级时 `logging.warning` 一行，并写进 `_state["install_failures"]`；`status()` 末尾打印 `patches skipped on this device: ...`。

`AttributeError` 放进白名单是刻意的：ComfyUI 版本漂移最常见的表现就是"属性不存在"。代价是它也可能掩盖我们自己写错的名字，所以失败必须出现在 `status()` 里，不能只留一条没人看的日志。

`kernels.safe_install()` 是这套策略的库版本（返回 `FAILED` 哨兵，用来区分"失败"和"patch 本来就返回 None"），`_install` 只是给它补上状态记录。

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
8. 改动内核后跑 `tools/verify_backends.py`（选择 parity + 数值 A/B，需要 GPU；无设备时会直接退出并说明原因），再记录性能。
9. 要固化实测结果就跑 `tools/bench.py --write`（默认写本地覆盖），让同型号的后续启动不必再探测。

## 8. 已知边界

- `kernel.configs` 是进程级共享；当前假设 ComfyUI 模型串行执行。并发执行时需要改成按调用传参。
- 未实测卡走候选模式时，每个新 `(m,n,k)` 第一次调用会编译/实测 7–8 个候选，首个 shape 会慢；之后按 `(m,n,k)` 缓存稳定。
- 静态 tile 表是“按卡型固化”的：驱动、编译器或实际 shape 变化后需要重新测。
- `COMFY_GFX906_TILE_CONFIG` 只影响 int8 tile，不会启用 DeltaNet/Qwen3.5/w4a8。
