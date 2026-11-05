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
- ComfyUI 0.39.0 或更新版本；逐块 CUDA graph 由节点自带的 `resident_graph.py` 以运行时包装提供，不需要改动 core 文件。

## 2. 功能分级：常开 / 可选开关 / 测量门

所有功能按“答案是否随卡变”分三类，这是整个包的门控总纲：

| 类 | 判据 | 内容 |
|---|---|---|
| **A. 常开** | 行为不随卡变、无权衡 | prefix kv、kvdecode、llama、text gemv、TE int8、amax guard |
| **B. 可选开关** | 代价不是速度，或内核质量有分层 | qkv（1.6 GB 常驻）、qwen35（等价类替换，env 默认开）、profiler、attention mode（随包的 `hip` 内核能加载时默认 `hip`：fp32 累加、各形状最快；本机 HIP 装不上时 `int8`，也可 node/env 手选） |
| **C. 测量门** | 最优解随卡变，没数据就没有默认值 | int8 tile、w4a8、deltanet、vae、convrot（mode + 开关）、gemv（backend + 开关） |

第三类是关键设计：**没跑过测量就是关闭，跑一次 `bench.py --all --write` 后由数据决定开或关**。数据有两个来源，本地覆盖出厂：

- **shipped 表**（`arch/<gcn>/manifest.toml`）：对已实测的架构（gfx906）随包分发，`grade = "measured"`、不带指纹，对这个架构的任何构建都生效——开箱即得；
- **本地表**（`~/.config/comfy-gfx-tune/arch/<gcn>/manifest.toml`）：`tools/bench.py --all --write` 写的，带本机指纹，逐形状覆盖 shipped 表。换 torch/驱动后本地表过期被忽略，回落 shipped 表。

C 类的门和取值都走同一批查询：

```python
def feature_on(item):         # env (on/off) > feature.<item> > off
    raw = _state.get(item + "_env", "")
    if raw in ("on", "off", ...): return raw == "on"
    table = measured_table()
    return table is not None and table.default_impl("feature." + item) == "on"
```

`tools/bench.py --all --write` 写这些 feature 与取值：`feature.w4a8`（和 int8 基线比）、`feature.gemv`（和 stock int8 线性比）、`feature.vae`（和 torch conv3d 或 kitchen 的 `fp16_conv3d` 比）、`feature.convrot` / `convrot.mode`（和 kitchen 的 matmul 旋转比）、`feature.vision`（和现状的 conv3d patch 投影比）、`feature.deltanet_prefill`（和 core 的逐行 UT 变换比）；完整清单见 §5.2。attention 不在其中——它的内核是质量分层（见 B 类），mode 由 node/env 手动选；`bench.py --attention` 只打印证据，不写任何记录。`deltanet` 是半例外：kitchen 没有 fused op 的卡上它结构性开启（那个事实不随测量过期），kitchen 有的卡上由 `feature.deltanet` 决定。bench 目前**不写** `feature.qwen35`（它的用例只覆盖 prefill，而收益主要在 decode 路径），所以 qwen35 留在 B 类。

`COMFY_GFX906_FORCE=deltanet,w4a8,...` 仍然是手动逃生口：跳过一切判定直接安装，启动日志会 WARNING 说明这些内核不是在当前卡上测的。

安装判定只属于这一层，int8 GEMM 的**取值**同样来自表：`int8_gemm.tile` 记录大/小 tile（`COMFY_GFX906_TILE_CONFIG` 优先）。attention 的 tile 用内置常量（gfx906 实测值），mode 走 node/env。

这里的“安装”是替换模块级符号或类，而不是设置一个关闭状态：

- `deltanet.install()` 替换 `comfy_kitchen.gated_delta_decode_fused` 等三个符号；
- `qwen35_model.install()` 替换 `comfy.text_encoders.qwen35.Qwen35Transformer`；
- w4a8 把模块指针放进 `_state["w4a8_module"]`。

所以未安装时 `status()` 显示 `n/a`，不是 `off (installed)`；节点再选 `textgen=on` 也打不开。

### 2.1 门控只做 dispatch 管不到的事

kitchen 的 dispatch 本身就回答了“内核跑谁的”：hip 注册成功时优先级被设为 `["hip", "cuda", "triton", "eager"]`（comfy_kitchen/__init__.py），而且 `get_capable_backend` 按**每次调用**校验 constraints——原生后端拒绝的形状（形状约束不满足）会落到 triton。据此分三类：

**int8 tile 补丁在所有 AMD 上安装，不看 capability 表。** dispatch 决定它何时运行：原生覆盖的卡上它只是 constraints 兜底路径，没覆盖的卡上（gfx906）是主路径。无条件安装是安全的，因为 tile 取值由 manifest 指纹自守——没数据的卡拿到的就是 stock 候选列表。

**dispatch 管不到的，归测量门或开关管（见 §2 的分级）：**

- **vae / convrot**：改写发生在 ComfyUI 或 quantization 调用路径里，在 dispatch 之前，无从仲裁 → `feature.vae` / `feature.convrot`（bench 写）；`convrot.mode` 记录 fused/butterfly 的赢家。
- **attention**：不是 kitchen op，替换的是 `comfy.attention.optimized_attention` 本身。所有 AMD 安装，mode 由 node/env 手动选；带 mask / GQA 的调用整段绕行到 core 的后端（内核没有 mask 参数）。**文本生成根本不经过它**：TE 用 `optimized_attention_for_device(small_input=True)` 解析（llama.py:971），拿到的是 core 的 attention_pytorch/basic 原始函数——所以测量门和 mode 只影响 DiT。NVIDIA 不装。
- **qkv**：1.6 GB 常驻的纯权衡 → env/node opt-in，默认关。
- 其余文本生成改写组（prefix kv / kvdecode / llama / text gemv / TE int8 / qwen35）与后端无关，全部安装；`deltanet` / `w4a8` 按 §2 判定。N 卡上 `deltanet` 的基线是 kitchen 的 CUDA fused op，大概率 `off`；`w4a8` 的 per-group `tl.dot` 走 tensor core，值得跑一次 bench。

bench 的 tile 站点在 `native_covers("int8_linear")` 时跳过：覆盖的卡上记录只在 constraints 兜底时被读到，几乎不值得测（`--tile` 可以强制测）。

两个实测发现的细节：**cuda 后端的 capability 集里没有 `int8_linear`**（只有 `w4a8_int8_linear`），所以 N 卡的 int8 线性本来就走 triton——但本包没在 N 卡测过 tile，补丁仍然不装，保持 stock autotune。manifest 的设备 key 已泛化：AMD 是 gcn target，NVIDIA 是 `sm<major><minor>`（`kernels.manifest.device_arch()`）。

## 3. 功能分级

### A. 常开 + 可选开关（对应 §2 的 A/B 类）

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

### B. 测量门（对应 §2 的 C 类：没数据就是关，bench 决定开或关）

| 功能 | 默认行为 | 需要做什么 |
|---|---|---|
| int8 linear tile | 未实测卡**只用 stock 6 个**候选；gfx906 的大/小 tile 只在 gfx906 上追加（别把 60 CU 的参数塞进不认识的卡，LDS/寄存器预算不同，可能编不过） | `tools/bench.py --all --write` 会扫出 winner 并写进 `int8_gemm.tile`；也可以只用 `tools/tile_sweep.py` 测，再设 `COMFY_GFX906_TILE_CONFIG` / `SMALL_TILE_CONFIG` |
| `small_batch` 窄 tile | 候选模式，由 autotune 决定 | 测 `m≤8`；更慢或不想付首次扫参成本时设 `small_batch=off` |
| `gemv`（模型 int8 解码 GEMV） | 没有实测数据就不安装；text gemv 不受影响（它有自己的开关） | `bench.py --all` 在 `int8_gemv.m1` 站点加计 stock int8 线性基线，赢了写 `feature.gemv="on"`；backend 照旧按形状选 |
| int8 线性大 m（`int8_gemm`） | C 类（性能）：≥128 行走自带 HIP dot4 GEMM（4 波/CU），否则回落 Triton；m≤1 仍走 GEMV | `libgfx906_int8.so` 能本机加载才生效（随包分发，按 HIP 7.18 构建；别的大版本要重编）；`bench.py --tile` 打印 hip vs triton 的证据（不写记录） |
| 注意力入口（`attention`） | B 类：mode 由 node/env 手动选（hip / int8 / int8qk / hand / gemm / sdpa）；tile 用内置常量；带 mask / GQA 的调用绕行到 core 后端 | `bench.py --attention` 只打印各内核 vs SDPA 的证据，不写记录。TE 不经过这条路径（见 §2.1） |
| `w4a8`（4-bit 权重 GEMV） | asm 内核用 gfx9 `v_dot4_i32_i8`；其它卡落到纯 Triton 的 per-group `tl.dot`，通常比 int8 慢，所以在没有实测数据的卡上不安装 | `bench.py --all` 会把它和 int8 基线（`decode_int8` + int8 GEMV/GEMM）同形状对比，赢了才写 `feature.w4a8="on"`。`group_size < 16` 时 `tl.dot` 不接受，会落到 `tl.sum` 保底内核（见 §5）；保底通常更慢，表里就会是 `off` |
| DeltaNet Triton 内核 | 只在 kitchen 的 fused op 不可用时划算（gfx906 就是没有 code object） | `bench.py --all` 直接和 `comfy_kitchen.gated_delta_decode_fused` 对比输出、state 和耗时，赢了才写 `feature.deltanet="on"` |
| `deltanet_prefill`（prefill UT 变换） | `feature.deltanet_prefill` 决定默认；装了以后 live gate 仍随 `textgen`/`deltanet` 走，UI 不加开关 | `bench.py --deltanet` 会同时测 decode 和 prefill：prefill 站点用 core 自己的 `torch_chunk_gated_delta_rule` 输出当参考，**每个形状跑 mild 与 coherent 两档输入取最差 rel 作门**（gfx906 实测 9.10→6.32 / 18.33→13.56 / 33.44→25.37 ms 每层，rel 5.1e-7、+26.0%），赢了写 `feature.deltanet_prefill="on"`。实现在 `deltanet.chunk_rule`：除 UT 变换外逐行照抄 core，**core 改了那段必须同步**；变换本身是 `deltanet.ut_inverse` 的 6 层分块递归。两档门是因为上一版用 Neumann 倍增级数：它在只测 mild 的旧门上是 rel 2.1e-7，但在模型自己的 24 个 prefill 调用里有一层算到 1.65e+16（core 同处 0.267），生文从第一个 token 起退化；改成 key 行间相干（`k_beta @ key^T` 逼近严格下三角全 1）的合成输入就能复现这一类错误（倍增 rel 4.65e+03，分块 1.6e-07） |
| `qwen35` 计算类替换 | **B 类（可选开关）**：`COMFY_GFX906_QWEN35` 默认开，它是逐方法对过 core 的等价改写 | `bench.py --qwen35` 只出证据不写 feature：合成小模型 fp32 A/B（1e-3 门槛）+ 计时；但用例只覆盖 prefill，而收益主要在 decode，测量覆盖不了决策面。未覆盖：recurrent 解码路径、真实 checkpoint 权重 |
| `convrot` | 没有实测数据就不安装 | `bench.py --convrot` 计时 kitchen 的 matmul 旋转 vs 我们的 butterfly vs fused，写 `feature.convrot` 和 `convrot.mode`（赢家 mode，`COMFY_GFX906_CONVROT` 可覆盖） |
| `vae` | 没有实测数据就不安装 | `bench.py --vae` 计时 depth-1 Conv3d：torch conv3d / conv2d squeeze / kitchen 的 `fp16_conv3d`（有原生后端的卡），写 `feature.vae` |
| `image_tower`（Qwen3.5 图像塔） | 包装器总是安装，但默认值看 `feature.vision`：没数据的卡保持 core 的行为，节点可以手动 on | `bench.py --vision` 计时 patch 投影三种走法：`conv3d_fp32`（现状）/ `conv3d`（只改 dtype）/ `gemm`（补丁），写 `vision.patch_embed` 和 `feature.vision`。gfx906 实测 gemm 比现状快 30-400 倍（0.58/2.0/8.4 ms vs 17.7/70.3/284 ms），而且 **bf16 的 conv3d 比 fp32 慢 12 倍**，所以 dtype 和 conv 必须一起换 |
| `qkv` | 默认关（1.6 GB 常驻，纯权衡，不归测量门） | 想用时设 `COMFY_GFX906_QKV=1` 或节点开关 |

**易踩的坑**：`configure(w4a8="on")` 会在节点执行时调用 `load_w4a8_gemv()`，把模块装上；这条路径绕过了 `feature.w4a8` 的安装判定。非 gfx906 上开着 `w4a8` 时 asm 探测会自动失败、落到 per-group `tl.dot`；如果显式设 `COMFY_GFX906_W4A8_ASM=on`，探测被跳过，asm 会在实际调用时报编译错误。节点默认值是“沿用 env”，不想用就保持 `COMFY_GFX906_W4A8=0`。

### D. 非目标设备

| 场景 | 原因 |
|---|---|
| NVIDIA（有 GPU） | `is_amd()` 为假 → triton 内核补丁与 attention 都不装（cuda 的 capability 集没有 `int8_linear`，N 卡的 int8 线性走 triton，但本包没在 N 卡测过 tile）。**文本生成重写组生效**：prefix KV、KV decode 折叠、llama 缓存、TE GEMV、TE int8 常驻；`w4a8`/`deltanet` 由本机 bench 的 `feature.*` 决定（没跑过就是关）；qwen35 随 env 默认开（B 类等价改写）。manifest 的 key 是 `sm<major><minor>`。**未在 N 卡实测**，性能结论要由有卡的人跑出来 |
| NVIDIA，权重非全量常驻 | 逐块 CUDA graph 不会捕获（权重按块换入时走 core 的 vbar 路径），只影响 launch 开销，结果不变 |
| CPU | `torch.cuda.is_available()` 为假，整包不安装 |
| ComfyUI 升级后 | 所有 core 接触点都是包装原函数，最坏降级为 eager 解码，不需要重新生成任何文件 |

AMD 上“kitchen 已经有后端”的卡不再归到这里：它们走 §2.1，只装四个与后端无关的改写，没有需要按卡验证的内核。

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

# 打印可直接用的启动变量
python .../tools/tile_sweep.py --emit
```

要**保存**结果而不只是打印，用 `tools/bench.py --all --write`：它调用同一套扫描，把赢家写进 `arch/<gcn>/manifest.toml` 的 `int8_gemm.tile`，之后这张卡启动就不再有 autotune。

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
- 两个变量都是启动期读取；节点 `int8_tile` / `small_batch` 的 `default` 会沿用它们；
- manifest 里的 `int8_gemm.tile` 与 env 同权（env 优先），所以固化结果不需要改启动脚本。

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
        kreg.Candidate("sum", kind="portable", priority=60),
    ],
)
```

`linear()` 不再写死 `hip > asm > triton`：它取 `_REGISTRY.choose()`，某个实现当场返回 `None` 时用 `_REGISTRY.next_after(...)` 沿同一条链往下走，直到一个实现接住这个形状。新增一个内核＝在列表里加一条：

1. **`w4a8_hip` 扩展**（`extension`）：有可导入的模块且提供 `linear()` 时优先；
2. **asm**（`asm`）：首次判定时编译探测 `v_dot4_i32_i8`；
3. **triton**（`portable`）：per-group `tl.dot`，要求 `group_size` ≥16 且是 16 的倍数；
4. **sum**（`portable`）：`tl.sum` 保底，不要求任何指令、也不要求 16 宽的 K 步。它是链的最后一项，也是“这张卡至少能跑 w4a8”的保证。

环境变量：

- `COMFY_GFX906_W4A8_ASM=auto`（默认）：编译探测；
- `=on` / `asm`：强制 asm（探测失败会在实际调用时报编译错误）；
- `=off` / `triton`：跳过 asm（这是"asm 关掉"，不是"只用 triton"）；
- `COMFY_GFX906_W4A8_HIP=auto`（默认）：有扩展就用；`=off` 忽略扩展，配合 `W4A8_ASM=off` 才真正落到 per-group `tl.dot`。

`int8_gemv.py` 用同一套机制，站点名 `int8_gemv.m1`，候选是 `asm` / `triton`，开关 `COMFY_GFX906_INT8_GEMV_ASM`。

环境变量只读一次、判定结果在进程内缓存（含"探测失败"），与重构前一致；`_REGISTRY.reset()` 用于工具和测试。

per-group `tl.dot` 需要 `group_size >= 16` 且是 16 的倍数；不满足时降到 `sum` 保底（首次使用会打一行 INFO）。只有 `sum` 也拒绝的形状（m>8、K 与权重不匹配、convrot 不整除）才返回 `None`，那时调用方才回退到 int8/`F.linear`。

保底比 `tl.dot` 慢（没有矩阵指令，靠 `tl.sum` 逐元素乘加），所以在没有 `v_dot4` 的卡上它就是“能跑但可能不如 int8”——这正是 `bench.py --w4a8-group <16` 要测的东西，测输就写 `feature.w4a8="off"`。

`status()` 会显示 `w4a8 on: N calls, backend=hip/asm/triton`，这里的 backend 是链上首选的实现；真正接住某次调用的是哪一个，bench 的站点输出会逐个列出。

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
| `w4a8_gemv` | hip > asm > per-group tl.dot > tl.sum 保底 | — | 已做 |
| `decode_int8`、`quantize_k/v`、`convrot`、`rotate` | 解包/量化/旋转，不是 dot | 不适用 | asm 帮不上 |
| QKV 融合 | 走 kitchen int8 GEMM | 同第一行 | 不建议 |

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

1. 列表最后一项必须是 `portable`，`resolve()` 才保证总能给出结果；可以有多个 `portable`（w4a8 就是 `triton` + `sum`），它们按声明顺序排在 asm 之后，最后一个必须无条件接受所有合法形状；
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

这个顾虑已经解决：shipped 的 gfx906 表重新烤成 `grade = "measured"`、不带指纹，对这张卡的任何构建生效——纯 manifest 判定不会让 gfx906 用户丢功能，而新架构在跑过 bench 前老实保持关闭。`COMFY_GFX906_ARCHS` 随之删除。

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

# 不是 (N, K, m) 的量用固定的 key，值仍然放在 impl 里：
[sites."int8_gemm.tile".shapes]
"large" = { impl = "block_m=128,block_n=256,block_k=32,group_size_m=8,warps=4,stages=2", ms = { 128x256x32g8w4s2 = 1.24 }, grade = "measured" }
"small" = { impl = "block_m=16,block_n=32,block_k=64,group_size_m=8,warps=4,stages=3", ms = { 16x32x64g8w4s3 = 0.31 }, grade = "measured" }

```

站点共有这几种形态，都由 `tools/bench.py --all --write` 写：

| 站点 | key | impl 是什么 | 谁读它 |
|---|---|---|---|
| `<op>.<形状类>` | `NxK@mM` | 实现名（asm/triton/hip） | 注册表按形状选后端 |
| `int8_gemm.tile` | `large` / `small` | 与 `COMFY_GFX906_TILE_CONFIG` 相同的 tile 文本 | `_triton_config()` / `_small_tile_config()` |
| `convrot.mode` | `mMxK` | `fused` / `butterfly` | `resolved_convrot_mode()` |
| `deltanet.decode` | `bBxhvHxdkDKxdvDVxsS` | `triton` / `kitchen` | 只作证据；开不开看 `feature.deltanet` |
| `deltanet.prefill` | `bBxhHxdkDKxdvDVxsS` | `block`（分块递归）/ `loop`（core 的逐行） | 只作证据；开不开看 `feature.deltanet_prefill` |
| `vae.decode` | `cCxSS` | `conv2d` / `torch` / `fp16_conv3d` | 只作证据；看 `feature.vae` |
| `vision.patch_embed` | `pPx dD` | `gemm` / `conv3d` / `conv3d_fp32` | 只作证据；看 `feature.vision` |
| `qwen35.block` | `bBxsSxhHxlL` | `ours` / `stock` | 只作证据；`feature.qwen35` 不由 bench 写 |
| `feature.<item>` | —（只有 `default`） | `on` / `off` | `install_ready()` |

只有 `<op>.<形状类>`、`int8_gemm.tile`、`convrot.mode` 和 `feature.*` 会被运行时读到；其余形状记录是给人看的证据（哪张卡上谁赢、数值差多少），删掉不影响调度。

`feature.*` 由 bench 写七项：`gemv` 和 `w4a8`（GEMV 站点的 `spec["feature"]`，[bench.py:1212](tools/bench.py#L1212)）、`deltanet`（[:743](tools/bench.py#L743)）、`deltanet_prefill`（[:831](tools/bench.py#L831)）、`vae`（[:979](tools/bench.py#L979)）、`vision`（[:1051](tools/bench.py#L1051)）、`convrot`（[:1133](tools/bench.py#L1133)）。**不写**三项：`attention`（mode 是质量取舍，`--attention` 只打印证据）、`qwen35`（用例只覆盖 prefill，收益在 decode，决定权在 `COMFY_GFX906_QWEN35`）、`te_dequant`（表里那行是手写的，bench 没有对应站点）。

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

**现状**：随节点分发的 `arch/gfx906/manifest.toml` 是 `grade = "measured"`、`fingerprint = "unmeasured"`，所以在 gfx906 上开箱就是 `in use`（`tools/status.py`：`23 shapes (23 measured), in use`）；没跑过 bench 的其它架构没有对应文件，走下面的 `absent`。本地 `--write` 会带上真实指纹，并按 shape 覆盖 shipped 表。

### 降级必须看得见

"节点跑了但没在用实测数据"是最糟的状态：表现和正常一模一样，只是慢。所以三种状态各有对应动作：

| 状态 | 含义 | 动作 |
|---|---|---|
| `measured` | 有文件、fingerprint 匹配 | 全量，无提示 |
| `stale` | 有文件、fingerprint 不匹配 | 启动 **WARNING**（打印双方指纹 + 修复命令）；`status()` 显示 `STALE: ...` |
| `absent` | 没有该架构的文件/数据 | 启动 **WARNING**（说明 tile 走 autotune、GEMV 走探测）；`status()` 不显示 manifest 行 |
| `off` | `COMFY_GFX906_MEASURED=0` | INFO 一行 |

没实测数据的卡只会拿到 stock 候选列表——把 60 CU 调出来的 tile 注入不认识的卡是在赌它能编译，而不是在帮它变快。想让新卡用上 tile，用 `COMFY_GFX906_TILE_CONFIG` 显式指定，或让 bench 在本机测出它自己的（shipped 表只对对应的 gcn target 生效）。

“文件匹配但没有 tile 记录”是新的降级态：fingerprint 对得上，于是不会报 stale，但 tile 仍然走 autotune。这种启动会有一条 WARNING，提示只跑过部分站点的 bench（例如 `--site w4a8.m1`），需要补 `--all`。

### 跑一次 bench

```bash
# 只打印，不写文件
.venv/bin/python custom_nodes/ComfyUI-gfx906-Tuning/tools/bench.py

# 一张卡需要的一切：GEMV 后端 + int8 tile + 各 feature 的 on/off
... --all --write

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

`--all` 追加六组测量：

- **int8 tile**：复用 `tile_sweep` 的扫描，默认 3 个 NxK（DiT 的 qkv/ffn、TE 的）乘 `m=1,8,128`，`m≤8` 与 `m>8` 两桶各取所有格子耗时的中位数赢家。这一层不用 `--min-gain`：跨形状的中位数已经压掉了单次噪声，而“上一次选的是谁”在这张卡上没有意义。
- **attention**：只出证据不写记录——mode 由 node/env 选，随包的 `hip` 内核可加载时默认取 `hip`（fp32 累加、rel ≈ 2.5–3.3e-4、三个实测形状都最快）。默认 `d128` 长/短 self+cross 各一个形状、`d64` 一个 SDXL 形状；每个 config 先和 `tools/refs.py` 的 fp32 参考比数值，再计时，最后打印各内核相对 SDPA 的中位数比例。
- **deltanet**：默认 `1x16x128x128x1` 与 `...xs4`（Qwen3.5-2B 的 16 个 value head、hidden 2048）。先和 `refs.delta_decode_reference` 比输出**和** `state`，再和 `comfy_kitchen.gated_delta_decode_fused` 同时计时；对端不可用（gfx906 就是没有 code object）时只记录我们自己的耗时，不写 `feature.deltanet`。
- **vae**：depth-1 Conv3d 三方计时（torch conv3d / conv2d squeeze / kitchen 的 `fp16_conv3d`，后者只在原生后端有这个 op 时出现），写 `feature.vae`。padding 0 以便 kitchen 内核可比。
- **convrot**：kitchen 的 matmul 旋转 + 量化 vs 我们的 butterfly 两步 vs fused 单内核，写 `feature.convrot` 和 `convrot.mode`。
- **qwen35**：建一个合成 Qwen3.5（默认 4 层、hidden 2048、seq 16，≈240M 参数），同一套权重跑两遍——gate 关 = 逐字调用 core 的方法，gate 开 = 我们的类。fp32 下 1e-3 门槛是真正的等价性检查。**只出证据，不写 `feature.qwen35`**（用例只覆盖 prefill，收益主要在 decode）。不需要 checkpoint，也不联网。

另外 `w4a8.m1` 这个站点现在还多计一条 `int8` 基线：把 4-bit 权重用 `decode_int8` 一次性解成 int8（计时循环外），再走 int8 GEMV（m=1）/ int8 GEMM（m>1）——那才是关掉 w4a8 时真正会跑的路径。默认用 16 级 codebook（随包分发的 checkpoint 就是这种），`--w4a8-uniform` 测均匀格点。`--w4a8-group` 默认 128；改成 8 就会跑 `tl.sum` 保底（`tl.dot` 会拒绝那个形状），用来确认“小 group 的 checkpoint 至少能跑，而且值不值得开”。

各组的形状/尺寸都可以用 `--tile-shapes` / `--tile-m` / `--attention-shapes` / `--deltanet-shapes` / `--vae-shapes` / `--convrot-shapes` / `--qwen35-layers` / `--qwen35-hidden` / `--qwen35-seq` 覆盖。表里没有的形状仍然回落到 `default`，没有 `default` 就回到 autotune/探测——所以形状表要覆盖你实际在跑的模型。

## 5.3 启动鲁棒性：单点失败不炸节点

`apply()` 里每个 patch 都经过 `_install(label, fn)`，语义是"能装就装，装不上就记一笔继续"：

- 只有**设备/编译级**异常会降级：`ImportError`、`AttributeError`、`OSError`，以及类名属于 `CompilationError` / `OutOfResources` / `PTXASError` / `HIPError` / `TritonError` 的那一类；
- 其它异常（`ValueError`、`TypeError`、`KeyError`…）照常抛出——那是我们的 bug，吞掉只会变成更难查的"某些功能莫名不生效"；
- 降级时 `logging.warning` 一行，并写进 `_state["install_failures"]`；`status()` 末尾打印 `patches skipped on this device: ...`。

`AttributeError` 放进白名单是刻意的：ComfyUI 版本漂移最常见的表现就是"属性不存在"。代价是它也可能掩盖我们自己写错的名字，所以失败必须出现在 `status()` 里，不能只留一条没人看的日志。

`kernels.safe_install()` 是这套策略的库版本（返回 `FAILED` 哨兵，用来区分"失败"和"patch 本来就返回 None"），`_install` 只是给它补上状态记录。

## 5.4 链路归因（`tools/chain_profile.py`）

站点微基准回答"这段里哪个实现快"，不回答"时间到底落在哪一段"。DiT 侧有 `profiler.py`（节点的 profile 开关）；**文本编码器它看不见**：TE 走 CLIP 链而不是 `BaseModel.apply_model`，而它的 attention 由 `optimized_attention_for_device(small_input=True)` 解析成 core 的 `attention_pytorch`/`attention_basic`（`comfy/text_encoders/llama.py:971`、`comfy/ldm/modules/attention.py:941-946`），profiler 包的是模块级 `optimized_attention`，两个 hook 都落空。`chain_profile.py` 补这一段：跑一次真实的 TextGenerate，用嵌套 CUDA event 归因。

```bash
cd /home/liu/ComfyUI-0.39.0
.venv/bin/python custom_nodes/ComfyUI-gfx906-Tuning/tools/chain_profile.py \
    --clip <text encoder 文件名> --set text_encoder=int8 --image <图> --max-length 32

... --pack off                     # 同一条链路只跑 core
... --set image_tower=off          # 任何 configure() 项，可重复
... --list-classes                 # 模块类普查，用来挑 --wrap 的名字
... --kitchen                      # 顺带装 profiler 的 hook，看 kitchen 层
... --disable-cuda-graphs          # 透传给 ComfyUI 的旗标
```

归因点是**结构性发现**的，不写死类名，所以换模型/换架构不用改工具：cond stage model、最宽 `ModuleList` 的 owner（它的 forward 每步一次，用来分 prefill 与 decode）、活的 generate 对象（每个 TE 都覆写 `generate()`，包基类会一次都记不到）、以及每一个 block；再各加一层直接子模块，图像塔这种兄弟子树才有自己的行。`comfy.ops` / `torch.nn` 的叶子层默认不包：包一个 `Linear` 等于包住全部 359 个实例，会把上面每个区域的 self 抽成 0；叶子视图用 `--kitchen`（profiler 的 `int8_linear`）或显式 `--wrap Linear`。

表里的 `self s` 是总时间减去直接嵌套在它里面的区域，所以整表相加等于链路时间而不是重复计数；`of run` 按 self 算。gfx906 上今天这两次运行：

| | pack on（图开、te=int8、32 token）+ 1160x679 图 | pack off（图关、te=default、16 token）无图 |
|---|---|---|
| 图像塔 | **34.9%**（1.56 s，一次） | — |
| GatedDeltaNet | 19.9% | 20.6% |
| MLP | 12.6% | **51.9%** |
| 每步 forward 自身 | 16.3% | 6.1% |
| `logits`（lm_head） | 1.28 ms/次 | **89.3 ms/次** |
| prefill / 每步 / tok/s | 1511 ms / 25.8 ms / 38.7 | 2208 ms / 670 ms / 1.49 |

两列之间差了**两个**变量（pack 与 CUDA graph），所以这不是 pack 的净收益。lm_head 那 89 ms 也不能记在 `text_gemv` 头上：`--pack off` 时 `--set text_encoder=int8` 根本不生效（工具会打印一行说明），那一列跑的是 te=default 的逐调用反量化。固定单一变量才有结论，例如 `text_gemv`（两边都 te=int8、图关、同 seed、同 prompt）：

| | `text_gemv=on` | `text_gemv=off` |
|---|---|---|
| 每步（sampling / greedy） | 48.2 / 49.8 ms | 62.6 / 64.4 ms |
| MLP 中位 | 0.45 ms | 0.64 ms |
| `logits` 中位 | 1.45 ms | 1.42 ms（不变：lm_head 的 1.4 ms 来自 TE int8 常驻，不是 text_gemv） |
| tok/s | 20.8 | 16.0 |

读数时注意三件事：

- **`graphed` 列**：捕获进 CUDA graph 的调用不能带 event，只计数不计时；而且**回放根本不进 python**，所以图开着时 block 以下的行只包含 prefill 和捕获那一遍。要看每步归因就加 `--disable-cuda-graphs`，但那次运行不能和图开着的比——图正是用来消掉每 block 的 dispatch 成本的（约文本生成的 37%，见 `resident_graph.py`）。
- **`unattributed`**：墙钟减去所有根区域的总时间，即 tokenize、python、launch gap 和没被包住的东西。它大就说明瓶颈不在 GPU 内核上，先别去调内核。
- **`md5`**：输出文本的哈希，是"变了没有"的探测器，不是正确性判据。上面那组 `text_gemv` A/B 在 `--sampling off` 下两边 md5 完全相同（数值等价），而 sampling 开着时同样的微小差异会逐 token 放大成完全不同的文本——比等价性就加 `--sampling off`。反例是上一版 prefill 变换：更快，但生文从第一个 token 就退化。

旗标透传靠 `comfy.options.enable_args_parsing(True)`：库模式下 `comfy/cli_args.py` 走 `parse_args([])`，不设这一句 `--disable-cuda-graphs` 之类会被**静默忽略**（工具自己在 import comfy 之前设好了）。textgen 子项没有 `configure()` 入口，A/B 用 env：`COMFY_GFX906_DELTANET` / `_KVDECODE` / `_LLAMA` / `_QWEN35` / `_GRAPH`（resident graph 也跟随 CLIP 节点的 `textgen` 开关）。

## 6. 开关优先级

**节点显式 on/off > `COMFY_GFX906_*` > 内置默认**。节点里的 `default` / `auto` 表示沿用 env，不会用硬编码值覆盖它。完整开关列表见 [README.md](README.md#L223-L240)。

## 7. 二次开发流程建议

1. 确认：AMD + HIP、`comfy-kitchen`、triton 可用；overlay 仅 0.39.0 可选。
2. 先用节点默认值跑通——没跑过 bench 的卡上测量门功能是关的，这是预期行为，不是坏了。
3. 跑一次 `tools/bench.py --all --write`：int8 tile、两个 GEMV 站点，以及 gemv / w4a8 / deltanet / deltanet prefill / vae / convrot / vision 的 on/off（convrot 还有 mode）都写进本机 manifest，之后启动不再 autotune。attention mode 不在此列——节点里手动选。只想看数字就先不加 `--write`。
4. 用 `tools/status.py` 确认 `measured ... in use`、`int8 tiles (measured tile)`、`gemv ... measured`，以及 `w4a8`/`textgen deltanet` 的 on/off；日志里不应再有 `no measured data`。
5. 归因，别猜：生文链路用 `tools/chain_profile.py`（§5.4），DiT 侧用节点的 profile 开关。归因出的热点如果在 bench 里没有对应站点，**先加站点再优化**——`vision.patch_embed` 和 `deltanet.prefill` 都是这么来的。
6. 逐项 A/B：`vae`、`gemv`、`convrot`、`qkv`、`text_gemv`。一次只动一个变量，否则结论没有归属（§5.4 里那张 pack on/off 表就是反例：它同时动了 pack 和 CUDA graph）。
7. `deltanet` 在 kitchen 的 fused op 不可用的卡上**结构性开启**——此时 bench 没有基线可比，不写 `feature.deltanet`（[gfx906_tuning.py:296-303](gfx906_tuning.py#L296-L303)）；kitchen 可用时由 bench 决定，大概率 `off`。两种都是正确结果。prefill 的基线是 core 自己的循环，所以 `feature.deltanet_prefill` 在任何卡上都能测出来。`qwen35` 不由 bench 写，决定权在 `COMFY_GFX906_QWEN35`；`--qwen35` 的计时是 4 层合成模型的代理，等价性检查才是主交付。
8. gfx906 上 `textgen` 四项都是 on（`kvdecode`/`llama` 常开、`qwen35` 由 env 默认开、`deltanet` 结构性开启）。别的卡上 `status()` 打出 `n/a` 是正常的：那一项没装上，节点里选 `on` 也打不开。
9. 需要 resident CUDA graph 时再装 overlay；不装不影响正确性。
10. 改动内核后跑 `tools/verify_backends.py`（选择 parity + 数值 A/B，需要 GPU；无设备时会直接退出并说明原因），再记录性能。
11. 测量数据换了 torch/驱动就失效，重跑第 3 步即可；表写错时 `COMFY_GFX906_MEASURED=0` 整体关掉。

## 8. 已知边界

- `kernel.configs` 是进程级共享；当前假设 ComfyUI 模型串行执行。并发执行时需要改成按调用传参。
- 未实测卡走候选模式时，每个新 `(m,n,k)` 第一次调用会编译/实测 7–8 个候选，首个 shape 会慢；之后按 `(m,n,k)` 缓存稳定。
- 静态 tile 表是“按卡型固化”的：驱动、编译器或实际 shape 变化后需要重新测。
- attention 的 tile 用内置常量、mode 手动选：同一 head dim 在 prefill 和 decode 下的最优 tile 不同（d128 decode 时 32x16 比 64x16 快 ~14%），单一常量两头兼顾不了；好在这些调用全在 DiT 侧，TE 不经过。
- `feature.<item>` 只有 `default`，没有 per-shape 记录，所以它们回答的是“这张卡上整体划不划算”，不是“这个形状划不划算”。`feature.w4a8` / `feature.deltanet` 的结论是各形状相对基线的中位数，所以 `--shapes` / `--deltanet-shapes` 要覆盖你实际在跑的尺寸。
- deltanet 只比了单次调用（含 seq>1 的循环）后的输出和最终 state；`snapshots` 中间态没有逐个比对。
- qwen35 的 A/B 只覆盖 prefill（`past_key_value=None`）：recurrent 解码路径要构造 `FixedKVBias` 缓存，还没有自动用例；权重是合成模型而不是 checkpoint。gate 关 = 逐字调用 core 的方法，所以这个对比是“我们的类 vs core 的类”，不是“对不对”。
- `feature.w4a8` 取决于权重本身：`s_rel` 的分格决定 int8 基线是否可比（bench 用 1..15.5，和 `tools/requant_uniform4.py` 一致），`--w4a8-group` 默认 128，`group_size` 必须整除 K。
- `group_size < 16` 时走 `tl.sum` 保底：它保证不崩、结果正确，但没有矩阵指令，通常比 int8 慢。表里 `feature.w4a8="on"` 只说明“在这张卡上按这次的形状测下来划算”，不代表保底一定快。
- `COMFY_GFX906_TILE_CONFIG` 只影响 int8 tile，不会启用 DeltaNet/Qwen3.5/w4a8。
