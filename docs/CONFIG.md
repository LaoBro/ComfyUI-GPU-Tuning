# 节点开关与注意力模式

[README](../README.md) · [PATCHES](PATCHES.md) · [GFX906](GFX906.md) · [PORTING](PORTING.md) · [HIP_KERNELS](HIP_KERNELS.md)

这一层回答“怎么调”：两个节点、每个开关的含义、注意力模式选择、int8 大 m 路径，
以及仍然保留的环境变量。

## 两个节点

| 节点 | 位置 | 作用 |
|---|---|---|
| **GPU Tuning Model Settings** | 模型链（加载器 → 采样器之间） | 注意力模式、连续布局、fp16 输入策略、convrot、int8 tile、`m≤8` 窄 tile、int8 大 m GEMM、int8 注意力与旋转、QKV、VAE、GEMV、prefix KV、Qwen2.1 int8 缓存、profiler |
| **GPU Tuning CLIP Settings** | CLIP 链（加载器 → 编码） | 文本编码器 int8、编码器注意力后端、TE GEMV、4-bit w4a8、文本生成补丁组 |

设置节点改动**下一次执行就生效**，不用重启。命令行复测 `benchmark()` 已删除：它没有节点注册、也没有任何调用者，长期是死代码；需要复测时直接用 `gfx906fix/` 下的独立脚本，避免再维护第三个节点。

多数开关是三态 combo（`default` / `on` / `off`）：`default` 沿用内置默认，有实测表的按
manifest 判定；`on` / `off` 显式覆盖，优先级最高。以下不是三态：`attention`
（`default` + `hip`/`gemm`/`hand` + core 后端名）、`convrot`（`default` + `fused`/`butterfly`/`off`）、
`fp16_attn`（`force`/`guarded`/`off`）、`int8_attn_rotate`（`default`/`off`/`64`/`128`）、
`text_encoder`（`int8`/`default`）、`text_attention`（只有 `default`）。

只有 `textgen`（对应 `DELTANET` / `KVDECODE` / `LLAMA` / `QWEN35` / `GRAPH`）、`int8_attn`、
`qwen21_int8_cache` 和 `int8_attn_rotate` 的 `default` 沿用启动时的环境变量值，其余开关
没有对应的环境变量。以前“节点默认值会覆盖 env”的问题已修掉。

### 节点开关清单

Model Settings：`attention`、`contiguous`、`fp16_attn`、`convrot`、`int8_tile`、`small_batch`、`int8_gemm`、`int8_attn`、`int8_attn_rotate`、`qkv`、`vae`、`gemv`、`prefix_kv`、`qwen21_int8_cache`、`profile`。

CLIP Settings：`text_encoder`、`text_attention`、`text_gemv`、`w4a8`、`image_tower`、`textgen`。

补充说明：

- `contiguous`：选中 `basic`/`pytorch`/`sub_quad`/`split` 等 core 后端时，调用前把输入转成连续的 `[B,H,S,D]`；关闭可复现 core 的原始转置视图调用。
- `fp16_attn`：三选一。`guarded`（默认）把 fp32/bf16 转 fp16，但 Q/K/V 绝对值 ≥60000 时放弃本次调用、
  回落 core（fp16 上限 65504，留余量）；`force` 无条件转，可能溢出；`off` 保持模型 dtype，自带
  hip/gemm/hand 是 fp16-only 会报错，需配 core 后端名使用。旧版 `fp16` / `amax_guard` 已并入此开关，
  旧工作流需重新连接。
- `int8_tile`：关闭后恢复上游 autotune 的 tile 列表，不再注入 gfx906 的 `128×256×32`；GEMV 仍由 `gemv` 单独控制。
- `small_batch`：gfx906 上把“`m≤8` 切到 `16×32×64` 窄 tile、`m>8` 切回 `128×256×32`”单独控制；其它 AMD 上表示“把窄 tile 追加为 autotune 候选”（是否真的用它由该卡实测决定）。需要 `int8_tile` 为 on，否则显示 n/a。GEMV 仍由 `gemv` 单独控制。
- `text_gemv`：单行文本编码器解码时，int8 权重直调 GEMV、未量化的小投影走 `torch.mv`；关闭后两者都回到分派层。
- `w4a8`：4-bit 权重解码；`status()` 会打印实际调用次数和当前后端（`hip` / `asm` / `triton`），换 checkpoint 后可据此确认是否真的走到了 w4a8 内核（`on: 0 calls ...` 表示权重布局没被识别）。能否用 `tl.dot` 替代 `v_dot4` 的实测结论见 [docs/PORTING.md 附录 E](PORTING.md)。
- `textgen`：一次控制 DeltaNet 内核、KV decode 折叠、llama RMSNorm/RoPE 缓存、Qwen3.5 计算类；关闭时逐调用委托回 core 实现（Qwen3.5 类符号本身不换回）。
- `int8_gemm`：行数 ≥ 128 的 int8 线性交给自带 HIP dot4 GEMM，`off` 强制回 Triton；详见下节。
- `int8_attn` / `int8_attn_rotate`：comfy-kitchen 的 int8 注意力契约（默认关）及其 Q/K 量化前的
  Hadamard 旋转（`off` / `64` / `128`，默认 128）；数值代价与实测见 [PATCHES.md](PATCHES.md)、
  [HIP_KERNELS.md](HIP_KERNELS.md)。
- `qwen21_int8_cache`：Qwen-Image 2.1 的 int8 prefix cache K/V 保持 int8、直接在 int8 内核上算
  注意力（默认关）；实测见 [HIP_KERNELS.md](HIP_KERNELS.md)。

### 在其它显卡上使用 / 扫描 int8 tile

其它 AMD 显卡上的功能分级（常开 / 可选开关 / 测量门 / 不可用）、门控与 dispatch 的分工、扫描工具 `tools/tile_sweep.py` 的完整用法，以及 `COMFY_GPU_TUNE_TILE_CONFIG` / `SMALL_TILE_CONFIG` 的使用方式，见：

**[PORTING.md — 其它显卡移植与二次开发指南](PORTING.md)**

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

开关都在节点里（GPU Tuning Model Settings / GPU Tuning CLIP Settings），`default` 用内置默认值
或 manifest 判定，`on/off` 显式覆盖，下一次执行生效。`textgen` / `int8_attn` /
`qwen21_int8_cache` / `int8_attn_rotate` 的 `default` 沿用启动时的环境变量值。

环境变量分三类：整包总闸、少数开关的启动值、以及没有节点入口的诊断/基准/移植旋钮。
下表是全集；移植与内核细节另见 [PORTING.md](PORTING.md) 和 [HIP_KERNELS.md](HIP_KERNELS.md)。
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
| `COMFY_GPU_TUNE_KVDECODE` | `1` | 同上：折叠 GQA 的 KV decode |
| `COMFY_GPU_TUNE_LLAMA` | `1` | 同上：llama RMSNorm 预计算 + RoPE 频率表 |
| `COMFY_GPU_TUNE_QWEN35` | `1` | 同上：Qwen3.5 计算类替换（等价改写，逐方法对过 core） |
| `COMFY_GPU_TUNE_GRAPH` | `1` | textgen 分项基准旋钮：权重常驻时的逐块 CUDA graph（也跟随 CLIP 节点的 `textgen` 开关） |
| `COMFY_GPU_TUNE_INT8_ATTN` | `0` | int8 注意力契约（checkpoint 标记 / int8 minimax VAE）的启动值；节点 `int8_attn=default` 沿用，`on`/`off` 覆盖 |
| `COMFY_GPU_TUNE_INT8_ATTN_ROTATE` | `128` | int8 注意力 Q/K 量化前的 Hadamard 旋转组：`off`/`0` 关，或 32/64/128；节点 `int8_attn_rotate=default` 沿用 |
| `COMFY_GPU_TUNE_QWEN21_INT8_CACHE` | `0` | Qwen-Image 2.1 int8 prefix cache 保持 K/V 为 int8 的启动值；节点 `qwen21_int8_cache=default` 沿用 |
| `COMFY_GPU_TUNE_F16_GEMM` | `1`/`0` | 非量化 fp16 linear 的 HIP GEMM；`off` 强制关掉做 A/B。缺省值由 manifest 的 `feature.f16_gemm` 决定，未实测卡为 `0`。见 [HIP_KERNELS.md](HIP_KERNELS.md) |
| `COMFY_GPU_TUNE_ROTATE` | `1` | 解码激活旋转的 HIP butterfly 内核；`off` 回落到 Triton 做 A/B。fp16 输入下 HIP 以 fp32 累加，量化激活更准但与 Triton 不等价 |
| `COMFY_GPU_TUNE_W4A16_HIP` | `1` | te=default 4-bit 反量化 GEMV 的 HIP 内核（`arch/gfx906/kernels/w4a16.cu`，m=1 与 m=2..8 两个入口）；`off` 回落到 Triton 的 `dequant_w4a16_kernel` 做 A/B。m>8 或不合形状的批仍走 Triton |
| `COMFY_GPU_TUNE_W4A16_F16ACT` | `0` | 只管**带 codebook** 的检查点是否用 fp16 激活 + `v_dot2_f32_f16` 的 feed（`w4a16_gemv_h`，m=1..8，比无损 feed 快 1.05-4.89x）；代价是 16 个格点退到 fp16（fp32 输出归一化 1.6-2.6e-4、fp16 输出约 1.5 ulp），开启前按 `--sampling off` 重验生文 md5。无 codebook 的检查点不看这个开关，默认就走该 feed（均匀格点在 fp16 里精确） |
| `COMFY_GPU_TUNE_W4A16_VALU` | 空 | **测量门**：缺省时由 `feature.w4a16_valu`（本机/出厂 manifest）决定，出厂表为 `on`；显式 `on`/`off` 覆盖它。均匀格点（无 codebook）的 m=2..4 GEMV 用 `v_cvt_f32_ubyteN` 在 VALU 里构造 16 个格点，替掉 `tab[16]` 的 LDS gather；逐位一致，实测快 3-7%（数字见上文与 docs/HIP_KERNELS.md）。M=8 自动回落表格路径（该形状已 spill），带 codebook 的检查点不看这个开关。`bench.py --w4a16-valu --write` 可重新测并改写判定 |
| `COMFY_GPU_TUNE_TRITON_BACKEND` | 空 | comfy-kitchen 的 triton 后端是否让 dispatch 选：空=跟随启动参数（`--disable-triton-backend` 就不开），`on`/`off` 覆盖它。`off` 让本包没覆盖的 kitchen 操作回落 eager，便于 A/B（实测 4096x4096 上 `quantize_int8_rowwise` 113.7→381.3 us、`dequantize_per_tensor_fp8` 666.9→863.4 us）；int8 linear 不受影响——两个后端的 `int8_linear` 都包了本包的路由，兜底始终是 triton 内核，因为 eager 的 `torch._int_mm` 在本卡每个 m 都抛 `HIPBLAS_STATUS_INVALID_VALUE` |

