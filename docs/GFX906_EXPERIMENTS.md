# gfx906 实验记录（归档）

[README](../README.md) · [CONFIG](CONFIG.md) · [GFX906](GFX906.md) · [PORTING](PORTING.md) · [HIP_KERNELS](HIP_KERNELS.md)

这里存放从 [HIP_KERNELS.md](HIP_KERNELS.md) 正文与附录移出的 gfx906 专属实测内容。
全部来自本机 MI50/MI60 实测，**不是其它架构的参数，也不能照搬**。

单次实验的过程叙述已精简：保留结论、数字表与「不要再做」的路径，删去逐项归因、
rocprofv3 消融过程与实现步骤；需要复现时见对应的 `gfx906fix/*/NOTES.md`、`.cu`
与 `tools/verify_*.py`。写内核的方法见 [HIP_KERNELS.md](HIP_KERNELS.md)，可直接沿用的
结论见 [GFX906.md](GFX906.md)。本文件是归档，不参与决策。

## gfx906 的硬件限制（其它架构不一定成立）

> 这一节是 **gfx906 专属**。RDNA2+/CDNA 有矩阵核心、不同的 wave 宽度和更大的
> LDS，下面每一条都要重新实测，不要照抄到别的卡。

- **没有 MFMA（矩阵核心）**。所谓 int8 / fp16 点积其实是 VALU 指令
  `v_dot4_i32_i8` / `v_dot2_f32_f16`，一条只做 4 或 2 个 MAC。`v_dot8`
  存在但要求两端都是 int4；`v_dot4c` / `v_dot8c` 不可用。
- **64 lane/wave**（不是 32），一个 CU 有 4 个 SIMD、64 KB LDS。
  只有 1 block/CU 才拿得到 256 VGPR/thread；2 blocks/CU 会把上限压到
  128 VGPR，64 累加器的 tile 直接 spill。
- 所以「**1 block/CU = 4 waves + 大寄存器 tile**」是这里常见的最优解，
  而不是「波次越多越好」。4 波的 VALU 天花板明显低于理论峰值。
- **纯 fp32 FMA 在逐次使用、转换成本摊不掉的场景反而快过 dot 路径**；
  只有像 GEMM 那样转换成本被复用摊薄时，dot 才赢。别看到理论 dot 吞吐就上 QDQ。
- **bf16 是软件模拟**，fp16 才是快速路径；conv3d 没有可用内核，torch 会
  降级成 im2col + bmm。
- HBM 实测远低于标称，交错布局还会丢掉约一半 sector。
- DPP / `ds_bpermute_b32` / SDWA / `v_perm_b32` 都可用且便宜，
  行内 shuffle 用 DPP 比 LDS 往返快约一倍；b128 的 LDS 访问远快于 b32。

## 硬件与指令天花板

| 项 | 值 |
|---|---|
| CU / 频率 | 60 CU @ 1725 MHz |
| wave | 64 lane，4 SIMD/CU |
| LDS | 64 KB/CU |
| VGPR | 256/thread（1 block/CU）；128（2 blocks/CU） |
| 理论 dot2 / dot4 / dot8 | 13.4 / 26.5 / 53.0 TMAC/s |
| 纯 dot2 实测（8 累加器，16-32 波） | 12.0 TMAC/s |
| 4 波 VALU 天花板 | dot2 7.23、dot4 12.8 TMAC/s（packed fp16 见下，更低） |
| 纯 dot4 循环（4 / 8 / 16 波） | 12.9 / 16.4 / 18.0 TMAC/s |
| 纯 packed fp16 `v_pk_fma_f16`（4 / 8 / 16 波） | 6.28 / 9.56 / 11.39 TMAC/s |
| 纯 `v_fma_f32`（4 / 8 / 16 波） | 3.64 / 4.80 / 5.69 TMAC/s（1 MAC/条） |
| HBM 实测 | 约 410 GB/s（交错布局）～ 674 GB/s（lm_head GEMV） |
| LDS b32 / b64 / b128 | 1.9-3.9 / 4.3-8.8 / 9.5-11.2 TB/s |
| 全局加载 dword / dwordx4 | 814 / 867-873 GB/s |
| 纯 FP32 FMA vs QDQ（逐次使用） | 5.95 / 4.19(dot2) / 2.00(dot4) TOPS |
| 同上（转换成本摊薄后） | 13.0 / 21.9 / 21.7 TOPS |

## 四个已落地的 HIP 内核

| 内核 | 结果 | 关键配置 |
|---|---|---|
| 融合注意力 `hip_attn.cu` | 5.0-6.0 TMAC/s，比 int8 快 1.3-1.7x。**「4 波 dot2 天花板的 75-78%」这句已作废**：消融实测两次矩阵乘占 ~95% 周期、且已贴着 4 波 dot2 天花板（旧表里「矩阵乘 21%」是段标签错位，见下面「注意力的相位分解」） | 128 Q 行 x 64 keys、256 线程、1 block/CU、32+64 fp32 累加器、XOR swizzled LDS 恰好 64 KB、**K 不做跨 tile 预取**（预取反而慢 6.7%，逐位相同） |
| 大 m int8 GEMM `int8_gemm.cu` | 16.56 TMAC/s @8192³（原 10.04，**1.6x**；对参考包 0.99x，已追平），节点级 1.10-1.16x 以上 | 128x128 tile、TM8/TN8 = 64 int32 累加器、BK=64、chunk-major LDS、**123-124 VGPR 无 spill、2 block/CU（8 波）**。操作数驻 LDS、逐 k-dword 即时读（B 跨 TM 行复用），s 循环 unroll 2、uint4 整块写 LDS、**`ALIGNED_K` 编译期特化去掉尾部掩码** |
| 激活旋转 `rotate.cu` | fp32 档约 12x（graph 内 23.6 → 2.0 us/call），端到端 38.4 → 51.6 tok/s | radix-4 butterfly，D=1/4/16 用 shfl_xor、D=64 走 LDS，无全局 Hadamard 流量 |
| W4A16 解码 GEMV `w4a16.cu` | 单行解码 3.6-4.4x（48-62 → 185-274 GB/s）；多行 `w4a16_gemv_m`（m=2..8）1.6-5.3x（11-45 → 18-238 GB/s） | 一行拆 32 lane（一个 wavefront 读 8 行 x 16 B）、grid 封顶 240 块 = 960 波、每 lane 一个 uint4、16 级 fp32 codebook 放 LDS、每个 dword 一条独立累加链、**激活按 quad 行落 LDS**（先 lane-major 修 bank 冲突 1.4x，再 b128 一条指令供 4 个权重又 1.15x）、**循环倒置**（每个 chunk 只 pack 一次激活，再遍历本块的 32 个 row-group，部分和走 LDS、每块一次 shuffle，寄存器不涨）；多行版沿用这套 lane 划分，但一个 dword 解出的 8 个格点跨 M 行复用，激活按 M 行整块落 LDS。同文件第三个入口 `w4a16_gemv_h` 把激活换成 fp16、乘加换成 `v_dot2_f32_f16`：m=1..8 再快 1.05-4.89x，寄存器从 94-240 掉到 29-59。**无 codebook 时它是默认**（`code-8` 在 fp16 里精确，数值与无损版同为 2.4e-7）；带 codebook 要把格点退到 fp16，故 `COMFY_GPU_TUNE_W4A16_F16ACT=1` 才开 |

两次占用率/归因实验的结论：int8 GEMM 从 211 VGPR/4 波（10.04 TMAC/s）改成操作数驻 LDS、
逐 k-dword 即时读、124 VGPR/2 block/CU（8 波），到 16.56 TMAC/s；`__launch_bounds__`
强压寄存器会 spill 163 B、慢 7.7x。**别用 launch_bounds 逼占用率，要减活跃区间让
分配器自己塞进去**。旧「8 波更慢」是带病对照（那次只能用 16 累加器，不是同一个 tile）；
最后 6% 来自 `ALIGNED_K` 编译期特化消除 K 尾部掩码。

### 自研 fp16 GEMM（`fp16_gemm.cu` / `gfx906_f16_gemm`）

把 int8 那套寄存器预算迁到 `v_dot2_f32_f16`：128x128 tile、64 fp32 累加器、124 VGPR /
0 spill / 2 block/CU。对 rocBLAS 逐形状 1.0–1.6x，SDXL UNet 端到端 609 → 515.6 ms（1.18x）。
与参考包仍差 ~0.93x，是内循环效率不是网格/tile 占用（每 dot2 指令构成：自研 total 12.56 /
非dot VALU 6.27 / 分支 1.125，参考 total 2.13 / VALU 0.73 / 分支 0.012）。全展开、half
unroll、inline-asm 原型都试过：静态指标全面优于 C++，动态却 0.94x 更慢 —— **wait 指令少
≠ 快，必须靠交错隐藏延迟**。折算到端到端只约 +4%，追平需复刻参考的整套流水+寄存器布局，
ROI 低，暂不做手写 GCN asm。`feature.f16_gemm` 出厂 on，`COMFY_GPU_TUNE_F16_GEMM` 可 A/B。

### 均匀格点的 VALU 化（`w4a16_gemv_m` / `gfx906_w4a16_mu`）

均匀格点（无 codebook）的 16 个 level 就是 `code - 8`，用 `v_cvt_f32_ubyteN` 在 VALU 里
构造，替掉 `tab[16]` 的 LDS gather，逐位一致。深队列 graph replay（fp32 out，us/call）：

| 形状 | 表格 | VALU | 比值 |
|---|---|---|---|
| 4096x4096 @m2 gs128 | 58.70 | 56.80 | 0.968 |
| 12288x4096 @m2 gs128 | 135.70 | 131.50 | 0.969 |
| 4096x4096 @m4 gs256 | 137.74 | 128.45 | 0.933 |
| 12288x4096 @m4 gs256 | 301.31 | 289.13 | 0.960 |
| 4096x4096 @m8 gs128 | 481.87 | 480.95 | 0.998 |
| 12288x4096 @m8 gs128 | 866.39 | 866.41 | 1.000 |

`feature.w4a16_valu` 进测量门，出厂 on，本机中位数快 6.1%；`bench.py --w4a16-valu --write`
可重测。`asm volatile` 版指令最少却最慢（m8 +15%，8 条转换被钉成一整块，编译器无法插进
FMA 链隐藏延迟），去掉 volatile 才有收益 —— 赢的是调度不是字节数。fp16 点积路径因
`v_cvt_pk_f16_f32` 在 gfx906 不存在、VALU 链净增 90 条指令，已删。

## 深队列下的 decode GEMV 余量

内核本身耗时：300 次连续提交、只在两端同步，convrot 关，激活按 fp32 进内核。GB/s 按
权重流量折算（int8 一字节、int4 半字节）；同形状 int8 GEMV 是这一档的实际参照。

| 内核 | 4096x4096 @m1 | 12288x4096 @m1 | 12288x4096 @m8 |
|---|---|---|---|
| `int8_gemv.asm_kernel`（w8a8，只接 m=1） | 41.9 us / 400 | 76.9 us / 654 | 不接此形状 |
| `w4a8_gemv.gemv_kernel`（w4a8 asm） | 42.4 us / 198 | 77.5 us / 325 | 320.5 us / 79 |
| `int8_gemv.dequant_gemv_kernel`（w8a16） | 46.9 us / 358 | 87.3 us / 576 | 337.1 us / 149 |
| `w4a8_gemv.dequant_w4a16_kernel`（w4a16 Triton） | 176.8 us / 47 | 455.3 us / 55 | 1870.7 us / 13 |
| `w4a16.cu`（w4a16 HIP，同一 `linear_dequant` 入口；m=8 走多行内核） | 45.2 us / 185 | 119.0 us / 211 | 862.0 us / 29 |
| `w4a16.cu` `w4a16_gemv_h`（fp16 激活 + dot2；均匀格点默认，codebook 档要 `F16ACT=1`） | 37.9 us / 221 | 105.2 us / 239 | 234.0 us / 108 |

- w8a16 在 m=1 已和 int8 GEMV 同级（358-576 对 400-654 GB/s），无内核余量；m≥4 掉到 149。
- w4a16 是余量最大的一档（Triton 13-55 GB/s）。落地顺序：lane-major 修 8-way bank 冲突
  1.37-1.55x → quad 行 + `ds_read_b128` 1.15x → 循环倒置 1.19-1.45x，最终 `w4a16.cu`
  单行 185-274 GB/s（3.6-4.4x），离读取地板 631-776 GB/s 仍差 2.3-4.3x。
- 已否：消 barrier（更慢）、预取下一块 uint4（-17%）、Triton fp16 大 dot（慢 1.6-7 倍，
  见下文「试过但没采用的路径」）、手写 HIP w4a8 只链式 dot4（190-224 GB/s，只在小 n 赢 44%）。

## 注意力的相位分解：矩阵乘占 ~95%，那个「40% 尾段」是标错的

旧 `s_memtime` 相位表给过「矩阵乘只占 21%」，是段标签错位：QK/PV 的时间被记进了 softmax
和尾段。2026-11 消融裁决（删 QK / PV 的 dot2 循环，看总时间降多少）：

| 形状 | base | QK | PV | QK+PV | 其余 |
|---|---|---|---|---|---|
| [1,4608,24,128] | 21.84 ms | 46.7% | 43.4% | **96.5%** | 3.5% |
| [2,4096,10,64] | 8.58 ms | 61.6% | 43.9% | **95.0%** | 5.0% |
| [1,4096,24,128] | 16.92 ms | 47.0% | 43.2% | **96.4%** | 3.6% |

真实 q/k/v（SDXL 两步采样实抓）对 fp32 参考的 relL2；`int8` / `int8qk` 两档已按此删除：

| 档 | [2,4096,10,64] | [2,1024,20,64] |
|---|---|---|
| fp16（hip 与 triton 同级） | 2.13e-04 | 2.14e-04 |
| int8 QK + fp16 PV（smooth） | 6.11e-03 | 6.96e-03 |
| int8 QK + fp16 PV（raw） | 6.46e-03 | 7.24e-03 |
| int8 QK + int8 PV | 7.73e-03 | 1.10e-02 |

结论：**注意力是 VALU 发射受限的矩阵乘**（QK+PV 占 ~95%），不是 softmax/barrier 受限。
由此 int8 注意力的理由变了（理论可省 ~40% 而非 10%），但拦路虎只剩 QK 量化的 30x relL2
损失；误差由 QK 量化主导（fp16→int8 是 29-33x），再量化 PV 只加 1.3-1.6x。
`2 block/CU` 本身救不了注意力：d64 的 TQ 4→2 真拿到 8 波、输出逐位一致，主图形状收益
0.99-1.07x（Q 块翻倍让 K/V 全局重读翻倍）；现有 Triton int8 档半字节操作数 3.29-3.56
TMAC/s，比 fp16 HIP 的 5.66 更慢，字节减半没兑现。删 K 跨 tile 预取快 6.7%（已落地）、
V 寄存器预取慢 5%，都自洽于「瓶颈是乘加吞吐」。

## 注意力后端的适用域（时间 + 峰值额外显存）

`peak extra` 是一次调用在 q/k/v 之外多占的字节。sdpa 的 flash / mem-efficient 在这张卡都是
`No available kernel`，表中 sdpa 一列即 math 回退。`int8` / `int8qk` 两档已按此表删除，
数字保留作证据；默认列是本机不传启动标志时 ComfyUI 实际走的 `attention_sub_quad`。

| 形状 | hip | gemm | hand | int8 / int8qk | comfy basic / sub_quad | sdpa math |
|---|---|---|---|---|---|---|
| [2,4096,10,64] 实抓 | **8.51 ms / 10 MiB** | 14.37 / 584 | 14.58 / 10 | 13.36·15.34 / 65·90 | 19.93 / 14.17，1310 | 31.24 / **2960** |
| [1,4096,10,64] | **4.63 / 5** | 7.06 / 372 | 7.28 / 5 | 6.98·7.94 / 33·45 | 10.13 / 9.49，655 | 16.78 / 1480 |
| [1,1024,20,64] | **0.56 / 2** | 0.87 / 77 | 1.86 / 5 | — | 1.02 / 0.91，88 | 3.75 / 400 |
| cross 1024x77 | **0.05 / 1** | 0.10 / 36 | 0.07 / 1 | 0.15·0.14 / 1·41 | 0.21 | — |
| CLIP 77x77 h20 | **0.03 / 0** | 0.09 / 33 | 0.04 / 0 | 0.15·0.13 | 0.18 / 0.22，33 | — |
| [1,4608,24,128] | **21.85 / 28** | 31.29 / 630 | 58.69 / 28 | 36.68·46.10 / 166·220 | 41.21 / **121.04**，1756 | 117.97 / **4590** |

hip 在这个域里严格 Pareto 最优（4.6-6.0 TMAC/s，其余最高 4.2），且是唯一不物化 score
的路径。**hip 不合用的三种情况都不是 hand / int8 能补位的**：head dim 不在 {64,128}
（只有 gemm 收）、decode sq=1..16（hip 比 gemm 慢 3.2-5.1x）、fp32 输入或 `fp16_attn`
guard 触发（只有 gemm 收）。

## 输入就是 int8 的注意力内核（`hip_attn_i8.cu`，2026-11）

与「把 fp16 临时量化成 int8」不同，这条的调用方手里已经是 int8，内核算不动点、只把结果
dequant 到累加器。两处真实输入：`comfy_kitchen.int8_attention` 契约、Qwen-Image 2.1 int8
前缀缓存。契约见 `kernels/hip/attention_i8.py`：Q/K 为 int8 + 每 (token,head) scale，V 打包
成 int32（4 key/字）并在 SK 方向补到 64 的倍数。内核沿用 fp16 版 64-key tile，QK 用
`v_dot4_i32_i8`、PV 用 `v_dot4_u32_u8`；`TQ=2` 时 d128 为 128 VGPR / 25 KB LDS / 2 block/CU
（8 波），靠降主循环展开砍寄存器得到（单行/单 quad 循环必须全展开，降 unroll 慢 2-3 倍）。

实测（`tools/verify_int8_attn.py`、`tools/bench_int8_attn.py`）：

| 项 | 数值 |
|---|---|
| 对「同一批 int8 张量」的 fp32 参考 relL2 | 5.4e-3（P 8 bit 的底，不是内核误差） |
| 对原始 fp32 输入 relL2 | 1.1-1.3e-2（fp16 内核 2.5e-4，约 50x） |
| 有离群通道（q/k 第 0 维 ×40）relL2，rotate off / 64 / 128 | 5.3e-2 / 3.3e-2 / **1.5e-2** |
| 内核本身 vs `hip_attn`（1024² h16 d128 / 256×1024 h8 d128 / 512² h20 d64 / 4096×256 d128） | **1.42x / 2.22x / 1.69x / 1.52x**（4 波版 1.20 / 1.60 / 1.26 / 1.33） |
| 同上，前缀缓存形状 4032×8090 h2 d128（fp16 4.65 ms） | 1.30x（4 波）→ 实测同批数据 8 波更快，见下节缓存表 |
| 含 torch 端量化打包的总时间 | 0.73x / 0.71x / 0.61x（形状同上） |

归因（rocprofv3 + 编译期宏消融，1024² d128 h8）：`VALUUtilization 99.8%` 但 `VALUBusy 38.7%`，
是喂不饱不是算力不足。短路宏（深队列总时间下降）：

| 短路项 | 1024² d128 h8 | 4096² d64 | Qwen 4058+4032 |
|---|---|---|---|
| 删 QK + PV 的 dot 循环（含其 LDS 读） | 79.6% | 74.1% | 81.2% |
| **只把 dot 循环的 LDS 读换成寄存器值**（dot4 全留） | **42.0%** | 39.1% | **43.5%** |
| 全局载入换常数 | 9.7% | 8.4% | 13.8% |
| 删光 `__syncthreads` | 0%（噪声） | — | ~9%（破坏正确性，存疑） |

**dot 循环里 LDS 读的吞吐是最大单项成本（39-44%），超过 dot4 指令本身**；全局内存、L2、
bank conflict、barrier 都不是瓶颈。根子是 `TQ=2` 时每 lane 只有 2 个 Q 行、操作数复用度低，
而 128 VGPR 上限不让 TQ 变大（每 tile 每 lane 152 条 `ds_read_b128` 喂 1024 条 dot4，约
2.5 B LDS/dot4，int8 GEMM 靠 TM=8 行复用只要约 1 B）。

已否：删非 dot 段指令（被 dot 发射完全掩盖，不提速）、P 不落 LDS 改 `__shfl`（慢 3-5%，
`ds_bpermute_b32` 同样占 LDS 单元、动态指令翻 4 倍）、累加器分时复用（`acci` 压进 PV 慢
4-11% 且 VGPR 几乎没降；`accp` 挪 LDS 掉到 4 波；即便真上 3 块/CU，扣掉不随波数变快的
约 15% 非 dot 段，收益上限只有 +4~5%）。下一轮该投 QK 的 `kw` 与 PV 的 `w` 复用，两者都
要先解决累加器占约 80 个寄存器、TQ 上不去的根子。唯一落地的指令级优化是 `VKEY` 分支把
`255.f/mx` 换成 `v_rcp_f32`（+1.4~3%，1 ulp 代价，verify 脚本数字逐行不变）。

接入点 `patch_int8_attention` 替换 comfy_kitchen 四个入口并注册 `comfy_kitchen_int8`，默认
**关**（节点 `int8_attn=on` 打开）。尚未覆盖：带 mask 的调用、D=256、comfy_kitchen 的 K 平移。

## Qwen-Image 2.1 int8 前缀缓存：直读 int8（2026-11）

`take()` 不再把缓存反量化成 fp16，K/V 保持 int8 直接进 `hip_attn_i8`；当前步 Q/K/V 由
`attn_quant.cu` 一次遍历完成 H64 蝶形旋转 + 每行 absmax + int8，V 从缓存前缀与当前步两个
来源直接打包。K 双源读省掉每块 16.6 MB 的前缀拷贝，输出在旋转基里最后用同一个 H 转回。

实测（`tools/verify_qwen21_int8_cache.py`）：

| 项 | 数值 |
|---|---|
| 输出 relL2 对「反量化缓存 + fp16 当前步」参考 | 1.0-1.1e-2（同一参考上 fp16 内核 2.6e-4） |
| 4058 前缀 + 4032 目标、32 head、d128 | shipped 51.3 ms/块 → **33.1 ms/块（1.55x）** |

每块分解：注意力内核 32.1 ms（94%，8.32 TMAC/s）、旋转+量化 0.67 ms、V 打包 0.10 ms。
三处改法都是访问模式而不是算法：旋转/量化与反旋转改 256 线程 4 行/块（各 1.6x / 1.9x）；
V 打包派块顺序从 `grid=(q, B*H)` 改成 `(B*H, q)`，32 个 head 变成 4 KB 连读（0.61 →
0.099 ms，667 GB/s，块粒度只贡献 5%）；输出反旋转融进 epilogue，省一次 66 MB 往返。

两个坑：内核按 `[B,SK,H,D]` 寻址并按 SK 掩码，传补到 64 的缓冲会让 b>0 批次整体错位
（b=1 看不出来，b=2 误差 0.8）；K 的行索引用 HKV 不是 H（无 GQA 的 Qwen-Image 2.1 看不出，
GQA 用例报 1.75 才暴露）。开关 `qwen21_int8_cache` 默认 off，关闭时 `take()` 与原路径逐位一致。

## 试过但没采用的路径

| 尝试 | 结果 |
|---|---|
| 8 波双 kernel 拆分（QK+stats / PV） | 4608x4608 d128 从 23.3 → 30.1 ms；Q 重读 2-4 GB、PV spill 84 B |
| 寄存器预取 V 一个 tile | 219 VGPR + 80 B 私有拷贝，23.6 ms（更慢）；load 是带宽不是延迟 |
| 去掉整个 softmax | 总时间只变 7%（成本在 P store + barrier，不在数学） |
| 旋转按列分块撑到 128 个 program | 比 16 个 program 更慢（7.8 vs 13.3 us in graph） |
| 用 per-group `tl.dot` 替代 `v_dot4` | 慢 1.6-7 倍（M 补齐到 16、256 次小 dot） |
| fp16 模拟 int8 linear | 慢 1.16-1.85 倍 |
| 激活量化+打包提到独立 Triton 内核 | m=1 净亏 53%（64.5 → 98.6 us：29 us 的启动税 > 内核省的 3.5 us）；m=8 才 1.87x |
| 预取下一块的 uint4（每 lane 两个在飞 load） | 中性偏负：175/217/223/236 GB/s 不变，k=12288（12 块）慢 17%。和消 barrier 的负结果同向 —— 这个内核不缺 MLP |
| 手写 HIP W4A8 GEMV（链式 dot4 + LUT + block 内打包激活） | 4096²@m1 +44%，但 12288²/4096×12288/152064² 落后 25-40%（212-224 对 282-373 GB/s）→ 不落地，`w4a8.m1` 的 `hip` 槽位继续空着 |
| 把 `KP` 从 32 降到 16/8（每 lane 2-4 个 uint4） | 更慢（142/75 GB/s）且数值坏掉（`red[]` 按 32-lane 分组，换 KP 就没写满）；`BK` 提到 2048/4096 同样更慢（168/113）|
| `__launch_bounds__(256, 4)` 逼出 4 块/CU | 2.4x **更慢**（161-193 → 66-81 GB/s）：64 VGPR 上限把 4 条独立累加链和 gather 的中间值全挤掉，占用率不是这个内核的瓶颈，`(..., 1)` 才对 |
| W4A16 多行：给单行内核直接加 M 组累加器（每行各走一遍 dword） | m=8 只有 0.9-1.1x（13 GB/s），因为 8 个格点每行都重新 gather 一次、激活 LDS 读按 M 线性涨。落地的 `w4a16_gemv_m` 改成「解码一次、跨 M 行复用」才有 1.6-5.3x，见上面两张表 |
| 注意力里给 V 加一套双缓冲寄存器、预取一个 tile（和 K 同等待遇） | 慢 5%（24.18 对 23.05 ms），QK 段反而从 254K 涨到 442K cycles：载入挤进发射窗口。顺带否掉了「V 缺预取」这个假设 |
| 删掉注意力 tile 顶的 barrier（诊断用） | 总时间与尾段都不变（23.00 ms，尾段 40.9%，四 wave 均匀）→ 那 40% 不是 barrier 收敛，也不是 memory drain |
| 为 dot4 的 1.77x 用 HIP 重写 int8 注意力 | 速度上已不能否掉：消融实测矩阵乘占 ~95%，dot4 理论上能省 ~40%（不是旧表算的 10%）。真正的否决理由是 **int8 QK 的 30x relL2 精度损失**（6-7e-3 vs 2.1e-4），对扩散默认路径不划算；且「2 block/CU 靠字节减半」已实测不加速（见下条），唯一的杠杆就是 dot4 本身。见「注意力的相位分解」 |
| 注意力 PV 换成 fp16 累加（`v_pk_fma_f16`，64 个累加器减到 32） | 没落地，动手前先被微基准否掉：4 波 6.28 对 dot2 7.11 TMAC/s（指令数同为 2 MAC/条，发射率更低），还要多一次 P 广播、丢掉键内两两归约；且 LDS 65536 B 已把占用率钉在 1 block/CU，省下的寄存器换不来波数 |
| 多行内核把激活块提到寄存器里跨列复用（省掉每列一次 b128 读） | 中性偏负：等 chunk 宽 0.85-1.08x，M=8 还得把 chunk 砍到 512 才装得下，0.85-0.89x。份额模型预测 2.3 TMAC/s，实测更慢 —— 那些 LDS 读被 dot2 发射掩盖了，不在关键路径上 |
| `tl.reduce(combine_fn=v_dot4)` 用满累加操作数 | 表达不了：reduce 的叶子值必须是待归约元素本身，`(w, a)` 双流喂进 c 槽会把权重原值当部分和加上。只剩先 `dot4(w,a,0)` 再求和，即现状。累加链在 HIP 里才免费（`int8_gemm.cu` 的 `__builtin_amdgcn_sdot4`） |
| `w4a16_gemv_h` 用 VALU 构造 fp16x2 格点替掉 `tab2` gather | 逐位一致但**慢 5-8%**（m1 1.03、m8 1.08）：消掉 16 次 LDS 读，换来 32 条 `v_cvt_f32_i32` + 33 条 `v_cvt_f16_f32` + 30 条加减的串行链（465→555 条指令）。`v_cvt_pk_f16_f32` 在 gfx906 不存在，链短不了；LDS gather 本来被 dot2 发射掩盖。已删除 |
| 同上，但格点用 `v_cvt_f32_ubyteN`（`w4a16_gemv_m`，fp32 FMA 路径） | 585 条指令，是三种形态里**最少**的，却最慢（m8 +15%）：`asm volatile` 把 8 条转换钉成一块，编译器无法把它们插进 FMA 链隐藏延迟。去掉 `volatile` 后 m2/m4 快 3-7%、m8 仍慢 9% —— M=8 本来就在 256 VGPR/spill 53，8 个 asm 输出的活跃区间再压上去。按 m≤4 门控后**默认开**（见上面「均匀格点的 VALU 化」）|
| d64 注意力重排 NQ=64 冲 2 block/CU | 真拿到了（LDS 40→24 KB、110 VGPR、8 波、逐位一致），但主图形状 1.00-1.07x（Q 块翻倍→K/V 重读翻倍抵消占用率），只有 sq=100 这种并行度受限的形状 1.46x（绝对 0.017 ms 可忽略）→ 不落地。**结论：注意力不是占用率受限，2 block/CU 不是它的价值来源** |

| `255/beta` 换 exp2 恒等式、`255/mx` 换 `v_rcp_f32`（消 fp32 除法链） | exp2 版 VK0 慢 1-2%（占 SFU 发射槽）；rcpf 版 VK0 也慢 1-2%，只有 VKEY 分支赢（+1.4~3%）→ 只在 VKEY 落地，见上节 |
| 把 `Sv` 的 LDS 读提到 `for t` 外；`SK%64==0` 时编译期消掉尾 tile 掩码（照 `ALIGNED_K`） | 前者编译器已做 CSE（`ds_read_b128` 不变，纯 no-op）；后者 `v_cndmask` 34→2 但计时中性、引入 ~2e-5 舍入差异 → 都不落地 |
| K/P 分离 LDS 缓冲少一个 barrier；K tile 行加 `+1 vec4` padding 破 32-bank 别名 | 都逐位一致但计时中性（0.98-1.01），前者还多占 4-8 KB LDS：barrier 与 LDS bank 都不是瓶颈，XOR swizzle 已经够 |
| P 不落 LDS：`ph[]` 留寄存器，PV 用 `__shfl`/`ds_bpermute` 取 P（±去两个 barrier） | 逐位等价但**慢 3-5%**。省 2 条静态 `ds_read_b128`、多 8 条 `ds_bpermute_b32`（动态 8→32 条/tile）；精细消融显示 P 的 LDS 读只占 5.1% 且是广播读。归因小节有完整数据 |
| `acci` 按 d 分片压进 PV 循环 | 逐位等价但**慢 4-11%**；VGPR d128 只省 2 个（编译器已分时复用 `accq`/`acci`）、d64 省 12 个到 103，都换不到第 3 块，反而让 `pw` 的 LDS 读重复 `DPT/4` 次 |
| `accp` 挪到 LDS 分时复用 | +32 KB LDS 让 d128 掉到 **1 块/4 波**（诊断实测 41728 B），d64 仍是 2 块零收益，且每 tile 多 32 次 LDS 读写 |

这些「不要做」的结论和 gfx906 的 LDS/VGPR 预算绑得很紧；换一张卡，
结论可能完全反过来。

## w4a16 的资源与相位（m=1）

`vgpr 94 / spill 0`，2 block/CU（8 波）；`-DMINB=4` 压到 64 VGPR 实测慢 2.4x，占用率不是
这条路的瓶颈。`pack` 相位从 44-46% 掉到 7-25% 只换来 1.19-1.45x：那一段里混着
`__syncthreads` 等最慢 wave 到齐的时间，**含 barrier 的段不能按份额折算收益**。先前
「整行一次 pack 反而更慢」的 NOSYNC 实验把 `xa` 撑到 32 KB、驻留掉到 1 block/CU，是 LDS
尺寸失败不是「提出来」失败 —— 一个结论被没控制的变量陪着，等于没测。

## fp16 激活变体（`w4a16_gemv_h`）

fp16 激活 + `v_dot2_f32_f16` 比 fp32 激活版快 1.05-4.89x，赢在寄存器（`vgpr 29-59` →
M=4 有 4 block/CU、M=8 有 2 block）而不是指令数减半；单行只有 1.05-1.20x，卡在读通路。
均匀格点（无 codebook）默认走它，数值与无损版同级（fp32 输出 2.4e-7、fp16 输出 1 ulp，
因为 `code-8` 在 fp16 里精确）；带 codebook 要把格点退到 fp16，故 `F16ACT=1` 才开，开启前
按 `--sampling off` 重验生文 md5。已否：寄存器驻留跨列复用（0.85-1.08x，省下的 b128 不在
关键路径）、`GB` 8→32（只在 152064 这类大 n 有效）。

## 从正文移来的 gfx906 专属条目

**融合注意力的 128 行 Q tile 在 sq < 128 时是结构性不合用。** grid 是
`ceil(sq/128) * B * H` 个 block：sq=1、B=1、H=10 时只有 10 个 block，等于
**0.17 block/CU**，60 个 CU 里 43 个闲着。实测 sq=1/sk=4096 是 0.80 ms，而 gemm
0.23 ms（sk=8192：1.49 对 0.63；d128：1.16 对 0.24）——差 3.2-5.1x，跟内核写得好不好
无关。图像模型不会出现这种形状（self/cross attention 的 sq 至少几百），文本生成的
decode 注意力走的是 KV cache 那条，所以今天它不是活路径；但「把 sq=1 的注意力交给
hip」是个真实的误判，`supported()` 只看 head dim 和连续性，不看 sq。

**「换更窄的累加器」在这条卡上几乎没有落脚点。** gfx906 每条 2 MAC 的 fp16 指令只有
`v_dot2_f32_f16`（累加端 fp32，ISA 写死）和 `v_pk_fma_f16`（累加端 packed fp16），
补测同 ILP、同占用率（`gfx906fix/hip_attn_v2/mb_pk.cu`）：4 波 6.28 对 7.11、
8 波 9.56 对 9.57、16 波 11.39 对 11.23
TMAC/s —— 换过去在注意力实际的 4 波反而慢 13%，指令数还一样是 2 MAC/条，另外要付
「把 P 广播进两个半字」并丢掉 dot2 免费的键内两两归约。int8 那侧 `v_dot4_i32_i8` 只出
int32，形态上就不可降；`rotate.cu` 12 VGPR、`w4a16.cu` 单行 5 个累加器，没有压力可解；
`hip_attn.cu` HDT=128 的 LDS 正好 65536 B = 整个 CU，占用率不由寄存器决定。fp16 成对
唯一还剩的价值是**替换 fp32 FMA 链**：`v_pk_fma_f16` 对 `v_fma_f32` 在 4/8 波是
1.7-2.0x，这才是 w4a16 接 m≥2 时该用的形态（它现在是 fp32 FMA）。

**本卡没有可用的 eager int8 矩阵乘。** `torch._int_mm` 在本机的 hipblasLt 上（缺
`TensileLibrary_lazy_gfx906.dat`）m=1/16/64/1024 全部抛 `HIPBLAS_STATUS_INVALID_VALUE`，
所以 comfy-kitchen 的 triton 后端在这里不是「可选加速」。`COMFY_GPU_TUNE_TRITON_BACKEND=off`
只该用于 A/B 本包没覆盖的操作（4096x4096：`quantize_int8_rowwise` 113.7→381.3 us、
`dequantize_per_tensor_fp8` 666.9→863.4 us）。本包在 triton 和 eager 两个后端的
`int8_linear` 上都装了同一层路由（GEMV → HIP int8 GEMM → triton 内核兜底），所以这个
开关不改变 int8 线性的落点，也不会让 2..127 行的批掉进 eager 报错 —— 那一档既不在
GEMV（只接 m=1）也不在 HIP GEMM（`MIN_ROWS=128`）里。

- **本仓库的实测记录**：`gfx906fix/hip_attn_v2/NOTES.md`（融合注意力全过程、
  微基准、8 波 split 的负面结果）、`gfx906fix/hip_rotate/NOTES.md`
  （butterfly 推导、graph 内计时方法）。
- **真实 q/k/v 的抓法**：`gfx906fix/capture_sdxl_qkv.py` 在 `optimized_attention`
  上挂 hook、跑两步 SDXL 采样，把 `[B,S,H,D]` fp16 的 self-attention 张量存成 .pt。
  注意力量化档的 relL2 和 softmax 参与率都是从这儿来的。
- **看内核真实资源**：`--save-temps` 留不下 GCN 汇编，得从 `.so` 的
  `.hip_fatbin` 里挖设备 ELF（找 `\x7fELF`，按 `e_shoff/e_shnum/e_shentsize`
  截尾），再 `llvm-readelf --notes` 读 `vgpr_count` / `vgpr_spill_count`。
  spill 非零说明累加器和临时值不够；但 **spill=0 不代表占用率健康**——
  int8 GEMM 211 VGPR 零 spill 却静默砍半占用率（2 block/CU 要 ≤128）。必须把
  `vgpr_count` 对着阈值算实际波数，用 `tools/occupancy_audit.py` 一次扫完所有
  `.so`，它会标出「dot 内核被 VGPR 压在 8 波以下」的项。
- **相位计时**：房内已有惯例 —— `gfx906fix/hip_attn/hip_attn_f16.cu:149` 的
  runtime-gated `s_memtime` 戳（`hip_attn_v2/mb_stamp2.cu` 验过 builtin）。
  三条必要约定：只有 block 0 记时；**每个 wave 一套槽**（写手 `(tid & 63) == 0`，
  索引带 `tid/64`）——共用槽会让跨块差值变负、四个 wave 互相覆盖；戳要打在大
  「消费上一次结果的语句之后」，否则量不到 load 等待。再加两个自检：跨块差值必须
  为正、四个 wave 的 span 应吻合在几个百分点内。还要把 span 和
  「实测时间 ÷ 每块 row-group 数」比一下：比值 <1 说明一个 CU 上排了多轮块，
  那是驻留/轮数信息，不是计时错了。**含 `__syncthreads` 的段不能按份额折算收益**：
  那一段里混着「等最慢的 wave 到齐」，把它要做的活搬走，等待不会消失，只会挪进
  下一个 barrier 所在的那一段重新分配（`w4a16.cu` 的 pack 相位就是这么从 44% 掉到
  7-25% 却只换来 1.19-1.45x，而按份额本该值 1.3-1.8x 的）。
  还有两条边界：戳的 `asm` 不带 `memory` clobber 时，调度器可以把载入/存储挪过戳，所以
  **段边界是近似的**，别把某一段的绝对值当机制；以及给同一内核加一版不带戳的构建对比墙钟
  （注意力那次是 23.05 对 23.27 ms，说明打戳本身不花钱）。
