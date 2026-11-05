# 给其它架构写 HIP 内核

这篇讲怎么给这个节点加一个手写 HIP 内核，以及 **gfx906 上哪些结论不能照搬**。
门控、测量数据和后端链请看 [PORTING.md](PORTING.md)。

本文里的数字都在末尾附录；正文只讲方法和判断。

## 先判断值不值得写

手写 HIP 只在一类问题上划算：**内核是 dot 密集、有 tile 级复用，而且能靠
LDS 布局或寄存器分配救回来**。我们在这里成功过两次（融合注意力、大 m int8
GEMM），也失败过几次：

- 带宽受限的 decode GEMV：换更宽的加载、更少的指令都没用，权重流式读取是硬上限；
- 绝对耗时本来就小的算子（DeltaNet 递推、量化、归一化）：快 2 倍也只省零点几毫秒；
- 把 Triton kernel 直译成 HIP：算法没变，只是换语言，收益接近 0。

一句话：**先分清是算法问题还是指令问题**。把 256x256 稠密 Hadamard 乘法换成
radix-4 butterfly 拿到约 12 倍，靠的是消掉 8 次串行全局读和 O(G^2) 的计算，
不是因为 HIP 比 Triton 快。

动手前先用 `tools/chain_profile.py` 找出时间落在哪一段，再用微基准把那一段
单独计时。如果热点在 bench 里没有对应站点，先加站点再优化。

## 基本流程

1. **量基线**。CUDA event 交替轮次取 min；如果节点用了 per-block CUDA graph，
   要在 graph 回放里量，单次 launch 的数字会被 Python 调度污染。
2. **写 `.cu`**。用 `extern "C"` 平铺 ABI：`void*` 指针加 `int` 尺寸
   加 `void* stream`，返回 `int` 错误码。不做 torch 扩展，用 ctypes 直接调。
3. **构建**。把源文件加进 `tools/build_hip_attn.py` 的 `TARGETS`，复用同一套
   hipcc 参数。产物是某个 `gcnArchName` 专用的代码对象，别的架构要各自编译。
4. **写 loader**。照 `hip_attention.py` / `hip_int8_gemm.py` / `hip_rotate.py`：
   检查 `torch.version.hip`、CUDA 可用、`gcnArchName`、`.so` 存在；用
   `RTLD_GLOBAL` 打开 `libamdhip64`；加载失败打一条 WARNING 并给出重编命令，
   返回 `None` 让调用方回落。
5. **接入站点**。在 `BackendRegistry` 加一个 Candidate，或在调用点分派。
   一定要留一条自动回落路径（Triton / core），再给一个能关掉的开关。
6. **验收**：数值等价（见坑点）、A/B 计时、开关 `off` 能复现旧行为、
   `.so` 缺失 / 加载失败 / 形状不符时节点照常启动。

## 常见坑点

**stream 必须在调用时读。** `torch.cuda.graph()` 会把 `current_stream` 切换到
捕获流；用捕获前缓存的 handle，launch 不会进图，测出来是「空图 + 0.04 us」
这种假数字。最容易踩、也最隐蔽。

**等价性要按消费者验收。** 旋转的输出会被量化成 int8：只比旋转值的 relL2
不够，要比量化后的 int8 是否逐元素一致。fp16 输入上 HIP 用 fp32 累加会更准，
但与 Triton 不等价 —— 要么刻意对齐精度，要么明确接受并重验生文 md5。

**A/B 一次只动一个变量**，并且用 `--sampling off` 让 md5 有意义。采样开启时
微小数值差会逐 token 放大成完全不同的文本。反过来，更快但生文从第一个
token 就退化，也是真失败。

**设备侧浮点函数用 `__builtin_*`。** HIP device code 里没有 `fmaxf` / `fabsf`
的匹配重载，要写 `__builtin_fmaxf` / `__builtin_fabsf`，否则 hipcc 报
`no matching function`。

**架构假设要显式探测。** 代码对象、wave 宽度、LDS 大小、VGPR 上限、可用指令
（dot4 / dot8 / MFMA）全都随架构变。加载失败不能让 ComfyUI 起不来；本包的
`_install` 只把设备/编译级异常降级，逻辑错误照常抛 —— 别用宽 `try/except`
把 bug 也吞了。失败项必须出现在 `status()` 里，不能只留一条没人看的日志。

**编译产物随包分发时要固定运行时。** 本机 HIP 大版本不同会 dlopen 失败；
节点会 WARNING 并回落 Triton，重编命令是
`tools/build_hip_attn.py --force`。`.so` 只含目标架构的代码对象，换卡时
loader 按 `gcnArchName` 直接跳过。

**LDS bank conflict 往往是第一瓶颈。** 行主序 padding 不一定够：列消费的
operand 用 `+1 vec4` padding，八行八列同时别名时只能 XOR swizzle。先用
微基准把 LDS 指令比隔离出来，再动主内核 —— 我们第一个 int8 GEMM 版本
就是被这个从 3.4 拖到 10.0 TMAC/s 的。

**别急着追占用率。** 试过把 16 个 program 拆成 128 个，反而更慢；真正的
瓶颈是 8 次串行的全局矩阵读。先量，再改。同理，寄存器预取在带宽受限的
load 阶段只会增加 VGPR 和私有拷贝，实测是负结果。

**注册与测量分开。** 实现列表是代码（`BackendRegistry` /
`arch/<gcn>/kernels/`），测量结果是数据（`manifest.toml`）。新内核加进去后
用 `bench.py --all --write` 决定它在这张卡上该不该开，别写死「某架构一律
用某内核」。

**留一个开关。** 任何手写内核都要能被单独关掉，用来 A/B、用来在本机 HIP
不兼容时回退。开关名跟现有的 `COMFY_GFX906_*` 保持一致的风格。

## 参考资料

- **本仓库的实测记录**：`gfx906fix/hip_attn_v2/NOTES.md`（融合注意力全过程、
  微基准、8 波 split 的负面结果）、`gfx906fix/hip_rotate/NOTES.md`
  （butterfly 推导、graph 内计时方法）。
- **可照抄的模板**：`hip_attention.py` / `hip_int8_gemm.py` / `hip_rotate.py`
  （loader 与回落）、`arch/gfx906/kernels/*.cu`（内核）、
  `tools/build_hip_attn.py`（构建）、`kernels/registry.py` / `contract.py`
  （后端链）、`arch/gfx906/kernels/README.md`（站点契约）。
- **架构与 ISA**：`rocm-libraries/wiki-gfx906/src/studies/2026-02-21/` 下的
  dot4/dot8、LDS 布局标准、SDWA/DPP 延迟隐藏、FP32 vs QDQ 几篇实测；
  LLVM 的 `AMDGPUAsmGFX906` 语法页（确认指令在某目标上是否真的存在）。
- **rocBLAS 源码**：`rocblas/library/src/blas3/Tensile/Logic/archive/vega20_*.yaml`
  （无 MFMA 时代的 VALU GEMM tile / LDS 配置）、
  `rocblas/library/src/blas2/rocblas_gemv_kernels.cpp`
  （gfx906 专用 512 线程 gemvn、skinny-m split-K 的注释很有参考价值）。
- **工具**：`tools/chain_profile.py`（链路归因）、`tools/verify_backends.py`
  （选择 parity + 数值 A/B）、`tools/bench.py`（站点计时）、
  `tools/status.py`（不开 ComfyUI 看判定）。

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

## 其它值得记住的

- **算法替换 > 指令替换**：这个节点里最大的几次收益都来自换算法（融合、
  butterfly、去掉重复全局读），不是把同一段循环写成汇编。
- **负面结果也要记**：试过不工作的路径（8 波 split、寄存器预取、
  列分块、两 kernel 拆分）比成功路径更能省下后来者的时间。
- **改动要可回退**：包装原函数而不是复制函数体，ComfyUI 升级时最坏是
  降级到 eager，不会静默跑错。
- **先数值后计时**：`bench.py` 已经把「超 rel 门槛不参与计时」写进流程，
  自定义微基准也要照做，否则「更快但算错了」会赢。

---

# 附录：gfx906 实测数字

正文只给了定性结论，数字集中在这里。全部来自本机 MI50/MI60 实测，
**不是其它架构的参数**。

## 硬件与指令天花板

| 项 | 值 |
|---|---|
| CU / 频率 | 60 CU @ 1725 MHz |
| wave | 64 lane，4 SIMD/CU |
| LDS | 64 KB/CU |
| VGPR | 256/thread（1 block/CU）；128（2 blocks/CU） |
| 理论 dot2 / dot4 / dot8 | 13.4 / 26.5 / 53.0 TMAC/s |
| 纯 dot2 实测（8 累加器，16-32 波） | 12.0 TMAC/s |
| 4 波 VALU 天花板 | dot2 7.23、dot4 12.8 TMAC/s |
| 纯 dot4 循环（4 / 8 / 16 波） | 12.9 / 16.4 / 18.0 TMAC/s |
| HBM 实测 | 约 410 GB/s（交错布局）～ 674 GB/s（lm_head GEMV） |
| LDS b32 / b64 / b128 | 1.9-3.9 / 4.3-8.8 / 9.5-11.2 TB/s |
| 全局加载 dword / dwordx4 | 814 / 867-873 GB/s |
| 纯 FP32 FMA vs QDQ（逐次使用） | 5.95 / 4.19(dot2) / 2.00(dot4) TOPS |
| 同上（转换成本摊薄后） | 13.0 / 21.9 / 21.7 TOPS |

## 三个已落地的 HIP 内核

| 内核 | 结果 | 关键配置 |
|---|---|---|
| 融合注意力 `hip_attn.cu` | 5.1-5.6 TMAC/s（4 波天花板的 75-78%），比 int8 快 1.3-1.6x | 128 Q 行 x 64 keys、256 线程、1 block/CU、32+64 fp32 累加器、XOR swizzled LDS 恰好 64 KB |
| 大 m int8 GEMM `int8_gemm.cu` | 10.04 TMAC/s，节点级 1.10-1.16x | 128x128 tile、TM8/TN8 = 64 int32 累加器、BK=32、chunk-major LDS、211 VGPR 无 spill |
| 激活旋转 `rotate.cu` | fp32 档约 12x（graph 内 23.6 → 2.0 us/call），端到端 38.4 → 51.6 tok/s | radix-4 butterfly，D=1/4/16 用 shfl_xor、D=64 走 LDS，无全局 Hadamard 流量 |

int8 GEMM 的波形 sweep（4032x12288x4096）：2 波 5.75、4 波 9.64、
8 波 9.14 TMAC/s（8 波只能用 16 累加器，与 4 波不是同一个 tile）。
配置 sweep：128 累加器 spill 364 B、BK=64 spill 376 B，只有 TM8/TN8
不 spill。bank conflict 是首版 3.4 TMAC/s 的全部原因。

## 试过但没采用的路径

| 尝试 | 结果 |
|---|---|
| 8 波双 kernel 拆分（QK+stats / PV） | 4608x4608 d128 从 23.3 → 30.1 ms；Q 重读 2-4 GB、PV spill 84 B |
| 寄存器预取 V 一个 tile | 219 VGPR + 80 B 私有拷贝，23.6 ms（更慢）；load 是带宽不是延迟 |
| 去掉整个 softmax | 总时间只变 7%（成本在 P store + barrier，不在数学） |
| 旋转按列分块撑到 128 个 program | 比 16 个 program 更慢（7.8 vs 13.3 us in graph） |
| 用 per-group `tl.dot` 替代 `v_dot4` | 慢 1.6-7 倍（M 补齐到 16、256 次小 dot），见 PORTING.md 附录 E |
| fp16 模拟 int8 linear | 慢 1.16-1.85 倍 |

这些「不要做」的结论和 gfx906 的 LDS/VGPR 预算绑得很紧；换一张卡，
结论可能完全反过来。
