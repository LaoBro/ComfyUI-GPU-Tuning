# 给其它架构写 HIP 内核

这篇讲怎么给这个节点加一个手写 HIP 内核，以及 **gfx906 上哪些结论不能照搬**。
门控、测量数据和后端链请看 [PORTING.md](PORTING.md)。

本文里的数字都在末尾附录；正文只讲方法和判断。

## 先判断值不值得写

手写 HIP 只在一类问题上划算：**内核是 dot 密集、有 tile 级复用，而且能靠
LDS 布局或寄存器分配救回来**。我们在这里成功过两次（融合注意力、大 m int8
GEMM），也失败过几次：

- 带宽受限的 decode GEMV：换更宽的加载、更少的指令都没用，权重流式读取是硬上限
  （这条只适用于已经贴顶的内核，int8 GEMV 深队列 400-654 GB/s 才是这一类）；
- 绝对耗时本来就小的算子（DeltaNet 递推、量化、归一化）：快 2 倍也只省零点几毫秒；
- 把 Triton kernel 直译成 HIP：算法没变，只是换语言，收益接近 0。

一句话：**先分清是算法问题还是指令问题**。把 256x256 稠密 Hadamard 乘法换成
radix-4 butterfly 拿到约 12 倍，靠的是消掉 8 次串行全局读和 O(G^2) 的计算，
不是因为 HIP 比 Triton 快。

动手前先用 `tools/chain_profile.py` 找出时间落在哪一段，再用微基准把那一段
单独计时。如果热点在 bench 里没有对应站点，先加站点再优化。

## 基本流程

1. **量基线**。CUDA event 交替轮次取 min；如果节点用了 per-block CUDA graph，
   要在 graph 回放里量，单次 launch 的数字会被 Python 调度污染。两种计时分的不是
   精度而是对象：单次调用的区间量用户付的代价（A/B、manifest 用它），判断内核离
   硬件上限还有多远必须用深队列（连续提交几百次、只在两端同步），见坑点。
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

**单次计时会把内核的余量算大 3 倍。** 单次调用的 CUDA event 区间包含 CPU 侧的启动和
分配空档：本卡三个启动加两次分配实测 29 us，而 GEMV 内核本身只有 42 us。按单次数字
反推带宽，w4a8 GEMV 是 72 GB/s，看着有 5-20x 的空间；深队列下它是 198，而同形状的
int8 GEMV 是 400，真实差距 2x。「字节减半而时间不变」这条判据两种计时下都成立，
方向没错，但相对排序被压平了（对照表见附录）。

**多一次启动就是多 20-30 us。** 提升如果不在「少一个启动」这一侧，就很难赢。把
GEMV 循环里的激活量化加打包提到独立内核，m=1 内核只快 8%，加上启动税净亏 53%；
同一改动在 m=8 才有 1.87x。所以这类提升要在同一次启动里拿到（HIP 内核顺手做完，
或落在 graph 捕获里），不要作为附加的 Triton 启动落地。

**组比一条 lane 宽时，每条 lane 都要自己落账。** 一行拆给 KP 条 lane 之后，一条
lane 的 k 区间（BK/KP）常常小于一个量化组（`s_rel` 的粒度）。这时不能只在「组边界
落在本 lane」时乘 scale 落账：同一个组里其余 lane 的部分和会被整段丢掉，而结果只是
「偏了一点」，不像错值。让每条 lane 都用自己所属组的 scale 结算自己的部分，组求和
交给最后的 shuffle。

**等价性要按消费者验收。** 旋转的输出会被量化成 int8：只比旋转值的 relL2
不够，要比量化后的 int8 是否逐元素一致。fp16 输入上 HIP 用 fp32 累加会更准，
但与 Triton 不等价 —— 要么刻意对齐精度，要么明确接受并重验生文 md5。

**量化档的误差要用真实 q/k/v 量，合成数据不保守也不通用。** 同一形状
[2,4096,10,64] 上：randn 给 int8 QK 报 1.42e-2、int8 QK+PV 报 1.79e-2，而 SDXL 实抓的是
6.11e-3 / 7.73e-3 —— 合成数据把这个档**高估了 2.3x**（fp16 档则两边一致，2.8e-4 对
2.1e-4，所以 fp16 的结论可以拿合成数据先看）。方向搞反就会把一个可用的档判死。
另外 P 量化的相对误差随 softmax 变平而**变大**：把 K 平移让参与率从 1066/4096 升到
3744/4096，int8 QK 从 6.11e-3 降到 3.71e-3，但 int8 PV 从 7.73e-3 升到 1.43e-2 ——
扁平平均把输出幅值压小，绝对量化误差的占比就升高。两个量的变化方向相反，所以「越平越差」
只对 PV 成立，对 QK 恰好相反，不能一句话概括成「softmax 平就完蛋」。

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

**LDS bank conflict 往往是第一瓶颈。** 行主序 padding 不一定够：列消费的operand 用 `+1 vec4` padding，八行八列同时别名时只能 XOR swizzle。先用
微基准把 LDS 指令比隔离出来，再动主内核 —— 我们第一个 int8 GEMM 版本
就是被这个从 3.4 拖到 10.0 TMAC/s 的。流式操作数也要小心：`w4a16.cu` 的激活按
行主序落进 LDS 后，一条 lane 的窗口宽度（ROWK=32 float）正好是 bank 数的约数，
同一个位置在 32 条 lane 上只落在 2 个 bank，8-way 冲突；把存储改成 lane-major
（word = lane + KP*位置）就是 **1.37-1.55x**，不用 padding 也不用 swizzle。

**别急着追占用率。** 试过把 16 个 program 拆成 128 个，反而更慢；真正的
瓶颈是 8 次串行的全局矩阵读。先量，再改。同理，寄存器预取在带宽受限的
load 阶段只会增加 VGPR 和私有拷贝，实测是负结果。

**注册与测量分开。** 实现列表是代码（`BackendRegistry` /
`arch/<gcn>/kernels/`），测量结果是数据（`manifest.toml`）。新内核加进去后
用 `bench.py --all --write` 决定它在这张卡上该不该开，别写死「某架构一律
用某内核」。

**留一个开关。** 任何手写内核都要能被单独关掉，用来 A/B、用来在本机 HIP
不兼容时回退。开关名跟现有的 `COMFY_GFX906_*` 保持一致的风格。

**本卡没有可用的 eager int8 矩阵乘。** `torch._int_mm` 在本机的 hipblasLt 上（缺
`TensileLibrary_lazy_gfx906.dat`）m=1/16/64/1024 全部抛 `HIPBLAS_STATUS_INVALID_VALUE`，
所以 comfy-kitchen 的 triton 后端在这里不是「可选加速」。`COMFY_GFX906_TRITON_BACKEND=off`
只该用于 A/B 本包没覆盖的操作（4096x4096：`quantize_int8_rowwise` 113.7→381.3 us、
`dequantize_per_tensor_fp8` 666.9→863.4 us）。本包在 triton 和 eager 两个后端的
`int8_linear` 上都装了同一层路由（GEMV → HIP int8 GEMM → triton 内核兜底），所以这个
开关不改变 int8 线性的落点，也不会让 2..127 行的批掉进 eager 报错 —— 那一档既不在
GEMV（只接 m=1）也不在 HIP GEMM（`MIN_ROWS=128`）里。

## 参考资料

- **本仓库的实测记录**：`gfx906fix/hip_attn_v2/NOTES.md`（融合注意力全过程、
  微基准、8 波 split 的负面结果）、`gfx906fix/hip_rotate/NOTES.md`
  （butterfly 推导、graph 内计时方法）。
- **真实 q/k/v 的抓法**：`gfx906fix/capture_sdxl_qkv.py` 在 `optimized_attention`
  上挂 hook、跑两步 SDXL 采样，把 `[B,S,H,D]` fp16 的 self-attention 张量存成 .pt。
  注意力量化档的 relL2 和 softmax 参与率都是从这儿来的。
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
- **看内核真实资源**：`--save-temps` 留不下 GCN 汇编，得从 `.so` 的
  `.hip_fatbin` 里挖设备 ELF（找 `\x7fELF`，按 `e_shoff/e_shnum/e_shentsize`
  截尾），再 `llvm-readelf --notes` 读 `vgpr_count` / `vgpr_spill_count`。
  spill 非零就是累加器和临时值不够用，比猜占用率快得多。
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
| 融合注意力 `hip_attn.cu` | 5.0-6.0 TMAC/s，比 int8 快 1.3-1.7x。**「4 波 dot2 天花板的 75-78%」这句已作废**：相位实测显示两次矩阵乘只占 21% 周期，那个比值是巧合不是机制（见下面「注意力的相位分解」） | 128 Q 行 x 64 keys、256 线程、1 block/CU、32+64 fp32 累加器、XOR swizzled LDS 恰好 64 KB、**K 不做跨 tile 预取**（预取反而慢 6.7%，逐位相同） |
| 大 m int8 GEMM `int8_gemm.cu` | 10.04 TMAC/s，节点级 1.10-1.16x | 128x128 tile、TM8/TN8 = 64 int32 累加器、BK=32、chunk-major LDS、211 VGPR 无 spill |
| 激活旋转 `rotate.cu` | fp32 档约 12x（graph 内 23.6 → 2.0 us/call），端到端 38.4 → 51.6 tok/s | radix-4 butterfly，D=1/4/16 用 shfl_xor、D=64 走 LDS，无全局 Hadamard 流量 |
| W4A16 解码 GEMV `w4a16.cu` | 单行解码 3.6-4.4x（48-62 → 185-274 GB/s）；多行 `w4a16_gemv_m`（m=2..8）1.6-5.3x（11-45 → 18-238 GB/s） | 一行拆 32 lane（一个 wavefront 读 8 行 x 16 B）、grid 封顶 240 块 = 960 波、每 lane 一个 uint4、16 级 fp32 codebook 放 LDS、每个 dword 一条独立累加链、**激活按 quad 行落 LDS**（先 lane-major 修 bank 冲突 1.4x，再 b128 一条指令供 4 个权重又 1.15x）、**循环倒置**（每个 chunk 只 pack 一次激活，再遍历本块的 32 个 row-group，部分和走 LDS、每块一次 shuffle，寄存器不涨）；多行版沿用这套 lane 划分，但一个 dword 解出的 8 个格点跨 M 行复用，激活按 M 行整块落 LDS。同文件第三个入口 `w4a16_gemv_h` 把激活换成 fp16、乘加换成 `v_dot2_f32_f16`：m=1..8 再快 1.05-4.89x，寄存器从 94-240 掉到 29-59。**无 codebook 时它是默认**（`code-8` 在 fp16 里精确，数值与无损版同为 2.4e-7）；带 codebook 要把格点退到 fp16，故 `COMFY_GFX906_W4A16_F16ACT=1` 才开 |

int8 GEMM 的波形 sweep（4032x12288x4096）：2 波 5.75、4 波 9.64、
8 波 9.14 TMAC/s（8 波只能用 16 累加器，与 4 波不是同一个 tile）。
配置 sweep：128 累加器 spill 364 B、BK=64 spill 376 B，只有 TM8/TN8
不 spill。bank conflict 是首版 3.4 TMAC/s 的全部原因。

## 深队列下的 decode GEMV 余量

内核本身的耗时：300 次连续提交、只在两端同步，convrot 关，`w8a16` / `w4a16` 的激活
按 fp32 进内核（`linear_dequant` 先 `x.float()`），`w4a16` 走 codebook。GB/s 按权重
流量折算（int8 一字节、int4 半字节）。
program 数够多时这张卡能吃满流式带宽（lm_head 的 int8 GEMV 连单次计时都有
726 GB/s，深队列只会更高），所以同形状的 int8 GEMV 才是这一档的实际参照。
均匀格点（无 codebook）一档另外算：Triton 在 m=1 是 144.5/348.1/4004 us（58-78 GB/s，
比 codebook 档快是因为它跳过 gather），m=8 是 691/1578/18062 us；默认已落到
`w4a16_gemv_h`，对应 39.2/103.5/1068.8 与 96.4/232.8/2663.6 us，3.4-7.2x。

| 内核 | 4096x4096 @m1 | 12288x4096 @m1 | 12288x4096 @m8 |
|---|---|---|---|
| `int8_gemv.asm_kernel`（w8a8，只接 m=1） | 41.9 us / 400 | 76.9 us / 654 | 不接此形状 |
| `w4a8_gemv.gemv_kernel`（w4a8 asm） | 42.4 us / 198 | 77.5 us / 325 | 320.5 us / 79 |
| `int8_gemv.dequant_gemv_kernel`（w8a16） | 46.9 us / 358 | 87.3 us / 576 | 337.1 us / 149 |
| `w4a8_gemv.dequant_w4a16_kernel`（w4a16 Triton） | 176.8 us / 47 | 455.3 us / 55 | 1870.7 us / 13 |
| `w4a16.cu`（w4a16 HIP，同一 `linear_dequant` 入口；m=8 走多行内核） | 45.2 us / 185 | 119.0 us / 211 | 862.0 us / 29 |
| `w4a16.cu` `w4a16_gemv_h`（fp16 激活 + dot2；均匀格点默认，codebook 档要 `F16ACT=1`） | 37.9 us / 221 | 105.2 us / 239 | 234.0 us / 108 |

读数：

- **w8a16 在 m=1 已经和同形状的 int8 GEMV 同级**（358-576 对 400-654 GB/s），没有
  内核余量可拿；m≥4 掉到 149 GB/s，成本在 `tl.static_range(M)` 加 `tl.where` 把
  M 行做成 M 次整 tile 重算。
- **w4a16 是这张卡上余量最大的内核**：读一半字节却比 w8a16 慢 3.8x（按字节算是
  7.5x）。每个 nibble 展成一个元素、每个 nibble 一次全局 gather
  （`tl.load(lut_ptr + c)`），加上 fp8 格点逐元素软解码，以及 BK 被
  `min(4 * group_size, 256)` 锁在 256。m=8 是 13 GB/s。
  已按这一格落地 `w4a16.cu`（单行 185-274 GB/s，3.6-4.4x；m=2..8 的多行内核 18-238
  GB/s，1.6-5.3x），离读取地板 631-776 GB/s
  还差 2.3-4.3x。实测的几段账：消掉 barrier（每块两次 `__syncthreads`）**更慢**，
  管线不是瓶颈；把激活从行主序改成 lane-major 落 LDS **1.37-1.55x**（行主序时 32 条 lane 的
  同一位置只落在 2 个 bank 上，8-way 冲突）；去掉 FMA 和一半 LDS 读（NOFMA）值
  1.5-1.7x；激活改 quad 行 + `ds_read_b128`（一条指令供 4 个权重）再 1.15x，已落地；
  循环倒置把 pack 从「每 8 行一次」摊到「每 chunk 一次、供 32 个 row-group 复用」，
  pack 相位从 44-46% 掉到 7-25%，同一进程内配对 A/B 实测 **1.19-1.45x**（原因见下面
  相位一节，比按份额折算的 1.3-1.8x 少）。
  剩下的 2.3-4.3x 里，最后一次每权重的 LDS 读是 codebook gather：换到 VALU（`v_perm_b32`
  解析 int8 格点 + `v_cvt_f32_i32_sdwa BYTE_n`，ISA 笔记确认这两条在 gfx906 都有）
  能把 LDS:global 从 5:1 压到 1:1，代价是 level 回到 int8 格点（~4e-3），要么刻意
  对齐精度要么按坑点重验 md5。这一条最终是用 fp16 格点表 + `v_dot2_f32_f16` 落地的
  （见下一节，`F16ACT` 开关），VALU 解析那条仍未试。**每 lane 在飞 load 数已试并被否**：预取下一块的
  uint4（+4 VGPR）没有收益，k=12288（12 个块）反而慢 17%。
- Triton 侧的替代方案已经量过并否掉（fp16 大 dot 慢 1.6-7 倍，PORTING.md 附录 E），
  而「nibble 留在字里 + packed fp16 成对点积」（`v_dot2_f32_f16`，两端本来就是
  fp16，没有转换成本）是 Triton 表达不了的形状，所以 w4a16 这一格是手写 HIP 的活。
- **w4a8 asm 那 2x 不在指令形态上，这一条已被否**：它读一半字节却和 int8 GEMV 同速
  （42.4 对 41.9 us），看着像指令受限；但按这一格的结构手写了一个只用链式
  `v_dot4_i32_i8`（累加操作数免费，正是 Triton 给不了的那条）+ block 内量化打包 +
  `v_perm_b32` LUT 的 HIP 内核，结果 190-224 GB/s，**只在小 n 赢 44%**（190 对 132），
  n≥12288 全部落后 Triton 的 282-373 GB/s。所以那 2x 不在解码/求和的指令流里，
  也不在 dot4 累加链上；剩下的差别在我这套结构与 Triton 的流式形状之间，没量出来
  之前不要再按「换指令形态」下注。

## 注意力的相位分解：int8 重写不是为了 dot4

`s_memtime` 打在每个 tile 的四个边界 + 循环末尾，只有 block (0,0)，每 wave 一套槽
（QK 段四个 wave 吻合到 0.0%，softmax 4%，PV 20%）。每格是 cycles/wave，72 个 tile 累计：

| 段 | 4608²@d128 | 4096²@d64 | 4608x256@d128 | cycles/tile |
|---|---|---|---|---|
| K store + V pack | 0.9% | 1.2% | 0.7% | 328 |
| QK（dot2） | 10.2% | 9.8% | 21.3% | 3.5-4.6K |
| softmax + P store + 两道 barrier | 39.7% | 36.5% | 34.7% | 13.7K |
| PV（dot2） | 8.6% | 14.9% | 7.4% | 3.0K |
| tile 顶 barrier 之前那段 | 40.6% | 37.6% | 35.9% | 14.1K |

- **两次矩阵乘加起来只有 19-21%**。dot4 对 dot2 是 1.77x 的 MAC 率，把 QK 和 PV 全换成
  dot4 的上限也就是省这 21% 的一半 ≈ **总时间的 10%**。所以「int8 注意力 = 1.7x」这个算式
  当场作废，聚合 TMAC/s 除掉点上限是巧合。
- 那 40% 的尾段**不是 barrier 收敛**：把顶 barrier 整个删掉（诊断构建，结果不采信），
  总时间 23.00 ms、尾段仍是 40.9% 且四个 wave 均匀（2.3%）。wave 均匀也排除了 wave 偏斜。
- 也**不是缺预取深度**：给 V 加一套双缓冲寄存器、让它和 K 一样提前一个 tile，实测
  **慢 5%**（24.18 对 23.05 ms），而且 QK 段从 254K 涨到 442K —— 载入挤进了 QK 的发射窗口。
  反过来，**删掉 K 的跨 tile 预取快了 6.7%**（4608²@d128 23.35 → 21.87 ms、
  4608x256 1.40 → 1.31 ms，d=64 中性，输出逐位相同），这条已落地。
- 剩下的 37-40% 我**没能归因**，而且要说清为什么不敢归因：戳的 `asm` 没有 `memory`
  clobber，调度器可以把载入/存储挪过戳，所以段边界是近似的；能确证的只有「不是那个
  barrier、也不是矩阵乘数学」。带戳构建的墙钟与不带戳的一致（23.05 对 23.27 ms），
  所以打戳本身不花钱。
- fp16 侧确实拿不到 2 block/CU：`Qs` 单项就是 32 KB，加上 Ks 16 + Vt 16 正好占满
  64 KB。只有操作数字节减半才可能降到 32 KB，这是 int8 路线唯一站得住的结构理由。
- 真实 q/k/v 的精度代价（SDXL 两步采样、hook `optimized_attention` 抓到的两层，
  relL2 对 fp32 参考）。这几档 int8 后端后来都删了，表里的数字就是删除依据：

  | 档 | [2,4096,10,64] | [2,1024,20,64] |
  |---|---|---|
  | fp16（hip 与 triton 同级） | 2.13e-04 | 2.14e-04 |
  | int8 QK + fp16 PV（smooth） | 6.11e-03 | 6.96e-03 |
  | int8 QK + fp16 PV（raw） | 6.46e-03 | 7.24e-03 |
  | int8 QK + int8 PV | 7.73e-03 | 1.10e-02 |

  **这一档的误差由 QK 量化主导，不是 P 量化**：QK 从 fp16 换成 int8 就是 29-33x
  （2.1e-4 → 6-7e-3），在此之上把 PV 也量化只再加 1.3-1.6x。`attention_kernel.py`
  里「INT8_PV=False 更准 2-30x」那句比的是 int8 档内部的两个 PV 形态，不是 int8 对
  fp16 的总代价，读的时候别混。顺带量到 softmax 的平坦度：参与率 1066/4096（平均最大
  概率 0.02）与 401/1024（0.09）——确实是扁平分布，所以 int8 P 的损失反而小，主导项
  是分数本身少掉的 7 位精度。
- **但字节减半这一条已经有反例**：现在的 Triton int8 档就是半字节操作数，实测
  3.29-3.56 TMAC/s，比 fp16 HIP 的 5.66 **更慢**——它并没有因此拿到 2 block/CU，
  字节优势也就没兑现。所以这件事值不值得建，取决于能否明确做出 2 block/CU 的布局，
  而不取决于 int8 本身；在尾段那 37-41% 归因清楚之前，这是一个大工程 + 每层每次注意力
  0.6-1.1% 的 relL2，换一个未证实的收益。

## 注意力后端的适用域（时间 + 峰值额外显存）

`peak extra` 是一次调用在 q/k/v 之外多占的字节。sdpa 的 flash 与 mem-efficient
在这张卡上都是 `No available kernel`，只剩 math 回退，所以下面 sdpa 一列就是 math。

（`int8` / `int8qk` 两档已按这一表删除，数字保留作证据。默认列是本机不传启动标志时 ComfyUI 实际走的 `attention_sub_quad`。）

| 形状 | hip | gemm | hand | int8 / int8qk | comfy basic / sub_quad | sdpa math |
|---|---|---|---|---|---|---|
| [2,4096,10,64] 实抓 | **8.51 ms / 10 MiB** | 14.37 / 584 | 14.58 / 10 | 13.36·15.34 / 65·90 | 19.93 / 14.17，1310 | 31.24 / **2960** |
| [1,4096,10,64] | **4.63 / 5** | 7.06 / 372 | 7.28 / 5 | 6.98·7.94 / 33·45 | 10.13 / 9.49，655 | 16.78 / 1480 |
| [1,1024,20,64] | **0.56 / 2** | 0.87 / 77 | 1.86 / 5 | — | 1.02 / 0.91，88 | 3.75 / 400 |
| cross 1024x77 | **0.05 / 1** | 0.10 / 36 | 0.07 / 1 | 0.15·0.14 / 1·41 | 0.21 | — |
| CLIP 77x77 h20 | **0.03 / 0** | 0.09 / 33 | 0.04 / 0 | 0.15·0.13 | 0.18 / 0.22，33 | — |
| [1,4608,24,128] | **21.85 / 28** | 31.29 / 630 | 58.69 / 28 | 36.68·46.10 / 166·220 | 41.21 / **121.04**，1756 | 117.97 / **4590** |

- 在这个域里 hip 是严格 Pareto 最优（TMAC/s 4.6-6.0，其余最高 4.2），而且它是唯一
  不物化 score 的路径：额外显存就是输出本身。int8 两档每一档都既慢又差 30x（见上节）。
- **hip 不合用的三种情况，都不是 hand 或 int8 补位的**：
  1. head dim 不在 {64,128}（SD1.5 的 40、SD2.1 的 80）—— hand 的
     `SUPPORTED_HEAD_DIMS` 是同一个限制，只有 gemm（`_gemm_ok` 不看 head dim）：
     d40 0.74 ms 对 comfy 的 0.76/0.88，d80@4096 17.29 对 22.61/26.11。
  2. decode 形状 sq=1..16 —— 见常见坑点那条，hip 比 gemm 慢 3.2-5.1x，hand 也一样慢
     （0.74 对 0.23 ms），因为它自己的 BLOCK_M=64。
  3. fp32 输入或 amax 保护触发 —— hand/int8 同样要求 fp16，只有 gemm 收。

## 试过但没采用的路径

| 尝试 | 结果 |
|---|---|
| 8 波双 kernel 拆分（QK+stats / PV） | 4608x4608 d128 从 23.3 → 30.1 ms；Q 重读 2-4 GB、PV spill 84 B |
| 寄存器预取 V 一个 tile | 219 VGPR + 80 B 私有拷贝，23.6 ms（更慢）；load 是带宽不是延迟 |
| 去掉整个 softmax | 总时间只变 7%（成本在 P store + barrier，不在数学） |
| 旋转按列分块撑到 128 个 program | 比 16 个 program 更慢（7.8 vs 13.3 us in graph） |
| 用 per-group `tl.dot` 替代 `v_dot4` | 慢 1.6-7 倍（M 补齐到 16、256 次小 dot），见 PORTING.md 附录 E |
| fp16 模拟 int8 linear | 慢 1.16-1.85 倍 |
| 激活量化+打包提到独立 Triton 内核 | m=1 净亏 53%（64.5 → 98.6 us：29 us 的启动税 > 内核省的 3.5 us）；m=8 才 1.87x |
| 预取下一块的 uint4（每 lane 两个在飞 load） | 中性偏负：175/217/223/236 GB/s 不变，k=12288（12 块）慢 17%。和消 barrier 的负结果同向 —— 这个内核不缺 MLP |
| 手写 HIP W4A8 GEMV（链式 dot4 + LUT + block 内打包激活） | 4096²@m1 +44%，但 12288²/4096×12288/152064² 落后 25-40%（212-224 对 282-373 GB/s）→ 不落地，`w4a8.m1` 的 `hip` 槽位继续空着 |
| 把 `KP` 从 32 降到 16/8（每 lane 2-4 个 uint4） | 更慢（142/75 GB/s）且数值坏掉（`red[]` 按 32-lane 分组，换 KP 就没写满）；`BK` 提到 2048/4096 同样更慢（168/113）|
| `__launch_bounds__(256, 4)` 逼出 4 块/CU | 2.4x **更慢**（161-193 → 66-81 GB/s）：64 VGPR 上限把 4 条独立累加链和 gather 的中间值全挤掉，占用率不是这个内核的瓶颈，`(..., 1)` 才对 |
| W4A16 多行：给单行内核直接加 M 组累加器（每行各走一遍 dword） | m=8 只有 0.9-1.1x（13 GB/s），因为 8 个格点每行都重新 gather 一次、激活 LDS 读按 M 线性涨。落地的 `w4a16_gemv_m` 改成「解码一次、跨 M 行复用」才有 1.6-5.3x，见上面两张表 |
| 注意力里给 V 加一套双缓冲寄存器、预取一个 tile（和 K 同等待遇） | 慢 5%（24.18 对 23.05 ms），QK 段反而从 254K 涨到 442K cycles：载入挤进发射窗口。顺带否掉了「V 缺预取」这个假设 |
| 删掉注意力 tile 顶的 barrier（诊断用） | 总时间与尾段都不变（23.00 ms，尾段 40.9%，四 wave 均匀）→ 那 40% 不是 barrier 收敛，也不是 memory drain |
| 为 dot4 的 1.77x 用 HIP 重写 int8 注意力 | 被相位实测否掉：两次矩阵乘只占 19-21% 周期，dot4 全换成不上也就 ~10%。int8 剩下的理由只有「K/V 字节减半 + LDS 降到 32 KB 换 2 block/CU」，见「注意力的相位分解」 |
| 注意力 PV 换成 fp16 累加（`v_pk_fma_f16`，64 个累加器减到 32） | 没落地，动手前先被微基准否掉：4 波 6.28 对 dot2 7.11 TMAC/s（指令数同为 2 MAC/条，发射率更低），还要多一次 P 广播、丢掉键内两两归约；且 LDS 65536 B 已把占用率钉在 1 block/CU，省下的寄存器换不来波数 |
| 多行内核把激活块提到寄存器里跨列复用（省掉每列一次 b128 读） | 中性偏负：等 chunk 宽 0.85-1.08x，M=8 还得把 chunk 砍到 512 才装得下，0.85-0.89x。份额模型预测 2.3 TMAC/s，实测更慢 —— 那些 LDS 读被 dot2 发射掩盖了，不在关键路径上 |
| `tl.reduce(combine_fn=v_dot4)` 用满累加操作数 | 表达不了：reduce 的叶子值必须是待归约元素本身，`(w, a)` 双流喂进 c 槽会把权重原值当部分和加上。只剩先 `dot4(w,a,0)` 再求和，即现状。累加链在 HIP 里才免费（`int8_gemm.cu` 的 `__builtin_amdgcn_sdot4`） |

这些「不要做」的结论和 gfx906 的 LDS/VGPR 预算绑得很紧；换一张卡，
结论可能完全反过来。

### 资源与相位实测（`w4a16.cu`，m=1）

- `vgpr_count 94 / vgpr_spill_count 0 / sgpr_count 46-48`（16 个实例化一致，倒置前是
  88）。256 线程 × 94 = 24064 → 驻留 2 block/CU（8 波），而纯读探针的最优点是
  4 block/CU（16 波）。但 `-DMINB=4` 压到 64 VGPR 实测**慢 2.4x**：占用率不是
  这条路的瓶颈，别再拿它试。
- 每 chunk 相位（`s_memtime`，每 wave 独立槽，四个形状一致）：**倒置前**激活 pack
  ≈ 44-46%（1702-2100 cyc）、load 等待+解码+LDS ≈ 31-41%（1272-2176）、group scale
  累账 ≈ 10-13%（464-546）、跨块固定开销 ≈ 8-11%（421-429）、**发出 load 只占
  1.5-2.1%**（78-80 cyc，等待全在下一段里）。最大的单项是 pack，而它当时只摊在
  BN=8 行上；m=1 的激活整块共享，所以把它提到 row-group 循环之外（循环倒置：每个
  chunk pack 一次，再遍历本块的所有 row-group，各组部分和走 LDS + 每块一次 shuffle
  归约，寄存器不涨）。按 44% 折算本该值 1.3-1.8x。
- **倒置后 pack 相位确实掉到 7.3-25.1%，但只换来 1.19-1.45x**（同一进程内配对 A/B：
  140-191 → 179-276 GB/s），load 发出仍是 80 cyc，各 wave 的总周期数彼此仍差 1% 以内。
  原因是 pack 那一段量到的 1700-2100 cyc 里混着**等最慢的 wave 到齐**（段首的
  `__syncthreads`），它是 barrier 收敛不是纯工作量；把它搬出循环，等待不会消失，
  只会挪到下一个 barrier 所在的那一段里重新分配。**一条相位的耗时只有在它做的是
  纯工作时才能按份额折算收益**；含 barrier 的段要先看 wave 间的离散度，离散度大
  就说明这段里有等待，收益要打折。
- 顺带纠正一次错误读数：先前「整行一次 pack 反而更慢」的 NOSYNC 实验把 `xa`
  撑到 32 KB，驻留从 2 block/CU 掉到 1 —— 那是 LDS 尺寸的失败，不是「提出来」
  的失败。一个结论如果被一个没控制的变量陪着，就等于没测。

### fp16 激活变体（`w4a16_gemv_h`）与多行的量到哪里为止

- fp32 激活版资源（实测）：`vgpr 122 / 186 / 240`（M=2/4/8），`spill 0`，LDS
  9.3 / 18.5 / 41 KB → M=4 起就是 1 block/CU（4 波），只有 M=2 还留得住 2 block。
  MAC 率 0.28-0.95 TMAC/s，离 4 波 fp32 FMA 地板 3.64 还差 4-13x，也就是说它不是
  FMA 受限。按每条指令占用的单元折算（b128 读 1 KB 波次、b32 gather 256 B、FMA
  4 cyc/波次）：M=8 时一个 (lane, chunk, 列) 做 256 FMA + 32 gather + 64 条 b128
  激活读 + 40 shuffle，份额约 FMA 36%、激活读 47%、gather 17%，上限约 1.3 TMAC/s。
- **fp16 激活 + `v_dot2_f32_f16` 版（`w4a16_gemv_h`）**：比 fp32 激活版快 1.05-4.89x，
  带 codebook 时 4.1-8.7x over Triton（87-260 GB/s）。**无 codebook 就默认走这一条**
  （均匀格点 3.4-7.2x over Triton，204-291 GB/s）；带 codebook 要 `F16ACT=1`。但赢的原因不是
  指令数减半，而是**寄存器**：`vgpr 29-59`、LDS 4-25.6 KB → M=4 有
  4 block/CU（16 波，正好是纯读探针的最优波数），M=8 有 2 block。fp32 版之所以吃到
  240 个寄存器，是因为它要把一个 dword 的 8 个格点和 8 个激活值都摊成 float 数组再
  乘；dot2 直接拿 packed dword 当操作数，这些中间值一个都不存在。MAC 率 1.49-1.65。
- 单行也走这一套（`M = 1` 实例化，不是拿 M=2 凑一个空行）：45.3 → 37.9 us
  （4096²，1.20x / 221 GB/s）、125.7 → 105.2（12288²，1.19x / 239）、1129.6 → 1075.3
  （152064²，1.05x / 290）。`vgpr 29-35`、LDS 4096 B → 4 block/CU（16 波，正是纯读
  探针的最优波数）。只有 1.05-1.20x 而不是 4-8x，因为单行档卡在读通路上，把指令
  减半解决不了读。
- 精度（实测，540 组路由+数值复验）：**均匀格点与无损版同级**（fp32 输出归一化最大
  2.4e-7，fp16 输出 5.0e-4 = 1 ulp，与无损版一致），因为 0..15 减 8 在 fp16 里精确可
  表示 —— 所以这一档默认走 fp16，没有质量问题。带 codebook 时格点退到 fp16，fp32 输出
  归一化 1.6-2.6e-4、rms 2.1-2.8e-4，fp16 输出最大 7.5e-4（约 1.5 ulp，无损版是 1 ulp），
  因此那一档要 `F16ACT=1` 才开，开启前按坑点那条用 `--sampling off` 重验生文 md5。
  convrot 把激活转成 fp32，只有无损 feed 收，所以它不跟这个开关走。
- **寄存器驻留（把 lane 的激活块提到寄存器里跨列复用）已被实测否掉**：省下的 b128
  根本不在关键路径上。等 chunk 宽（1024）下 M=2/4 是 0.85-1.08x，即中性；M=8 装不下
  （要 128 个寄存器）只能把 chunk 砍到 512，结果 0.85-0.89x，比原版更慢。份额模型
  当时预测它能到 2.3 TMAC/s —— 这是同一个模型第二次在同一个内核上判错方向（第一次
  是 pack 相位按份额折算）。**减少指令数的提案，要先证明那些指令在关键路径上**；
  这里 LDS 读被 dot2 的发射掩盖了，省下它们换来的是更多 chunk 轮次和更高寄存器压力。
- `GB` 从 8 提到 32（把 M 行的 pack 摊给更多列组）实测只在大 n 有效：152064@m8
  2.18 → 2.70x、@m4 4.13 → 4.49x，而 4096² 一档不变 —— 那里每块只摊到 2 个 row-group，
  摊不摊无所谓。又一个「按份额折算」要打折的例子。
