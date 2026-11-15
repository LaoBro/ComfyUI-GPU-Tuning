# 给其它架构写 HIP 内核

这篇讲怎么给这个节点加一个手写 HIP 内核。功能分级、测量数据和换环境重测请看
[PORTING.md](PORTING.md)；后端链与注册表见 `kernels/registry.py`；节点开关见 [CONFIG.md](CONFIG.md)。

**本文只讲方法和判断，不放实测数字。** gfx906 的实测边界与「不要再做」的结论见
[GFX906.md](GFX906.md)，实测数字、结论摘要和 gfx906 专属硬件限制见
[GFX906_EXPERIMENTS.md](GFX906_EXPERIMENTS.md)（归档，单次实验过程已精简，不参与判断）。

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
3. **构建**。把源文件加进 `tools/build_hip_attn.py` 的 `TARGETS`（逻辑名，不带
   平台前缀），用 `--arch <目标>` 指定 `gcnArchName`。产物落在
   `arch/<目标>/kernels/`，文件名由 `kernels/runtime.py` 的 `library_name()` 决定：
   Linux 是 `lib<目标>_<名字>.so`，Windows 是 `<目标>_<名字>.dll`。一份 checkout
   因此可以同时放几个平台/架构的产物。代码对象按目标绑定，别的架构要各自编译。
   Windows 由构建脚本生成 `.def` 导出 `extern "C"` 入口并链接 `amdhip64.lib`。
4. **写 loader**。照 `kernels/hip/attention.py`：调 `kernels/runtime.py` 的 `load(名字)`，
   它按当前平台/架构定位产物、预载运行时、加载失败打 WARNING 并返回 `None` 让调用方
   回落；loader 只设 `argtypes` 和形状校验。不要把架构名或库文件名写进 loader。
5. **接入站点**。在 `BackendRegistry` 加一个 Candidate，或在调用点分派。
   一定要留一条自动回落路径（Triton / core），再给一个能关掉的开关。
6. **验收**：数值等价（见坑点）、A/B 计时、开关 `off` 能复现旧行为、
   产物缺失 / 加载失败 / 形状不符时节点照常启动。

## 常见坑点

**stream 必须在调用时读。** `torch.cuda.graph()` 会把 `current_stream` 切换到
捕获流；用捕获前缓存的 handle，launch 不会进图，测出来是「空图 + 0.04 us」
这种假数字。最容易踩、也最隐蔽。

**单次计时会把内核的余量算大 3 倍。** 单次调用的 CUDA event 区间包含 CPU 侧的启动和
分配空档：本卡三个启动加两次分配实测 29 us，而 GEMV 内核本身只有 42 us。按单次数字
反推带宽，w4a8 GEMV 是 72 GB/s，看着有 5-20x 的空间；深队列下它是 198，而同形状的
int8 GEMV 是 400，真实差距 2x。「字节减半而时间不变」这条判据两种计时下都成立，
方向没错，但相对排序被压平了（对照表见 [GFX906_EXPERIMENTS.md](GFX906_EXPERIMENTS.md)）。

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

**别急着追占用率——但先分清内核类型。** 带宽受限的算子（GEMV、预取阶段）
里占用率确实不是杠杆：把 16 个 program 拆成 128 个反而更慢，真正瓶颈是 8 次
串行的全局矩阵读；寄存器预取在 load 阶段只增加 VGPR，实测为负。**计算受限的
dot 内核则相反**：占用率直接定发射天花板（4 波 dot4 12.9 vs 8 波 16.4 TMAC/s），
int8 GEMM 就是被 211 VGPR 钉在 4 波、后来靠 124 VGPR 上到 8 波快 1.6x。所以这条
经验只对内存受限场景成立，别套到 dot 密集的内核上。

**注册与测量分开。** 实现列表是代码（`BackendRegistry` /
`arch/<gcn>/kernels/`），测量结果是数据（`manifest.toml`）。新内核加进去后
用 `bench.py --all --write` 决定它在这张卡上该不该开，别写死「某架构一律
用某内核」。

**留一个开关。** 任何手写内核都要能被单独关掉，用来 A/B、用来在本机 HIP
不兼容时回退。开关名跟现有的 `COMFY_GPU_TUNE_*` 保持一致的风格。

## 参考资料

- **可照抄的模板**：`kernels/hip/attention.py` / `kernels/hip/int8_gemm.py` / `kernels/hip/rotate.py`
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
- **无 GPU 自检**：`tools/registry_selftest.py`（后端链决策）、
  `tools/manifest_selftest.py`（manifest 解析与信任）、
  `tools/settings_selftest.py`（节点三态开关，需要 venv 里的 torch）。

## 其它值得记住的

- **算法替换 > 指令替换**：这个节点里最大的几次收益都来自换算法（融合、
  butterfly、去掉重复全局读），不是把同一段循环写成汇编。
- **负面结果也要记**：试过不工作的路径（8 波 split、寄存器预取、
  列分块、两 kernel 拆分）比成功路径更能省下后来者的时间。
- **改动要可回退**：包装原函数而不是复制函数体，ComfyUI 升级时最坏是
  降级到 eager，不会静默跑错。
- **先数值后计时**：`bench.py` 已经把「超 rel 门槛不参与计时」写进流程，
  自定义微基准也要照做，否则「更快但算错了」会赢。

## 数字与结论在哪

| 要找什么 | 去哪 |
|---|---|
| 写内核的方法与跨架构坑点 | 本文 |
| 移植流程、测量门、进一步调优 | [PORTING.md](PORTING.md) |
| gfx906 的实测边界与已否路径 | [GFX906.md](GFX906.md) |
| gfx906 实测数字、结论摘要、硬件限制 | [GFX906_EXPERIMENTS.md](GFX906_EXPERIMENTS.md) |
| 补丁与文本生成实现 | [PATCHES.md](PATCHES.md) |
