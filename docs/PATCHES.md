# 补丁与文本生成实现

[README](../README.md) · [CONFIG](CONFIG.md) · [GFX906](GFX906.md) · [PORTING](PORTING.md) · [HIP_KERNELS](HIP_KERNELS.md)

这一层回答“它为什么快”：八个补丁各改了什么，以及单 token 文本生成的完整推导。
安装与节点开关见 [README](../README.md) 与 [CONFIG.md](CONFIG.md)。

## 八个补丁

1. **int8 linear tile**：上游 autotune 列表是给 CDNA/CUDA 的（128×256×64、8 warps），
   在 gfx906 上换成 128×256×32、4 warps、2 stages。单 token 解码（m ≤ 8）另走一套窄 tile，
   见下面「文本生成」。
2. **convrot 融合**：蝶形 Hadamard 旋转与 int8 量化合成一个 kernel（原本两次全局读写）。
   fp16/bf16 现在也走融合路径（K 为 2 的幂且 ≤8192）；K=12288 的 MLP 下投影因寄存器上限
   仍走 butterfly+quantize。M=4032、fp16 实测 K=4096 0.196 → 0.149 ms、K=2048 0.108 → 0.084 ms（1.3x）。
3. **注意力入口**：把 `[B,S,H,D]` 统一转 fp16 + 连续布局，再分发给选中的后端。
   可选 `hip` / `hand` / `gemm`，以及 ComfyUI 自带的 `basic` /
   `pytorch` / `sub_quad` / `split` 等全部后端。
4. **文本编码器 int8**：权重保持量化（每次调用不再反量化）。
5. **QKV 三合一**：q/k/v 三个投影合成一次 int8 GEMM（逐行 scale 拼接，bit-exact）。
   只省 1.5%（5.24 s 步里 -81 ms），却常驻 1.6 GB 合并权重，所以**默认关**，
   Model Settings 节点的 `qkv` 开关打开。
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
- **qwen21 fp16 全前向**：`qwen21_fp16`（默认 on）在模型本身是 bf16 时把整个 DiT forward
  一次转 fp16，出口转回；64×64 latent 实测 7.92 → 3.83 s（2.07x），输出比 bf16 更接近 fp32。
- **int8 linear_act 保持量化**：`int8_linear_act`（默认 on）包住所有
  `comfy.ops.linear_input_act` 调用点，权重 orig_dtype 与激活 dtype 不一致（gfx906 上 bf16
  激活 + fp32 manual cast）时不再 requantize→dequantize，SwiGLU/GeGLU 下投影留在 int8
  内核上；Qwen-Image 2.1 实测 7.92 → 3.34 s（2.37x），dtype 一致的调用不受影响。
- **qwen21 residual 融合**：`qwen21_residual_fuse`（默认 on）把 `x += gate * layer_out` 融进
  `libgfx906_int8.so` 的 v2 residual epilogue（attention `to_out` + MLP down），省掉单独的
  `addcmul`；4032 token 实测两个 GEMM 各快约 0.6%，整步约 0.1%（低于测量噪声）。
- **int8 前缀缓存**：`qwen21_int8_cache` 让 Qwen-Image 2.1 的 int8 缓存 K/V 保持
  int8 直接进注意力内核（原路径每块反量化成 fp16）：4058+4032 token、32 head、
  d128 实测 51.2 → 34.2 ms/块（1.50x），输出 relL2 1.1e-2。默认关。
- **TE GEMV**：`text_gemv` 控制文本编码器的直调 int8 GEMV 与未量化小投影的 `torch.mv`。
- **4-bit w4a8**：`w4a8` 控制 4-bit 权重的 decode GEMV 与 prefill 现解 int8。
- **int8 注意力**：`int8_attn` 让 comfy-kitchen 的 int8 注意力契约（checkpoint 标记
  `comfy_kitchen_int8` 的模型、int8 minimax VAE）跑本包的 int8 输入内核（默认关：
  输出 relL2 1.1–1.3e-2，fp16 内核 2.5e-4）；`int8_attn_rotate` 是它的量化旋转
  精度开关（0/64/128）。
- **文本生成组**：`textgen` 默认 on，一次控制 `deltanet`、`kvdecode`、`llama`、`qwen35` 四项；
  不连该节点时才按各自环境变量独立生效。
- **fp16 输入策略**：`fp16_attn` 三选一（`guarded` / `force` / `off`），合并了旧版的 `fp16` 与 `amax_guard`。

## 文本生成（单 token 解码）

LLM 逐 token 解码时每次乘的是 **m=1** 的矩阵，和 DiT 的 M≈4096 是两回事，
DiT 的最优 tile 在这里反而更慢：128 行的 tile 对一行输入要算 64 倍的 MAC，
256 宽的 N tile 又让 4096 宽的权重只剩 16 个 program 铺在 60 个 CU 上。

所以 m ≤ 8 时改用 `16×32×64`（4 warps / 3 stages），大 batch 仍是 `128×256×32`，
按次切换、不触发 autotune 扫参。m=1 实测（4096×4096 / 4096×12288 / 152k lm_head）：
`0.30 / 0.48 / 2.66 ms`，比 256 宽的 tile 快 2.1x / 1.5x / 1.6x。

| 配置（Qwen3.5-9B int8，decode 稳态） | it/s | ms/token |
|---|---|---|
| 不装本包 | 1.27 | 791 |
| 装包，默认启动（动态显存分块） | 6.3 | 155 |
| 装包 + 权重常驻（`--disable-dynamic-vram` 或 `--highvram`），graph 关 | 17–18 | 55–58 |
| 装包 + 权重常驻 + 逐块 CUDA graph（默认开） | **35–37** | **~28** |
| 同上，`text_encoder=default`（fp16 路径） | 34.6 | 29 |

权重常驻是这里最大的一跳，但它不是补丁的功能：默认启动时 CLIP 被 `load_models_gpu` 判成
"dynamic VRAM loading"（日志 `9031MB Staged`），9 GB 权重按块换入，GPU 时间从 37.5 涨到
~135 ms/step。`--disable-dynamic-vram`（或 `--highvram`）让权重全量常驻（`current: cuda:0`），
GPU 回到 37.5 ms；随后本包默认开启的逐块 CUDA graph 吸收每步约 800 次 eager launch，
稳态到 35–37 it/s。512 token 交错复测（不同 seed、预热+计测）中位 34.6 it/s。

### m=1 走专用 GEMV

tile 只能把分块 GEMM 调到 55 GB/s；解码的权重是"读一遍、每个元素用一次"，正确形状是不分块。
`kernels/triton/int8_gemv.py` 里一程序负责 16 行输出，沿 K 以 1024 为步长累加 `int32`（整数求和可结合，
所以结果与 GEMM 路径同值）。旋转与量化也折进来：旋转内核顺手写出每个 256-group 的最大值，
GEMV 内核自己合成行 scale 并在线量化，一条 linear 从 3 次启动降到 2 次。

旋转本身也换成了 HIP radix-4 butterfly（`kernels/hip/rotate.py` + `arch/gfx906/kernels/rotate.cu`）：
Hadamard 符号在寄存器里生成，不再逐 tile 读 256x256 矩阵。fp32 档量化激活与 Triton 逐元素一致，
`COMFY_GPU_TUNE_ROTATE=off` 回退到 Triton 做 A/B；fp16 档按 fp32 累加，结果更准但不等价。

te=default 的 4-bit 解码（激活不量化、权重在寄存器内反量化）换成 HIP 内核
（`kernels/hip/w4a16.py` + `arch/gfx906/kernels/w4a16.cu`）：nibble 不再展成 8 个 fp32 元素，
16 级 codebook 走 LDS 索引读，一行拆给 32 条 lane、grid 封顶 240 块，每个 k 块的激活
只打包一次（一条 `ds_read_b128` 供 4 个权重）再摊给整块的 row-group 复用。单行与
多行（m=2..8，一个 dword 解出的 8 个格点跨行复用）两个入口，fp32 累加，数值与
Triton 一致。
bench 的 `w4a16.m1` 站点 266.6 → 93.3 us（4096x4096@m1）、496.8 → 172.0 us
（12288x4096@m1），深队列下内核本身 m=1 为 48-62 → 185-274 GB/s（3.6-4.4x）、
m=2..8 为 11-45 → 18-238 GB/s（1.6-5.3x）；与 Triton 路径的差在 fp32
舍入内（归一化 ≤3.0e-7），fp16 输出另含 1 ulp 舍入（≤4.9e-4）。
`COMFY_GPU_TUNE_W4A16_HIP=off` 回退 Triton；m>8、fp16 激活的多行批等不合形状的批
仍走 Triton（多行内核按 fp32 激活在 LDS 里摆 M 行，见 [HIP_KERNELS.md](HIP_KERNELS.md) 附录）。
fp16 激活 + `v_dot2_f32_f16` 的内核（`w4a16_gemv_h`）现在**是均匀格点（无 codebook）
检查点的默认 feed**：3.4-7.2x over Triton（204-291 GB/s），数值与无损版同级（fp32 输出
2.4e-7、fp16 输出 1 ulp），因为 `code-8` 的 16 个值在 fp16 里精确可表示。带 codebook 的
检查点会看到格点退到 fp16（fp32 输出归一化 1.6-2.6e-4、rms 2.1-2.8e-4，fp16 输出约
1.5 ulp），所以那一档要 `COMFY_GPU_TUNE_W4A16_F16ACT=1` 才开，开启前按 `--sampling off`
重验生文 md5。convrot 路径的激活是 fp32，只走无损 feed，不受这个开关影响。

均匀格点（无 codebook）、fp32 激活的 m=2..4 多行解码现在**默认**把 `tab[16]` 的格点 gather
换成 `v_cvt_f32_ubyteN`：一个权重 dword 的 8 个 nibble 先在 VALU 里变成 fp32
格点，不再每权重读一次 LDS 表。深队列实测（graph replay，fp32 out，us/call）
4096²@m2 58.7→56.8、12288×4096@m2 135.7→131.5、4096²@m4 137.7→128.4、
12288×4096@m4 301.3→289.1（快 3-7%），m=8 的 481.9/866.4 → 481.0/866.4 打平；
6 组形状 × fp32/fp16 输出全部逐位一致。`COMFY_GPU_TUNE_W4A16_VALU=off` 回落表格做 A/B。
M=8 自动回落：它本来就在 256 VGPR / spill 53，8 个 asm 输出的活跃区间再压上去会慢 9%。

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
`kernels/triton/w4a8_gemv.py` 用 inline asm 直接发；nibble 在寄存器里展开
（`w & 0x0F0F0F0F` / `w >> 4`），零点用「组内和 − 8·Σq」修正（逐字节减 8 会借位）。

**优先从 convrot int8 转**：只量化一次，误差最小。已打包的 `asym_w4a8_int8`
checkpoint 也能转，但它带 16 级 Lloyd-Max codebook，等于再量化一次，误差会多出来。

```bash
python custom_nodes/ComfyUI-GPU-Tuning/tools/requant_uniform4.py \
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

### 时间分解（逐块 graph 之前）

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
所以那一档的墙是 GPU 而不是 CPU——直调内核只对权重常驻（`--disable-dynamic-vram` 或 `--highvram`）有效，
默认启动仍是 6.3 it/s。

CUDA graph 已落地并默认开启（`patches/textgen/resident_graph.py`）：每块录成一张图，回放一次
就是一次提交，吸收掉每步约 800 次 eager launch 和 python 调度开销，稳态墙钟从 56 ms 降到
~28 ms（35–37 it/s）。CLIP Settings 节点的 `text_gemv=off` 可关掉 GEMV 与直调做对比。

## 实验记录（仓库外）

以下推导与微基准不在仓库里，留在本机实验目录：

- `gfx906fix/textgen-tuning.md` — **文本生成调优记录**：现状、时间分解、五个坑、后续方向
- `gfx906fix/qwen-image21-remaining-ops.md` — 各补丁的收益与测量方法
- `gfx906fix/gfx906-triton-attention-tuning.md` — Triton 注意力调优全过程
- `gfx906fix/gfx906-int8-ceiling.md` — int8 在这张卡上的理论上限
- `gfx906fix/hip_attn/NOTES_F16.md` — 第一版 HIP fp16 内核（dot2 只有 ILP2–4，实测 2.7 TMAC/s，未进入本包）
- `gfx906fix/hip_attn_v2/` — 本包内核的推导与微基准：指令吞吐/占用率曲线、每 lane tile 扫描、阶段消融

