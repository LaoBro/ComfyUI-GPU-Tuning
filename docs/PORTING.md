# 移植到其它显卡

装好就能用：大部分改写与显卡型号无关，一次 `bench.py --all --write` 会把这张卡上
「最优解随卡变」的部分测出来并写进本机 manifest。本文只讲两件事：**怎么跑通**，以及
**跑通之后怎么继续调优**。给别的架构写 HIP 内核、改后端、加 bench 站点属于二次开发，
见文末「实现细节在哪」，不在本文展开。

## 适用环境

- 需要一张 CUDA 设备；CPU 上整包跳过。AMD 按 `gcnArchName`、NVIDIA 按
  `sm<major><minor>` 分表，同一厂商的不同卡也不会共用一张表。
- 只有 `gfx906` 随包带实测表。其它架构装好后测量门默认关、走通用路径，这是预期行为。
- comfy-kitchen 的 triton 后端由本包在启动时自行启用，不需要用户先配置；常驻权重的逐块
  CUDA graph 由节点安装，core 已覆盖的 aimdo/VBAR 流式路径不重复捕获。
- Qwen3.5 文本编码器是**版本锁定**的：`vendor/qwen35/` 自带一份 0.39.0 的计算实现
  （`FORK_BASE` 标注来源 tag），启动时把 `comfy.text_encoders.qwen35.te/tokenizer` 这两个
  工厂换成自带版本，让所有 core 版本都跑同一套优化。`te()` 的签名从 0.39 到 0.45.4 不变，
  是稳定的注入边界。0.44+ core 改用 `CombinedLinear` / `FixedKVCache` / deferred DeltaNet，
  这些在 gfx906 上回退到 torch、实测慢约 3 倍，所以不再让位给 core。gfx906 的解码优化
  （split-half RoPE、合并 A/B GEMV、depthwise conv tap 求和、stacked q/k 归一化、KV decode
  的 GQA group 折叠、RMSNorm 预计算权重、频率表缓存）已**直接烘焙进** `model.py` /
  `llama_base.py`——不再靠运行时按类打补丁（那样打在 core 类上、够不到 fork 的类副本）。
  因此旧的 `kv decode` 独立补丁项已移除：`FixedKVBias.decode` 的 seq==1 flash 路径本就是
  stock 0.39 自带、seq>1 的 group 折叠已烘焙进 fork。`comfy_kitchen.*` 上的 Triton 内核和
  `comfy.model_prefetch` 上的图补丁是全局符号，自带模型照常受益。
  `COMFY_GPU_TUNE_QWEN35=off` 可退回 core 实现。参考副本在 `vendor/qwen35/reference/`，
  用 `tools/vendor_selftest.py` 自检烘焙标记与边界。
- 0.45 的 `CombinedLinear` / `premerge` / `PrefixCache` 已纳入兼容探测：节点只在 core
  没有对应能力时安装旧补丁（仅在没有自带 Qwen3.5 实现时的回退路径）。
- 不要求 ROCm/HIP 构建：fingerprint 对 CUDA 构建同样成立。N 卡上 int8 tile 与注意力
  内核补丁不安装、文本生成改写组照常生效，`bench.py --all` 会按设备跳过用不到的
  tile / attention 站点；本包未在 N 卡实测，性能结论由有卡的人跑出来。

## 三步跑通

1. 按 README 装到 `custom_nodes/`，重启 ComfyUI。不改任何 core 文件。
2. 用节点默认值先跑一遍。没跑过 bench 的卡上测量门是关的，不是坏了；
   `tools/status.py` 会写 `absent`。
3. 从 ComfyUI 根目录跑一次 `bench.py --all --write`（命令见附录）。它测出 int8 tile
   和各 feature 的 on/off，写进 `~/.config/comfy-gpu-tune/arch/<arch>/manifest.toml`，
   本地覆盖出厂表。重启后 `status.py` 应显示 `measured ... in use`，启动日志不该再有
   `no measured data`。

到此为止不需要手写内核，也不需要读实现。剩下的工作才是本文的重点。

## 哪些自动，哪些要自己定

| 类 | 判据 | 怎么处理 |
|---|---|---|
| **A 常开** | 行为不随卡变、没有取舍 | 直接装，发现更慢再单独关 |
| **B 手工** | 精度分层、显存取舍，或测量覆盖不了决策面 | 默认给一个合理值，按实际模型自己 A/B |
| **C 测量门** | 最优解随卡变 | 没数据就是关，跑一次 bench 由数据决定 |

原则一句话：**dispatch 能回答的交给 dispatch，管不到的才用门或开关**。C 类的「没跑过就
别猜」是这套包最关键的一条：数据来自随包出厂表或本地覆盖，本地优先；两者都按
fingerprint 自守，指纹不符整张表被忽略（不是逐条赌）并启动 WARNING。

还有一点容易误判：这里的「安装」是替换模块级符号或类，不是设一个开关位。没装上的项
`status()` 显示 `n/a`，节点里把对应开关选成 `on` 也打不开它。
`COMFY_GPU_TUNE_FORCE` 是跳过判定强装的逃生口，启动会 WARNING 说明这些内核没在本卡
测过。

`deltanet` 是半例外：comfy-kitchen 没有 fused gated-delta op 的卡上它结构性开启，这个
事实不随测量过期，所以没跑 bench 也可能看到它是 on。

### 要自己定的四项

bench 不写这四项，需要人按自己的模型和显存判断：

| 项 | 为什么不能自动 | 怎么定 |
|---|---|---|
| `attention` mode | 精度分层（int8 PV rel ~2e-2 对 `hand` 2.5e-4），是质量取舍 | `bench.py --attention` 只打印证据不写记录；在节点里手选 `hip` / `gemm` / `hand`，见 [CONFIG.md](CONFIG.md) 的「注意力模式怎么选」 |
| `qwen35` | bench 用例只覆盖 prefill，收益主要在 decode，测量判不了 | 等价类替换，env / 节点默认开；用真实 checkpoint 和实际解码长度 A/B 后再定 |
| `qkv` | 1.6 GB 常驻换约 1.5%，纯显存权衡 | 默认关；显存够且确认有收益再开 |
| `te_dequant` | 对端是逐调用反量化，bench 的两个站点只出证据、不写 `feature.*` | `COMFY_GPU_TUNE_TE_DEQUANT=on/off` 或 `COMFY_GPU_TUNE_FORCE=te_dequant`，也可在表里手写 |

## 进一步调优

### 先归因，再动手

- 生文链路用 `tools/chain_profile.py`，DiT 侧用节点的 `profile` 开关。`unattributed`
  占比大说明瓶颈不在 GPU 内核上，继续换内核是白费。
- 微基准只回答「这一段里谁快」，不回答「时间落在哪」。两者别混用。

### 一次只动一个变量

- 同时改 pack、CUDA graph、text encoder、token 数和有无图的一组对比，两个配置之间差了
  五个变量，38.7 对 1.49 tok/s 没有任何归属。固定单变量的例子才有意义（`text_gemv`
  开关：两边都 te=int8、图关、同 seed、同 prompt，才能把 16.0 → 20.8 tok/s 算给它）。
- `md5` 只在 `--sampling off` 下才是等价判据。采样开启时微小的数值差会逐 token 放大
  成完全不同的文本；反过来，更快但生文从第一个 token 就退化，也是真失败。
- 数值门槛先于计时：`bench.py` 每个形状先和 `tools/refs.py` 的 torch 参考比对，超
  `--max-rel` 的实现不参与计时；`--min-gain`（默认 3%）是噪声下限，差距小于它不宣布
  赢家。

### 让测量覆盖你实际在跑的形状

`feature.*` 只有 `default`、没有 per-shape 记录，回答的是「这张卡整体划不划算」，不是
「这个形状划不划算」。所以 `bench.py --shapes NxK@mM`（以及 `--tile-shapes`、
`--vae-shapes`、`--convrot-shapes`、`--deltanet-shapes` 等）要照实际在跑的尺寸给。
表里没有的形状回落 `default`，再没有就回到 autotune / 探测；未实测卡走候选模式时每个
新形状第一次调用会现测候选，首个 shape 偏慢是正常现象。

### 热点没有站点就先加站点

`vision.patch_embed` 和 `deltanet.prefill` 都是这么来的：归因发现某段占时间，但 bench
里没有对应站点，先补一个能出证据的站点，再谈优化。加站点、写内核、接入后端链的做法见
[HIP_KERNELS.md](HIP_KERNELS.md) 的「基本流程」和
[arch/gfx906/kernels/README.md](../arch/gfx906/kernels/README.md)；本文不展开实现契约。
改完内核先跑 `tools/verify_backends.py`（选择 parity + 数值 A/B，需要 GPU），再记录性能。

### 换环境重测

fingerprint 含 ROCm 版本、Torch 构建号、Triton 版本和目标架构。内核是那个编译器产出的，
同版本 ROCm 的另一次构建也可能挪动几个百分点。**换一次 torch 就要重测**；指纹不符时
整张表被忽略，不会拿旧数字赌。

## 验证与救急

- `tools/status.py`（不开 ComfyUI）给出 `measured` / `stale` / `absent` / `off`
  四态，每态都有对应的启动日志；`measured 12/14` 是本机表的命中计数。
- 「节点跑了但没在用实测数据」是最糟的状态：表现和正常一模一样，只是慢。先确认
  `measured ... in use`，再谈性能。
- 表写错或 kernel 改坏了，用 `COMFY_GPU_TUNE_MEASURED=0` 整体关掉这一层回到纯探测；
  `COMFY_GPU_TUNE=off` 是整包总闸。

## 实现细节在哪

本文刻意不讲节点内部实现。需要动手时按主题去这几处：

| 主题 | 位置 |
|---|---|
| 节点开关、注意力模式、环境变量全表 | [CONFIG.md](CONFIG.md) |
| 八个补丁、文本生成实现 | [PATCHES.md](PATCHES.md) |
| 给其它架构写 HIP 内核、内核契约 | [HIP_KERNELS.md](HIP_KERNELS.md)、[arch/gfx906/kernels/README.md](../arch/gfx906/kernels/README.md) |
| gfx906 实测边界、数字与结论摘要 | [GFX906.md](GFX906.md)、[GFX906_EXPERIMENTS.md](GFX906_EXPERIMENTS.md) |
| manifest 结构、站点与 grade 语义 | `kernels/manifest.py`、`tools/bench.py` |
| 门控分层、后端注册表、启动降级策略 | `kernels/registry.py`、`gpu_tuning.py` |

---

# 附录：命令速查

从 ComfyUI 根目录运行：

```
# 只打印，不写文件
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/bench.py

# 一张卡需要的一切：tile + 各 feature 的 on/off，写入本机覆盖
... --all --write

# 只测一个站点 / 指定形状
... --site w4a8.m1 --shapes 4096x4096@m1,4096x4096@m8 --write

# 只出证据、不写记录的站点
... --attention
... --qwen35

# 直接改随包分发的表（准备 PR 时用）
... --write --out custom_nodes/ComfyUI-GPU-Tuning/arch/gfx906/manifest.toml

# 扫单个 tile 候选、打印可直接用的环境变量
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/tile_sweep.py --emit

# 生文链路归因
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/chain_profile.py --clip <文件名> --max-length 32

# 不开 ComfyUI 看判定 / 改内核后的 parity 与数值 A/B
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/status.py
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/verify_backends.py

# 自带 Qwen3.5 的 fork 自检；以及上游改动摘要（struct 要移植 / compute 我们自带）
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/vendor_selftest.py
.venv/bin/python custom_nodes/ComfyUI-GPU-Tuning/tools/vendor_forward_diff.py
#   --repo /path/to/ComfyUI --ref origin/master --save-diff /tmp/up.diff
```

`vendor_forward_diff.py` 在 `FORK_BASE` 与上游 ref 之间 diff `qwen35.py`，按 AST 把每个
hunk 归到它所属的定义，再按层分组：**STRUCT**（config / tokenizer / `te` 工厂，core 调用或
决定权重键 —— 必须移植）、**VISION**（编码路径，图像条件退化才移植）、**COMPUTE**
（解码数学，我们锁定 —— 只有实测更快才搬单个函数）。它还会校验注入边界（`te`/`tokenizer`
签名）和权重键配置是否变动；变动会在开头以 `[!!]` 标出，这就是 fork 会失效的信号。

注入边界做了防御：core 以后给 `te()` 传自带 fork 不认识的新 kwarg（如 master 的
`vocab_size`/`tokenizer_json`）时，节点不会崩——未知参数被丢弃并打一次 WARNING，同时在
`status()` / `textgen_summary()` 里以 `DRIFT(...)` 常驻可见。STRUCT 层已移植上游 master 的
`vocab_size` / `tokenizer_json`（含嵌入 tokenizer 的 stop token 推导、按词表取 image/pad
token id、以及 tokenizer 数据目录重定向到 core 安装位置），这两个参数现在会被识别并转发，
不会再报 DRIFT；词表裁剪 checkpoint 可正常加载。节点开关关掉时走 core 原生工厂，参数原样
透传。将来 core 若再加**新的** kwarg，同样由这套兜底 + 告警机制保护，届时按工具提示继续
移植即可。

手工指定 tile：`COMFY_GPU_TUNE_TILE_CONFIG` 让该卡视为「已实测」、列表只放这一个配置、
不跑 autotune；`COMFY_GPU_TUNE_SMALL_TILE_CONFIG` 只换窄 tile 候选。两者都是启动期读取，
与 manifest 里的 `int8_gemm.tile` 同权（env 优先）。manifest 根目录用
`COMFY_GPU_TUNE_MANIFEST_DIR` 改，`COMFY_GPU_TUNE_ARCH` 强制指定目标目录（没插卡时
开发用），`COMFY_GPU_TUNE_ARCH_KERNELS` 追加 `.so` 搜索目录。环境变量全集见
[CONFIG.md](CONFIG.md)。
