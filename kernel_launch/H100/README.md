# H100 Kernel Launch 开销实验

本目录用于在 **单张 H100** 上，测量 **SGLang（FlashInfer 后端）** 在 Qwen3.5 系列模型上的
**kernel launch（内核启动）数量与开销**，并分析其对不同模型规模、不同推理场景的影响。

## 1. 实验目标

对每一个 `(模型, 执行模式, 测试用例)` 组合，提取三类指标：

1. **kernel launch 总次数**；
2. **kernel launch 的时间开销**（绝对耗时，以及占端到端时间的百分比）；
3. **耗时最高的 Top-5 kernel**。

## 2. 实验维度

| 维度 | 取值 |
|------|------|
| 框架 | SGLang（使用 `envs/` 中约定的 `sgl_env` 环境，见 `envs/envs.md`） |
| 模型 | 稠密：Qwen3.5-27B / 9B / 4B / 2B / 0.8B；MoE：Qwen3-30B-A3B（30B 总参 / 3B 激活） |
| Attention 后端 | FlashInfer |
| 执行模式 | eager（关闭 CUDA Graph）、cudagraph（开启 CUDA Graph） |
| Batch Size | 1 |

### 测试用例（沿用需求编号，无 case4）

| 用例 | Prompt 长度 | Decode 长度 | 说明 |
|------|------------|------------|------|
| case1 | 256 | 0 | 纯 prefill |
| case2 | 1024 | 0 | 纯 prefill |
| case3 | 8192 | 0 | 纯 prefill（长上下文） |
| case5 | 16 | 128 | 短 prompt + 中等 decode |
| case6 | 16 | 512 | 短 prompt + 较长 decode |
| case7 | 16 | 1024 | 短 prompt + 长 decode |

> **关于 "Decode 0"**：解码器至少产出 1 个 token，因此 prefill-only 用例强制
> `max_new_tokens=1`，该 token 与 prefill 属于同一次前向，trace 捕获到的即 prefill 阶段的 kernel。

默认共 `6 模型 × 2 模式 × 6 用例 = 72` 个实验组合。加上两个减层外推的大 MoE
（`--extrap`，见 3.4 节）则为 `8 × 2 × 6 = 96`。

> **为什么 MoE 用 Qwen3-30B-A3B 而非 Qwen3.5-35B-A3B**：最小的 Qwen3.5 MoE 是
> 35B-A3B，bf16 权重约 70GB，加上 KV cache / 激活 / CUDA Graph 缓冲后放不进单张 80GB
> H100；下一档 Qwen3.5 MoE 更是 122B 起步。因此 MoE 对照选用能装下的 Qwen3-30B-A3B
> （约 60GB，留约 20GB 余量），它与 Qwen3.5 MoE 同为 3B 激活、架构同源。更大的
> Qwen3.5 MoE 通过 3.4 节的减层外推来估计。

## 3. 方法论

### 3.1 warmup 与测量分离

每个组合执行 **1 次 warmup + 1 次正式测量**：

- warmup 触发 CUDA Graph 捕获（cudagraph 模式）以及 JIT / autotune 缓存；
- 正式测量的 `generate()` 被包在 NVTX range `measure` 中。

`worker.py` 里关闭了 radix（prefix）cache（`disable_radix_cache=True`）。否则 warmup 与
测量使用相同 prompt 时，测量阶段会命中前缀缓存，导致 prefill kernel 消失、测量失真。

### 3.2 为什么用 NVTX 时间窗口过滤

SGLang 的模型前向运行在独立的 scheduler 子进程中，而 NVTX `measure` 由主进程发出。由于
`generate()` 是阻塞调用，它在 nsys 的统一时间线上完整地"包住"了子进程的 GPU 工作。因此
分析阶段按 **时间窗口重叠**（而非线程归属）筛选 kernel，可跨进程正确归集测量区间的 kernel。

### 3.3 指标定义

| 指标 | 定义 |
|------|------|
| 端到端时间 e2e | NVTX `measure` 窗口的时长 |
| kernel 总数 | 窗口内 `CUPTI_ACTIVITY_KIND_KERNEL` 记录数 |
| launch 次数 | 窗口内 kernel 启动类主机 API 调用数：`cudaLaunchKernel*` / `cudaGraphLaunch*` / `cuLaunchKernel*`（`CUPTI_ACTIVITY_KIND_RUNTIME`），另在 `launch_by_api` 中给出分项 |
| launch 开销 | 上述 launch API 调用的累计主机耗时 |
| launch 占比 | launch 开销 / e2e × 100% |
| Top-5 kernel | 窗口内按累计 GPU 时间排序的前 5 个 kernel |

> **eager vs cudagraph 的解读**：eager 模式下每个 kernel 对应一次 `cudaLaunchKernel`，
> 因此 launch 次数≈kernel 数、launch 开销占比高；cudagraph 模式下同样数量的 kernel
> 由极少量 `cudaGraphLaunch`（外加少量未纳入图的 eager 调用，如采样）重放完成，
> launch 次数和开销都大幅下降——这正是本实验要量化的核心对比。`total_kernels`（GPU
> 端执行次数）在两种模式下口径一致，可直接比较。

### 3.4 大 MoE 的减层外推

对放不进 80GB 的大 MoE（Qwen3.5-122B-A10B、Qwen3.5-397B-A17B），采用**只加载前若干层、
再外推到全模型**的方法：

1. **部分下载**：`download_models.py --model <大MoE> --layers K` 先取回权重索引
   （`model.safetensors.index.json`），据此只下载"前 K 层 + embedding / lm_head / 末层
   norm"所在的分片，避免下载数百 GB 的完整权重。
2. **减层加载**：`worker.py --num-layers K` 通过 SGLang 的 `json_model_override_args`
   把 `num_hidden_layers` 覆盖为 K，只实例化前 K 层再采集。
3. **线性外推**：Transformer 每层启动的 kernel 数基本恒定，故按
   `scale = 全层数 / K` 将实测 kernel 数、launch 数、launch 开销外推到全模型。
   全层数从模型 `config.json` 的 `num_hidden_layers` 读取。

该外推为**一阶近似**：它把全部实测 kernel 都当作"按层线性增长"，未单独扣除 embedding、
采样、lm_head 等与层数无关的固定开销，因此会略微高估。默认 `K=4`（见
`config.py: EXTRAP_MODELS`）。此路径为**可选**（`run_profiling.py --extrap` /
`run_all.sh ... --extrap`），因为即使部分下载体量仍较大。结果单列在
`results/README.md` 的"减层外推估计"表中，与实测结果区分开。

## 4. 目录结构

```
kernel_launch/H100/
├── README.md            # 本文件：实验总览与方法论
├── scripts/             # 全部实验脚本（详见 scripts/README.md）
├── results/
│   ├── nsys/            # 压缩后的 .nsys-rep.gz 与 .sqlite.gz（供本地可视化分析）
│   ├── metrics/         # 每个组合的指标 JSON + summary.csv
│   └── README.md        # 运行后由 summarize.py 生成的结果汇总（中文表格）
└── logs/                # 每个组合的运行日志
```

`results/nsys/` 按 `模型/模式/用例` 组织，例如
`results/nsys/Qwen3.5-9B/eager/case3.nsys-rep.gz`。将其下载到本地后 `gunzip`，即可用
Nsight Systems GUI 打开做可视化分析。

## 5. 如何运行

前置：H100 实例上已按 `envs/envs.md` 建好 `~/envs/sgl_env`，且 `nsys` 在 `PATH` 上
（NVIDIA PyTorch 镜像位于 `/usr/local/cuda/bin`）。

```bash
# 一键跑完：下载模型（5 个 Qwen3.5 稠密 + Qwen3-30B-A3B）-> 逐组合 nsys 采集 -> 汇总
bash kernel_launch/H100/scripts/run_all.sh

# 可选：只跑部分组合（参数透传给 run_profiling.py）
bash kernel_launch/H100/scripts/run_all.sh --models Qwen3.5-0.8B --modes eager

# 可选：额外跑大 MoE 的减层外推（会先部分下载 122B / 397B 的前 4 层分片）
bash kernel_launch/H100/scripts/run_all.sh --extrap
```

脚本可重入：已完成的组合（对应 `results/metrics/*.json` 已存在）会被跳过。

## 6. 路径约定

所有脚本均使用**相对路径**，并要求以项目根目录（`profiling/`）为工作目录运行；
`run_all.sh` 会自动 `cd` 到项目根目录。这样在 vast.ai 实例上克隆仓库后即可直接运行，
不受绝对路径差异影响。
