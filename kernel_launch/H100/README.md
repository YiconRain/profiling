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
| 模型 | Qwen3.5-27B / 9B / 4B / 2B / 0.8B |
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

共 `5 模型 × 2 模式 × 6 用例 = 60` 个实验组合。

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
| launch 次数 | 窗口内 `cudaLaunchKernel*` 主机 API 调用数（`CUPTI_ACTIVITY_KIND_RUNTIME`） |
| launch 开销 | 上述 launch API 调用的累计主机耗时 |
| launch 占比 | launch 开销 / e2e × 100% |
| Top-5 kernel | 窗口内按累计 GPU 时间排序的前 5 个 kernel |

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
# 一键跑完：下载模型 -> 逐组合 nsys 采集 -> 汇总
bash kernel_launch/H100/scripts/run_all.sh

# 可选：只跑部分组合（参数透传给 run_profiling.py）
bash kernel_launch/H100/scripts/run_all.sh --models Qwen3.5-0.8B --modes eager
```

脚本可重入：已完成的组合（对应 `results/metrics/*.json` 已存在）会被跳过。

## 6. 路径约定

所有脚本均使用**相对路径**，并要求以项目根目录（`profiling/`）为工作目录运行；
`run_all.sh` 会自动 `cd` 到项目根目录。这样在 vast.ai 实例上克隆仓库后即可直接运行，
不受绝对路径差异影响。
