# H100 Kernel Launch 开销实验（Experiment 2）

本目录用于在 **单张 H100** 上，测量 **SGLang（FlashInfer 后端）** 在 **Qwen3 / Qwen3.5**模型上的
**kernel launch 数量、开销与其造成的 GPU 空闲(bubble)**，并分析其对不同模型规模、prefill/decode、
以及 **batch size** 的影响。核心目的:用 **eager → cudagraph** 的对比,量化 kernel launch / host
开销的真实影响,为 megakernel(大 kernel 减 launch)这一 motivation 提供数据。

> 第一次正式实验(Qwen3.5 系列)的数据已用 git tag **`experiment-1`** 归档。

## 1. 实验目标

对每一个 `(模型, 执行模式, 测试用例)` 组合,提取:

1. **GPU bubble 占比**(核心)= `(e2e − gpu_busy) / e2e`,即 GPU 因 launch/host/sync 而空转的比例;
2. **kernel launch 次数 / launch 开销**,以及 **kernel 执行数、GPU 忙碌时间(区间并集)**;
3. **耗时最高的 Top-5 kernel**;
4. 由此得出 **eager → cudagraph** 的 Δbubble 与 e2e 加速比。

## 2. 实验维度

| 维度 | 取值 |
|------|------|
| 框架 | SGLang（使用 `envs/` 中约定的 `sgl_env` 环境，见 `envs/envs.md`） |
| 模型 | Qwen3 稠密：0.6B / 1.7B / 8B / 14B；Qwen3 MoE：30B-A3B；Qwen3.5 稠密补充：0.8B / 2B / 9B / 27B |
| Attention 后端 | FlashInfer |
| 执行模式 | eager（关闭 CUDA Graph）、cudagraph（开启 CUDA Graph） |
| Batch Size | 1（prefill+decode）；4 / 8 / 16（仅 decode，做 batch 扫描） |

### 测试用例（13 个 = BS×prompt×decode 组合）

| 用例 id | BS | Prompt | Decode | 说明 |
|------|----|--------|--------|------|
| bs1_p16_d0 / p256 / p1k / p4k / p8k | 1 | 16/256/1k/4k/8k | 0 | 纯 prefill（不同上下文长度） |
| bs1_p16_d128 / d512 | 1 | 16 | 128/512 | 纯 decode |
| bs4_p16_d128 / d512 | 4 | 16 | 128/512 | decode，batch 扫描 |
| bs8_p16_d128 / d512 | 8 | 16 | 128/512 | decode，batch 扫描 |
| bs16_p16_d128 / d512 | 16 | 16 | 128/512 | decode，batch 扫描 |

> **关于 "Decode 0"**：解码器至少产出 1 个 token，因此 prefill-only 用例强制
> `max_new_tokens=1`，该 token 与 prefill 同属一次前向，trace 捕获到的即 prefill 阶段的 kernel。
> **BS>1**：把同一 prompt 复制 N 份组成 batch，N 条序列 lockstep 解码 —— 用于观察 launch 影响
> **随 batch 增大而衰减**的趋势(batch 越大,kernel 越大、GPU 越饱和,launch 越藏得住)。

完整配置共 `9 模型 × 2 模式 × 13 用例 = 234` 个实验组合。MoE 选 Qwen3-30B-A3B(约 60GB,单卡
80GB 留约 20GB 余量;更大的 MoE 放不下)。补充实验可用 `--models` / `--cases` 只跑少量组合。

## 3. 方法论

### 3.1 warmup 与测量分离

每个组合执行 **1 次 warmup + 1 次正式测量**：

- warmup(不采集)触发 CUDA Graph 捕获（cudagraph 模式）以及 JIT / autotune 缓存；
- 正式测量的 `generate()` 用 `cudaProfilerStart/Stop` 括起来(见 3.2)。

`worker.py` 里关闭了 radix（prefix）cache（`disable_radix_cache=True`）。否则 warmup 与
测量使用相同 prompt 时，测量阶段会命中前缀缓存，导致 prefill kernel 消失、测量失真。

### 3.2 用 cudaProfilerApi 限定采集范围(关键)

`worker.py` 在被测 `generate()` 前后调用 `torch.cuda.cudart().cudaProfilerStart()/Stop()`，
nsys 用 `--capture-range=cudaProfilerApi --capture-range-end=stop`，于是**只录制被测的这一次
generate**(模型加载、warmup 都在 Start 之前，不入库)。关键点:

- **`cudaProfilerStop` 会强制 CUPTI flush**，此时 SGLang 的 scheduler 子进程仍然存活,所以
  子进程里的 GPU kernel 活动(包括 **CUDA graph 内部**、以及大 MoE)都能被可靠捕获 ——
  不依赖缓冲区填满、也不依赖子进程干净退出(这解决了早期"0 kernel"的采集失败)。
- 因为采集范围本身就是被测区间,分析阶段**直接统计库里的全部 kernel/launch**,无需再按时间
  窗口过滤。
- nsys 加 **`--cuda-graph-trace=node`**:记录 CUDA graph 内的**每个 kernel 节点**(默认只记一次
  graph launch),否则 cudagraph 的 kernel 数会被严重少算甚至为 0。eager 下该参数无副作用。

### 3.3 指标定义

| 指标 | 定义 |
|------|------|
| 端到端时间 e2e | 采集区间(被测 generate)的墙钟跨度:主机 API 事件的 `max(end) − min(start)` |
| **gpu_busy** | 所有 GPU 活动区间(kernel+memcpy+memset)取**并集**的忙碌时间(≤e2e,并发安全) |
| **gpu_bubble_ratio**(核心) | `(e2e − gpu_busy) / e2e` = GPU 空闲占比(launch+host+sync 造成的气泡) |
| kernel 总数 | 库内 `CUPTI_ACTIVITY_KIND_KERNEL` 记录数(含 CUDA graph 内节点) |
| launch 次数 | kernel 启动类主机 API 调用数：`cudaLaunchKernel*` / `cudaGraphLaunch*` / `cuLaunchKernel*`（`CUPTI_ACTIVITY_KIND_RUNTIME`），另在 `launch_by_api` 中给出分项 |
| launch 开销 / 占比 | 上述 launch API 累计主机耗时,及其 / e2e(**仅诊断**,见下方口径坑) |
| Top-5 kernel | 按累计 GPU 时间排序的前 5 个 kernel |

**核心分析 = eager → cudagraph 对比**:`Δbubble = bubble(eager) − bubble(cudagraph)`(cudagraph 消掉的
GPU 空闲)和 `speedup = e2e(eager)/e2e(cudagraph)`。Δ/加速越大 → 该场景越受 kernel launch/host 开销
主导(decode/小模型/MoE 大),越小 → compute-bound(长 prefill/大模型 ≈1)。

> **eager vs cudagraph 的解读**：eager 模式下每个 kernel 对应一次 `cudaLaunchKernel`，
> 因此 launch 次数≈kernel 数、launch 开销占比高；cudagraph 模式下同样数量的 kernel
> 由极少量 `cudaGraphLaunch`（外加少量未纳入图的 eager 调用，如采样）重放完成，
> launch 次数大幅下降——这正是本实验要量化的核心对比。`total_kernels`（GPU 端执行次数,
> 靠 `--cuda-graph-trace=node` 保证两模式口径一致）可直接比较。

> **两个度量口径的坑**(实测发现,重要):
> - `launch_overhead / e2e`(launch 占比)**不适合**直接当"launch 影响程度":compute-bound 时
>   launch 与 GPU 计算 overlap、且被队列反压撑大(高估);launch-bound(小模型/decode)时 launch
>   API 很短但 GPU 大量空转(低估)。衡量真实影响更应看 **GPU 空闲率** 或 **eager→cudagraph 的
>   e2e 加速比**。
> - `total_kernel_gpu_ms` 是各 kernel 时长的**简单求和**,kernel 跨多 stream 并发时会 **>e2e**;
>   真实"GPU 忙碌时间"应取所有 kernel 区间的**并集**(≤e2e)。

## 4. 目录结构

```
kernel_launch/H100/
├── README.md            # 本文件：实验总览与方法论
├── scripts/             # 全部实验脚本（详见 scripts/README.md）
├── results/
│   ├── nsys/            # 压缩后的 .nsys-rep.gz 与 .sqlite.gz（本地保留，不提交 GitHub）
│   ├── metrics/         # 每个组合的指标 JSON + summary.csv（可提交 GitHub）
│   └── README.md        # 运行后由 summarize.py 生成的结果汇总（中文表格）
└── logs/                # 每个组合的运行日志（本地保留，不提交 GitHub）
```

`results/nsys/` 按 `模型/模式/用例` 组织，例如
`results/nsys/Qwen3-8B/eager/bs1_p8k_d0.nsys-rep.gz`。将其下载到本地后 `gunzip`，即可用
Nsight Systems GUI 打开做可视化分析。**每个 case 的 rep 和 sqlite 都会压缩保留在本地，但被
`.gitignore` 排除**。

## 5. 如何运行

前置：H100 实例上已按 `envs/envs.md` 建好 `~/envs/sgl_env`，且 `nsys` 在 `PATH` 上
（NVIDIA PyTorch 镜像位于 `/usr/local/cuda/bin`）。

```bash
# 一键跑完：下载全部配置模型 -> 逐组合 nsys 采集 -> 汇总
bash kernel_launch/H100/scripts/run_all.sh

# 可选：只跑部分组合（run_all.sh 会按 --models 只下载被选模型）
bash kernel_launch/H100/scripts/run_all.sh --models Qwen3-8B --modes eager --cases bs1_p8k_d0
```

脚本可重入：已完成的组合（对应 `results/metrics/*.json` 已存在）会被跳过。

## 6. 路径约定

所有脚本均使用**相对路径**，并要求以项目根目录（`profiling/`）为工作目录运行；
`run_all.sh` 会自动 `cd` 到项目根目录。这样在 vast.ai 实例上克隆仓库后即可直接运行，
不受绝对路径差异影响。
