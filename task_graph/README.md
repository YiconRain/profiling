# Qwen3 Task Graph 测试说明

本目录用于在 Vast.ai 单卡 H100 上提取 Mirage MPK 的 Qwen3 模型任务图，并输出任务数、事件数、每层算子任务数以及 tile 切分信息。

主测试脚本 [run_task_graph.py](run_task_graph.py) 内的所有注释、帮助文本和报告字段均使用英文；本文档使用中文说明环境、参数和输出格式。

## 1. 实现范围

脚本支持以下模型，默认为 `Qwen3-8B`：

| 命令行别名 | Hugging Face 模型 |
|---|---|
| `Qwen3-0.6B` | `Qwen/Qwen3-0.6B` |
| `Qwen3-1.7B` | `Qwen/Qwen3-1.7B` |
| `Qwen3-8B` | `Qwen/Qwen3-8B` |
| `Qwen3-14B` | `Qwen/Qwen3-14B` |
| `Qwen3-30B-A3B` | `Qwen/Qwen3-30B-A3B` |

默认测试矩阵为：

| Prefill prompt 长度 | Decode 长度 |
|---:|---:|
| 8 | 128 |
| 16 | 128 |
| 1024（`1k`） | 128 |
| 8192（`8k`） | 128 |

默认 `mbt=8`，`max_num_batched_requests=1`。两者都可通过命令行修改。

Split-KV 是强制要求：

- Qwen3 稠密模型直接使用 Mirage `main` 的 `demo_hopper.py --split-kv-cache` 路径。
- Mirage `main` 当前的 `demo_30B_A3B_hopper.py` 没有 Split-KV 参数。脚本不修改 Mirage 源码，而是在当前 Python 进程内将该 demo 的 `paged_attention_layer` 调用精确替换为 Mirage 已有的 `paged_attention_split_kv_layer` 和 `paged_attention_split_kv_merge_layer`。
- 生成报告前会检查任务图中同时存在 Split-KV attention 任务和 merge 任务；检查失败时脚本直接报错，不会把普通 paged attention 误报为 Split-KV。

## 2. Vast.ai H100 环境准备

脚本不会创建、租用或销毁 Vast.ai 实例。请先手工启动一台单 H100 实例，登录后确认只有一张可见 GPU：

```bash
nvidia-smi
```

建议使用带 CUDA Toolkit、NVCC 和 PyTorch 的 NVIDIA PyTorch 镜像。默认执行模式会编译 MPK launcher，因此仅有 CUDA runtime 而没有 `nvcc` 的镜像不够。

在远端克隆 profiling 仓库并切换到 `task_graph` 分支：

```bash
git clone git@github.com:YiconRain/profiling.git
cd profiling
git switch task_graph
```

使用配套脚本创建独立 Python 3.12 环境，并从指定 SSH 地址克隆 Mirage `main`：

```bash
bash task_graph/setup_mirage.sh
```

默认位置：

- Mirage 源码：`~/envs/mirage`
- Python 环境：`~/envs/mpk_env`

也可指定 Mirage 目录：

```bash
bash task_graph/setup_mirage.sh /workspace/mirage
```

安装后建议记录实际 commit：

```bash
git -C ~/envs/mirage branch --show-current
git -C ~/envs/mirage rev-parse HEAD
```

测试脚本会再次检查 Mirage 必须处于 `main`，并将 commit 写入每份报告。

## 3. 快速运行

### 3.1 默认完整测试

以 Qwen3-8B、`mbt=8`、`BS=1`运行 8/16/1k/8k prefill，每个场景 decode 128 token：

```bash
export MIRAGE_HOME=$HOME/envs/mirage
$HOME/envs/mpk_env/bin/python task_graph/run_task_graph.py
```

默认会完成以下操作：

1. 每个 prompt 长度启动一个独立子进程，避免模型/CUDA 内存跨场景残留。
2. 加载模型并构造 Mirage MPK 图。
3. 生成 `task_graph_0.json`。
4. 编译 MPK launcher，实际执行指定 prompt 长度和 decode 长度的离线推理调度。
5. 校验 Split-KV，然后生成 JSON、TXT、CSV 报告。

### 3.2 只提取任务图

如果只需要任务图和统计，不需要 NVCC 编译与实际 persistent-kernel 执行：

```bash
$HOME/envs/mpk_env/bin/python task_graph/run_task_graph.py --graph-only
```

`--graph-only` 仍需要 H100 和模型权重，因为 Mirage 会根据 GPU compute capability、模型维度和 CUDA tensor 构造真实的 H100 任务图；它只是在生成 JSON 后提前停止。

### 3.3 运行单个场景

```bash
$HOME/envs/mpk_env/bin/python task_graph/run_task_graph.py \
  --model Qwen3-8B \
  --prompt-lengths 1k \
  --decode-len 128 \
  --mbt 8 \
  --max-num-batched-requests 1
```

### 3.4 运行 MoE 模型

```bash
$HOME/envs/mpk_env/bin/python task_graph/run_task_graph.py \
  --model Qwen3-30B-A3B \
  --prompt-lengths 8 16 1k 8k \
  --decode-len 128 \
  --mbt 8 \
  --max-num-batched-requests 1 \
  --continue-on-error
```

Qwen3-30B-A3B 加载权重、合并 MoE expert tensor 和编译都需要更多时间与显存。请使用 80 GB H100，并确保没有其他进程占用 GPU。如果只需要图统计，优先加 `--graph-only`。

### 3.5 使用本地模型目录

```bash
$HOME/envs/mpk_env/bin/python task_graph/run_task_graph.py \
  --model Qwen3-8B \
  --model-path "$PWD/models/Qwen3-8B"
```

`--model` 仍然必须填正确的架构别名；`--model-path` 只替换权重和 tokenizer 的加载位置。

## 4. 命令行参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | `Qwen3-8B` | 模型别名，可选值见上表。 |
| `--model-path` | 空 | 本地 Hugging Face 模型目录。 |
| `--prompt-lengths` | `8 16 1k 8k` | 一个或多个 prefill token 长度；`k` 按 1024 解析。 |
| `--decode-len` | `128` | 每个 request 的 decode token 数。 |
| `--mbt` | `8` | Mirage `max_num_batched_tokens`，即每次图迭代可调度的最大 token 数。 |
| `--max-num-batched-requests` | `1` | 最大 batch request 数，即 BS。 |
| `--page-size` | `4096` | KV cache page size。Mirage `main` 的稠密 Qwen3 H100 模型当前硬性要求 4096。 |
| `--max-num-pages` | 稠密模型 `16`；MoE 自动 | Mirage `main` 的稠密 Qwen3 模型内部当前硬编码检查 16 页；Qwen3-30B-A3B 按序列长度自动计算并每个 request 额外保留一页。 |
| `--mirage-dir` | `$MIRAGE_HOME` 或 `~/envs/mirage` | Mirage 源码目录。 |
| `--output-dir` | `task_graph/results` | 结果根目录。 |
| `--graph-only` | 关闭 | 跳过 NVCC 编译和实际执行，仅生成/分析图。 |
| `--force` | 关闭 | 删除匹配的已有 case 目录并重跑。 |
| `--continue-on-error` | 关闭 | 某个 prompt 场景失败后继续后续场景。 |
| `--keep-build-artifacts` | 关闭 | 保留大型生成 CUDA 文件、`.so` 和编译目录。 |
| `--allow-mirage-non-main` | 关闭 | 允许使用经人工审查的 Mirage 兼容分支。 |

参数约束：

- `mbt >= max_num_batched_requests`。Decode 阶段每个 request 每轮至少需要一个 token slot。
- `max_seq_length = prompt_len + decode_len`，由脚本自动计算。
- 手工设置的 `max_num_pages` 不能小于当前 BS 和最大序列长度的容量需求。
- Qwen3-0.6B/1.7B/8B/14B 必须使用 `page_size=4096, max_num_pages=16`。这是 Mirage `main` 的 `demo/qwen3/models/modeling_qwen3.py` 当前硬编码约束，脚本会自动应用；不需要手工传参。
- 脚本检查且只允许一张 compute capability 9.0 GPU，即 H100/Hopper 目标。

## 5. 输出目录

默认输出到 `task_graph/results/`，每个 case 独立存储。例如：

```text
task_graph/results/
├── summary.csv
└── Qwen3-8B__p1k_d128__mbt8_bs1__execute/
    ├── run.log
    ├── raw/
    │   └── task_graph_0.json
    └── report/
        ├── mirage_summary.txt
        ├── report.json
        ├── report.txt
        └── operations.csv
```

默认会删除生成的 `kernel_0.cu` 和 compiled launcher 目录，以减少磁盘占用。需要调试编译时加 `--keep-build-artifacts`。

`raw/` 下的原始 JSON 和编译产物默认被 `.gitignore` 排除；适合提交到 profiling 仓库的是 `report/` 和 `summary.csv`。

### 5.1 `mirage_summary.txt`

这是 Mirage `main` 自带的 `scripts/parse_task_graph.py` 对原始任务图产生的标准摘要，包含总 task/event/first-task 数和 TaskType/EventType 分组。脚本会先运行该官方 parser，再进行本实验要求的每层与 tile 扩展分析。

### 5.2 `summary.csv`

所有已完成 case 的总表，主要字段包括：

- `total_tasks_including_runtime`：静态图中 `all_tasks` 的总数，包含 terminate 和 begin-graph 控制任务。
- `compute_tasks_per_graph_iteration`：排除上述运行时控制任务后，每次图迭代的计算任务数。
- `total_events_per_graph_iteration`：静态图 `all_events` 的事件数。
- `split_kv_task_count`：每次图迭代的 Split-KV attention 任务数。
- `split_kv_merge_task_count`：每次图迭代的 Split-KV merge 任务数。
- `estimated_prefill_iterations`：按 `ceil(prompt_len * BS / mbt)` 计算的 prefill 图迭代数。
- `decode_iterations`：默认为 128。
- `estimated_compute_task_dispatches`：`compute_tasks_per_graph_iteration * (prefill_iterations + decode_iterations)`。
- `estimated_event_dispatches`：`total_events_per_graph_iteration * (prefill_iterations + decode_iterations)`。

注意：`estimated_*` 是基于 MPK 调度容量的统计估算，不是 Nsight/CUPTI 实测的 GPU kernel launch 数。静态任务图总数与整段请求的动态 dispatch 总数是两个不同指标，报告会同时保留两者。

### 5.3 `report.json`

机器可读的完整报告，顶层字段为：

```json
{
  "metadata": {},
  "statistics": {},
  "task_type_counts": [],
  "event_type_counts": [],
  "layers": {}
}
```

`metadata` 包含模型、Mirage remote/branch/commit、GPU、执行模式、Split-KV 适配路径和全部场景参数。

`layers` 按以下顺序组织：

```text
runtime
model_input
layer_0
layer_1
...
model_output
```

每个算子项包含：

- `operation`：可读名称，如 `RMSNorm`、`Split-KV attention`、`Linear + residual`。
- `task_type_name`：Mirage `TaskType` 枚举名，如 `TASK_RMS_NORM_HOPPER`。
- `variant_id`：Mirage 代码生成 variant。
- `task_count`：该层该算子的任务数。
- `tile_groups`：按输入/输出 tile 形状分组的详细信息。

每个 tensor tile 会报告：

- `base_ptr`：图中 tensor 名称，通常可用于识别 layer 和 weight。
- `tile_dims`：单个任务看到的局部 tensor 形状。
- `strides`：该 tensor descriptor 的 stride。
- `unique_offsets`：该组任务中出现的不同字节偏移数。
- `offset_min_bytes` / `offset_max_bytes`：字节偏移范围。
- `placement`：`split` 表示任务指向不同 tile；`replicated` 表示多个任务读取同一个 tensor 位置。

### 5.4 `report.txt`

面向人工阅读的完整文本报告。它首先列出全局任务/事件统计，然后展开每层算子。RMSNorm 的格式示意如下：

```text
[layer_0]
  RMSNorm [TASK_RMS_NORM_HOPPER, variant=0]: tasks=8
    tile-group 1: tasks=8
      input:  layer_0_input_layernorm tile=4096 ... offsets=1 (replicated, ...)
      output: rmsnorm_out tile=1x4096 ... offsets=8 (split, ...)
```

### 5.5 `operations.csv`

每行对应一个 `layer + task_type + variant` 组合，方便用 pandas/表格工具比较不同模型、prompt 长度、mbt 和 BS。`tile_groups_json` 字段保留完整 tile 信息。

## 6. 场景和统计的解读

Mirage MPK 的 `task_graph.json` 描述一次 persistent graph iteration 中的静态任务和事件。完整 prefill + decode 请求会多次复用该图。

因此报告分为两层：

1. **静态图指标**：`total_tasks_including_runtime`、`compute_tasks_per_graph_iteration`、`total_events_per_graph_iteration`、每层算子任务数和 tile 切分。
2. **请求级估算**：根据 prompt、decode、mbt 和 BS 计算图迭代数，再估算 compute task/event dispatch 总数。

对默认 `BS=1, mbt=8, decode=128`：

| Prompt | 估算 prefill iteration | Decode iteration | 总 graph iteration |
|---:|---:|---:|---:|
| 8 | 1 | 128 | 129 |
| 16 | 2 | 128 | 130 |
| 1024 | 128 | 128 | 256 |
| 8192 | 1024 | 128 | 1152 |

Split-KV 的静态切分数由 Mirage `main` 当前 H100 demo 逻辑决定：`max(1, max_seq_length // 256)`，其中 `max_seq_length = prompt_len + decode_len`。实际数值应以 `report.json` 和 `report.txt` 为准。

## 7. 长上下文执行说明

Mirage 当前两个 H100 demo 在附加 RoPE tensor 时硬编码为前 4096 行。为了使 8k 调度/任务图测试不越界，本脚本在进程内将已有 RoPE 行重复扩展到 `max_seq_length`。

这一适配有两个重要边界：

- 对任务图结构、任务数、事件数、Split-KV 切分和调度执行安全性是足够的。
- 4096 token 以后的 RoPE 值不是模型真实位置编码，所以本脚本不用于验证生成文本的数值正确性或质量。

同样，prompt token 由真实 tokenizer 产生后重复/截断到指定长度，并强制忽略 EOS，目的是可重复地跑满指定 decode 长度，而不是做语义质量评测。

## 8. Mirage `main` 失败时的兼容分支

当前实现优先使用原始 Mirage `main`，MoE Split-KV 和 8k RoPE 兼容都在 profiling 进程内完成，因此不需要修改 Mirage 分支。

如果未来 Mirage `main` 发生 API 变化，必须修改 Mirage 才能运行，按以下方式建立最小兼容分支：

```bash
cd ~/envs/mirage
git switch main
git pull --ff-only origin main
git switch -c task_graph_compat
```

仅修改与以下内容直接相关的部分：

1. H100 Qwen3 demo 的 graph-only 停止点；
2. MoE demo 的 Split-KV attention + merge 路径；
3. 8k 上下文需要的完整 RoPE tensor 长度。

审查并提交后，显式允许该分支：

```bash
$HOME/envs/mpk_env/bin/python task_graph/run_task_graph.py \
  --mirage-dir "$HOME/envs/mirage" \
  --allow-mirage-non-main
```

报告会记录兼容分支名和 commit，便于复现。

## 9. 本地 CPU 验证

不需要 GPU 即可检查 Python 语法、分析器单元测试和 shell 语法：

```bash
python3 -m py_compile task_graph/run_task_graph.py task_graph/test_analyzer.py
python3 task_graph/test_analyzer.py
bash -n task_graph/setup_mirage.sh
```

完整 Mirage graph build/NVCC/persistent-kernel 执行必须在 Vast.ai 单 H100 上验证。
