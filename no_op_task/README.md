# MPK 超编译开销实验（单卡 H100）

本目录用于测量 Mirage Persistent Kernel 在“实际工作负载固定，但编译期容量参数增大”时的 GPU 端到端执行延迟。使用的 MPK 代码库是 [YiconRain/mirage](https://github.com/YiconRain/mirage)，不是 `mirage-project/mirage`。

## 实验定义

所有 case 的真实 workload 完全一致：

- GPU：单张 H100。
- 真实 request 数：`1`。`max_num_batched_requests` 只是编译期容量，不会将 workload 复制成 8 或 16 个请求。
- Prompt：恰好 `1` 个 token。Harness 保留 Mirage demo/tokenizer 产生的第一个合法 token，然后将 `prompt_lengths[0]` 强制设为 1。
- Decode：强制生成 `512` 个 token，开启 `--ignore-eos`。
- `max_seq_length = 1 + 512 = 513`。
- 不做 warmup。
- 每个 case 编译一次，在同一进程、同一个已编译 MPK 上重置 request/KV-page 状态后连续运行 5 次。
- 每次使用 CUDA Event 直接围住 `mpk()`，记录 GPU 端完整执行时间，单位为 ms。
- 5 次测量结束后，另起一个 nsys 进程，重新加载同一 case 的已编译 kernel，再执行 1 次。nsys 只捕获 `cudaProfilerStart/Stop` 之间的这一次 MPK。

实验矩阵：

| Case | `max_num_batched_requests` | `max_num_batched_tokens` |
|---|---:|---:|
| `mbr1_mbt1` | 1 | 1 |
| `mbr1_mbt8` | 1 | 8 |
| `mbr8_mbt8` | 8 | 8 |
| `mbr8_mbt16` | 8 | 16 |
| `mbr16_mbt16` | 16 | 16 |

模型范围：

```python
QWEN3_SPECS = {
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
}
```

`Qwen3-0.6B` / `1.7B` / `8B` 运行全部 5 个 case。`Qwen3-14B` 和 `Qwen3-30B-A3B` 跳过所有 `MBT=16` case，只运行前 3 个 case。总计 21 个 cell。

## Mirage 分支与改动

请使用 `YiconRain/mirage` 的 `no_op_task` 分支。该分支没有修改 MPK task、kernel 或 scheduler 逻辑，只包含为实验复用/重置和可配置 KV cache 所需的调试修复：

1. 普通 `compile()` 后保存生成扩展中的 `init_request_func`，使 5 次独立 decode 之间可以重置 request 和 page queue。
2. `load_mpk_kernel()` 重载已编译 kernel 时补上 `paged_kv_indices_snapshot` 这个第 11 个 meta tensor。
3. Qwen3 H100 demo 不再将 KV cache 强制断言为历史默认值 `16 pages × page_size 4096`，而是校验由 demo 构造器参数实际分配的动态 shape。实验使用的 `page_size=64` 与 H100 attention kernel 的 `KV_TILE_SIZE=64` 兼容。

这些改动不影响 task graph 数量和算子执行逻辑。

## 目录与脚本

- `config.py`：模型、case 和固定 workload 参数。
- `worker.py`：在 YiconRain Mirage H100 demo 外包一层实验 harness，负责真实 BS=1、prompt=1、状态重置、CUDA Event 计时和 decode 长度校验。
- `run_experiments.py`：运行完整矩阵，管理编译、5 次测量、nsys 和 SQLite export。
- `run_all.sh`：日常入口。
- `smoke_test.sh`：运行并自动校验 `Qwen3-0.6B + mbr1_mbt1` 的完整测量、编译和 nsys 产物。
- `command.md`：从 clone、环境安装、模型下载、smoke test 到完整实验的可复制命令。
- `remote_bootstrap.sh`：已经由用户创建好的 Vast.ai H100 实例上的引导脚本；它不会创建或销毁 Vast.ai 实例。

## Vast.ai 上的准备

建议将两个 repo 放在同一父目录：

```text
workdir/
├── profiling/   # branch: no_op_task
└── mirage/     # branch: no_op_task
```

基本要求：

- 单张 H100，`CUDA_VISIBLE_DEVICES` 中只暴露目标 GPU 更稳妥。
- `nvidia-smi` 和 `nsys` 在 `PATH` 中。
- Python 环境已安装与 YiconRain Mirage 兼容的 PyTorch、Transformers、Safetensors 等依赖。
- Hugging Face token/cache 按需配置；模型由 Hugging Face ID 加载并复用 `HF_HOME` 缓存。
- Mirage 建议 editable install，以确保使用的正是 `no_op_task` 分支。

如果 repo 已经 clone：

```bash
cd profiling
git switch no_op_task

git -C ../mirage switch no_op_task
python -m pip install -e ../mirage -v

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PWD/no_op_task/artifacts/hf_cache"
bash no_op_task/run_all.sh
```

如果希望由脚本 clone/update Mirage 并做 editable install：

```bash
cd profiling
export CUDA_VISIBLE_DEVICES=0
export MPK_PYTHON=python
bash no_op_task/remote_bootstrap.sh
```

`remote_bootstrap.sh` 优先使用 GitHub SSH，失败后回退到 HTTPS。它只管理 repo 和 Python 安装，不会操作 Vast.ai 实例生命周期。

## 常用命令

全部实验：

```bash
bash no_op_task/run_all.sh
```

先用最小 case 做完整且自校验的 smoke test：

```bash
bash no_op_task/smoke_test.sh
```

只打印完整命令而不加载模型：

```bash
bash no_op_task/run_all.sh --dry-run
```

已有测量和 compile artifact，只补 nsys：

```bash
bash no_op_task/run_all.sh --only-nsys
```

只跑 5 次延迟，暂时不跑 nsys：

```bash
bash no_op_task/run_all.sh --skip-nsys
```

覆盖已有结果：

```bash
bash no_op_task/run_all.sh --force
```

脚本可断点续跑：已有 `metrics.json` 的测量会跳过，已同时存在 `.nsys-rep` 和 `.sqlite` 的 nsys 阶段也会跳过。

## 产物布局

所有大文件和运行结果都位于 `no_op_task/artifacts/`，整个目录已加入 `.gitignore`：

```text
no_op_task/artifacts/
├── summary.csv
├── hf_cache/                         # 如果 HF_HOME 指向这里
└── Qwen3-0.6B/
    └── mbr1_mbt1/
        ├── metrics.json                 # 5 次延迟+平均值
        ├── measure.log
        ├── nsys.log
        ├── compile/
        │   ├── test_rank0.cu
        │   ├── task_graph_rank0.json
        │   ├── kernel_metadata_rank0.json
        │   └── mpk_launcher_rank0.cpython-*.so
        └── nsys/
            ├── profile.nsys-rep
            ├── profile.sqlite
            └── profile_run.json
```

`metrics.json` 中的关键字段：

- `mpk_gpu_ms`：5 次 `mpk()` GPU 端端到端时间。
- `average_mpk_gpu_ms`：5 次的算术平均。
- `average_ms_per_generated_token`：平均总时间除以 512。
- `generated_tokens`：必须为 512，否则 worker 直接报错。
- `num_tasks` / `num_events`：从该 case 的 task graph JSON 读取。
- `mirage_revision`：实际运行的 Mirage commit。

## 计时口径

每次测量的核心等价于：

```python
reset_request_and_page_state()
start.record()
mpk()
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

因此结果包含 `prepare_kernel`、GPU scheduler/worker persistent kernels、GPU 上的 `prepare_next_batch`、全部 task graph traversal和跨 stream event 依赖。它不是 CPU wall-clock，不用来表示 Python/CUDA API 的纯 CPU launch overhead。

测量前没有额外 warmup。但是模型加载、task graph 生成和 NVCC 编译会在计时区间之前完成，这些不属于 `mpk()` GPU 执行时间。

## 自检与故障排查

1. 首先执行 `--dry-run`，检查 21 个 cell 的模型/case 筛选。
2. 用 `Qwen3-0.6B + mbr1_mbt1` 做 smoke test。
3. 确认 `metrics.json` 中 `actual_num_requests=1`、`prompt_len=1`、`generated_tokens=512`、`measured_runs=5`。
4. 确认 `nsys/profile.nsys-rep` 和 `nsys/profile.sqlite` 都存在且非空。
5. 如果 nsys 阶段提示 kernel metadata 不兼容，检查 measure 和 nsys 是否使用同一 Mirage commit、Python 版本、模型和 case。
6. `Qwen3-30B-A3B` 占用显存较大，运行前应确保 H100 上没有其他进程占用 GPU。
