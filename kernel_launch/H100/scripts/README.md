# scripts/ 脚本说明

本目录包含 H100 kernel launch 实验的全部脚本。所有脚本以项目根目录（`profiling/`）
为工作目录运行；`run_all.sh` 会自动切换到该目录。

## 运行入口

```bash
bash kernel_launch/H100/scripts/run_all.sh [过滤参数...]
```

## 各脚本功能

| 脚本 | 作用 |
|------|------|
| `config.py` | 集中定义模型 `MODELS`（Qwen3 4 稠密 + Qwen3-30B-A3B + Qwen3.5-0.8B/2B/9B/27B）、执行模式、Attention 后端、`CASES`(含 `batch_size`/`prompt_len`/`decode_len` 的 14 个用例)、结果目录，以及组合遍历 `combos()` / 命名 `run_tag()`。 |
| `worker.py` | 单次采集的被测进程。构造 `batch_size` 条相同 prompt,启动 SGLang `Engine`（FlashInfer，按模式开/关 CUDA Graph，关闭 radix cache），执行 **1 次 warmup + 1 次测量** 的 `generate()`，用 `cudaProfilerStart/Stop` 括住测量调用(配合 nsys capture-range,只录被测那一次)。 |
| `run_profiling.py` | 采集编排器。对每个组合：① `nsys profile`(`--capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node`) 运行 `worker.py`(传入 `--batch-size`)；② `nsys export` 导出 SQLite；③ 调 `analyze_nsys.py` 解析指标 JSON；④ 将 `.nsys-rep` 与 `.sqlite` **都压缩**进 `results/nsys/` 并删原始大文件。已有指标 JSON 的组合自动跳过（可重入）。 |
| `analyze_nsys.py` | 离线分析。采集范围已限定在被测 generate,直接统计库内全部：**gpu_busy(区间并集)、gpu_bubble_ratio(核心)**、`unhidden_launch_api_ms`、`other_host_idle_ms`、kernel 数、launch 类 API 次数/开销、launch 占比(诊断)、Top-5 kernel;e2e 取主机 API 时间跨度。 |
| `summarize.py` | 汇总。读取全部 `results/metrics/*.json`，生成 `results/metrics/summary.csv` 与 `results/README.md`（每模型指标表 + **eager→cudagraph 对比表(Δbubble/加速)** + 各组合 Top-5 kernel）。 |
| `run_all.sh` | 一键驱动：按 `--models` 过滤下载被选模型（无过滤时下载全部配置模型）→ 跑采集 → 汇总。环境变量 `SGL_PY` 可指定 SGLang 解释器（默认 `~/envs/sgl_env/bin/python`）。 |
| `remote_bootstrap.sh` | 裸实例引导：缺失时按 `envs/envs.md` 建 `sgl_env`，验证 SGLang 可导入，再调 `run_all.sh`。 |

## 关键设计点

- **warmup / 测量分离**：CUDA Graph 捕获、JIT、autotune 都放在 warmup；测量只跑一次干净的前向。
- **关闭 prefix cache**：`disable_radix_cache=True`，避免测量阶段命中 warmup 的前缀缓存。
- **cudaProfilerApi 限定采集范围**：`worker.py` 用 `cudaProfilerStart/Stop` 括住被测 generate,
  nsys 用 `--capture-range=cudaProfilerApi --capture-range-end=stop` 只录这一段。`cudaProfilerStop`
  会在 SGLang 子进程仍存活时强制 CUPTI flush,从而可靠捕获子进程/CUDA graph/大 MoE 的 kernel
  (解决早期"0 kernel"问题),分析时也无需再按时间窗口过滤。
- **CUDA graph 内核记录**：`--cuda-graph-trace=node` 记录 graph 内每个 kernel 节点,否则 cudagraph
  的 kernel 数会被少算(默认只记一次 graph launch)。
- **nsys 采集参数**：`--trace=cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi
  --capture-range-end=stop --sample=none --cpuctxsw=none`。
- **batch size**：`worker.py --batch-size N` 把同一 prompt 复制 N 份组 batch(BS>1 只用于 decode 用例)。
- **`d0` 用例的实际 decode 数**：`bs1_p*_d0` 是历史 case id；`worker.py` 会执行
  `max_new_tokens=max(decode_len, 1)`，所以这些用例实际是 prefill + 1 个 decode token。
  小 prompt 下 CUDA Graph 仍有收益，是因为这 1 个 decode token 及固定 shape 的 graph-eligible
  路径占比高；长 prompt 下 prefill 主体计算占比上升，收益被摊薄。
- **核心指标 gpu_bubble_ratio**：`(e2e − 并集 gpu_busy)/e2e`,比 launch 占比更能反映真实的 launch/host 影响。
- **bubble 拆分**：`unhidden_launch_api_ms` 是 launch API 与 GPU idle 的交集时长；
  `other_host_idle_ms = gpu_bubble_ms − unhidden_launch_api_ms`,用于区分裸 launch API 和
  scheduler/sampling/sync 等其他 host idle。
- **相对路径**：模型路径形如 `models/Qwen3-8B`，结果落在 `kernel_launch/H100/results/`。
- **结果提交策略**：`results/metrics/*.json`、`summary.csv` 可提交；`results/nsys/` 与 `logs/`
  本地保留但不提交 GitHub。

## 常用过滤参数（透传给 `run_profiling.py`）

```bash
--models Qwen3-0.6B Qwen3-8B       # 仅指定模型
--modes eager                     # 仅指定模式
--cases bs1_p8k_d0 bs16_p16_d512  # 仅指定用例
```
