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
| `config.py` | 集中定义全量模型 `MODELS`（5 个 Qwen3.5 稠密 + Qwen3-30B-A3B）、减层外推模型 `EXTRAP_MODELS`（Qwen3.5-122B/397B，含 `profile_layers`）、执行模式、Attention 后端、测试用例、结果目录，以及组合遍历 `combos(include_extrap)` / 命名 `run_tag()`。 |
| `worker.py` | 单次采集的被测进程。构造指定长度输入 token，启动 SGLang `Engine`（FlashInfer，按模式开/关 CUDA Graph，关闭 radix cache），执行 **1 次 warmup + 1 次测量** 的 `generate()`，用 NVTX range `measure` 框住测量调用。`--num-layers K` 可用 `json_model_override_args` 只加载前 K 层（大 MoE 减层采集）。 |
| `run_profiling.py` | 采集编排器。对每个组合：① `nsys profile` 运行 `worker.py`；② `nsys export` 导出 SQLite；③ 调 `analyze_nsys.py` 解析指标 JSON；④ 将 `.nsys-rep` 与 `.sqlite` 压缩进 `results/nsys/` 并删原始大文件。已有指标 JSON 的组合自动跳过（可重入）。`--extrap` 纳入减层外推模型。 |
| `analyze_nsys.py` | 离线分析。定位 NVTX `measure` 窗口，按时间窗口统计：kernel 总数、kernel GPU 总时间、launch 类 API（`cudaLaunchKernel*` / `cudaGraphLaunch*` / `cuLaunchKernel*`）次数与累计开销、launch 占比、Top-5 kernel；减层运行时另按全层数外推。 |
| `summarize.py` | 汇总。读取全部 `results/metrics/*.json`，生成 `results/metrics/summary.csv` 与 `results/README.md`（中文表格 + 各组合 Top-5 kernel + 减层外推估计表）。 |
| `run_all.sh` | 一键驱动：下载模型（Qwen3.5 稠密 + Qwen3-30B-A3B；`--extrap` 时再部分下载 122B/397B 的前 4 层）→ 跑采集 → 汇总。环境变量 `SGL_PY` 可指定 SGLang 解释器（默认 `~/envs/sgl_env/bin/python`）。 |
| `remote_bootstrap.sh` | 裸实例引导：缺失时按 `envs/envs.md` 建 `sgl_env`，验证 SGLang 可导入，再调 `run_all.sh`。 |

## 关键设计点

- **warmup / 测量分离**：CUDA Graph 捕获、JIT、autotune 都放在 warmup；测量只跑一次干净的前向。
- **关闭 prefix cache**：`disable_radix_cache=True`，避免测量阶段命中 warmup 的前缀缓存。
- **跨进程时间窗口过滤**：SGLang 前向在子进程执行，分析按 nsys 统一时间线上 `measure`
  窗口的时间范围过滤 kernel/launch，与线程归属无关。
- **nsys 采集参数**：`--trace=cuda,nvtx --sample=none --cpuctxsw=none`，聚焦 CUDA 与 NVTX，
  减小 trace 体积。
- **相对路径**：模型路径形如 `models/Qwen3.5-27B`，结果落在 `kernel_launch/H100/results/`。
- **减层外推**：大 MoE 装不下时只加载前 K 层实测，再按 `全层数 / K` 线性外推（一阶近似，
  略高估）。见上级 `README.md` 3.4 节。

## 常用过滤参数（透传给 `run_profiling.py`）

```bash
--models Qwen3.5-0.8B Qwen3.5-2B   # 仅指定模型
--modes eager                      # 仅指定模式
--cases case1 case5                # 仅指定用例
--extrap                           # 额外纳入大 MoE 的减层外推（122B/397B）
```
