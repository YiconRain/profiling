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
| `config.py` | 集中定义模型 `MODELS`（5 个 Qwen3.5 稠密 + Qwen3-30B-A3B）、执行模式、Attention 后端、测试用例、结果目录，以及组合遍历 `combos()` / 命名 `run_tag()`。其它脚本统一从这里读取配置。 |
| `worker.py` | 单次采集的被测进程。构造指定长度输入 token，启动 SGLang `Engine`（FlashInfer，按模式开/关 CUDA Graph，关闭 radix cache），执行 **1 次 warmup + 1 次测量** 的 `generate()`，用 `cudaProfilerStart/Stop` 括住测量调用(配合 nsys 的 capture-range,只录被测那一次)。 |
| `run_profiling.py` | 采集编排器。对每个组合：① `nsys profile`(`--capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node`) 运行 `worker.py`；② `nsys export` 导出 SQLite；③ 调 `analyze_nsys.py` 解析指标 JSON；④ 将 `.nsys-rep` 与 `.sqlite` 压缩进 `results/nsys/` 并删原始大文件。已有指标 JSON 的组合自动跳过（可重入）。 |
| `analyze_nsys.py` | 离线分析。因采集范围已限定在被测 generate,直接统计库内全部：kernel 总数、kernel GPU 总时间、launch 类 API（`cudaLaunchKernel*` / `cudaGraphLaunch*` / `cuLaunchKernel*`）次数与累计开销、launch 占比、Top-5 kernel;e2e 取主机 API 事件的时间跨度。 |
| `summarize.py` | 汇总。读取全部 `results/metrics/*.json`，生成 `results/metrics/summary.csv` 与 `results/README.md`（中文表格 + 各组合 Top-5 kernel）。 |
| `run_all.sh` | 一键驱动：下载模型（Qwen3.5 稠密 + Qwen3-30B-A3B）→ 跑采集 → 汇总。环境变量 `SGL_PY` 可指定 SGLang 解释器（默认 `~/envs/sgl_env/bin/python`）。 |
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
- **相对路径**：模型路径形如 `models/Qwen3.5-27B`，结果落在 `kernel_launch/H100/results/`。

## 常用过滤参数（透传给 `run_profiling.py`）

```bash
--models Qwen3.5-0.8B Qwen3.5-2B   # 仅指定模型
--modes eager                      # 仅指定模式
--cases case1 case5                # 仅指定用例
```
