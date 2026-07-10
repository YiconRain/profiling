# profiling

推理系统 profiling 实验仓库。实验在 vast.ai 的 GPU 实例上运行，仓库通过 GitHub 同步，
脚本一律使用**相对路径**（以本仓库根目录 `profiling/` 为工作目录）。

## 目录

| 目录 | 说明 |
|------|------|
| `assets/figs/` | blog / report 使用的图片资产，统一提交到 GitHub。 |
| `envs/` | 环境准备与模型下载。`envs.md` 记录 MPK / vLLM / SGLang 三套独立环境的安装方式；`download_models.py` 下载 Qwen3 / Qwen3.5 模型。 |
| `kernel_launch/H100/` | H100 上的 kernel launch 开销实验（SGLang + FlashInfer，当前实验用 Qwen3 + Qwen3.5 补充模型）。详见其 `README.md`。 |
| `kernel_launch/blog_kernel_launch.md` | kernel launch / host overhead 分析 blog。 |

> 说明:第一次正式实验(Qwen3.5 系列)的数据已用 git tag **`experiment-1`** 归档;当前 main 为
> Experiment 2(Qwen3 系列 + Qwen3.5-0.8B/2B/9B/27B 补充模型 + batch 扫描 + GPU bubble 指标)。

## 快速开始（kernel_launch 实验）

在已按 `envs/envs.md` 建好 `~/envs/sgl_env` 且 `nsys` 在 `PATH` 上的 H100 实例上：

```bash
# 一键运行(内部按 --models 过滤下载模型；无过滤时下载全部实验模型 -> 采集 -> 汇总)
bash kernel_launch/H100/scripts/run_all.sh
```

## 模型下载脚本用法

```bash
python envs/download_models.py --series qwen3        # 整个 Qwen3 系列（当前实验用:0.6/1.7/8/14B + 30B-A3B）
python envs/download_models.py --series qwen3.5      # 当前实验用 Qwen3.5:0.8/2/9/27B
python envs/download_models.py --series all          # 两个系列全部
python envs/download_models.py --model Qwen3-30B-A3B # 仅下载单个模型
python envs/download_models.py --models Qwen3-8B Qwen3-14B   # 指定若干个
```

模型默认下载到 `models/`（本仓库根目录下）。当前 kernel_launch 实验使用 4 个 Qwen3 稠密
模型（0.6/1.7/8/14B）+ Qwen3-30B-A3B（MoE）+ Qwen3.5 稠密补充模型（0.8/2/9/27B），
详见 `kernel_launch/H100/README.md`。

## 结果提交策略

- `kernel_launch/H100/results/metrics/*.json`、`summary.csv`、`results/README.md` 可以提交到 GitHub。
- `kernel_launch/H100/results/nsys/` 下的 `.nsys-rep.gz` / `.sqlite.gz`、`kernel_launch/H100/logs/` 不提交。
