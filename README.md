# profiling

推理系统 profiling 实验仓库。实验在 vast.ai 的 GPU 实例上运行，仓库通过 GitHub 同步，
脚本一律使用**相对路径**（以本仓库根目录 `profiling/` 为工作目录）。

## 目录

| 目录 | 说明 |
|------|------|
| `envs/` | 环境准备与模型下载。`envs.md` 记录 MPK / vLLM / SGLang 三套独立环境的安装方式；`download_models.py` 下载 Qwen3 / Qwen3.5 模型。 |
| `kernel_launch/H100/` | H100 上的 kernel launch 开销实验（SGLang + FlashInfer，Qwen3.5 系列）。详见其 `README.md`。 |

## 快速开始（kernel_launch 实验）

在已按 `envs/envs.md` 建好 `~/envs/sgl_env` 且 `nsys` 在 `PATH` 上的 H100 实例上：

```bash
# 下载模型（默认下载整个 Qwen3.5 系列）
~/envs/sgl_env/bin/python envs/download_models.py --series qwen3.5

# 一键运行 kernel launch 实验（下载 -> 采集 -> 汇总）
bash kernel_launch/H100/scripts/run_all.sh
```

## 模型下载脚本用法

```bash
python envs/download_models.py --series qwen3.5      # 整个 Qwen3.5 系列（默认）
python envs/download_models.py --series qwen3        # 整个 Qwen3 系列（含 Qwen3-30B-A3B）
python envs/download_models.py --series all          # 两个系列全部
python envs/download_models.py --model Qwen3-30B-A3B # 仅下载单个模型
python envs/download_models.py --models Qwen3.5-4B Qwen3.5-2B   # 指定若干个
```

模型默认下载到 `models/`（本仓库根目录下）。kernel_launch 实验使用 5 个 Qwen3.5
稠密模型 + Qwen3-30B-A3B（MoE），详见 `kernel_launch/H100/README.md`。
