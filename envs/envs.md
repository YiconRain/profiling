# 环境准备

本目录下的全部实验涉及三套推理系统，在 A100/H100 配置上使用同样的三个 Qwen3 模型
（Qwen3-0.6B / 1.7B / 8B）。三套系统各用独立环境，互不污染：

| 系统 | 角色 | 环境 |
|------|------|------|
| **MPK**（Mirage persistent-kernel）| MegaKernel 被测对象 | uv venv + 源码 editable 安装 |
| **vLLM** 0.23.0（FlashInfer 后端）| 基线 / 对照 | 预编译 wheel + 独立 venv |
| **SGLang** 0.5.12（FlashInfer 后端）| 基线 / 对照 | 预编译 wheel + 独立 venv |

> 哪个实验用哪套环境，见各实验目录下的 README。简单说：
> `01_motivation/` 与各实验的 `baselines/` 用 vLLM / SGLang 环境；
> `02_prefill_decode/{A100,H100}/exp*/mpk/` 与 `03_mpk_internals/` 用 MPK 环境。

---

## 一、MPK（Mirage）安装

MPK 也用独立的 uv venv。源码仍然需要 editable 安装，并切到带本实验补丁的分支。

```bash
mkdir -p ~/envs
cd ~/envs

uv venv mpk_env --python 3.12
source ~/envs/mpk_env/bin/activate

git clone --recursive --branch exp_prefill_decode git@github.com:YiconRain/mirage.git
cd mirage

uv pip install --upgrade pip
uv pip install -e . -v
export MIRAGE_HOME=$(pwd)
```

后续运行 MPK 实验时，继续使用 `~/envs/mpk_env` 里的解释器，并保证 `MIRAGE_HOME`
指向这个 mirage 源码目录。

模型下载脚本：`02_prefill_decode/common/download_models.sh`（三个子实验共用，
把 Qwen3-0.6B / 1.7B / 8B 拉进 HF cache）。

---

## 二、vLLM / SGLang 安装

用**预编译 release wheel**，不要 editable 源码构建。release wheel 自带与其代码、
与它拉进来的 torch 匹配的编译扩展（`.so`），因此没有源码构建、不需要 Rust 工具链、
也不会出现源码 / 二进制不匹配的问题。

这套栈针对 **CUDA 13**。vLLM 0.23.0 钉死 `torch==2.11.0`，而 torch 2.11.0 自带
CUDA 13（`nvidia-*-cu13`）。NVIDIA PyTorch CUDA 13 镜像（如
`nvcr.io/nvidia/pytorch:26.01-py3`）是天然匹配；`--torch-backend=auto` 能干净地
解析出 cu130 构建。

两个环境要**串行安装**，不要并行——每个都要拉好几 GB，两个大下载叠在一起能把
50 GB 磁盘塞满。每个框架用各自的 venv，依赖集永不冲突。

### 为什么不用源码 editable（之前踩过的坑）

- vLLM：在 v0.23.0 源码树上跑 `VLLM_USE_PRECOMPILED=1 uv pip install -e .`，
  没有匹配的 cu130 预编译 wheel，于是它悄悄退回到一个 cu130 *nightly* wheel，
  其扩展与源码不匹配（`import vllm._C` 永久失败）。直接装 `vllm==0.23.0` wheel
  自洽。
- SGLang：v0.5.12 源码构建需要 Rust 工具链（`srt.grpc._core` 扩展），且源码里
  未钉版本的 `kernels` 会解析到与 `transformers==5.6.0` 不兼容的版本。wheel 自带
  编译好的扩展和一致的依赖集。

### vLLM env

```bash
mkdir -p ~/envs
cd ~/envs

uv venv vllm_env --python 3.12
source ~/envs/vllm_env/bin/activate

# --torch-backend=auto 选出与本机 CUDA runtime 匹配的 torch 构建
# （CUDA 13 镜像上即 cu130）。vLLM wheel 自带编译好的扩展。
uv pip install --upgrade pip
uv pip install "vllm==0.23.0" --torch-backend=auto

deactivate
```

### SGLang env

```bash
cd ~/envs

uv venv sgl_env --python 3.12
source ~/envs/sgl_env/bin/activate

uv pip install --upgrade pip
# sglang 0.5.12 依赖 flash-attn-4，它只发布为预发布版（>=4.0.0b9），
# 所以必须加 --prerelease=allow，否则 resolve 会以 "No solution found" 失败。
# 离线 Engine 推理用普通 `sglang` 包即可，不需要 `[all]` 扩展。
uv pip install --prerelease=allow "sglang==0.5.12"

# sglang 0.5.12 会拉进 transformers 5.6.0，它要求 kernels<0.13。解析器可能反而
# 选了更新的 kernels（如 0.15.x），其 LayerRepository API 会在
# `from sglang import Engine` 时破坏 transformers 导入。钉回去。
uv pip install "kernels>=0.12,<0.13"

deactivate
```

### 验证

```bash
# vLLM
~/envs/vllm_env/bin/python -c "import torch; from vllm import LLM, SamplingParams; print('vllm ok', torch.__version__, torch.cuda.is_available())"

# SGLang
~/envs/sgl_env/bin/python -c "import torch; from sglang import Engine; print('sglang ok', torch.__version__, torch.cuda.is_available())"
```

### 注意

- profiling 的 Phase B 需要 `nsys` 在 `PATH` 上。NVIDIA PyTorch 镜像把它放在
  `/usr/local/cuda/bin`。
- 磁盘：一次装一个 env。空间紧张时 `uv cache clean` 可释放 uv 下载缓存
  （可能涨到 10 GB 以上）。
- 用各 env 的解释器直接跑 profiling 脚本，例如
  `~/envs/vllm_env/bin/python 01_motivation/kernel_launch/A100/run_vllm_profiling.py ...`，
  让 worker 子进程继承正确的环境。
