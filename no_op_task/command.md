# MPK 超编译实验执行命令

本文档给出一台全新 Vast.ai 单卡 H100 实例上，从 clone repo、安装环境、预下载模型、运行 smoke test 到执行完整实验的可复制命令。

建议准备至少 250 GB 可用 SSD。下面假设大容量 SSD 挂载在 `/workspace`；如果实际挂载点不同，请替换 `/workspace`。

## 1. 检查 Vast.ai 实例

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
nvcc --version
nsys --version
python --version
df -h /workspace
```

GPU 名称必须包含 `H100`，并且 `nsys` 必须在 `PATH` 中。

## 2. 创建 workspace 并 clone 两个 repo

```bash
export WORK=/workspace/no_op_task_work
mkdir -p "$WORK"
cd "$WORK"

git clone --branch no_op_task https://github.com/YiconRain/profiling.git
git clone --branch no_op_task https://github.com/YiconRain/mirage.git
```

检查 profiling repo：

```bash
git -C "$WORK/profiling" branch --show-current
git -C "$WORK/profiling" log -1 --oneline
git -C "$WORK/profiling" status --short
```

当前分支应为：

```text
no_op_task
```

检查 YiconRain Mirage repo：

```bash
git -C "$WORK/mirage" branch --show-current
git -C "$WORK/mirage" rev-parse HEAD
git -C "$WORK/mirage" status --short
```

Mirage 的预期 commit 为：

```text
cec749afa1e2e089b68bfa2c921924ed7e9d4ac2
```

可以强制检查：

```bash
test "$(git -C "$WORK/mirage" rev-parse HEAD)" = \
  "cec749afa1e2e089b68bfa2c921924ed7e9d4ac2"
```

## 3. 创建 Python 环境

建议保留 NVIDIA PyTorch 镜像已安装的 CUDA PyTorch：

```bash
cd "$WORK"
python -m venv --system-site-packages .venv
source "$WORK/.venv/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install ninja
```

检查 PyTorch 和 GPU：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
assert "H100" in torch.cuda.get_device_name(0)
PY
```

## 4. 配置路径和 cache

将模型 cache 放在 profiling 结果目录之外，避免打包实验结果时同时打包 100 GB 以上的模型。

```bash
mkdir -p "$WORK/hf_cache" "$WORK/pip_cache" "$WORK/tmp"

export CUDA_VISIBLE_DEVICES=0
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export HF_HOME="$WORK/hf_cache"
export HF_HUB_CACHE="$WORK/hf_cache/hub"
export PIP_CACHE_DIR="$WORK/pip_cache"
export TMPDIR="$WORK/tmp"
export MIRAGE_ROOT="$WORK/mirage"
export MPK_PYTHON="$WORK/.venv/bin/python"
export TOKENIZERS_PARALLELISM=false
```

重新登录 SSH 或创建新 tmux shell 后，需要重新执行上述 `export`。

## 5. 安装 YiconRain Mirage

Mirage `setup.py` 会在 build 期间 import 部分依赖，因此先安装 requirements：

```bash
python -m pip install -r "$WORK/mirage/requirements.txt"
python -m pip install safetensors sentencepiece

cd "$WORK/mirage"
python -m pip install -e . --no-deps -v
```

检查实际 import 的 Mirage 路径：

```bash
python - <<'PY'
import inspect
import mirage

print(inspect.getfile(mirage))
PY
```

输出应指向：

```text
/workspace/no_op_task_work/mirage/python/mirage/...
```

检查 `no_op_task` 分支中的两处实验支持修复：

```bash
grep -n 'self.init_request_func = getattr(mod, "init_request_func")' \
  "$WORK/mirage/python/mirage/mpk/persistent_kernel.py"

grep -n 'meta_tensors.append(self.meta_tensors\["paged_kv_indices_snapshot"\])' \
  "$WORK/mirage/python/mirage/mpk/persistent_kernel.py"

grep -n 'assert self.value_cache.shape == self.key_cache.shape' \
  "$WORK/mirage/demo/qwen3/models/modeling_qwen3.py"
```

第一个 `grep` 应输出两行，后两个应各输出一行。第三个检查确保 Qwen3 demo 不再将 KV cache shape 写死为 `16 × 4096`。

## 6. 预下载全部模型

Qwen3 模型通常可以直接下载。如果 Hugging Face 要求认证，先执行：

```bash
python -c "from huggingface_hub import login; login()"
```

下载五个完整 snapshot：

```bash
cd "$WORK"
python - <<'PY'
from huggingface_hub import snapshot_download

models = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B",
]

for index, model in enumerate(models, start=1):
    print(f"[{index}/{len(models)}] Downloading {model}", flush=True)
    path = snapshot_download(repo_id=model)
    print(f"Cached at {path}", flush=True)
PY
```

检查 cache 大小和剩余 SSD：

```bash
du -sh "$HF_HOME"
df -h "$WORK"
```

## 7. 检查 21 个 cell 的命令矩阵

此命令不加载模型，不运行 GPU workload，只打印所有命令：

```bash
cd "$WORK/profiling"
bash no_op_task/run_all.sh --dry-run
```

开头应显示：

```text
[plan] 21 cells
```

## 8. 运行专用 smoke test

Smoke test 会执行完整的最小 cell：

- 模型：`Qwen3-0.6B`
- 真实 request 数：1
- `max_num_batched_requests=1`
- `max_num_batched_tokens=1`
- Prompt length：1
- Decode length：512
- Warmup：0
- CUDA Event 测量：5 次
- nsys：额外捕获 1 次
- 自动校验 metrics、编译产物、`.nsys-rep` 和 `.sqlite`

运行：

```bash
cd "$WORK/profiling"
bash no_op_task/smoke_test.sh
```

强制删除已有 smoke cell 并重新运行：

```bash
bash no_op_task/smoke_test.sh --force
```

如果只想排查 MPK 测量，暂时不跑 nsys：

```bash
bash no_op_task/smoke_test.sh --force --skip-nsys
```

完整 smoke test 成功时必须显示：

```text
Smoke test passed.
[smoke] PASS
```

## 9. 检查 smoke test 结果

```bash
cat no_op_task/artifacts/Qwen3-0.6B/mbr1_mbt1/metrics.json

ls -lh no_op_task/artifacts/Qwen3-0.6B/mbr1_mbt1/compile
ls -lh no_op_task/artifacts/Qwen3-0.6B/mbr1_mbt1/nsys

nsys stats \
  no_op_task/artifacts/Qwen3-0.6B/mbr1_mbt1/nsys/profile.nsys-rep
```

`metrics.json` 中应满足：

```text
actual_num_requests = 1
prompt_len = 1
decode_len = 512
generated_tokens = 512
warmup_runs = 0
measured_runs = 5
len(mpk_gpu_ms) = 5
```

## 10. 运行完整实验

建议使用 tmux：

```bash
tmux new -s no_op_task
```

在 tmux 中重新配置环境：

```bash
export WORK=/workspace/no_op_task_work
source "$WORK/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=0
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export HF_HOME="$WORK/hf_cache"
export HF_HUB_CACHE="$WORK/hf_cache/hub"
export PIP_CACHE_DIR="$WORK/pip_cache"
export TMPDIR="$WORK/tmp"
export MIRAGE_ROOT="$WORK/mirage"
export MPK_PYTHON="$WORK/.venv/bin/python"
export TOKENIZERS_PARALLELISM=false

cd "$WORK/profiling"
bash no_op_task/run_all.sh
```

已成功的 smoke cell 会被自动跳过。

从 tmux detach：

```text
Ctrl-b d
```

重新进入：

```bash
tmux attach -t no_op_task
```

## 11. 监控进度和断点续跑

```bash
find no_op_task/artifacts -name metrics.json | wc -l
find no_op_task/artifacts -name '*.nsys-rep' | wc -l
find no_op_task/artifacts -name '*.sqlite' | wc -l
du -sh no_op_task/artifacts
df -h "$WORK"
```

完整实验的三个数量都应为 21。

如果中途中断，直接重新执行：

```bash
bash no_op_task/run_all.sh
```

脚本会跳过已有完整产物的 cell。

先跑完全部延迟测量，之后再补 nsys：

```bash
bash no_op_task/run_all.sh --skip-nsys
bash no_op_task/run_all.sh --only-nsys
```

## 12. 最终完整性检查

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("no_op_task/artifacts")
metrics = sorted(root.glob("*/*/metrics.json"))
reports = sorted(root.glob("*/*/nsys/profile.nsys-rep"))
sqlites = sorted(root.glob("*/*/nsys/profile.sqlite"))

assert len(metrics) == 21, len(metrics)
assert len(reports) == 21, len(reports)
assert len(sqlites) == 21, len(sqlites)

for path in metrics:
    data = json.loads(path.read_text())
    assert data["actual_num_requests"] == 1, path
    assert data["prompt_len"] == 1, path
    assert data["decode_len"] == 512, path
    assert data["generated_tokens"] == 512, path
    assert data["warmup_runs"] == 0, path
    assert data["measured_runs"] == 5, path
    assert len(data["mpk_gpu_ms"]) == 5, path

for path in reports + sqlites:
    assert path.stat().st_size > 0, path

print("All 21 experiment cells passed integrity checks.")
PY
```

## 13. 打包结果

Model cache 位于 `$WORK/hf_cache`，不会进入下面的结果压缩包。

```bash
cd "$WORK/profiling"
tar -czf "$WORK/no_op_task_artifacts.tar.gz" no_op_task/artifacts
ls -lh "$WORK/no_op_task_artifacts.tar.gz"
```

从本地机器下载：

```bash
scp -P YOUR_VAST_PORT \
  root@YOUR_VAST_HOST:/workspace/no_op_task_work/no_op_task_artifacts.tar.gz \
  .
```
