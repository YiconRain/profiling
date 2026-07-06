#!/usr/bin/env bash
# Bring a fresh vast.ai H100 instance from a bare git clone to full results.
#
# Assumes: repo already cloned on the instance, running on an NVIDIA PyTorch
# image (nsys in /usr/local/cuda/bin). Builds the SGLang venv if missing
# (per envs/envs.md), then runs the full experiment sweep.
#
# Usage (on the instance, from the repo root):
#   bash kernel_launch/H100/scripts/remote_bootstrap.sh [run_all filters...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"

SGL_ENV="${SGL_ENV:-$HOME/envs/sgl_env}"

if [ ! -x "$SGL_ENV/bin/python" ]; then
  echo "[bootstrap] building SGLang env at $SGL_ENV"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv "$SGL_ENV" --python 3.12
  uv pip install --python "$SGL_ENV/bin/python" --upgrade pip
  # sglang 0.5.12 needs flash-attn-4 prerelease; pin kernels for transformers 5.6.0.
  uv pip install --python "$SGL_ENV/bin/python" --prerelease=allow "sglang==0.5.12"
  uv pip install --python "$SGL_ENV/bin/python" "kernels>=0.12,<0.13"
fi

echo "[bootstrap] verifying SGLang import"
"$SGL_ENV/bin/python" -c "import torch; from sglang import Engine; print('sglang ok', torch.__version__, torch.cuda.is_available())"

export SGL_PY="$SGL_ENV/bin/python"
bash kernel_launch/H100/scripts/run_all.sh "$@"
