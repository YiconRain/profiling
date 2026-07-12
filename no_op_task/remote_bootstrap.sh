#!/usr/bin/env bash
# Prepare a user-provided Vast.ai H100 instance and start the experiment.
# This script never creates or destroys Vast.ai instances.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
PYTHON_BIN="${MPK_PYTHON:-python}"
MIRAGE_ROOT="${MIRAGE_ROOT:-$ROOT/../mirage}"

command -v nvidia-smi >/dev/null
command -v nsys >/dev/null
command -v git >/dev/null
"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

if [ ! -d "$MIRAGE_ROOT/.git" ]; then
  echo "[bootstrap] cloning YiconRain/mirage into $MIRAGE_ROOT"
  git clone --branch no_op_task git@github.com:YiconRain/mirage.git "$MIRAGE_ROOT" || \
    git clone --branch no_op_task https://github.com/YiconRain/mirage.git "$MIRAGE_ROOT"
else
  echo "[bootstrap] using existing Mirage checkout at $MIRAGE_ROOT"
  git -C "$MIRAGE_ROOT" fetch origin no_op_task
  git -C "$MIRAGE_ROOT" switch no_op_task
  git -C "$MIRAGE_ROOT" pull --ff-only origin no_op_task
fi

echo "[bootstrap] installing Mirage editable"
"$PYTHON_BIN" -m pip install -e "$MIRAGE_ROOT" -v

export MIRAGE_ROOT
export MPK_PYTHON="$PYTHON_BIN"
exec bash no_op_task/run_all.sh "$@"
