#!/usr/bin/env bash
# End-to-end driver for the H100 kernel-launch experiments.
#
# Steps: download requested models -> profile every (model, mode, case) cell
# under nsys -> aggregate metrics. Safe to re-run: completed cells are skipped.
#
# Requirements on the instance:
#   * SGLang venv at ~/envs/sgl_env (see envs/envs.md), or set SGL_PY.
#   * nsys on PATH (NVIDIA PyTorch image ships it in /usr/local/cuda/bin).
#
# Usage (from anywhere):  bash kernel_launch/H100/scripts/run_all.sh
# Optional filters are forwarded to run_profiling.py, e.g.:
#   bash .../run_all.sh --models Qwen3-8B --modes eager --cases bs1_p8k_d0
set -euo pipefail

# Resolve project root = three levels up from this script (scripts/H100/kernel_launch/..).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"

export PATH="/usr/local/cuda/bin:$PATH"
PY="${SGL_PY:-$HOME/envs/sgl_env/bin/python}"

echo "[run_all] project root : $ROOT"
echo "[run_all] python        : $PY"
echo "[run_all] nsys          : $(command -v nsys || echo 'NOT FOUND')"

extract_requested_models() {
  local collect=0
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "--models" ]]; then
      collect=1
      continue
    fi
    if [[ "$arg" == --models=* ]]; then
      printf '%s\n' "${arg#--models=}"
      collect=0
      continue
    fi
    if [[ "$arg" == --* ]]; then
      collect=0
      continue
    fi
    if [[ "$collect" == 1 ]]; then
      printf '%s\n' "$arg"
    fi
  done
}

REQUESTED_MODELS=()
while IFS= read -r model; do
  REQUESTED_MODELS+=("$model")
done < <(extract_requested_models "$@")

# 1. Models. If --models is present, download only those aliases; otherwise
# download the full configured experiment set (Qwen3 + selected Qwen3.5).
if (( ${#REQUESTED_MODELS[@]} )); then
  "$PY" envs/download_models.py --models "${REQUESTED_MODELS[@]}"
else
  "$PY" envs/download_models.py --series all
fi

# 2. Profiling sweep (extra args act as filters).
"$PY" kernel_launch/H100/scripts/run_profiling.py --python "$PY" "$@"

# 3. Aggregate metrics -> CSV + Chinese README.
"$PY" kernel_launch/H100/scripts/summarize.py

echo "[run_all] Done. Results under kernel_launch/H100/results/."
