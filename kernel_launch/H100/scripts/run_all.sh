#!/usr/bin/env bash
# End-to-end driver for the H100 kernel-launch experiments.
#
# Steps: download Qwen3 models -> profile every (model, mode, case) cell
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

# 1. Models: the Qwen3 series (dense 0.6B/1.7B/8B/14B + MoE Qwen3-30B-A3B).
"$PY" envs/download_models.py --series qwen3

# 2. Profiling sweep (extra args act as filters).
"$PY" kernel_launch/H100/scripts/run_profiling.py --python "$PY" "$@"

# 3. Aggregate metrics -> CSV + Chinese README.
"$PY" kernel_launch/H100/scripts/summarize.py

echo "[run_all] Done. Results under kernel_launch/H100/results/."
