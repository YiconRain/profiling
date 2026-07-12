#!/usr/bin/env bash
# Run the complete MPK over-compilation experiment from any working directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export PATH="/usr/local/cuda/bin:$PATH"
PYTHON_BIN="${MPK_PYTHON:-python}"
MIRAGE_ROOT="${MIRAGE_ROOT:-$ROOT/../mirage}"

exec "$PYTHON_BIN" no_op_task/run_experiments.py \
  --python "$PYTHON_BIN" \
  --mirage-root "$MIRAGE_ROOT" \
  "$@"
