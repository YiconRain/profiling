#!/usr/bin/env bash
# Prepare the Mirage main branch and an isolated MPK Python environment.
#
# Usage from the profiling repository root:
#   bash task_graph/setup_mirage.sh [mirage_checkout]
set -euo pipefail

MIRAGE_DIR="${1:-$HOME/envs/mirage}"
MPK_ENV="${MPK_ENV:-$HOME/envs/mpk_env}"
MIRAGE_REMOTE="git@github.com:YiconRain/mirage.git"

mkdir -p "$(dirname "$MIRAGE_DIR")" "$(dirname "$MPK_ENV")"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$MIRAGE_DIR/.git" ]; then
  git clone --recursive --branch main "$MIRAGE_REMOTE" "$MIRAGE_DIR"
else
  current_remote="$(git -C "$MIRAGE_DIR" remote get-url origin)"
  if [ "$current_remote" != "$MIRAGE_REMOTE" ]; then
    echo "error: $MIRAGE_DIR origin is $current_remote, expected $MIRAGE_REMOTE" >&2
    exit 1
  fi
  if [ -n "$(git -C "$MIRAGE_DIR" status --porcelain)" ]; then
    echo "error: $MIRAGE_DIR has local changes; clean or preserve them before updating main" >&2
    exit 1
  fi
  git -C "$MIRAGE_DIR" switch main
  git -C "$MIRAGE_DIR" pull --ff-only origin main
  git -C "$MIRAGE_DIR" submodule update --init --recursive
fi

if [ ! -x "$MPK_ENV/bin/python" ]; then
  uv venv "$MPK_ENV" --python 3.12
fi

uv pip install --python "$MPK_ENV/bin/python" --upgrade pip
uv pip install --python "$MPK_ENV/bin/python" -e "$MIRAGE_DIR" -v

echo "Mirage commit: $(git -C "$MIRAGE_DIR" rev-parse HEAD)"
echo "MPK Python: $MPK_ENV/bin/python"
echo "Run:"
echo "  MIRAGE_HOME=$MIRAGE_DIR $MPK_ENV/bin/python task_graph/run_task_graph.py"
