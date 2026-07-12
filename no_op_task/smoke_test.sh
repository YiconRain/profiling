#!/usr/bin/env bash
# Run and validate the smallest complete MPK experiment cell on one H100.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export PATH="/usr/local/cuda/bin:$PATH"
PYTHON_BIN="${MPK_PYTHON:-python}"
MIRAGE_ROOT="${MIRAGE_ROOT:-$ROOT/../mirage}"
ARTIFACTS_DIR="${NO_OP_TASK_ARTIFACTS_DIR:-$ROOT/no_op_task/artifacts}"

SKIP_NSYS=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --skip-nsys)
      SKIP_NSYS=1
      ;;
    --force)
      FORCE=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: bash no_op_task/smoke_test.sh [--force] [--skip-nsys]" >&2
      exit 2
      ;;
  esac
done

RUN_ARGS=(
  --models Qwen3-0.6B
  --cases mbr1_mbt1
  --artifacts-dir "$ARTIFACTS_DIR"
)
if [[ "$SKIP_NSYS" == "1" ]]; then
  RUN_ARGS+=(--skip-nsys)
fi
if [[ "$FORCE" == "1" ]]; then
  RUN_ARGS+=(--force)
fi

echo "[smoke] project root  : $ROOT"
echo "[smoke] Mirage root  : $MIRAGE_ROOT"
echo "[smoke] Python       : $PYTHON_BIN"
echo "[smoke] artifacts    : $ARTIFACTS_DIR"
echo "[smoke] nsys enabled : $((1 - SKIP_NSYS))"

bash no_op_task/run_all.sh "${RUN_ARGS[@]}"

export NO_OP_TASK_ARTIFACTS_DIR="$ARTIFACTS_DIR"
export NO_OP_TASK_EXPECT_NSYS="$((1 - SKIP_NSYS))"
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
from pathlib import Path


root = Path(os.environ["NO_OP_TASK_ARTIFACTS_DIR"])
cell = root / "Qwen3-0.6B" / "mbr1_mbt1"
metrics_path = cell / "metrics.json"
compile_dir = cell / "compile"

assert metrics_path.is_file() and metrics_path.stat().st_size > 0, metrics_path
metrics = json.loads(metrics_path.read_text())

assert metrics["mode"] == "measure", metrics
assert metrics["model_alias"] == "Qwen3-0.6B", metrics
assert metrics["actual_num_requests"] == 1, metrics
assert metrics["max_num_batched_requests"] == 1, metrics
assert metrics["max_num_batched_tokens"] == 1, metrics
assert metrics["prompt_len"] == 1, metrics
assert metrics["decode_len"] == 512, metrics
assert metrics["generated_tokens"] == 512, metrics
assert metrics["warmup_runs"] == 0, metrics
assert metrics["measured_runs"] == 5, metrics
assert metrics["compile_mode"] == "compile", metrics
assert metrics["num_tasks"] and metrics["num_tasks"] > 0, metrics
assert metrics["num_events"] and metrics["num_events"] > 0, metrics

runs = metrics["mpk_gpu_ms"]
assert len(runs) == 5, runs
assert all(isinstance(value, (int, float)) and value > 0 for value in runs), runs
assert math.isclose(
    metrics["average_mpk_gpu_ms"],
    statistics.fmean(runs),
    rel_tol=1e-9,
    abs_tol=1e-9,
), metrics

required_compile_files = [
    compile_dir / "test_rank0.cu",
    compile_dir / "task_graph_rank0.json",
    compile_dir / "kernel_metadata_rank0.json",
]
for path in required_compile_files:
    assert path.is_file() and path.stat().st_size > 0, path

launchers = list(compile_dir.glob("mpk_launcher_rank0.cpython-*.so"))
assert len(launchers) == 1 and launchers[0].stat().st_size > 0, launchers

if os.environ["NO_OP_TASK_EXPECT_NSYS"] == "1":
    nsys_dir = cell / "nsys"
    rep = nsys_dir / "profile.nsys-rep"
    sqlite = nsys_dir / "profile.sqlite"
    profile_json = nsys_dir / "profile_run.json"
    for path in (rep, sqlite, profile_json):
        assert path.is_file() and path.stat().st_size > 0, path

    profile = json.loads(profile_json.read_text())
    assert profile["mode"] == "nsys", profile
    assert profile["compile_mode"] == "load", profile
    assert profile["actual_num_requests"] == 1, profile
    assert profile["prompt_len"] == 1, profile
    assert profile["generated_tokens"] == 512, profile
    assert profile["measured_runs"] == 1, profile
    assert profile["mirage_revision"] == metrics["mirage_revision"], profile

    with sqlite3.connect(sqlite) as connection:
        kernel_table = connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL'"
        ).fetchone()
        assert kernel_table is not None, "nsys SQLite has no kernel table"
        kernel_count = connection.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
        assert kernel_count > 0, "nsys captured zero GPU kernels"

print("Smoke test passed.")
print("Five MPK GPU runs (ms):", runs)
print("Average MPK GPU latency (ms):", metrics["average_mpk_gpu_ms"])
print("Artifacts:", cell)
PY

echo "[smoke] PASS"
