"""Shared configuration for the H100 kernel-launch profiling experiments.

All paths are RELATIVE to the project root (the `profiling/` directory). Every
script is expected to be invoked with the project root as the current working
directory (run_all.sh guarantees this).
"""

from __future__ import annotations

# Target models that fit on a single 80GB H100. Alias -> local model directory
# (relative to project root); fetched by envs/download_models.py.
# Dense: 0.8B..27B. MoE: Qwen3-30B-A3B (30B total / 3B active) -- the smallest
# MoE that fits (the Qwen3.5 35B MoE is ~70GB and does not), used for the
# MoE-vs-dense launch comparison.
MODELS = {
    "Qwen3.5-0.8B": "models/Qwen3.5-0.8B",
    "Qwen3.5-2B": "models/Qwen3.5-2B",
    "Qwen3.5-4B": "models/Qwen3.5-4B",
    "Qwen3.5-9B": "models/Qwen3.5-9B",
    "Qwen3.5-27B": "models/Qwen3.5-27B",
    "Qwen3-30B-A3B": "models/Qwen3-30B-A3B",
}

# Execution modes. "eager" disables CUDA graphs; "cudagraph" keeps them on.
MODES = ["eager", "cudagraph"]

# Attention backend under test.
ATTENTION_BACKEND = "flashinfer"

# Test cases. Each case fixes a (prompt_len, decode_len) pair at batch size 1.
# decode_len == 0 means prefill-only: the worker still emits exactly one token
# (the minimum a decoder produces), so the trace captures the prefill kernels.
# The original numbering (no case 4) from the experiment brief is preserved.
CASES = [
    {"id": "case1", "prompt_len": 256, "decode_len": 0},
    {"id": "case2", "prompt_len": 1024, "decode_len": 0},
    {"id": "case3", "prompt_len": 8192, "decode_len": 0},
    {"id": "case5", "prompt_len": 16, "decode_len": 128},
    {"id": "case6", "prompt_len": 16, "decode_len": 512},
    {"id": "case7", "prompt_len": 16, "decode_len": 1024},
]

BATCH_SIZE = 1

# NVTX range name that marks the single profiled generate() call. Analysis
# counts only the kernels/launches that fall inside this time window.
MEASURE_RANGE = "measure"

# Result / working directories (relative to project root).
RESULTS_DIR = "kernel_launch/H100/results"
NSYS_DIR = "kernel_launch/H100/results/nsys"       # compressed .nsys-rep + .sqlite
METRICS_DIR = "kernel_launch/H100/results/metrics"  # per-run parsed metrics (json)
WORK_DIR = "kernel_launch/H100/results/_work"       # transient raw rep/sqlite
LOGS_DIR = "kernel_launch/H100/logs"


def combos():
    """Yield (model_alias, model_path, mode, case) for every experiment cell."""
    for model_alias, model_path in MODELS.items():
        for mode in MODES:
            for case in CASES:
                yield model_alias, model_path, mode, case


def run_tag(model_alias: str, mode: str, case_id: str) -> str:
    """Stable identifier used for filenames, e.g. Qwen3.5-9B__eager__case3."""
    return f"{model_alias}__{mode}__{case_id}"
