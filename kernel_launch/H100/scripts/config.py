"""Shared configuration for the H100 kernel-launch profiling experiments.

All paths are RELATIVE to the project root (the `profiling/` directory). Every
script is expected to be invoked with the project root as the current working
directory (run_all.sh guarantees this).

Experiment 2 scope: Qwen3 + selected Qwen3.5 models, SGLang, eager + cudagraph, FlashInfer.
Workloads:
  * BS=1 pure prefill  : prompt in {16, 256, 1k, 4k, 8k}, decode 0
  * BS=1 pure decode   : prompt 16, decode in {128, 512}
  * BS in {4,8,16}     : pure decode, prompt 16, decode in {128, 512}
"""

from __future__ import annotations

# Target models. Alias -> local model directory (relative to project root);
# fetched by envs/download_models.py.
# Qwen3 dense: 0.6B/1.7B/8B/14B. MoE: Qwen3-30B-A3B (30B total / 3B active).
# Qwen3.5 supplement: dense 0.8B/2B/9B/27B.
MODELS = {
    "Qwen3-0.6B": "models/Qwen3-0.6B",
    "Qwen3-1.7B": "models/Qwen3-1.7B",
    "Qwen3-8B": "models/Qwen3-8B",
    "Qwen3-14B": "models/Qwen3-14B",
    "Qwen3-30B-A3B": "models/Qwen3-30B-A3B",
    "Qwen3.5-0.8B": "models/Qwen3.5-0.8B",
    "Qwen3.5-2B": "models/Qwen3.5-2B",
    "Qwen3.5-9B": "models/Qwen3.5-9B",
    "Qwen3.5-27B": "models/Qwen3.5-27B",
}

# Execution modes. "eager" disables CUDA graphs; "cudagraph" keeps them on.
MODES = ["eager", "cudagraph"]

# Attention backend under test.
ATTENTION_BACKEND = "flashinfer"

# Test cases. Each fixes (batch_size, prompt_len, decode_len).
# decode_len == 0 means prefill-only: the worker still emits exactly one token
# (the minimum a decoder produces), so the trace captures the prefill kernels.
CASES = [
    # BS=1, pure prefill (decode 0)
    {"id": "bs1_p16_d0", "batch_size": 1, "prompt_len": 16, "decode_len": 0},
    {"id": "bs1_p256_d0", "batch_size": 1, "prompt_len": 256, "decode_len": 0},
    {"id": "bs1_p1k_d0", "batch_size": 1, "prompt_len": 1024, "decode_len": 0},
    {"id": "bs1_p4k_d0", "batch_size": 1, "prompt_len": 4096, "decode_len": 0},
    {"id": "bs1_p8k_d0", "batch_size": 1, "prompt_len": 8192, "decode_len": 0},
    # BS=1, pure decode (prompt 16)
    {"id": "bs1_p16_d128", "batch_size": 1, "prompt_len": 16, "decode_len": 128},
    {"id": "bs1_p16_d512", "batch_size": 1, "prompt_len": 16, "decode_len": 512},
    # BS=4/8/16, pure decode (prompt 16) -- batch-size sweep
    {"id": "bs4_p16_d128", "batch_size": 4, "prompt_len": 16, "decode_len": 128},
    {"id": "bs4_p16_d512", "batch_size": 4, "prompt_len": 16, "decode_len": 512},
    {"id": "bs8_p16_d128", "batch_size": 8, "prompt_len": 16, "decode_len": 128},
    {"id": "bs8_p16_d512", "batch_size": 8, "prompt_len": 16, "decode_len": 512},
    {"id": "bs16_p16_d128", "batch_size": 16, "prompt_len": 16, "decode_len": 128},
    {"id": "bs16_p16_d512", "batch_size": 16, "prompt_len": 16, "decode_len": 512},
]

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
    """Stable identifier used for filenames, e.g. Qwen3-8B__eager__bs1_p8k_d0."""
    return f"{model_alias}__{mode}__{case_id}"
