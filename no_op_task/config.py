"""Configuration for the MPK over-compilation decode experiment.

The runtime workload is always one real request with one prompt token followed
by 512 generated tokens.  Only the compile-time capacity knobs vary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


QWEN3_SPECS = {
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
}

SMALL_MODELS = {"Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-8B"}

PROMPT_LEN = 1
DECODE_LEN = 512
MAX_SEQ_LENGTH = PROMPT_LEN + DECODE_LEN
ACTUAL_NUM_REQUESTS = 1
MEASURE_REPEATS = 5
PAGE_SIZE = 64
MAX_NUM_PAGES = 16


@dataclass(frozen=True)
class Case:
    max_num_batched_requests: int
    max_num_batched_tokens: int

    @property
    def id(self) -> str:
        return (
            f"mbr{self.max_num_batched_requests}_"
            f"mbt{self.max_num_batched_tokens}"
        )


CASES = (
    Case(1, 1),
    Case(1, 8),
    Case(8, 8),
    Case(8, 16),
    Case(16, 16),
)


def cases_for_model(model_alias: str) -> tuple[Case, ...]:
    """Return the supported cases for one model.

    Models larger than Qwen3-8B intentionally skip every MBT=16 case.
    """

    if model_alias in SMALL_MODELS:
        return CASES
    return tuple(case for case in CASES if case.max_num_batched_tokens < 16)


def iter_cells(
    selected_models: set[str] | None = None,
    selected_cases: set[str] | None = None,
) -> Iterator[tuple[str, str, Case]]:
    """Yield every selected (model alias, Hugging Face id, case) cell."""

    for model_alias, model_id in QWEN3_SPECS.items():
        if selected_models and model_alias not in selected_models:
            continue
        for case in cases_for_model(model_alias):
            if selected_cases and case.id not in selected_cases:
                continue
            yield model_alias, model_id, case


def demo_path(mirage_root: Path, model_alias: str) -> Path:
    """Return the H100 demo used to build the requested MPK graph."""

    if model_alias == "Qwen3-30B-A3B":
        return mirage_root / "demo/qwen3/demo_30B_A3B_hopper.py"
    return mirage_root / "demo/qwen3/demo_hopper.py"
