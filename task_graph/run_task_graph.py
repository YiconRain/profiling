#!/usr/bin/env python3
"""Extract and analyze Mirage MPK task graphs for Qwen3 models on H100.

The public entry point orchestrates one subprocess per prompt-length scenario so
that CUDA/model memory is released between cases.  A worker invokes Mirage's
current H100 Qwen3 demo, forces an exact synthetic prompt length, enables
split-KV, and analyzes the emitted task_graph_0.json file.

Dense Qwen3 models use Mirage's existing --split-kv-cache path.  The upstream
Qwen3-30B-A3B H100 demo currently calls the non-split paged-attention API, so
this script installs a narrow in-process adapter that replaces that single API
call with Mirage's existing split-KV and merge layers.  Mirage source files are
not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import runpy
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


QWEN3_SPECS = {
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
}

DEFAULT_MODEL = "Qwen3-8B"
DEFAULT_PROMPT_LENGTHS = (8, 16, 1024, 8192)
DEFAULT_DECODE_LENGTH = 128
DEFAULT_MBT = 8
DEFAULT_BATCH_SIZE = 1
DEFAULT_PAGE_SIZE = 4096
DENSE_QWEN3_PAGE_SIZE = 4096
DENSE_QWEN3_MAX_NUM_PAGES = 16

LAYER_RE = re.compile(r"(?:^|_)layer_(\d+)(?:_|$)")
TASK_ENUM_RE = re.compile(r"enum\s+TaskType\s*\{(.*?)\};", re.DOTALL)
EVENT_ENUM_RE = re.compile(r"enum\s+EventType\s*\{(.*?)\};", re.DOTALL)
ENUM_ITEM_RE = re.compile(r"\b([A-Z][A-Z0-9_]*)\b(?:\s*=\s*([^,]+))?")


class GraphOnlyComplete(RuntimeError):
    """Internal sentinel used to stop the Mirage demo before NVCC/runtime."""


@dataclass(frozen=True)
class Scenario:
    prompt_len: int
    decode_len: int
    mbt: int
    batch_size: int
    page_size: int
    max_num_pages: int

    @property
    def max_seq_length(self) -> int:
        return self.prompt_len + self.decode_len


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_mirage_dir() -> Path:
    if os.environ.get("MIRAGE_HOME"):
        return Path(os.environ["MIRAGE_HOME"]).expanduser().resolve()
    return Path.home() / "envs" / "mirage"


def parse_length(value: str) -> int:
    """Parse token counts such as 8, 1024, 1k, or 8K."""
    text = value.strip().lower().replace("_", "")
    multiplier = 1
    if text.endswith("k"):
        multiplier = 1024
        text = text[:-1]
    try:
        parsed = int(text) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid token length: {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Token lengths must be positive")
    return parsed


def length_label(length: int) -> str:
    return (
        f"{length // 1024}k" if length >= 1024 and length % 1024 == 0 else str(length)
    )


def run_checked(command: Sequence[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def git_value(repo: Path, *args: str) -> str:
    try:
        return run_checked(["git", "-C", str(repo), *args])
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def validate_mirage_checkout(mirage_dir: Path, allow_non_main: bool) -> dict[str, str]:
    required = [
        mirage_dir / "demo" / "qwen3" / "demo_hopper.py",
        mirage_dir / "demo" / "qwen3" / "demo_30B_A3B_hopper.py",
        mirage_dir / "python" / "mirage" / "mpk" / "persistent_kernel.py",
        mirage_dir / "include" / "mirage" / "persistent_kernel" / "runtime_header.h",
        mirage_dir / "scripts" / "parse_task_graph.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Mirage checkout is incomplete. Missing:\n  "
            + "\n  ".join(missing)
            + "\nRun task_graph/setup_mirage.sh first."
        )

    branch = git_value(mirage_dir, "branch", "--show-current")
    commit = git_value(mirage_dir, "rev-parse", "HEAD")
    remote = git_value(mirage_dir, "remote", "get-url", "origin")
    if not allow_non_main and branch != "main":
        raise RuntimeError(
            f"Mirage must be on main (found {branch!r} at {commit}). "
            "Use --allow-mirage-non-main only for an explicitly reviewed compatibility branch."
        )
    return {"branch": branch, "commit": commit, "remote": remote}


def resolve_model(model: str, model_path: str | None) -> tuple[str, str]:
    if model_path:
        path = str(Path(model_path).expanduser().resolve())
        if not Path(path).exists():
            raise FileNotFoundError(f"--model-path does not exist: {path}")
        return model, path
    if model in QWEN3_SPECS:
        return model, QWEN3_SPECS[model]
    if model in QWEN3_SPECS.values():
        alias = next(alias for alias, repo in QWEN3_SPECS.items() if repo == model)
        return alias, model
    raise ValueError(
        f"Unsupported model {model!r}. Choose one of: {', '.join(QWEN3_SPECS)}"
    )


def auto_max_num_pages(max_seq_length: int, batch_size: int, page_size: int) -> int:
    # One extra page per request avoids a boundary allocation failure at the
    # final decode step while keeping KV-cache allocation modest.
    pages_per_request = math.ceil(max_seq_length / page_size) + 1
    return pages_per_request * batch_size


def resolve_kv_cache_geometry(
    *,
    is_moe: bool,
    max_seq_length: int,
    batch_size: int,
    requested_page_size: int,
    requested_max_num_pages: int | None,
) -> tuple[int, int, str]:
    """Resolve cache geometry, including Mirage main's dense-demo constraint.

    demo/qwen3/models/modeling_qwen3.py accepts max_num_pages/page_size in its
    constructor but currently asserts (16, 4096) inside every attention layer.
    Use that required geometry for dense models until the upstream assertion is
    made dynamic.  The separate MoE demo has no such hard-coded assertion.
    """
    if not is_moe:
        if requested_page_size != DENSE_QWEN3_PAGE_SIZE:
            raise ValueError(
                "Mirage main's dense Qwen3 H100 model currently requires "
                f"--page-size {DENSE_QWEN3_PAGE_SIZE}; got {requested_page_size}"
            )
        if (
            requested_max_num_pages is not None
            and requested_max_num_pages != DENSE_QWEN3_MAX_NUM_PAGES
        ):
            raise ValueError(
                "Mirage main's dense Qwen3 H100 model currently requires "
                f"--max-num-pages {DENSE_QWEN3_MAX_NUM_PAGES}; got "
                f"{requested_max_num_pages}"
            )
        return (
            DENSE_QWEN3_PAGE_SIZE,
            DENSE_QWEN3_MAX_NUM_PAGES,
            "mirage_main_dense_fixed_16x4096",
        )

    max_num_pages = requested_max_num_pages or auto_max_num_pages(
        max_seq_length, batch_size, requested_page_size
    )
    return requested_page_size, max_num_pages, "automatic"


def validate_scenario(scenario: Scenario) -> None:
    if scenario.mbt < scenario.batch_size:
        raise ValueError(
            f"mbt ({scenario.mbt}) must be >= max_num_batched_requests "
            f"({scenario.batch_size}) so every decode iteration can schedule one token per request"
        )
    required_pages = (
        math.ceil(scenario.max_seq_length / scenario.page_size) * scenario.batch_size
    )
    if scenario.max_num_pages < required_pages:
        raise ValueError(
            f"max_num_pages={scenario.max_num_pages} is too small; at least {required_pages} "
            f"pages are required for BS={scenario.batch_size}, max_seq_length={scenario.max_seq_length}, "
            f"page_size={scenario.page_size}"
        )


def parse_enum(header_text: str, enum_re: re.Pattern[str]) -> dict[int, str]:
    match = enum_re.search(header_text)
    if not match:
        raise ValueError("Could not find the requested enum in runtime_header.h")
    body = re.sub(
        r"//.*?$|/\*.*?\*/", "", match.group(1), flags=re.MULTILINE | re.DOTALL
    )
    result: dict[int, str] = {}
    current = -1
    names_to_values: dict[str, int] = {}
    for raw_item in body.split(","):
        item = raw_item.strip()
        if not item:
            continue
        item_match = ENUM_ITEM_RE.match(item)
        if not item_match:
            continue
        name, expression = item_match.groups()
        if expression is None:
            current += 1
        else:
            expression = expression.strip()
            if re.fullmatch(r"0[xX][0-9a-fA-F]+|\d+", expression):
                current = int(expression, 0)
            elif expression in names_to_values:
                current = names_to_values[expression]
            else:
                raise ValueError(
                    f"Unsupported enum expression for {name}: {expression}"
                )
        names_to_values[name] = current
        result[current] = name
    return result


def task_tensors(task: dict[str, Any], side: str) -> list[dict[str, Any]]:
    value = task.get(side)
    return value if isinstance(value, list) else []


def tensor_names(task: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for side in ("inputs", "outputs"):
        for tensor in task_tensors(task, side):
            base_ptr = tensor.get("base_ptr")
            if isinstance(base_ptr, str):
                names.append(base_ptr)
    return names


def explicit_layer(task: dict[str, Any]) -> str | None:
    names = tensor_names(task)
    layers = {
        int(match.group(1)) for name in names for match in LAYER_RE.finditer(name)
    }
    if layers:
        return f"layer_{min(layers)}"
    joined = " ".join(names).lower()
    if any(
        name in joined for name in ("model_norm", "lm_head", "argmax", "output_token")
    ):
        return "model_output"
    if any(name in joined for name in ("embed_tokens", "input_token", "embed_out")):
        return "model_input"
    return None


def infer_layers(tasks: list[dict[str, Any]], task_names: dict[int, str]) -> list[str]:
    layers: list[str | None] = [explicit_layer(task) for task in tasks]
    previous: list[str | None] = []
    current: str | None = None
    for layer in layers:
        if layer is not None:
            current = layer
        previous.append(current)

    following: list[str | None] = [None] * len(tasks)
    current = None
    for index in range(len(tasks) - 1, -1, -1):
        if layers[index] is not None:
            current = layers[index]
        following[index] = current

    inferred: list[str] = []
    for index, task in enumerate(tasks):
        if layers[index] is not None:
            inferred.append(layers[index] or "unassigned")
            continue
        task_name = task_names.get(int(task.get("task_type", -1)), "")
        if task_name in {"TASK_TERMINATE", "TASK_BEGIN_TASK_GRAPH"}:
            inferred.append("runtime")
        elif "EMBEDDING" in task_name:
            inferred.append("model_input")
        elif any(token in task_name for token in ("ARGMAX", "SAMPLING")):
            inferred.append("model_output")
        elif previous[index] == following[index] and previous[index] is not None:
            inferred.append(previous[index] or "unassigned")
        elif previous[index] and previous[index].startswith("layer_"):
            # Boundary-local operations such as split-KV merge and SiLU do not
            # necessarily carry a layer-specific tensor name.  Graph order
            # places them after another task from the same transformer layer.
            inferred.append(previous[index] or "unassigned")
        elif following[index] is not None:
            inferred.append(following[index] or "unassigned")
        else:
            inferred.append("unassigned")
    return inferred


def friendly_operation(enum_name: str) -> str:
    name = enum_name.removeprefix("TASK_")
    for suffix in ("_HOPPER", "_SM90", "_SM100"):
        name = name.removesuffix(suffix)
    replacements = {
        "BEGIN_TASK_GRAPH": "Runtime begin",
        "TERMINATE": "Runtime terminate",
        "RMS_NORM": "RMSNorm",
        "RMS_NORM_LINEAR": "RMSNorm + Linear",
        "PAGED_ATTENTION_SPLIT_KV_MERGE": "Split-KV merge",
        "PAGED_ATTENTION_SPLIT_KV": "Split-KV attention",
        "PAGED_ATTENTION": "Paged attention",
        "LINEAR_WITH_RESIDUAL": "Linear + residual",
        "SILU_MUL": "SiLU multiply",
        "MOE_TOPK_SOFTMAX": "MoE Top-K softmax",
        "MOE_MUL_SUM_ADD": "MoE weighted sum + residual",
        "MOE_W13_LINEAR": "MoE W1/W3 linear",
        "MOE_W2_LINEAR": "MoE W2 linear",
        "ARGMAX_PARTIAL": "Argmax partial",
        "ARGMAX_REDUCE": "Argmax reduce",
    }
    if name in replacements:
        return replacements[name]
    return name.replace("_", " ").title()


def tensor_tile_signature(tensor: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tensor.get("base_ptr", "?"),
        tuple(tensor.get("dims", [])),
        tuple(tensor.get("strides", [])),
        tensor.get("data_type"),
    )


def task_tile_signature(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(tensor_tile_signature(item) for item in task_tensors(task, "inputs")),
        tuple(tensor_tile_signature(item) for item in task_tensors(task, "outputs")),
    )


def summarize_tensor_group(
    task_group: list[dict[str, Any]], side: str, tensor_index: int
) -> dict[str, Any]:
    tensors = [task_tensors(task, side)[tensor_index] for task in task_group]
    first = tensors[0]
    offsets = sorted({int(tensor.get("offset", 0)) for tensor in tensors})
    return {
        "base_ptr": first.get("base_ptr", "?"),
        "tile_dims": first.get("dims", []),
        "strides": first.get("strides", []),
        "data_type": first.get("data_type"),
        "unique_offsets": len(offsets),
        "offset_min_bytes": offsets[0] if offsets else 0,
        "offset_max_bytes": offsets[-1] if offsets else 0,
        "placement": "replicated"
        if len(offsets) == 1 and len(task_group) > 1
        else "split",
    }


def summarize_tile_group(task_group: list[dict[str, Any]]) -> dict[str, Any]:
    first = task_group[0]
    return {
        "task_count": len(task_group),
        "inputs": [
            summarize_tensor_group(task_group, "inputs", index)
            for index in range(len(task_tensors(first, "inputs")))
        ],
        "outputs": [
            summarize_tensor_group(task_group, "outputs", index)
            for index in range(len(task_tensors(first, "outputs")))
        ],
    }


def layer_sort_key(layer: str) -> tuple[int, int | str]:
    if layer == "runtime":
        return (0, 0)
    if layer == "model_input":
        return (1, 0)
    match = re.fullmatch(r"layer_(\d+)", layer)
    if match:
        return (2, int(match.group(1)))
    if layer == "model_output":
        return (3, 0)
    return (4, layer)


def analyze_graph(
    graph: dict[str, Any],
    header_text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    tasks = list(graph.get("all_tasks", []))
    events = list(graph.get("all_events", []))
    first_tasks = list(graph.get("first_tasks", []))
    task_names = parse_enum(header_text, TASK_ENUM_RE)
    event_names = parse_enum(header_text, EVENT_ENUM_RE)
    layers = infer_layers(tasks, task_names)

    runtime_task_types = {0, 10}
    compute_tasks = [
        task
        for task in tasks
        if int(task.get("task_type", -1)) not in runtime_task_types
    ]
    split_task_count = sum(
        1
        for task in compute_tasks
        if "PAGED_ATTENTION_SPLIT_KV"
        in task_names.get(int(task.get("task_type", -1)), "")
        and "MERGE" not in task_names.get(int(task.get("task_type", -1)), "")
    )
    merge_task_count = sum(
        1
        for task in compute_tasks
        if "PAGED_ATTENTION_SPLIT_KV_MERGE"
        in task_names.get(int(task.get("task_type", -1)), "")
    )
    if split_task_count == 0 or merge_task_count == 0:
        present = sorted(
            {
                task_names.get(int(task.get("task_type", -1)), "UNKNOWN")
                for task in tasks
            }
        )
        raise RuntimeError(
            "Split-KV validation failed: the graph must contain both split attention and merge tasks. "
            f"Present task types: {present}"
        )

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for layer, task in zip(layers, tasks):
        key = (layer, int(task.get("task_type", -1)), int(task.get("variant_id", 0)))
        grouped[key].append(task)

    layer_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (layer, task_type, variant_id), task_group in grouped.items():
        tile_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for task in task_group:
            tile_groups[task_tile_signature(task)].append(task)
        enum_name = task_names.get(task_type, f"TASK_TYPE_{task_type}")
        layer_details[layer].append(
            {
                "operation": friendly_operation(enum_name),
                "task_type": task_type,
                "task_type_name": enum_name,
                "variant_id": variant_id,
                "task_count": len(task_group),
                "tile_groups": [
                    summarize_tile_group(group) for group in tile_groups.values()
                ],
            }
        )

    for operations in layer_details.values():
        operations.sort(key=lambda item: (item["task_type"], item["variant_id"]))

    scenario = metadata["scenario"]
    prefill_iterations = math.ceil(
        scenario["prompt_len"] * scenario["max_num_batched_requests"] / scenario["mbt"]
    )
    decode_iterations = scenario["decode_len"]
    estimated_iterations = prefill_iterations + decode_iterations

    type_counts = Counter(int(task.get("task_type", -1)) for task in tasks)
    event_type_counts = Counter(int(event.get("event_type", -1)) for event in events)
    return {
        "metadata": metadata,
        "statistics": {
            "total_tasks_including_runtime": len(tasks),
            "compute_tasks_per_graph_iteration": len(compute_tasks),
            "runtime_control_tasks": len(tasks) - len(compute_tasks),
            "total_events_per_graph_iteration": len(events),
            "first_task_count": len(first_tasks),
            "split_kv_task_count": split_task_count,
            "split_kv_merge_task_count": merge_task_count,
            "estimated_prefill_iterations": prefill_iterations,
            "decode_iterations": decode_iterations,
            "estimated_graph_iterations": estimated_iterations,
            "estimated_compute_task_dispatches": len(compute_tasks)
            * estimated_iterations,
            "estimated_event_dispatches": len(events) * estimated_iterations,
            "estimate_note": (
                "Dispatch estimates assume equal prompt lengths, prefill packs up to mbt tokens per "
                "iteration across the batch, and decode schedules one token per request per iteration."
            ),
        },
        "task_type_counts": [
            {
                "task_type": task_type,
                "task_type_name": task_names.get(task_type, f"TASK_TYPE_{task_type}"),
                "operation": friendly_operation(
                    task_names.get(task_type, f"TASK_TYPE_{task_type}")
                ),
                "count": count,
            }
            for task_type, count in sorted(type_counts.items())
        ],
        "event_type_counts": [
            {
                "event_type": event_type,
                "event_type_name": event_names.get(
                    event_type, f"EVENT_TYPE_{event_type}"
                ),
                "count": count,
            }
            for event_type, count in sorted(event_type_counts.items())
        ],
        "layers": {
            layer: layer_details[layer]
            for layer in sorted(layer_details, key=layer_sort_key)
        },
    }


def format_dims(dims: Iterable[Any]) -> str:
    values = list(dims)
    return "x".join(str(value) for value in values) if values else "scalar"


def format_tensor_tile(tensor: dict[str, Any]) -> str:
    return (
        f"{tensor['base_ptr']} tile={format_dims(tensor['tile_dims'])} "
        f"strides={tensor['strides']} offsets={tensor['unique_offsets']} "
        f"({tensor['placement']}, bytes {tensor['offset_min_bytes']}..{tensor['offset_max_bytes']})"
    )


def render_text_report(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    scenario = metadata["scenario"]
    stats = report["statistics"]
    lines = [
        "Mirage MPK Qwen3 Task Graph Report",
        "=" * 36,
        f"Model: {metadata['model_alias']} ({metadata['model_ref']})",
        f"Mirage: {metadata['mirage']['branch']} @ {metadata['mirage']['commit']}",
        f"GPU: {metadata['gpu']}",
        f"Mode: {metadata['execution_mode']}",
        f"Prompt/decode: {scenario['prompt_len']} / {scenario['decode_len']} tokens",
        f"mbt / max_num_batched_requests: {scenario['mbt']} / {scenario['max_num_batched_requests']}",
        f"max_seq_length / page_size / max_num_pages: "
        f"{scenario['max_seq_length']} / {scenario['page_size']} / {scenario['max_num_pages']}",
        "Split-KV: enabled and validated",
        "",
        "Global statistics",
        "-----------------",
        f"Total tasks (including runtime controls): {stats['total_tasks_including_runtime']}",
        f"Compute tasks per graph iteration: {stats['compute_tasks_per_graph_iteration']}",
        f"Runtime control tasks: {stats['runtime_control_tasks']}",
        f"Total events per graph iteration: {stats['total_events_per_graph_iteration']}",
        f"First tasks: {stats['first_task_count']}",
        f"Split-KV attention tasks: {stats['split_kv_task_count']}",
        f"Split-KV merge tasks: {stats['split_kv_merge_task_count']}",
        f"Estimated prefill graph iterations: {stats['estimated_prefill_iterations']}",
        f"Decode graph iterations: {stats['decode_iterations']}",
        f"Estimated compute task dispatches (prefill + decode): "
        f"{stats['estimated_compute_task_dispatches']}",
        f"Estimated event dispatches (prefill + decode): {stats['estimated_event_dispatches']}",
        f"Estimate definition: {stats['estimate_note']}",
        "",
        "Task-type totals",
        "----------------",
    ]
    for item in report["task_type_counts"]:
        lines.append(f"{item['task_type_name']} ({item['operation']}): {item['count']}")

    lines.extend(
        [
            "",
            "Per-layer operation and tile details",
            "------------------------------------",
        ]
    )
    for layer, operations in report["layers"].items():
        lines.extend(["", f"[{layer}]"])
        for operation in operations:
            lines.append(
                f"  {operation['operation']} [{operation['task_type_name']}, "
                f"variant={operation['variant_id']}]: tasks={operation['task_count']}"
            )
            for index, tile_group in enumerate(operation["tile_groups"], start=1):
                lines.append(
                    f"    tile-group {index}: tasks={tile_group['task_count']}"
                )
                for tensor in tile_group["inputs"]:
                    lines.append(f"      input:  {format_tensor_tile(tensor)}")
                for tensor in tile_group["outputs"]:
                    lines.append(f"      output: {format_tensor_tile(tensor)}")
    return "\n".join(lines) + "\n"


def write_operation_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer",
                "operation",
                "task_type_name",
                "task_type",
                "variant_id",
                "task_count",
                "tile_group_count",
                "tile_groups_json",
            ],
        )
        writer.writeheader()
        for layer, operations in report["layers"].items():
            for operation in operations:
                writer.writerow(
                    {
                        "layer": layer,
                        "operation": operation["operation"],
                        "task_type_name": operation["task_type_name"],
                        "task_type": operation["task_type"],
                        "variant_id": operation["variant_id"],
                        "task_count": operation["task_count"],
                        "tile_group_count": len(operation["tile_groups"]),
                        "tile_groups_json": json.dumps(
                            operation["tile_groups"], separators=(",", ":")
                        ),
                    }
                )


def write_report_files(report: dict[str, Any], case_dir: Path) -> None:
    report_dir = case_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (report_dir / "report.txt").write_text(render_text_report(report), encoding="utf-8")
    write_operation_csv(report, report_dir / "operations.csv")


def run_canonical_mirage_parser(
    mirage_dir: Path, graph_path: Path, report_dir: Path
) -> str:
    """Run Mirage's canonical raw task-graph parser before richer analysis."""
    parser_path = mirage_dir / "scripts" / "parse_task_graph.py"
    if not parser_path.is_file():
        raise FileNotFoundError(f"Missing canonical Mirage parser: {parser_path}")
    environment = os.environ.copy()
    environment["MIRAGE_HOME"] = str(mirage_dir)
    completed = subprocess.run(
        [sys.executable, str(parser_path), str(graph_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = completed.stdout
    (report_dir / "mirage_summary.txt").write_text(summary, encoding="utf-8")
    return summary


def update_summary(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(output_root.glob("*/report/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metadata = report["metadata"]
        scenario = metadata["scenario"]
        stats = report["statistics"]
        rows.append(
            {
                "case": report_path.parents[1].name,
                "model": metadata["model_alias"],
                "prompt_len": scenario["prompt_len"],
                "decode_len": scenario["decode_len"],
                "mbt": scenario["mbt"],
                "max_num_batched_requests": scenario["max_num_batched_requests"],
                "max_seq_length": scenario["max_seq_length"],
                "execution_mode": metadata["execution_mode"],
                "total_tasks_including_runtime": stats["total_tasks_including_runtime"],
                "compute_tasks_per_graph_iteration": stats[
                    "compute_tasks_per_graph_iteration"
                ],
                "total_events_per_graph_iteration": stats[
                    "total_events_per_graph_iteration"
                ],
                "split_kv_task_count": stats["split_kv_task_count"],
                "split_kv_merge_task_count": stats["split_kv_merge_task_count"],
                "estimated_prefill_iterations": stats["estimated_prefill_iterations"],
                "decode_iterations": stats["decode_iterations"],
                "estimated_compute_task_dispatches": stats[
                    "estimated_compute_task_dispatches"
                ],
                "estimated_event_dispatches": stats["estimated_event_dispatches"],
                "mirage_commit": metadata["mirage"]["commit"],
            }
        )
    if not rows:
        return
    with (output_root / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ExactLengthTokenizer:
    """Delegate tokenizer behavior but synthesize an exact input length."""

    def __init__(self, tokenizer: Any, prompt_len: int):
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch
        from transformers import BatchEncoding

        original = self._tokenizer(*args, **kwargs)
        ids = original["input_ids"][0].detach().cpu().to(torch.long)
        if ids.numel() == 0:
            fallback = self._tokenizer.bos_token_id
            ids = torch.tensor([0 if fallback is None else fallback], dtype=torch.long)
        repeats = math.ceil(self._prompt_len / ids.numel())
        exact = ids.repeat(repeats)[: self._prompt_len].unsqueeze(0).contiguous()
        return BatchEncoding(
            {
                "input_ids": exact,
                "attention_mask": torch.ones_like(exact),
            }
        )


@contextmanager
def patched_tokenizer(prompt_len: int) -> Iterator[None]:
    from transformers import AutoTokenizer

    original = AutoTokenizer.from_pretrained

    def from_pretrained(*args: Any, **kwargs: Any) -> ExactLengthTokenizer:
        return ExactLengthTokenizer(original(*args, **kwargs), prompt_len)

    AutoTokenizer.from_pretrained = from_pretrained
    try:
        yield
    finally:
        AutoTokenizer.from_pretrained = original


@contextmanager
def patched_position_embeddings(max_seq_length: int) -> Iterator[None]:
    """Extend the demos' hard-coded 4096-row RoPE attachment for 8k tests.

    The graph and scheduling experiment does not evaluate model quality.  For
    execution safety beyond row 4096, the existing rows are repeated.  This
    keeps the adapter independent of model internals while avoiding an out-of-
    bounds access in the upstream demo.  Graph-only mode never reads them.
    """
    from mirage.mpk.persistent_kernel import PersistentKernel

    original = PersistentKernel.attach_input

    def attach_input(self: Any, torch_tensor: Any, name: str | None = None) -> Any:
        if (
            name in {"cos_position_embedding", "sin_position_embedding"}
            and torch_tensor.ndim == 2
            and torch_tensor.shape[0] < max_seq_length
        ):
            repeats = math.ceil(max_seq_length / torch_tensor.shape[0])
            torch_tensor = torch_tensor.repeat((repeats, 1))[
                :max_seq_length
            ].contiguous()
        return original(self, torch_tensor=torch_tensor, name=name)

    PersistentKernel.attach_input = attach_input
    try:
        yield
    finally:
        PersistentKernel.attach_input = original


@contextmanager
def patched_moe_split_kv(enabled: bool) -> Iterator[None]:
    """Route the MoE demo's paged-attention call through split-KV on H100."""
    if not enabled:
        yield
        return

    import mirage as mi
    from mirage.mpk.persistent_kernel import PersistentKernel

    original = PersistentKernel.paged_attention_layer
    layer_counter = 0

    def split_kv_attention(
        self: Any,
        input: Any,
        k_cache: Any,
        v_cache: Any,
        q_norm: Any,
        k_norm: Any,
        cos_pos_embed: Any,
        sin_pos_embed: Any,
        output: Any,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ) -> None:
        nonlocal layer_counter
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim
        num_qo_per_kv = num_q_heads // num_kv_heads
        num_kv_chunks = max(1, self.max_seq_length // 256)
        prefix = f"task_graph_layer_{layer_counter}_split_kv"

        lse = self.new_tensor(
            dims=(
                self.max_num_batched_tokens,
                num_kv_chunks * num_qo_per_kv,
                num_kv_heads,
            ),
            strides=(num_kv_chunks * num_q_heads, 1, num_kv_chunks * num_qo_per_kv),
            dtype=mi.float32,
            name=f"{prefix}_lse",
            io_category="cuda_tensor",
        )
        output_tmp = self.new_tensor(
            dims=(
                self.max_num_batched_tokens,
                num_kv_chunks * num_qo_per_kv * head_dim,
                num_kv_heads,
            ),
            strides=(
                # Match demo_hopper.py on Mirage main exactly.  The layout is
                # consumed by the existing split-KV/merge task pair.
                num_kv_chunks * num_q_heads,
                1,
                num_kv_chunks * num_qo_per_kv * head_dim,
            ),
            dtype=mi.bfloat16,
            name=f"{prefix}_output_tmp",
            io_category="cuda_tensor",
        )
        self.paged_attention_split_kv_layer(
            input=input,
            k_cache=k_cache,
            v_cache=v_cache,
            q_norm=q_norm,
            k_norm=k_norm,
            cos_pos_embed=cos_pos_embed,
            sin_pos_embed=sin_pos_embed,
            lse=lse,
            output=output_tmp,
            attention_params=(num_q_heads, num_kv_chunks),
            grid_dim=(self.max_num_batched_requests, num_kv_heads, num_kv_chunks),
            block_dim=block_dim,
        )
        self.paged_attention_split_kv_merge_layer(
            lse=lse,
            output_tmp=output_tmp,
            output=output,
            attention_params=(num_q_heads, head_dim),
            grid_dim=(self.max_num_batched_requests, num_kv_heads, 1),
            block_dim=block_dim,
        )
        layer_counter += 1

    PersistentKernel.paged_attention_layer = split_kv_attention
    try:
        yield
    finally:
        PersistentKernel.paged_attention_layer = original


@contextmanager
def patched_graph_only(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    from mirage.mpk.persistent_kernel import PersistentKernel

    original = PersistentKernel.compile

    def stop_before_compile(self: Any, **kwargs: Any) -> None:
        raise GraphOnlyComplete(
            "Task graph generated; graph-only mode skips NVCC and execution"
        )

    PersistentKernel.compile = stop_before_compile
    try:
        yield
    finally:
        PersistentKernel.compile = original


def gpu_description() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; run this script on the Vast.ai H100 instance"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"This experiment requires exactly one visible GPU; found {torch.cuda.device_count()}"
        )
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (9, 0):
        raise RuntimeError(
            f"This test targets H100 (compute capability 9.0); found {props.name} "
            f"with capability {props.major}.{props.minor}"
        )
    return f"{props.name} (compute capability {props.major}.{props.minor})"


def demo_arguments(
    demo_path: Path,
    model_ref: str,
    scenario: Scenario,
    build_dir: Path,
    is_moe: bool,
) -> list[str]:
    args = [
        str(demo_path),
        "--use-mirage",
        "--model",
        model_ref,
        "--max-num-batched-tokens",
        str(scenario.mbt),
        "--max-num-batched-requests",
        str(scenario.batch_size),
        "--page-size",
        str(scenario.page_size),
        "--max-num-pages",
        str(scenario.max_num_pages),
        "--max-seq-length",
        str(scenario.max_seq_length),
        "--ignore-eos",
        "--output-dir",
        str(build_dir),
    ]
    if not is_moe:
        args.append("--split-kv-cache")
    return args


def run_worker(args: argparse.Namespace) -> int:
    import torch

    mirage_dir = Path(args.mirage_dir).expanduser().resolve()
    mirage_info = validate_mirage_checkout(mirage_dir, args.allow_mirage_non_main)
    model_alias, model_ref = resolve_model(args.model, args.model_path)
    is_moe = model_alias == "Qwen3-30B-A3B"
    prompt_len = int(args._worker_prompt_len)
    max_seq_length = prompt_len + args.decode_len
    page_size, max_num_pages, kv_cache_compatibility = resolve_kv_cache_geometry(
        is_moe=is_moe,
        max_seq_length=max_seq_length,
        batch_size=args.max_num_batched_requests,
        requested_page_size=args.page_size,
        requested_max_num_pages=args.max_num_pages,
    )
    scenario = Scenario(
        prompt_len=prompt_len,
        decode_len=args.decode_len,
        mbt=args.mbt,
        batch_size=args.max_num_batched_requests,
        page_size=page_size,
        max_num_pages=max_num_pages,
    )
    validate_scenario(scenario)
    if not is_moe:
        print(
            "[compat] Mirage main dense Qwen3 requires KV cache geometry "
            f"max_num_pages={DENSE_QWEN3_MAX_NUM_PAGES}, "
            f"page_size={DENSE_QWEN3_PAGE_SIZE}."
        )

    case_dir = Path(args._worker_case_dir).resolve()
    raw_dir = case_dir / "raw"
    build_dir = raw_dir / "compiled"
    raw_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    gpu = gpu_description()

    demo_name = "demo_30B_A3B_hopper.py" if is_moe else "demo_hopper.py"
    demo_path = mirage_dir / "demo" / "qwen3" / demo_name
    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    old_mirage_home = os.environ.get("MIRAGE_HOME")
    os.environ["MIRAGE_HOME"] = str(mirage_dir)
    sys.path.insert(0, str(demo_path.parent))
    sys.argv = demo_arguments(demo_path, model_ref, scenario, build_dir, is_moe)

    execution_mode = "graph_only" if args.graph_only else "execute"
    execution_status = "not_started"
    run_error: BaseException | None = None
    try:
        os.chdir(raw_dir)
        with (
            torch.inference_mode(),
            patched_tokenizer(prompt_len),
            patched_position_embeddings(max_seq_length),
            patched_moe_split_kv(is_moe),
            patched_graph_only(args.graph_only),
        ):
            try:
                runpy.run_path(str(demo_path), run_name="__main__")
                execution_status = "completed"
            except GraphOnlyComplete as exc:
                print(f"[graph-only] {exc}")
                execution_status = "graph_generated"
            except BaseException as exc:  # preserve a graph/report when runtime fails
                execution_status = "failed_after_or_during_graph_generation"
                run_error = exc
                traceback.print_exc()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        if sys.path and sys.path[0] == str(demo_path.parent):
            sys.path.pop(0)
        if old_mirage_home is None:
            os.environ.pop("MIRAGE_HOME", None)
        else:
            os.environ["MIRAGE_HOME"] = old_mirage_home

    graph_path = raw_dir / "task_graph_0.json"
    if not graph_path.is_file():
        if run_error is not None:
            raise RuntimeError(
                "Mirage failed before producing task_graph_0.json"
            ) from run_error
        raise FileNotFoundError(f"Mirage did not produce {graph_path}")

    # The MPK workflow designates scripts/parse_task_graph.py as the canonical
    # raw JSON parser.  Run it first and retain its task/event summary; the
    # analysis below adds the requested per-layer and tile aggregation.
    canonical_summary = run_canonical_mirage_parser(
        mirage_dir, graph_path, case_dir / "report"
    )
    print(canonical_summary)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    header_path = (
        mirage_dir / "include" / "mirage" / "persistent_kernel" / "runtime_header.h"
    )
    metadata = {
        "model_alias": model_alias,
        "model_ref": model_ref,
        "mirage": mirage_info,
        "gpu": gpu,
        "execution_mode": execution_mode,
        "execution_status": execution_status,
        "split_kv": True,
        "split_kv_adapter": "upstream_dense_demo"
        if not is_moe
        else "moe_runtime_api_adapter",
        "kv_cache_compatibility": kv_cache_compatibility,
        "scenario": {
            "prompt_len": scenario.prompt_len,
            "decode_len": scenario.decode_len,
            "mbt": scenario.mbt,
            "max_num_batched_requests": scenario.batch_size,
            "max_seq_length": scenario.max_seq_length,
            "page_size": scenario.page_size,
            "max_num_pages": scenario.max_num_pages,
        },
    }
    report = analyze_graph(graph, header_path.read_text(encoding="utf-8"), metadata)
    write_report_files(report, case_dir)

    if not args.keep_build_artifacts:
        shutil.rmtree(build_dir, ignore_errors=True)
        for kernel_path in raw_dir.glob("kernel_*.cu"):
            kernel_path.unlink(missing_ok=True)

    print(render_text_report(report))
    if run_error is not None:
        raise RuntimeError(
            f"The task graph was analyzed, but Mirage execution failed. Report: "
            f"{case_dir / 'report' / 'report.json'}"
        ) from run_error
    return 0


def case_tag(
    model_alias: str,
    prompt_len: int,
    decode_len: int,
    mbt: int,
    batch_size: int,
    graph_only: bool,
) -> str:
    mode = "graph_only" if graph_only else "execute"
    return (
        f"{model_alias}__p{length_label(prompt_len)}_d{decode_len}__"
        f"mbt{mbt}_bs{batch_size}__{mode}"
    )


def worker_command(
    args: argparse.Namespace, prompt_len: int, case_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--mirage-dir",
        str(Path(args.mirage_dir).expanduser().resolve()),
        "--mbt",
        str(args.mbt),
        "--max-num-batched-requests",
        str(args.max_num_batched_requests),
        "--decode-len",
        str(args.decode_len),
        "--page-size",
        str(args.page_size),
        "--_worker-prompt-len",
        str(prompt_len),
        "--_worker-case-dir",
        str(case_dir),
    ]
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    if args.max_num_pages:
        command.extend(["--max-num-pages", str(args.max_num_pages)])
    if args.graph_only:
        command.append("--graph-only")
    if args.allow_mirage_non_main:
        command.append("--allow-mirage-non-main")
    if args.keep_build_artifacts:
        command.append("--keep-build-artifacts")
    return command


def tee_subprocess(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def run_parent(args: argparse.Namespace) -> int:
    mirage_dir = Path(args.mirage_dir).expanduser().resolve()
    validate_mirage_checkout(mirage_dir, args.allow_mirage_non_main)
    model_alias, _ = resolve_model(args.model, args.model_path)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for prompt_len in args.prompt_lengths:
        tag = case_tag(
            model_alias,
            prompt_len,
            args.decode_len,
            args.mbt,
            args.max_num_batched_requests,
            args.graph_only,
        )
        case_dir = output_root / tag
        report_path = case_dir / "report" / "report.json"
        if report_path.exists() and not args.force:
            print(f"[skip] {tag}: {report_path} already exists (use --force to rerun)")
            continue
        if case_dir.exists() and args.force:
            shutil.rmtree(case_dir)
        print(f"[run ] {tag}")
        return_code = tee_subprocess(
            worker_command(args, prompt_len, case_dir), case_dir / "run.log"
        )
        if return_code != 0:
            failures.append(tag)
            print(f"[fail] {tag}: exit code {return_code}; see {case_dir / 'run.log'}")
            if not args.continue_on_error:
                break
        else:
            print(f"[done] {tag}")
        update_summary(output_root)

    update_summary(output_root)
    if failures:
        print(f"Failed cases ({len(failures)}): {', '.join(failures)}")
        return 1
    print(f"All requested cases completed. Summary: {output_root / 'summary.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and analyze split-KV Mirage task graphs for Qwen3 on one H100."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(QWEN3_SPECS),
        help=f"Qwen3 model alias (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--model-path",
        help="Optional local Hugging Face model directory; the selected --model still determines architecture.",
    )
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=parse_length,
        default=list(DEFAULT_PROMPT_LENGTHS),
        help="Prompt lengths in tokens; k means 1024 (default: 8 16 1k 8k).",
    )
    parser.add_argument(
        "--decode-len",
        type=int,
        default=DEFAULT_DECODE_LENGTH,
        help=f"Decode tokens per request (default: {DEFAULT_DECODE_LENGTH}).",
    )
    parser.add_argument(
        "--mbt",
        type=int,
        default=DEFAULT_MBT,
        help=f"Mirage max_num_batched_tokens / micro-batch token capacity (default: {DEFAULT_MBT}).",
    )
    parser.add_argument(
        "--max-num-batched-requests",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Maximum request batch size / BS (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"KV-cache page size (default: {DEFAULT_PAGE_SIZE}).",
    )
    parser.add_argument(
        "--max-num-pages",
        type=int,
        help="KV-cache page count. Default: automatically sized for max_seq_length and BS plus one page/request.",
    )
    parser.add_argument(
        "--mirage-dir",
        default=str(default_mirage_dir()),
        help="Mirage main checkout (default: $MIRAGE_HOME, otherwise ~/envs/mirage).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root() / "task_graph" / "results"),
        help="Result root (default: task_graph/results under this repository).",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Generate/analyze the graph but skip NVCC compilation and persistent-kernel execution.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rerun matching result directories.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue later prompt-length cases after a worker failure.",
    )
    parser.add_argument(
        "--keep-build-artifacts",
        action="store_true",
        help="Keep generated CUDA and compiled launcher files (large).",
    )
    parser.add_argument(
        "--allow-mirage-non-main",
        action="store_true",
        help="Allow a reviewed Mirage compatibility branch instead of main.",
    )
    parser.add_argument("--_worker-prompt-len", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-case-dir", help=argparse.SUPPRESS)
    return parser


def validate_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive = {
        "decode_len": args.decode_len,
        "mbt": args.mbt,
        "max_num_batched_requests": args.max_num_batched_requests,
        "page_size": args.page_size,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_num_pages is not None and args.max_num_pages <= 0:
        parser.error("--max-num-pages must be positive")
    if args.mbt < args.max_num_batched_requests:
        parser.error("--mbt must be >= --max-num-batched-requests")
    if bool(args._worker_prompt_len) != bool(args._worker_case_dir):
        parser.error("internal worker arguments must be supplied together")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_cli(args, parser)
    if args._worker_prompt_len is not None:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
