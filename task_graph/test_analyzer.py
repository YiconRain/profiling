#!/usr/bin/env python3
"""CPU-only unit tests for the task-graph report analyzer."""

from __future__ import annotations

import unittest

from run_task_graph import analyze_graph, parse_length, resolve_kv_cache_geometry


HEADER = r"""
enum TaskType {
  TASK_TERMINATE = 0,
  TASK_BEGIN_TASK_GRAPH = 10,
  TASK_RMS_NORM_HOPPER = 154,
  TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER = 164,
  TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100 = 264,
};
enum EventType {
  EVENT_LAUNCH_TASKS = 901,
  EVENT_END_OF_TASK_GRAPH = 910,
  EVENT_TERMINATION = 911,
};
"""


def tensor(name: str, offset: int, dims: list[int]) -> dict:
    return {
        "base_ptr": name,
        "offset": offset,
        "dims": dims,
        "strides": [dims[-1], 1] if len(dims) == 2 else [1],
        "data_type": 1,
    }


def task(task_type: int, inputs: list[dict], outputs: list[dict]) -> dict:
    return {
        "task_type": task_type,
        "variant_id": 0,
        "inputs": inputs,
        "outputs": outputs,
    }


class AnalyzerTest(unittest.TestCase):
    def test_parse_lengths(self) -> None:
        self.assertEqual(parse_length("1k"), 1024)
        self.assertEqual(parse_length("8K"), 8192)
        self.assertEqual(parse_length("16"), 16)

    def test_dense_qwen3_uses_upstream_fixed_cache_geometry(self) -> None:
        page_size, max_num_pages, compatibility = resolve_kv_cache_geometry(
            is_moe=False,
            max_seq_length=136,
            batch_size=1,
            requested_page_size=4096,
            requested_max_num_pages=None,
        )
        self.assertEqual(page_size, 4096)
        self.assertEqual(max_num_pages, 16)
        self.assertEqual(compatibility, "mirage_main_dense_fixed_16x4096")

    def test_moe_qwen3_keeps_automatic_cache_geometry(self) -> None:
        page_size, max_num_pages, compatibility = resolve_kv_cache_geometry(
            is_moe=True,
            max_seq_length=8320,
            batch_size=1,
            requested_page_size=4096,
            requested_max_num_pages=None,
        )
        self.assertEqual(page_size, 4096)
        self.assertEqual(max_num_pages, 4)
        self.assertEqual(compatibility, "automatic")

    def test_analyze_split_kv_graph(self) -> None:
        graph = {
            "all_tasks": [
                task(0, [], []),
                task(10, [], []),
                task(
                    154,
                    [tensor("layer_0_input_layernorm", 0, [4096])],
                    [tensor("rms", 0, [1, 4096])],
                ),
                task(
                    154,
                    [tensor("layer_0_input_layernorm", 0, [4096])],
                    [tensor("rms", 8192, [1, 4096])],
                ),
                task(
                    164,
                    [tensor("layer_0_k_cache", 0, [1, 128])],
                    [tensor("split_tmp", 0, [1, 128])],
                ),
                task(
                    164,
                    [tensor("layer_0_k_cache", 256, [1, 128])],
                    [tensor("split_tmp", 256, [1, 128])],
                ),
                task(
                    264,
                    [tensor("split_tmp", 0, [1, 128])],
                    [tensor("attn_out", 0, [1, 128])],
                ),
            ],
            "all_events": [
                {"event_type": 911},
                {"event_type": 901},
                {"event_type": 910},
            ],
            "first_tasks": [1],
        }
        metadata = {
            "model_alias": "Qwen3-8B",
            "model_ref": "Qwen/Qwen3-8B",
            "mirage": {"branch": "main", "commit": "abc", "remote": "origin"},
            "gpu": "H100",
            "execution_mode": "graph_only",
            "execution_status": "graph_generated",
            "split_kv": True,
            "split_kv_adapter": "test",
            "scenario": {
                "prompt_len": 16,
                "decode_len": 128,
                "mbt": 8,
                "max_num_batched_requests": 1,
                "max_seq_length": 144,
                "page_size": 4096,
                "max_num_pages": 2,
            },
        }
        report = analyze_graph(graph, HEADER, metadata)
        stats = report["statistics"]
        self.assertEqual(stats["total_tasks_including_runtime"], 7)
        self.assertEqual(stats["compute_tasks_per_graph_iteration"], 5)
        self.assertEqual(stats["split_kv_task_count"], 2)
        self.assertEqual(stats["split_kv_merge_task_count"], 1)
        self.assertEqual(stats["estimated_prefill_iterations"], 2)
        self.assertEqual(stats["estimated_graph_iterations"], 130)
        rmsnorm = next(
            operation
            for operation in report["layers"]["layer_0"]
            if operation["operation"] == "RMSNorm"
        )
        self.assertEqual(rmsnorm["task_count"], 2)
        self.assertEqual(rmsnorm["tile_groups"][0]["outputs"][0]["unique_offsets"], 2)


if __name__ == "__main__":
    unittest.main()
