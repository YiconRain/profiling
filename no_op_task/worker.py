"""Run one MPK capacity case through the upstream YiconRain Mirage demo.

This worker deliberately does not duplicate Mirage's model-specific graph
builders.  Instead, it applies a narrow runtime harness around
``PersistentKernel`` and executes the appropriate H100 demo with ``runpy``.

The harness makes three experiment-specific changes without touching kernels:

1. the compiled capacity is MBR, but the runtime has exactly one real request;
2. the prompt metadata is reset to exactly one valid token before every run;
3. one compiled kernel is reset and invoked repeatedly with CUDA Event timing.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


RESULT_PREFIX = "NO_OP_TASK_RESULT="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirage-root", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--mbr", type=int, required=True)
    parser.add_argument("--mbt", type=int, required=True)
    parser.add_argument("--prompt-len", type=int, default=1)
    parser.add_argument("--decode-len", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=513)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--max-num-pages", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mode", choices=("measure", "nsys"), required=True)
    parser.add_argument(
        "--compile-mode", choices=("compile", "load"), required=True
    )
    parser.add_argument("--compile-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception:
        return "unknown"


def cuda_profiler(action: str) -> None:
    """Start or stop a cudaProfilerApi capture range."""

    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    fn = cudart.cudaProfilerStart if action == "start" else cudart.cudaProfilerStop
    result = fn()
    if result not in (0, None):
        raise RuntimeError(f"cudaProfiler{action.title()} returned {result}")


def task_graph_counts(compile_dir: Path) -> dict[str, int | None]:
    graph_path = compile_dir / "task_graph_rank0.json"
    if not graph_path.exists():
        return {"num_tasks": None, "num_events": None}
    with graph_path.open() as f:
        graph = json.load(f)
    return {
        "num_tasks": len(graph.get("all_tasks", [])),
        "num_events": len(graph.get("all_events", [])),
    }


class ExperimentHarness:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.compile_seconds: float | None = None
        self.result_written = False

    def install(self) -> None:
        """Monkey-patch only the orchestration surface of PersistentKernel."""

        from mirage.mpk.persistent_kernel import PersistentKernel

        original_init = PersistentKernel.__init__
        original_compile = PersistentKernel.compile
        original_call = PersistentKernel.__call__
        harness = self

        def patched_init(kernel_self, *init_args, **init_kwargs):
            original_init(kernel_self, *init_args, **init_kwargs)
            # Keep all MBR-sized metadata buffers and the MBR-specialized graph,
            # but tell the offline runtime that only request 0 actually exists.
            kernel_self.total_num_requests = 1

        def patched_compile(kernel_self, **compile_kwargs):
            output_dir = Path(
                compile_kwargs.get("output_dir") or harness.args.compile_dir
            ).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            if harness.args.compile_mode == "load":
                kernel_self.load_mpk_kernel(
                    output_dir=str(output_dir),
                    eos_token_id=compile_kwargs.get("eos_token_id", -1),
                )
            else:
                compile_kwargs["output_dir"] = str(output_dir)
                original_compile(kernel_self, **compile_kwargs)
            harness.compile_seconds = time.perf_counter() - start

        def patched_call(kernel_self, **call_kwargs):
            if harness.result_written:
                raise RuntimeError("The selected Mirage demo invoked MPK more than once")
            harness.run_repeated(kernel_self, original_call, call_kwargs)
            harness.result_written = True

        PersistentKernel.__init__ = patched_init
        PersistentKernel.compile = patched_compile
        PersistentKernel.__call__ = patched_call

    def reset_request(self, kernel: Any, prompt_token_id: int) -> None:
        tensors = kernel.meta_tensors
        tensors["tokens"].zero_()
        tensors["tokens"][0, 0] = prompt_token_id
        tensors["input_tokens"].zero_()
        tensors["output_tokens"].zero_()
        tensors["num_new_tokens"].fill_(1)
        tensors["prompt_lengths"].zero_()
        tensors["prompt_lengths"][0] = self.args.prompt_len
        # This GPU helper resets steps, active-request slots, page queues, and
        # request/KV indptrs.  It synchronizes before returning.
        kernel.init_request_func()
        torch.cuda.synchronize()

    def run_once(
        self,
        kernel: Any,
        original_call: Any,
        call_kwargs: dict[str, Any],
        prompt_token_id: int,
        profile: bool,
    ) -> tuple[float, int]:
        self.reset_request(kernel, prompt_token_id)
        if profile:
            cuda_profiler("start")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        original_call(kernel, **call_kwargs)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start_event.elapsed_time(end_event))

        if profile:
            cuda_profiler("stop")

        step = int(kernel.meta_tensors["step"][0].item())
        generated_tokens = step + 1 - self.args.prompt_len
        if generated_tokens != self.args.decode_len:
            raise RuntimeError(
                "Unexpected decode length: "
                f"expected {self.args.decode_len}, got {generated_tokens} "
                f"(step={step}, max_seq_length={self.args.max_seq_length})"
            )
        return elapsed_ms, generated_tokens

    def run_repeated(
        self,
        kernel: Any,
        original_call: Any,
        call_kwargs: dict[str, Any],
    ) -> None:
        if self.args.prompt_len != 1:
            raise ValueError("This experiment requires prompt_len=1")
        if kernel.total_num_requests != 1:
            raise RuntimeError(
                f"Runtime request count must be 1, got {kernel.total_num_requests}"
            )

        # The upstream demo has already produced valid model-specific tokens.
        # Preserve its first token, then discard the remainder of the prompt.
        prompt_token_id = int(kernel.meta_tensors["tokens"][0, 0].item())
        run_ms: list[float] = []
        generated_tokens = 0

        if self.args.mode == "nsys":
            elapsed_ms, generated_tokens = self.run_once(
                kernel,
                original_call,
                call_kwargs,
                prompt_token_id,
                profile=True,
            )
            run_ms.append(elapsed_ms)
        else:
            for _ in range(self.args.repeats):
                elapsed_ms, generated_tokens = self.run_once(
                    kernel,
                    original_call,
                    call_kwargs,
                    prompt_token_id,
                    profile=False,
                )
                run_ms.append(elapsed_ms)

        result = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": self.args.mode,
            "model_alias": self.args.model_alias,
            "model_id": self.args.model_id,
            "actual_num_requests": 1,
            "max_num_batched_requests": self.args.mbr,
            "max_num_batched_tokens": self.args.mbt,
            "prompt_len": self.args.prompt_len,
            "decode_len": self.args.decode_len,
            "generated_tokens": generated_tokens,
            "max_seq_length": self.args.max_seq_length,
            "page_size": self.args.page_size,
            "max_num_pages": self.args.max_num_pages,
            "warmup_runs": 0,
            "measured_runs": len(run_ms),
            "mpk_gpu_ms": run_ms,
            "average_mpk_gpu_ms": statistics.fmean(run_ms),
            "average_ms_per_generated_token": statistics.fmean(run_ms)
            / self.args.decode_len,
            "prompt_token_id": prompt_token_id,
            "compile_mode": self.args.compile_mode,
            "compile_or_load_seconds": self.compile_seconds,
            "mirage_revision": git_revision(self.args.mirage_root),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            **task_graph_counts(self.args.compile_dir),
        }
        self.args.result_json.parent.mkdir(parents=True, exist_ok=True)
        with self.args.result_json.open("w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)


def main() -> int:
    args = parse_args()
    args.mirage_root = args.mirage_root.resolve()
    args.demo = args.demo.resolve()
    args.compile_dir = args.compile_dir.resolve()
    args.result_json = args.result_json.resolve()

    if args.max_seq_length != args.prompt_len + args.decode_len:
        raise ValueError(
            "max_seq_length must equal prompt_len + decode_len for an exact "
            "fixed-length ignore-EOS run"
        )
    if args.mode == "measure" and args.repeats != 5:
        raise ValueError("Measurement mode requires exactly five runs")
    if args.mode == "nsys" and args.repeats != 1:
        raise ValueError("nsys mode requires exactly one captured run")
    if not args.demo.is_file():
        raise FileNotFoundError(args.demo)

    os.environ.setdefault("MIRAGE_HOME", str(args.mirage_root))
    sys.path.insert(0, str(args.mirage_root))
    sys.path.insert(0, str(args.mirage_root / "demo/qwen3"))

    harness = ExperimentHarness(args)
    harness.install()

    demo_argv = [
        str(args.demo),
        "--use-mirage",
        "--model",
        args.model_id,
        "--max-num-batched-requests",
        str(args.mbr),
        "--max-num-batched-tokens",
        str(args.mbt),
        "--max-seq-length",
        str(args.max_seq_length),
        "--page-size",
        str(args.page_size),
        "--max-num-pages",
        str(args.max_num_pages),
        "--output-dir",
        str(args.compile_dir),
        "--ignore-eos",
    ]
    old_argv = sys.argv
    try:
        sys.argv = demo_argv
        runpy.run_path(str(args.demo), run_name="__main__")
    finally:
        sys.argv = old_argv

    if not harness.result_written or not args.result_json.exists():
        raise RuntimeError("Mirage demo completed without invoking the MPK harness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
