"""SGLang inference worker for a single profiling run.

Launched (under nsys) by run_profiling.py once per (model, mode, case) cell.
It performs exactly ONE warmup generate() followed by ONE measured generate().

The measured generate() is bracketed by cudaProfilerStart/Stop. Combined with
nsys `--capture-range=cudaProfilerApi --capture-range-end=stop`, nsys records
ONLY the measured request (model load + warmup happen before Start). Crucially,
cudaProfilerStop forces a CUPTI flush while the SGLang scheduler subprocess is
still alive, so GPU kernel activity -- including kernels inside CUDA graphs and
for large MoEs -- is reliably captured (no dependence on buffer-fill or on a
clean subprocess teardown).
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from sglang import Engine
from transformers import AutoTokenizer


def build_input_ids(model_path: str, prompt_len: int) -> list[int]:
    """Return a token id list of exactly prompt_len tokens."""
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    seed = tok.encode("The quick brown fox jumps over the lazy dog. ", add_special_tokens=False)
    if not seed:
        seed = [tok.eos_token_id or 0]
    ids = (seed * (prompt_len // len(seed) + 1))[:prompt_len]
    return ids


def cuda_profiler(action: str) -> None:
    """Start/stop the CUDA profiler; nsys keys its capture range off these.

    A synchronize() before each call ensures all prior GPU work is done, so the
    measured region starts clean and cudaProfilerStop flushes a complete trace.
    """
    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    fn = cudart.cudaProfilerStart if action == "start" else cudart.cudaProfilerStop
    ret = fn()
    if ret not in (0, None):
        print(f"[warn] cudaProfiler{action} returned {ret}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--mode", choices=["eager", "cudagraph"], required=True)
    p.add_argument("--prompt-len", type=int, required=True)
    p.add_argument("--decode-len", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--attention-backend", default="flashinfer")
    p.add_argument("--tp", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # One prompt of the requested length, replicated to the batch size. Batched
    # sequences decode in lockstep -- the point of the BS>1 cases.
    ids = build_input_ids(args.model_path, args.prompt_len)
    input_ids = [ids for _ in range(args.batch_size)]

    engine = Engine(
        model_path=args.model_path,
        attention_backend=args.attention_backend,
        disable_cuda_graph=(args.mode == "eager"),
        # Identical prompts across warmup + measure would otherwise be served
        # from the prefix cache, hiding the real prefill kernels.
        disable_radix_cache=True,
        tp_size=args.tp,
        trust_remote_code=True,
        log_level="warning",
    )

    # decode_len == 0 -> prefill-only, still emit the single unavoidable token.
    max_new_tokens = max(args.decode_len, 1)
    sampling_params = {"temperature": 0.0, "max_new_tokens": max_new_tokens, "ignore_eos": True}

    # Warmup (not profiled): builds CUDA graphs and JIT/autotune caches.
    engine.generate(input_ids=input_ids, sampling_params=sampling_params)

    # Measured run, captured between cudaProfilerStart/Stop.
    cuda_profiler("start")
    t0 = time.perf_counter()
    engine.generate(input_ids=input_ids, sampling_params=sampling_params)
    t1 = time.perf_counter()
    cuda_profiler("stop")

    print(json.dumps({
        "measure_wall_s": t1 - t0,
        "prompt_len": args.prompt_len,
        "decode_len": args.decode_len,
        "batch_size": args.batch_size,
        "mode": args.mode,
    }))

    # Teardown is best-effort and happens after nsys has already stopped and
    # flushed; a large-MoE kill_process_tree failure here cannot lose data.
    try:
        engine.shutdown()
    except Exception as e:
        print(f"[warn] engine.shutdown() non-fatal error: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
