"""SGLang inference worker for a single profiling run.

Launched (under nsys) by run_profiling.py once per (model, mode, case) cell.
It performs exactly ONE warmup generate() followed by ONE measured generate().
The measured call is wrapped in an NVTX range so the offline analysis can
isolate its kernels on the nsys timeline.

The SGLang Engine runs the model in a separate scheduler process, but a
blocking generate() in this process fully encloses that subprocess work on the
shared nsys timeline, so the NVTX window emitted here is a valid time filter.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--mode", choices=["eager", "cudagraph"], required=True)
    p.add_argument("--prompt-len", type=int, required=True)
    p.add_argument("--decode-len", type=int, required=True)
    p.add_argument("--attention-backend", default="flashinfer")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--measure-range", default="measure")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_ids = build_input_ids(args.model_path, args.prompt_len)

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

    # Warmup: builds CUDA graphs (in cudagraph mode) and JIT/autotune caches.
    engine.generate(input_ids=input_ids, sampling_params=sampling_params)

    # Measured run, bracketed by the NVTX range that analysis keys off.
    torch.cuda.nvtx.range_push(args.measure_range)
    t0 = time.perf_counter()
    engine.generate(input_ids=input_ids, sampling_params=sampling_params)
    t1 = time.perf_counter()
    torch.cuda.nvtx.range_pop()

    print(json.dumps({
        "measure_wall_s": t1 - t0,
        "prompt_len": args.prompt_len,
        "decode_len": args.decode_len,
        "mode": args.mode,
    }))

    # Force CUPTI device-activity buffers to flush. On CUDA 11+ nsys only saves
    # a device buffer when it FILLS; a short measure region (e.g. a single
    # prefill) never fills one, so its kernel records would be dropped when the
    # SGLang scheduler subprocess is killed -- the export then lacks the
    # CUPTI_ACTIVITY_KIND_KERNEL table entirely. Running extra cheap generates
    # after the NVTX window fills the buffer; the resulting buffer-full flush
    # also delivers the measured kernels. This work is outside 'measure', so the
    # analysis window (filtered by NVTX time) excludes it.
    flush_ids = input_ids[:16]
    flush_sp = {"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}
    for _ in range(40):
        engine.generate(input_ids=flush_ids, sampling_params=flush_sp)
    time.sleep(2)
    engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
