"""Orchestrate the single-H100 MPK over-compilation experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import config as C


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKER = SCRIPT_DIR / "worker.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mirage-root",
        type=Path,
        default=PROJECT_ROOT.parent / "mirage",
        help="Path to a YiconRain/mirage checkout (default: ../mirage).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=SCRIPT_DIR / "artifacts",
        help="Ignored output tree for metrics, compile artifacts, logs, and nsys files.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter with PyTorch, Transformers, and Mirage installed.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(C.QWEN3_SPECS),
        help="Restrict the sweep to selected model aliases.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(case.id for case in C.CASES),
        help="Restrict the sweep to selected capacity cases.",
    )
    parser.add_argument("--skip-nsys", action="store_true")
    parser.add_argument(
        "--only-nsys",
        action="store_true",
        help="Skip measurement and profile cells whose compiled artifacts already exist.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_logged(cmd: list[str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {command_string(cmd)}", flush=True)
    if dry_run:
        return
    with log_path.open("ab") as log:
        log.write(("\n$ " + command_string(cmd) + "\n").encode())
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}; see {log_path}"
        )


def check_environment(args: argparse.Namespace) -> None:
    if not args.mirage_root.is_dir():
        raise FileNotFoundError(
            f"Mirage checkout not found: {args.mirage_root}. "
            "Clone YiconRain/mirage and check out the no_op_task branch."
        )
    if not WORKER.is_file():
        raise FileNotFoundError(WORKER)
    persistent_kernel_py = (
        args.mirage_root / "python/mirage/mpk/persistent_kernel.py"
    )
    if not persistent_kernel_py.is_file():
        raise FileNotFoundError(persistent_kernel_py)
    source = persistent_kernel_py.read_text()
    if source.count('self.init_request_func = getattr(mod, "init_request_func")') < 2:
        raise RuntimeError(
            "Mirage is missing the no_op_task compile-path init_request_func fix. "
            "Check out YiconRain/mirage:no_op_task."
        )
    load_section = source[source.find("def load_mpk_kernel"):]
    if 'meta_tensors.append(self.meta_tensors["paged_kv_indices_snapshot"])' not in load_section:
        raise RuntimeError(
            "Mirage is missing the no_op_task compiled-kernel reload fix. "
            "Check out YiconRain/mirage:no_op_task."
        )
    if not args.dry_run and not args.skip_nsys and shutil.which("nsys") is None:
        raise RuntimeError("nsys is required but was not found on PATH")
    if shutil.which(args.python) is None and not Path(args.python).is_file():
        raise FileNotFoundError(f"Python interpreter not found: {args.python}")

    if not args.dry_run:
        gpu_name = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                "--id=0",
            ],
            text=True,
        ).strip()
        print(f"[env] GPU: {gpu_name}")
        if "H100" not in gpu_name:
            raise RuntimeError(f"This experiment requires one H100 GPU, found: {gpu_name}")


def worker_command(
    args: argparse.Namespace,
    model_alias: str,
    model_id: str,
    case: C.Case,
    mode: str,
    compile_mode: str,
    compile_dir: Path,
    result_json: Path,
) -> list[str]:
    demo = C.demo_path(args.mirage_root, model_alias)
    repeats = C.MEASURE_REPEATS if mode == "measure" else 1
    return [
        args.python,
        str(WORKER),
        "--mirage-root",
        str(args.mirage_root),
        "--demo",
        str(demo),
        "--model-alias",
        model_alias,
        "--model-id",
        model_id,
        "--mbr",
        str(case.max_num_batched_requests),
        "--mbt",
        str(case.max_num_batched_tokens),
        "--prompt-len",
        str(C.PROMPT_LEN),
        "--decode-len",
        str(C.DECODE_LEN),
        "--max-seq-length",
        str(C.MAX_SEQ_LENGTH),
        "--page-size",
        str(C.PAGE_SIZE),
        "--max-num-pages",
        str(C.MAX_NUM_PAGES),
        "--repeats",
        str(repeats),
        "--mode",
        mode,
        "--compile-mode",
        compile_mode,
        "--compile-dir",
        str(compile_dir),
        "--result-json",
        str(result_json),
    ]


def remove_cell_outputs(cell_dir: Path) -> None:
    if cell_dir.exists():
        shutil.rmtree(cell_dir)


def run_measurement(
    args: argparse.Namespace,
    model_alias: str,
    model_id: str,
    case: C.Case,
    cell_dir: Path,
) -> None:
    compile_dir = cell_dir / "compile"
    result_json = cell_dir / "metrics.json"
    log_path = cell_dir / "measure.log"
    if result_json.exists() and not args.force:
        print(f"[skip measure] {model_alias} {case.id}")
        return
    compile_dir.mkdir(parents=True, exist_ok=True)
    cmd = worker_command(
        args,
        model_alias,
        model_id,
        case,
        mode="measure",
        compile_mode="compile",
        compile_dir=compile_dir,
        result_json=result_json,
    )
    run_logged(cmd, log_path, args.dry_run)


def run_nsys(
    args: argparse.Namespace,
    model_alias: str,
    model_id: str,
    case: C.Case,
    cell_dir: Path,
) -> None:
    compile_dir = cell_dir / "compile"
    nsys_dir = cell_dir / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    profile_json = nsys_dir / "profile_run.json"
    rep_prefix = nsys_dir / "profile"
    rep_file = nsys_dir / "profile.nsys-rep"
    sqlite_file = nsys_dir / "profile.sqlite"
    log_path = cell_dir / "nsys.log"

    if rep_file.exists() and sqlite_file.exists() and not args.force:
        print(f"[skip nsys] {model_alias} {case.id}")
        return
    if not args.dry_run and not compile_dir.exists():
        raise FileNotFoundError(
            f"Compiled kernel missing for {model_alias} {case.id}: {compile_dir}"
        )

    worker_cmd = worker_command(
        args,
        model_alias,
        model_id,
        case,
        mode="nsys",
        compile_mode="load",
        compile_dir=compile_dir,
        result_json=profile_json,
    )
    profile_cmd = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "--output",
        str(rep_prefix),
        *worker_cmd,
    ]
    run_logged(profile_cmd, log_path, args.dry_run)

    export_cmd = [
        "nsys",
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        "--output",
        str(sqlite_file),
        str(rep_file),
    ]
    run_logged(export_cmd, log_path, args.dry_run)


def write_summary(artifacts_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(artifacts_dir.glob("*/*/metrics.json")):
        with metrics_path.open() as f:
            data = json.load(f)
        rows.append(
            {
                "model": data["model_alias"],
                "case": metrics_path.parent.name,
                "actual_requests": data["actual_num_requests"],
                "mbr": data["max_num_batched_requests"],
                "mbt": data["max_num_batched_tokens"],
                "prompt_len": data["prompt_len"],
                "decode_len": data["decode_len"],
                "run_1_ms": data["mpk_gpu_ms"][0],
                "run_2_ms": data["mpk_gpu_ms"][1],
                "run_3_ms": data["mpk_gpu_ms"][2],
                "run_4_ms": data["mpk_gpu_ms"][3],
                "run_5_ms": data["mpk_gpu_ms"][4],
                "average_ms": data["average_mpk_gpu_ms"],
                "average_ms_per_generated_token": data[
                    "average_ms_per_generated_token"
                ],
                "num_tasks": data.get("num_tasks"),
                "num_events": data.get("num_events"),
                "mirage_revision": data.get("mirage_revision"),
            }
        )
    if not rows:
        return

    summary_path = artifacts_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] {summary_path}")


def main() -> int:
    args = parse_args()
    args.mirage_root = args.mirage_root.resolve()
    args.artifacts_dir = args.artifacts_dir.resolve()
    if args.only_nsys and args.skip_nsys:
        raise ValueError("--only-nsys and --skip-nsys are mutually exclusive")
    check_environment(args)

    selected_models = set(args.models) if args.models else None
    selected_cases = set(args.cases) if args.cases else None
    cells = list(C.iter_cells(selected_models, selected_cases))
    if not cells:
        raise RuntimeError("No experiment cells selected")

    print(f"[plan] {len(cells)} cells")
    failures: list[str] = []
    for index, (model_alias, model_id, case) in enumerate(cells, start=1):
        tag = f"{model_alias}/{case.id}"
        cell_dir = args.artifacts_dir / model_alias / case.id
        print(f"[{index}/{len(cells)}] {tag}", flush=True)
        if args.force and not args.only_nsys and not args.dry_run:
            remove_cell_outputs(cell_dir)
        try:
            if not args.only_nsys:
                run_measurement(args, model_alias, model_id, case, cell_dir)
            if not args.skip_nsys:
                run_nsys(args, model_alias, model_id, case, cell_dir)
        except Exception as exc:
            failures.append(tag)
            print(f"[failed] {tag}: {exc}", file=sys.stderr, flush=True)

    if not args.dry_run:
        write_summary(args.artifacts_dir)
    if failures:
        print(f"Failed cells ({len(failures)}): {', '.join(failures)}", file=sys.stderr)
        return 1
    print("All selected cells completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
