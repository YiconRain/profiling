"""Orchestrate the nsys profiling sweep over models x modes x cases.

For every experiment cell this script:
  1. runs worker.py under `nsys profile` (1 warmup + 1 measured generate),
  2. exports the .nsys-rep to SQLite,
  3. parses the SQLite into a metrics JSON (analyze_nsys.py),
  4. gzips both the .nsys-rep and .sqlite into results/nsys/ and drops the raw
     files to save disk.

It is resumable: a cell whose metrics JSON already exists is skipped. Invoke
from the project root (run_all.sh does this) using the SGLang venv interpreter,
e.g. `~/envs/sgl_env/bin/python kernel_launch/H100/scripts/run_profiling.py`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
WORKER = SCRIPTS / "worker.py"
ANALYZER = SCRIPTS / "analyze_nsys.py"


def sh(cmd: list[str], log_path: Path) -> None:
    """Run a command, teeing combined output to a log file; raise on failure."""
    with open(log_path, "ab") as log:
        log.write(("\n$ " + " ".join(cmd) + "\n").encode())
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}), see {log_path}: {' '.join(cmd)}")


def cleanup_work(tag: str) -> None:
    """Drop any partial raw rep/sqlite for a tag so a retry starts clean."""
    work = Path(C.WORK_DIR)
    for p in work.glob(f"{tag}.*"):
        p.unlink()


def gzip_into(src: Path, dst_gz: Path) -> None:
    dst_gz.parent.mkdir(parents=True, exist_ok=True)
    if dst_gz.exists():
        dst_gz.unlink()
    # -c to stdout so we control the destination path and remove the raw file.
    with open(dst_gz, "wb") as out:
        subprocess.run(["gzip", "-c", str(src)], stdout=out, check=True)
    src.unlink()


def run_cell(model_alias, model_path, mode, case, python_bin: str) -> None:
    tag = C.run_tag(model_alias, mode, case["id"])

    metrics_dir = Path(C.METRICS_DIR)
    work_dir = Path(C.WORK_DIR)
    logs_dir = Path(C.LOGS_DIR)
    nsys_out = Path(C.NSYS_DIR) / model_alias / mode
    for d in (metrics_dir, work_dir, logs_dir, nsys_out):
        d.mkdir(parents=True, exist_ok=True)

    metrics_json = metrics_dir / f"{tag}.json"
    if metrics_json.exists():
        print(f"[skip] {tag}: metrics already present")
        return

    log_path = logs_dir / f"{tag}.log"
    rep = work_dir / f"{tag}"          # nsys appends .nsys-rep
    rep_file = work_dir / f"{tag}.nsys-rep"
    sqlite = work_dir / f"{tag}.sqlite"

    print(f"[run ] {tag}")

    # 1. Profile: 1 warmup + 1 measured generate inside worker.py.
    #    --capture-range=cudaProfilerApi + --capture-range-end=stop: nsys records
    #    only the region between the worker's cudaProfilerStart/Stop (the measured
    #    generate), and cudaProfilerStop forces a CUPTI flush while the SGLang
    #    subprocess is still alive -- robust for CUDA graphs and large MoEs.
    #    --cuda-graph-trace=node records each kernel node inside CUDA graphs
    #    (the default logs only one entry per graph launch). Harmless in eager.
    sh([
        "nsys", "profile",
        "--trace=cuda",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--sample=none", "--cpuctxsw=none",
        "--force-overwrite=true",
        "--output", str(rep),
        python_bin, str(WORKER),
        "--model-path", model_path,
        "--mode", mode,
        "--prompt-len", str(case["prompt_len"]),
        "--decode-len", str(case["decode_len"]),
        "--batch-size", str(case["batch_size"]),
        "--attention-backend", C.ATTENTION_BACKEND,
    ], log_path)

    # 2. Export to SQLite.
    sh(["nsys", "export", "--type", "sqlite", "--force-overwrite=true",
        "--output", str(sqlite), str(rep_file)], log_path)

    # 3. Parse metrics.
    sh([python_bin, str(ANALYZER), str(sqlite), str(metrics_json),
        "--model", model_alias, "--mode", mode, "--case", case["id"],
        "--prompt-len", str(case["prompt_len"]),
        "--decode-len", str(case["decode_len"]),
        "--batch-size", str(case["batch_size"])], log_path)

    # 4. Compress rep + sqlite for download, drop the raw files.
    gzip_into(rep_file, nsys_out / f"{case['id']}.nsys-rep.gz")
    gzip_into(sqlite, nsys_out / f"{case['id']}.sqlite.gz")
    print(f"[done] {tag}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter used to run worker.py under nsys (default: this one).")
    p.add_argument("--models", nargs="+", help="Restrict to these model aliases.")
    p.add_argument("--modes", nargs="+", help="Restrict to these modes.")
    p.add_argument("--cases", nargs="+", help="Restrict to these case ids.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    failures = []
    for model_alias, model_path, mode, case in C.combos():
        if args.models and model_alias not in args.models:
            continue
        if args.modes and mode not in args.modes:
            continue
        if args.cases and case["id"] not in args.cases:
            continue
        tag = C.run_tag(model_alias, mode, case["id"])
        # Retry once: nsys/CUPTI flush can occasionally drop the kernel table.
        for attempt in (1, 2):
            try:
                run_cell(model_alias, model_path, mode, case, args.python)
                break
            except Exception as e:
                print(f"[fail {attempt}/2] {tag}: {e}")
                cleanup_work(tag)
                if attempt == 2:
                    failures.append(tag)
    if failures:
        print(f"Completed with {len(failures)} failed cell(s): {failures}")
    else:
        print("All requested cells complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
