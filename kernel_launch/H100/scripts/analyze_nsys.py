"""Parse an nsys-exported SQLite database into kernel-launch metrics.

The nsys report was captured with --capture-range=cudaProfilerApi, so it holds
ONLY the single measured generate() (model load + warmup are excluded). We can
therefore count every kernel / launch in the database directly.

Metrics:
  * total_kernels        - number of GPU kernel executions
  * total_kernel_gpu_ns  - summed GPU time of those kernels
  * launch_count         - number of kernel-launch host API calls
  * launch_overhead_ns   - summed host time spent in those launch calls
  * launch_overhead_pct  - launch_overhead_ns / e2e_ns * 100
  * e2e_ns               - span of the captured region (host API first..last)
  * top5                 - five kernels with the largest summed GPU time

Usage: analyze_nsys.py <db.sqlite> <output.json> [metadata flags]
"""

from __future__ import annotations

import argparse
import json
import sqlite3

# Host APIs that launch GPU kernels. cudaLaunchKernel* dominates eager mode;
# cudaGraphLaunch* replaces most of them in cudagraph mode (one call replays
# many kernels) -- counting both keeps the two modes comparable.
LAUNCH_API_PATTERNS = ("cudaLaunchKernel%", "cudaGraphLaunch%", "cuLaunchKernel%")


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def kernel_stats(con: sqlite3.Connection):
    rows = con.execute(
        """
        SELECT s.value AS name, COUNT(*) AS n, SUM(k.end - k.start) AS dur
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        GROUP BY name
        ORDER BY dur DESC
        """
    ).fetchall()
    total_kernels = sum(r[1] for r in rows)
    total_gpu_ns = sum(r[2] or 0 for r in rows)
    top5 = [
        {"name": r[0], "count": r[1], "gpu_ns": int(r[2] or 0), "gpu_ms": (r[2] or 0) / 1e6}
        for r in rows[:5]
    ]
    return total_kernels, total_gpu_ns, top5


def launch_stats(con: sqlite3.Connection):
    like = " OR ".join(["s.value LIKE ?"] * len(LAUNCH_API_PATTERNS))
    rows = con.execute(
        f"""
        SELECT s.value AS api, COUNT(*) AS n, SUM(r.end - r.start) AS dur
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        WHERE ({like})
        GROUP BY api
        """,
        LAUNCH_API_PATTERNS,
    ).fetchall()
    count = sum(r[1] for r in rows)
    dur = sum(r[2] or 0 for r in rows)
    by_api = {r[0]: {"count": r[1], "overhead_ns": int(r[2] or 0)} for r in rows}
    return count, dur, by_api


def e2e_span_ns(con: sqlite3.Connection) -> int:
    """Wall span of the captured region: earliest..latest host API call.

    RUNTIME events bracket the CPU side of the measured generate(); their span
    approximates its end-to-end wall time. Fall back to kernel span if needed.
    """
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL"):
        if not table_exists(con, table):
            continue
        row = con.execute(f"SELECT MIN(start), MAX(end) FROM {table}").fetchone()
        if row and row[0] is not None:
            return int(row[1] - row[0])
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("db")
    p.add_argument("out")
    p.add_argument("--model")
    p.add_argument("--mode")
    p.add_argument("--case")
    p.add_argument("--prompt-len", type=int)
    p.add_argument("--decode-len", type=int)
    args = p.parse_args()

    con = sqlite3.connect(args.db)
    if not table_exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        raise SystemExit("No CUPTI_ACTIVITY_KIND_KERNEL table: capture produced no kernels.")
    e2e_ns = e2e_span_ns(con)
    total_kernels, total_gpu_ns, top5 = kernel_stats(con)
    launch_count, launch_ns, launch_by_api = launch_stats(con)
    con.close()

    metrics = {
        "model": args.model,
        "mode": args.mode,
        "case": args.case,
        "prompt_len": args.prompt_len,
        "decode_len": args.decode_len,
        "e2e_ns": e2e_ns,
        "e2e_ms": e2e_ns / 1e6,
        "total_kernels": total_kernels,
        "total_kernel_gpu_ns": total_gpu_ns,
        "total_kernel_gpu_ms": total_gpu_ns / 1e6,
        "launch_count": launch_count,
        "launch_overhead_ns": launch_ns,
        "launch_overhead_ms": launch_ns / 1e6,
        "launch_overhead_pct": (launch_ns / e2e_ns * 100) if e2e_ns else 0.0,
        "launch_by_api": launch_by_api,
        "top5_kernels": top5,
    }

    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[analyze] {args.out}: {total_kernels} kernels, launch_count {launch_count}, "
          f"launch {metrics['launch_overhead_pct']:.2f}% of e2e")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
