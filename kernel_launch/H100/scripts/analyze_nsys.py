"""Parse an nsys-exported SQLite database into kernel-launch metrics.

Metrics extracted for the single measured generate() (the NVTX 'measure'
window):
  * total_kernels        - number of GPU kernel executions
  * total_kernel_gpu_ns  - summed GPU time of those kernels
  * launch_count         - number of cudaLaunchKernel* host API calls
  * launch_overhead_ns   - summed host time spent in those launch calls
  * launch_overhead_pct  - launch_overhead_ns / e2e_ns * 100
  * e2e_ns               - duration of the measure window (end-to-end)
  * top5                 - five kernels with the largest summed GPU time

Usage: analyze_nsys.py <db.sqlite> <output.json> [--measure-range measure]
       plus metadata flags recorded verbatim into the output.
"""

from __future__ import annotations

import argparse
import json
import sqlite3


def measure_window(con: sqlite3.Connection, range_name: str) -> tuple[int, int]:
    """Return (start_ns, end_ns) of the NVTX measure range.

    NVTX text is stored either inline (NVTX_EVENTS.text) or via a StringIds id
    (NVTX_EVENTS.textId); handle both.
    """
    row = con.execute(
        """
        SELECT e.start, e.end
        FROM NVTX_EVENTS e
        LEFT JOIN StringIds s ON s.id = e.textId
        WHERE e.end IS NOT NULL AND COALESCE(e.text, s.value) = ?
        ORDER BY e.start
        LIMIT 1
        """,
        (range_name,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"NVTX range {range_name!r} not found in database.")
    return int(row[0]), int(row[1])


def kernel_stats(con: sqlite3.Connection, w0: int, w1: int):
    rows = con.execute(
        """
        SELECT s.value AS name, COUNT(*) AS n, SUM(k.end - k.start) AS dur
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        WHERE k.start >= ? AND k.start < ?
        GROUP BY name
        ORDER BY dur DESC
        """,
        (w0, w1),
    ).fetchall()
    total_kernels = sum(r[1] for r in rows)
    total_gpu_ns = sum(r[2] or 0 for r in rows)
    top5 = [
        {"name": r[0], "count": r[1], "gpu_ns": int(r[2] or 0), "gpu_ms": (r[2] or 0) / 1e6}
        for r in rows[:5]
    ]
    return total_kernels, total_gpu_ns, top5


def launch_stats(con: sqlite3.Connection, w0: int, w1: int):
    row = con.execute(
        """
        SELECT COUNT(*), SUM(r.end - r.start)
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        WHERE s.value LIKE 'cudaLaunchKernel%'
          AND r.start >= ? AND r.start < ?
        """,
        (w0, w1),
    ).fetchone()
    return int(row[0] or 0), int(row[1] or 0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("db")
    p.add_argument("out")
    p.add_argument("--measure-range", default="measure")
    p.add_argument("--model")
    p.add_argument("--mode")
    p.add_argument("--case")
    p.add_argument("--prompt-len", type=int)
    p.add_argument("--decode-len", type=int)
    args = p.parse_args()

    con = sqlite3.connect(args.db)
    w0, w1 = measure_window(con, args.measure_range)
    e2e_ns = w1 - w0
    total_kernels, total_gpu_ns, top5 = kernel_stats(con, w0, w1)
    launch_count, launch_ns = launch_stats(con, w0, w1)
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
        "top5_kernels": top5,
    }

    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[analyze] {args.out}: {total_kernels} kernels, "
          f"launch {metrics['launch_overhead_pct']:.2f}% of e2e")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
