"""Parse an nsys-exported SQLite database into kernel-launch metrics.

The nsys report was captured with --capture-range=cudaProfilerApi, so it holds
ONLY the single measured generate() (model load + warmup are excluded). We can
therefore count every kernel / launch in the database directly.

Metrics:
  * total_kernels        - number of GPU kernel executions
  * total_kernel_gpu_ns  - SUMMED GPU time of kernels (double counts overlap)
  * gpu_busy_ns          - UNION of GPU activity intervals (kernels+memcpy+memset);
                           the real "GPU busy" time (<= e2e), overlap-safe
  * gpu_bubble_ns        - e2e_ns - gpu_busy_ns
  * gpu_bubble_ratio     - (e2e_ns - gpu_busy_ns) / e2e_ns; fraction of e2e the
                           GPU sat idle (launch + host + sync bubbles)
  * launch_count         - number of kernel-launch host API calls
  * launch_overhead_ns   - summed host time spent in those launch calls
  * launch_overhead_pct  - launch_overhead_ns / e2e_ns * 100 (diagnostic only)
  * unhidden_launch_api_ns - wall-time union of launch API intervals that overlap
                             GPU idle intervals; CPU launch time not hidden by GPU work
  * other_host_idle_ns   - remaining GPU idle wall time not covered by launch APIs
  * e2e_ns               - span of the captured region (host API first..last)
  * top5                 - five kernels with the largest summed GPU time

Note on metric choice: prefer gpu_bubble_ratio (and eager->cudagraph deltas of
it) over launch_overhead_pct -- the latter is inflated by launch-queue back-
pressure in compute-bound cases and misses the host-framework gaps that dominate
launch-bound decode.

Usage: analyze_nsys.py <db.sqlite> <output.json> [metadata flags]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterable

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


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    clean = sorted((int(b), int(e)) for b, e in intervals if b is not None and e is not None and e > b)
    if not clean:
        return []
    merged = []
    cs, ce = clean[0]
    for b, e in clean[1:]:
        if b <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = b, e
    merged.append((cs, ce))
    return merged


def interval_duration_ns(intervals: Iterable[tuple[int, int]]) -> int:
    return int(sum(e - b for b, e in intervals))


def intersect_duration_ns(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Wall-time duration covered by both sorted, non-overlapping interval lists."""
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        ab, ae = a[i]
        bb, be = b[j]
        total += max(0, min(ae, be) - max(ab, bb))
        if ae <= be:
            i += 1
        else:
            j += 1
    return int(total)


def gpu_activity_intervals(con: sqlite3.Connection) -> list[tuple[int, int]]:
    """Union (merged) coverage of all GPU activity intervals.

    Sums kernel + memcpy + memset execution windows with overlaps merged, so it
    is the true GPU-busy time (<= e2e) even when kernels run concurrently across
    streams (where a naive sum of durations would exceed e2e).
    """
    intervals = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL",
                  "CUPTI_ACTIVITY_KIND_MEMCPY",
                  "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table_exists(con, table):
            intervals += con.execute(f"SELECT start, end FROM {table}").fetchall()
    return merge_intervals(intervals)


def gpu_busy_ns(con: sqlite3.Connection) -> int:
    return interval_duration_ns(gpu_activity_intervals(con))


def e2e_bounds_ns(con: sqlite3.Connection) -> tuple[int, int]:
    """Wall bounds of the captured region: earliest..latest host API call.

    RUNTIME events bracket the CPU side of the measured generate(); their span
    approximates its end-to-end wall time. Fall back to kernel span if needed.
    """
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL"):
        if not table_exists(con, table):
            continue
        row = con.execute(f"SELECT MIN(start), MAX(end) FROM {table}").fetchone()
        if row and row[0] is not None:
            return int(row[0]), int(row[1])
    return 0, 0

def e2e_span_ns(con: sqlite3.Connection) -> int:
    start, end = e2e_bounds_ns(con)
    return int(end - start)


def complement_intervals(bounds: tuple[int, int], busy: list[tuple[int, int]]) -> list[tuple[int, int]]:
    start, end = bounds
    if end <= start:
        return []
    idle = []
    cursor = start
    for b, e in busy:
        if e <= start or b >= end:
            continue
        b = max(b, start)
        e = min(e, end)
        if b > cursor:
            idle.append((cursor, b))
        cursor = max(cursor, e)
    if cursor < end:
        idle.append((cursor, end))
    return idle


def launch_api_intervals(con: sqlite3.Connection) -> list[tuple[int, int]]:
    if not table_exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    like = " OR ".join(["s.value LIKE ?"] * len(LAUNCH_API_PATTERNS))
    rows = con.execute(
        f"""
        SELECT r.start, r.end
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        WHERE ({like})
        """,
        LAUNCH_API_PATTERNS,
    ).fetchall()
    return merge_intervals(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("db")
    p.add_argument("out")
    p.add_argument("--model")
    p.add_argument("--mode")
    p.add_argument("--case")
    p.add_argument("--prompt-len", type=int)
    p.add_argument("--decode-len", type=int)
    p.add_argument("--batch-size", type=int)
    args = p.parse_args()

    con = sqlite3.connect(args.db)
    if not table_exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        raise SystemExit("No CUPTI_ACTIVITY_KIND_KERNEL table: capture produced no kernels.")
    e2e_bounds = e2e_bounds_ns(con)
    e2e_ns = int(e2e_bounds[1] - e2e_bounds[0])
    busy_intervals = gpu_activity_intervals(con)
    busy_ns = interval_duration_ns(busy_intervals)
    bubble_ns = max(0, e2e_ns - busy_ns)
    idle_intervals = complement_intervals(e2e_bounds, busy_intervals)
    unhidden_launch_ns = intersect_duration_ns(launch_api_intervals(con), idle_intervals)
    other_host_idle_ns = max(0, bubble_ns - unhidden_launch_ns)
    total_kernels, total_gpu_ns, top5 = kernel_stats(con)
    launch_count, launch_ns, launch_by_api = launch_stats(con)
    con.close()

    metrics = {
        "model": args.model,
        "mode": args.mode,
        "case": args.case,
        "prompt_len": args.prompt_len,
        "decode_len": args.decode_len,
        "batch_size": args.batch_size,
        "e2e_ns": e2e_ns,
        "e2e_ms": e2e_ns / 1e6,
        "gpu_busy_ns": busy_ns,
        "gpu_busy_ms": busy_ns / 1e6,
        "gpu_bubble_ns": bubble_ns,
        "gpu_bubble_ms": bubble_ns / 1e6,
        "gpu_bubble_ratio": bubble_ns / e2e_ns if e2e_ns else 0.0,
        "unhidden_launch_api_ns": unhidden_launch_ns,
        "unhidden_launch_api_ms": unhidden_launch_ns / 1e6,
        "unhidden_launch_api_pct": (unhidden_launch_ns / e2e_ns * 100) if e2e_ns else 0.0,
        "other_host_idle_ns": other_host_idle_ns,
        "other_host_idle_ms": other_host_idle_ns / 1e6,
        "other_host_idle_pct": (other_host_idle_ns / e2e_ns * 100) if e2e_ns else 0.0,
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
          f"gpu_bubble {metrics['gpu_bubble_ratio']*100:.1f}%, "
          f"unhidden_launch {metrics['unhidden_launch_api_ms']:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
