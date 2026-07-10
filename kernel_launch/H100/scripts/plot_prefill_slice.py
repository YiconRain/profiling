#!/usr/bin/env python3
"""Plot BS=1 prefill-only slice metrics from Nsight SQLite traces.

The historical bs1_p*_d0 cases execute prefill + one decode token.  To isolate
prefill, this script cuts the trace at the GPU start time of the second
resolve_future_token_ids_kernel, which marks the beginning of the decode tail.

Do not cut at the CPU launch API for that kernel: the CPU may enqueue decode
work much earlier while the GPU is still draining prefill work.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sqlite3
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODELS = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-30B-A3B",
    "Qwen3.5-27B",
]

DEFAULT_CASES = [
    ("16", "bs1_p16_d0"),
    ("256", "bs1_p256_d0"),
    ("1k", "bs1_p1k_d0"),
    ("4k", "bs1_p4k_d0"),
    ("8k", "bs1_p8k_d0"),
]

LAUNCH_API_PATTERNS = (
    "cudaLaunchKernel%",
    "cudaGraphLaunch%",
    "cuLaunchKernel%",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def valid_sqlite(path: Path) -> bool:
    if not path.exists():
        return False
    if path.suffix == ".gz":
        return True
    try:
        con = sqlite3.connect(path)
        ok = table_exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME") and table_exists(
            con, "CUPTI_ACTIVITY_KIND_KERNEL"
        )
        con.close()
        return ok
    except sqlite3.Error:
        return False


def open_sqlite(path: Path) -> tuple[sqlite3.Connection, str | None]:
    if path.suffix == ".gz":
        data = gzip.open(path, "rb").read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
        tmp.write(data)
        tmp.close()
        return sqlite3.connect(tmp.name), tmp.name
    return sqlite3.connect(path), None


def find_db(
    nsys_roots: list[Path], model: str, mode: str, case: str
) -> Path | None:
    for root in nsys_roots:
        candidates = [
            root / model / mode / f"{case}.sqlite",
            root / model / mode / f"{case}.sqlite.gz",
        ]
        for path in candidates:
            if valid_sqlite(path):
                return path
    return None


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    clean = sorted(
        (int(s), int(e))
        for s, e in intervals
        if s is not None and e is not None and e > s
    )
    if not clean:
        return []
    merged: list[list[int]] = []
    for s, e in clean:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]


def interval_duration(intervals: list[tuple[int, int]]) -> int:
    return int(sum(e - s for s, e in intervals))


def clip_intervals(
    intervals: list[tuple[int, int]], bounds: tuple[int, int]
) -> list[tuple[int, int]]:
    bs, be = bounds
    clipped = []
    for s, e in intervals:
        if e <= bs or s >= be:
            continue
        cs, ce = max(s, bs), min(e, be)
        if ce > cs:
            clipped.append((cs, ce))
    return clipped


def complement_intervals(
    bounds: tuple[int, int], busy: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    start, end = bounds
    cursor = start
    idle = []
    for s, e in busy:
        if e <= start or s >= end:
            continue
        s, e = max(s, start), min(e, end)
        if s > cursor:
            idle.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < end:
        idle.append((cursor, end))
    return idle


def intersect_duration(
    a: list[tuple[int, int]], b: list[tuple[int, int]]
) -> int:
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return int(total)


def second_resolve_boundary(con: sqlite3.Connection) -> int:
    rows = con.execute(
        """
        SELECT k.start
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE s.value LIKE '%resolve_future_token_ids_kernel%'
        ORDER BY k.start
        """
    ).fetchall()
    if len(rows) >= 2:
        return int(rows[1][0])
    # Fallback keeps the script usable for traces without the marker.
    return int(con.execute("SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0])


def prefill_slice_metrics(path: Path) -> dict[str, float | int | str]:
    con, tmp = open_sqlite(path)
    try:
        runtime_start = con.execute(
            "SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        ).fetchone()[0]
        if runtime_start is None:
            raise RuntimeError(f"No runtime events in {path}")
        runtime_start = int(runtime_start)
        boundary = second_resolve_boundary(con)
        bounds = (runtime_start, boundary)
        if boundary <= runtime_start:
            raise RuntimeError(f"Bad prefill slice bounds in {path}: {bounds}")

        activity_intervals: list[tuple[int, int]] = []
        for table in (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
        ):
            if table_exists(con, table):
                activity_intervals.extend(
                    con.execute(f"SELECT start, end FROM {table}").fetchall()
                )

        busy = merge_intervals(clip_intervals(activity_intervals, bounds))
        idle = complement_intervals(bounds, busy)

        kernel_rows = con.execute(
            """
            SELECT start, end, correlationId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE start < ?
            """,
            (boundary,),
        ).fetchall()
        corr_ids = sorted({r[2] for r in kernel_rows if r[2] is not None})

        like_clause = " OR ".join(["s.value LIKE ?"] * len(LAUNCH_API_PATTERNS))
        launch_rows: list[tuple[int, int, str, int]] = []
        for i in range(0, len(corr_ids), 500):
            chunk = corr_ids[i : i + 500]
            if not chunk:
                continue
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT r.start, r.end, s.value, r.correlationId
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN StringIds s ON r.nameId = s.id
                WHERE ({like_clause})
                  AND r.correlationId IN ({placeholders})
            """
            launch_rows.extend(
                con.execute(query, (*LAUNCH_API_PATTERNS, *chunk)).fetchall()
            )

        launch_intervals = merge_intervals([(r[0], r[1]) for r in launch_rows])
        e2e_ns = bounds[1] - bounds[0]
        busy_ns = interval_duration(busy)
        bubble_ns = max(0, e2e_ns - busy_ns)
        unhidden_ns = intersect_duration(launch_intervals, idle)
        total_kernel_gpu_ns = sum(
            max(0, min(end, boundary) - max(start, runtime_start))
            for start, end, _ in kernel_rows
        )

        cuda_graph_launches = sum(
            1 for _, _, api, _ in launch_rows if api.startswith("cudaGraphLaunch")
        )

        return {
            "path": str(path),
            "e2e_ms": e2e_ns / 1e6,
            "gpu_busy_ms": busy_ns / 1e6,
            "bubble_ms": bubble_ns / 1e6,
            "bubble_ratio": bubble_ns / e2e_ns if e2e_ns else 0.0,
            "kernel_count": len(kernel_rows),
            "total_kernel_gpu_ms": total_kernel_gpu_ns / 1e6,
            "launch_count": len(launch_rows),
            "cudaGraphLaunch": cuda_graph_launches,
            "launch_overhead_ms": sum(end - start for start, end, _, _ in launch_rows)
            / 1e6,
            "unhidden_launch_ms": unhidden_ns / 1e6,
            "other_host_idle_ms": max(0, bubble_ns - unhidden_ns) / 1e6,
        }
    finally:
        con.close()
        if tmp is not None:
            os.unlink(tmp)


def plot_figures(
    labels: list[str],
    models: list[str],
    metrics: dict[tuple[str, str], dict[str, float | int | str]],
    out_dir: Path,
    log_e2e: bool,
    write_pdf: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
        }
    )
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
    markers = ["o", "s", "^", "D", "v", "P"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    for i, model in enumerate(models):
        y = [metrics[(model, label)]["bubble_ratio"] * 100 for label in labels]
        ax.plot(
            x,
            y,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("GPU bubble ratio (%)")
    ax.set_title("BS=1 Prefill: GPU Bubble Ratio")
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, loc="upper right")
    fig.savefig(out_dir / "fig3_prefill_bubble.png")
    if write_pdf:
        fig.savefig(out_dir / "fig3_prefill_bubble.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    for i, model in enumerate(models):
        y = [metrics[(model, label)]["e2e_ms"] for label in labels]
        ax.plot(
            x,
            y,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("E2E latency (ms)")
    ax.set_title("BS=1 Prefill: E2E Latency")
    if log_e2e:
        ax.set_yscale("log")
    ax.legend(ncol=3, loc="upper left")
    fig.savefig(out_dir / "fig3_prefill_e2e.png")
    if write_pdf:
        fig.savefig(out_dir / "fig3_prefill_e2e.pdf")
    plt.close(fig)


def print_p8k_table(
    models: list[str],
    metrics: dict[tuple[str, str], dict[str, float | int | str]],
) -> None:
    print("\nBS=1, prompt=8k, eager prefill-only slice:\n")
    print(
        "| Model | e2e ms | launch count | cudaGraphLaunch | bubble ms | bubble% | "
        "unhidden launch ms | other host idle ms |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in models:
        row = metrics[(model, "8k")]
        print(
            f"| {model} | {row['e2e_ms']:.2f} | {int(row['launch_count']):,} | "
            f"{int(row['cudaGraphLaunch']):,} | {row['bubble_ms']:.2f} | "
            f"{row['bubble_ratio'] * 100:.1f}% | {row['unhidden_launch_ms']:.2f} | "
            f"{row['other_host_idle_ms']:.2f} |"
        )


def print_topology_table(
    models: list[str],
    nsys_roots: list[Path],
) -> None:
    print("\nBS=1, prompt=8k, prefill-only topology eager/cudagraph:\n")
    print(
        "| Model | e2e ms eager/cg | launch count eager/cg | "
        "cudaGraphLaunch eager/cg | bubble% eager/cg |"
    )
    print("| --- | ---: | ---: | ---: | ---: |")
    for model in models:
        eager_path = find_db(nsys_roots, model, "eager", "bs1_p8k_d0")
        cg_path = find_db(nsys_roots, model, "cudagraph", "bs1_p8k_d0")
        if eager_path is None or cg_path is None:
            print(f"| {model} | missing | missing | missing | missing |")
            continue
        eager = prefill_slice_metrics(eager_path)
        cg = prefill_slice_metrics(cg_path)
        print(
            f"| {model} | {eager['e2e_ms']:.2f} / {cg['e2e_ms']:.2f} | "
            f"{int(eager['launch_count']):,} / {int(cg['launch_count']):,} | "
            f"{int(eager['cudaGraphLaunch']):,} / {int(cg['cudaGraphLaunch']):,} | "
            f"{eager['bubble_ratio'] * 100:.1f}% / {cg['bubble_ratio'] * 100:.1f}% |"
        )


def parse_args() -> argparse.Namespace:
    root = project_root()
    default_nsys_root = root / "kernel_launch/H100/results/nsys"
    default_extra = Path("/Users/rain/vastai/79/profiling/kernel_launch/H100/results/nsys")
    default_roots = []
    if default_extra.exists():
        default_roots.append(default_extra)
    default_roots.append(default_nsys_root)

    p = argparse.ArgumentParser()
    p.add_argument(
        "--nsys-root",
        action="append",
        type=Path,
        default=None,
        help="Nsight root(s) to search. Can be passed multiple times.",
    )
    p.add_argument("--out-dir", type=Path, default=root / "assets/figs")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument(
        "--cases",
        nargs="+",
        default=[case for _, case in DEFAULT_CASES],
        help="Case ids to plot, e.g. bs1_p16_d0 bs1_p8k_d0.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=[label for label, _ in DEFAULT_CASES],
        help="X-axis labels. Must match --cases length.",
    )
    p.add_argument("--mode", default="eager", choices=["eager", "cudagraph"])
    p.add_argument("--linear-e2e", action="store_true")
    p.add_argument("--pdf", action="store_true", help="Also write PDF copies.")
    p.add_argument("--no-topology", action="store_true")
    args = p.parse_args()
    if args.nsys_root is None:
        args.nsys_root = default_roots
    return args


def main() -> int:
    args = parse_args()
    if len(args.cases) != len(args.labels):
        raise SystemExit("--cases and --labels must have the same length")

    nsys_roots = [p for p in args.nsys_root if p.exists()]
    if not nsys_roots:
        raise SystemExit("No valid nsys roots found")

    metrics: dict[tuple[str, str], dict[str, float | int | str]] = {}
    for model in args.models:
        for label, case in zip(args.labels, args.cases):
            path = find_db(nsys_roots, model, args.mode, case)
            if path is None:
                raise SystemExit(f"Missing trace for {model} {args.mode} {case}")
            metrics[(model, label)] = prefill_slice_metrics(path)

    plot_figures(
        labels=args.labels,
        models=args.models,
        metrics=metrics,
        out_dir=args.out_dir,
        log_e2e=not args.linear_e2e,
        write_pdf=args.pdf,
    )
    print(f"Wrote {args.out_dir / 'fig3_prefill_bubble.png'}")
    print(f"Wrote {args.out_dir / 'fig3_prefill_e2e.png'}")

    if "8k" in args.labels and args.mode == "eager":
        print_p8k_table(args.models, metrics)
    if not args.no_topology:
        print_topology_table(args.models, nsys_roots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
