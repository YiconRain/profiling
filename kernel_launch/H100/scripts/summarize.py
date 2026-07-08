"""Aggregate per-run metrics JSON into a CSV and a Chinese Markdown report.

Reads every kernel_launch/H100/results/metrics/*.json produced by
run_profiling.py and writes:
  * results/metrics/summary.csv        - one row per (model, mode, case)
  * results/README.md                  - Chinese tables + top-5 kernel listings
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

CSV_FIELDS = [
    "model", "mode", "case", "prompt_len", "decode_len",
    "e2e_ms", "total_kernels", "total_kernel_gpu_ms",
    "launch_count", "launch_overhead_ms", "launch_overhead_pct",
]


def load_metrics() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(Path(C.METRICS_DIR) / "*.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})


def case_label(case_id: str) -> str:
    case = next((c for c in C.CASES if c["id"] == case_id), None)
    if not case:
        return case_id
    return f"{case_id} (P{case['prompt_len']}/D{case['decode_len']})"


def write_readme(rows: list[dict], out: Path) -> None:
    lines: list[str] = []
    lines.append("# H100 Kernel Launch 实验结果\n")
    lines.append("本文档由 `scripts/summarize.py` 自动生成，汇总 SGLang（FlashInfer 后端）"
                 "在 Qwen3.5 系列上的 kernel launch 数量与开销。\n")
    lines.append("## 指标说明\n")
    lines.append("> 采集用 `cudaProfilerStart/Stop` + nsys `--capture-range=cudaProfilerApi`，"
                 "只录被测的那一次 `generate()`；`--cuda-graph-trace=node` 记录 CUDA graph 内的每个 kernel。\n")
    lines.append("- **e2e_ms**：被测 `generate()` 的端到端时间（采集区间内主机 API 的时间跨度）。\n"
                 "- **total_kernels**：GPU kernel 执行次数（含 CUDA graph 内节点）。\n"
                 "- **launch_count**：`cudaLaunchKernel*` / `cudaGraphLaunch*` / `cuLaunchKernel*` 主机侧 API 调用次数。\n"
                 "- **launch_overhead_ms / pct**：这些 launch 调用累计的主机耗时，以及其占端到端时间的比例。\n"
                 "- 口径提醒：launch 占比在 compute-bound 时会被 overlap/反压高估、launch-bound 时低估，"
                 "衡量真实影响更宜看 GPU 空闲率或 eager→cudagraph 加速比；`total_kernel_gpu_ms` 是求和，"
                 "kernel 并发时可能 > e2e。\n")

    # Main table grouped by model.
    models = [m for m in C.MODELS if any(r["model"] == m for r in rows)]
    for model in models:
        lines.append(f"\n## {model}\n")
        lines.append("| Case | Mode | e2e (ms) | Kernels | Launches | "
                     "Launch 开销 (ms) | Launch 占比 (%) |")
        lines.append("|---|---|---|---|---|---|---|")
        for mode in C.MODES:
            for case in C.CASES:
                r = next((x for x in rows if x["model"] == model
                          and x["mode"] == mode and x["case"] == case["id"]), None)
                if not r:
                    continue
                lines.append(
                    f"| {case_label(case['id'])} | {mode} | {r['e2e_ms']:.3f} | "
                    f"{r['total_kernels']} | {r['launch_count']} | "
                    f"{r['launch_overhead_ms']:.3f} | {r['launch_overhead_pct']:.2f} |"
                )

    # Top-5 kernels per run.
    lines.append("\n## 各组合 Top-5 耗时 Kernel\n")
    for r in rows:
        lines.append(f"\n### {r['model']} / {r['mode']} / {case_label(r['case'])}\n")
        lines.append("| # | Kernel | 次数 | GPU 时间 (ms) |")
        lines.append("|---|---|---|---|")
        for i, k in enumerate(r.get("top5_kernels", []), 1):
            lines.append(f"| {i} | `{k['name']}` | {k['count']} | {k['gpu_ms']:.3f} |")

    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    rows = load_metrics()
    if not rows:
        print("No metrics found; run run_profiling.py first.")
        return 0
    Path(C.METRICS_DIR).mkdir(parents=True, exist_ok=True)
    write_csv(rows, Path(C.METRICS_DIR) / "summary.csv")
    write_readme(rows, Path(C.RESULTS_DIR) / "README.md")
    print(f"Wrote {C.METRICS_DIR}/summary.csv and {C.RESULTS_DIR}/README.md "
          f"({len(rows)} runs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
