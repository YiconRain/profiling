"""Aggregate per-run metrics JSON into a CSV and a Chinese Markdown report.

Reads every kernel_launch/H100/results/metrics/*.json produced by
run_profiling.py and writes:
  * results/metrics/summary.csv  - one row per (model, mode, case)
  * results/README.md            - Chinese tables: per-model metrics, the
    eager->cudagraph comparison (bubble drop / speedup), and top-5 kernels.
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
    "model", "mode", "case", "batch_size", "prompt_len", "decode_len",
    "e2e_ms", "gpu_busy_ms", "gpu_bubble_ratio",
    "total_kernels", "total_kernel_gpu_ms",
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
    c = next((c for c in C.CASES if c["id"] == case_id), None)
    if not c:
        return case_id
    return f"{case_id} (BS{c['batch_size']}/P{c['prompt_len']}/D{c['decode_len']})"


def get(rows, model, mode, case_id):
    return next((x for x in rows if x["model"] == model
                 and x["mode"] == mode and x["case"] == case_id), None)


def write_readme(rows: list[dict], out: Path) -> None:
    L: list[str] = []
    L.append("# H100 Kernel Launch 实验结果（Experiment 2）\n")
    L.append("由 `scripts/summarize.py` 自动生成。SGLang（FlashInfer 后端）在 Qwen3 / Qwen3.5 模型上，"
             "eager vs cudagraph 的 kernel launch 与 GPU 空闲(bubble)分析。\n")
    L.append("## 指标说明\n")
    L.append("> 采集：`cudaProfilerStart/Stop` + nsys `--capture-range=cudaProfilerApi`（只录被测那一次 "
             "`generate()`）；`--cuda-graph-trace=node`（记录 CUDA graph 内每个 kernel）。\n")
    L.append("- **e2e_ms**：被测 generate 的端到端时间（采集区间内主机 API 的时间跨度）。\n"
             "- **gpu_busy_ms**：所有 GPU 活动区间(kernel+memcpy+memset)取**并集**的忙碌时间（≤e2e，避免并发重复计）。\n"
             "- **bubble%** = `(e2e − gpu_busy) / e2e`：GPU 空闲占比（launch + host 框架 + sync 造成的气泡）。**核心指标**。\n"
             "- **total_kernels / launch_count**：kernel 执行数 / launch 类 API 调用数（`cudaLaunchKernel*`+`cudaGraphLaunch*`+`cuLaunchKernel*`）。\n"
             "- **launch_overhead_pct**：仅作诊断——compute-bound 时被 overlap/反压高估、launch-bound 时低估,勿当真实影响。\n")

    # Per-model tables.
    models = [m for m in C.MODELS if any(r["model"] == m for r in rows)]
    for model in models:
        L.append(f"\n## {model}\n")
        L.append("| Case | Mode | e2e (ms) | gpu_busy (ms) | bubble% | Kernels | Launches | launch% |")
        L.append("|---|---|---|---|---|---|---|---|")
        for case in C.CASES:
            for mode in C.MODES:
                r = get(rows, model, mode, case["id"])
                if not r:
                    continue
                L.append(
                    f"| {case_label(case['id'])} | {mode} | {r['e2e_ms']:.2f} | "
                    f"{r.get('gpu_busy_ms', 0):.2f} | {r.get('gpu_bubble_ratio', 0)*100:.1f} | "
                    f"{r['total_kernels']} | {r['launch_count']} | {r['launch_overhead_pct']:.1f} |"
                )

    # Core comparison: eager vs cudagraph.
    L.append("\n## eager → cudagraph 对比（核心）\n")
    L.append("> Δbubble = eager 的 bubble% − cudagraph 的 bubble%（cudagraph 消掉的 GPU 空闲）；"
             "speedup = e2e(eager) / e2e(cudagraph)。Δ/speedup 越大 → 该场景越受 kernel launch/host 开销主导。\n")
    L.append("| Model | Case | bubble% eager→cg | Δbubble (pt) | e2e eager→cg (ms) | speedup |")
    L.append("|---|---|---|---|---|---|")
    for model in models:
        for case in C.CASES:
            e = get(rows, model, "eager", case["id"])
            g = get(rows, model, "cudagraph", case["id"])
            if not e or not g:
                continue
            eb, gb = e.get("gpu_bubble_ratio", 0) * 100, g.get("gpu_bubble_ratio", 0) * 100
            sp = e["e2e_ms"] / g["e2e_ms"] if g["e2e_ms"] else 0
            L.append(f"| {model} | {case_label(case['id'])} | {eb:.1f}→{gb:.1f} | {eb-gb:+.1f} | "
                     f"{e['e2e_ms']:.0f}→{g['e2e_ms']:.0f} | {sp:.2f}× |")

    # Top-5 kernels per run.
    L.append("\n## 各组合 Top-5 耗时 Kernel\n")
    for r in rows:
        L.append(f"\n### {r['model']} / {r['mode']} / {case_label(r['case'])}\n")
        L.append("| # | Kernel | 次数 | GPU 时间 (ms) |")
        L.append("|---|---|---|---|")
        for i, k in enumerate(r.get("top5_kernels", []), 1):
            L.append(f"| {i} | `{k['name']}` | {k['count']} | {k['gpu_ms']:.3f} |")

    out.write_text("\n".join(L) + "\n")


def main() -> int:
    rows = load_metrics()
    if not rows:
        print("No metrics found; run run_profiling.py first.")
        return 0
    Path(C.METRICS_DIR).mkdir(parents=True, exist_ok=True)
    write_csv(rows, Path(C.METRICS_DIR) / "summary.csv")
    write_readme(rows, Path(C.RESULTS_DIR) / "README.md")
    print(f"Wrote {C.METRICS_DIR}/summary.csv and {C.RESULTS_DIR}/README.md ({len(rows)} runs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
