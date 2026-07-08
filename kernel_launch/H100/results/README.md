# results/ 结果目录（Experiment 2，待运行）

> 本文件为占位说明。实验运行后，`scripts/summarize.py` 会用**结果汇总**覆盖本文件
> （每模型指标表 + eager→cudagraph 对比表 + 各组合 Top-5 kernel）。

## 结构

```
results/
├── nsys/       # 每个 case 的 <模型>/<模式>/<用例>.nsys-rep.gz 与 .sqlite.gz(均保留)
├── metrics/    # 每个组合的指标 JSON + summary.csv
└── README.md   # 运行后生成的中文汇总
```

## 核心指标

- **gpu_bubble_ratio** = `(e2e − gpu_busy)/e2e`,GPU 空闲占比(gpu_busy 为所有 GPU 活动区间的并集)。
- **eager → cudagraph** 的 Δbubble 与 e2e 加速比,用来量化 kernel launch / host 开销的真实影响。
- 口径说明见上级 `../README.md` 第 3 节。

## 产物用途

- `nsys/*.nsys-rep.gz`:`gunzip` 后用 Nsight Systems GUI 可视化;`*.sqlite.gz` 供 SQL 自定义查询。
- `metrics/summary.csv`:所有组合关键指标,便于二次绘图。
