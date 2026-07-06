# results/ 结果目录

> 本文件为占位说明。实验运行后，`scripts/summarize.py` 会用**结果汇总表**覆盖本文件。

## 结构

```
results/
├── nsys/       # 压缩后的采集产物：<模型>/<模式>/<用例>.nsys-rep.gz 与 .sqlite.gz
├── metrics/    # 每个组合的指标 JSON（<模型>__<模式>__<用例>.json）+ summary.csv
└── README.md   # 运行后生成的中文汇总（指标表格 + 各组合 Top-5 kernel）
```

## 用途

- `nsys/*.nsys-rep.gz`：下载到本地 `gunzip` 后用 Nsight Systems GUI 打开，做可视化分析。
- `nsys/*.sqlite.gz`：解压后可用 SQL 自定义查询（表 `CUPTI_ACTIVITY_KIND_KERNEL`、
  `CUPTI_ACTIVITY_KIND_RUNTIME`、`NVTX_EVENTS`、`StringIds` 等）。
- `metrics/summary.csv`：所有组合的关键指标，便于二次绘图。
- 各 `metrics/*.json`：单组合完整指标，含 Top-5 kernel 明细。

## 指标口径

见上级目录 `../README.md` 第 3.3 节。
