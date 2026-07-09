# H100 Kernel Launch 实验结果（Experiment 2）

由 `scripts/summarize.py` 自动生成。SGLang（FlashInfer 后端）在 Qwen3 / Qwen3.5 模型上，eager vs cudagraph 的 kernel launch 与 GPU 空闲(bubble)分析。

## 指标说明

> 采集：`cudaProfilerStart/Stop` + nsys `--capture-range=cudaProfilerApi`（只录被测那一次 `generate()`）；`--cuda-graph-trace=node`（记录 CUDA graph 内每个 kernel）。

- **e2e_ms**：被测 generate 的端到端时间（采集区间内主机 API 的时间跨度）。
- **gpu_busy_ms**：所有 GPU 活动区间(kernel+memcpy+memset)取**并集**的忙碌时间（≤e2e，避免并发重复计）。
- **bubble%** = `(e2e − gpu_busy) / e2e`：GPU 空闲占比（launch + host 框架 + sync 造成的气泡）。**核心指标**。
- **unhidden_launch_api_ms**：GPU idle 区间中，CPU 正在执行 launch API 的墙钟时间；表示未被 GPU work overlap 的 launch API 时间。
- **other_host_idle_ms** = `gpu_bubble_ms − unhidden_launch_api_ms`：GPU idle 但 CPU 不在 launch API 中的时间，通常来自 scheduler/sampling/sync/框架逻辑。
- **total_kernels / launch_count**：kernel 执行数 / launch 类 API 调用数（`cudaLaunchKernel*`+`cudaGraphLaunch*`+`cuLaunchKernel*`）。
- **launch_overhead_pct**：仅作诊断——compute-bound 时被 overlap/反压高估、launch-bound 时低估,勿当真实影响。

> `bs1_p*_d0` 是历史 case id；实际 `worker.py` 会执行 `max_new_tokens=max(decode_len, 1)`，所以这些行是 **prefill + 1 个 decode token**，表格中标为 `D1 effective`。


## Qwen3-0.6B

| Case | Mode | e2e (ms) | gpu_busy (ms) | bubble (ms/%) | unhidden launch (ms) | other host idle (ms) | Kernels | Launches | launch% |
|---|---|---|---|---|---|---|---|---|---|
| bs1_p16_d0 (BS1/P16/D1 effective) | eager | 33.09 | 2.60 | 30.49 / 92.1 | 3.65 | 26.84 | 758 | 505 | 11.5 |
| bs1_p16_d0 (BS1/P16/D1 effective) | cudagraph | 20.01 | 2.68 | 17.33 / 86.6 | 2.56 | 14.77 | 801 | 181 | 13.0 |
| bs1_p256_d0 (BS1/P256/D1 effective) | eager | 22.39 | 3.59 | 18.80 / 84.0 | 2.55 | 16.25 | 798 | 573 | 13.0 |
| bs1_p256_d0 (BS1/P256/D1 effective) | cudagraph | 15.71 | 3.57 | 12.13 / 77.2 | 1.79 | 10.34 | 801 | 209 | 12.9 |
| bs1_p1k_d0 (BS1/P1024/D1 effective) | eager | 24.21 | 6.26 | 17.95 / 74.1 | 1.44 | 16.51 | 786 | 561 | 10.8 |
| bs1_p1k_d0 (BS1/P1024/D1 effective) | cudagraph | 14.55 | 6.30 | 8.25 / 56.7 | 0.97 | 7.28 | 801 | 209 | 14.4 |
| bs1_p4k_d0 (BS1/P4096/D1 effective) | eager | 35.84 | 18.97 | 16.87 / 47.1 | 1.49 | 15.39 | 770 | 545 | 8.3 |
| bs1_p4k_d0 (BS1/P4096/D1 effective) | cudagraph | 25.85 | 19.00 | 6.85 / 26.5 | 0.24 | 6.61 | 773 | 181 | 8.3 |
| bs1_p8k_d0 (BS1/P8192/D1 effective) | eager | 65.15 | 45.28 | 19.88 / 30.5 | 1.70 | 18.18 | 770 | 545 | 5.3 |
| bs1_p8k_d0 (BS1/P8192/D1 effective) | cudagraph | 53.98 | 44.90 | 9.08 / 16.8 | 0.34 | 8.74 | 773 | 181 | 6.0 |
| bs1_p16_d128 (BS1/P16/D128) | eager | 1130.55 | 159.63 | 970.92 / 85.9 | 155.72 | 815.20 | 46050 | 45797 | 14.5 |
| bs1_p16_d128 (BS1/P16/D128) | cudagraph | 247.15 | 164.64 | 82.51 / 33.4 | 52.16 | 30.35 | 49570 | 2341 | 35.3 |
| bs1_p16_d512 (BS1/P16/D512) | eager | 4461.67 | 680.69 | 3780.98 / 84.7 | 596.93 | 3184.04 | 192354 | 192101 | 14.7 |
| bs1_p16_d512 (BS1/P16/D512) | cudagraph | 806.77 | 680.16 | 126.61 / 15.7 | 41.45 | 85.15 | 197026 | 8869 | 30.7 |
| bs4_p16_d128 (BS4/P16/D128) | eager | 1181.11 | 172.27 | 1008.84 / 85.4 | 162.60 | 846.24 | 49626 | 49373 | 14.4 |
| bs4_p16_d128 (BS4/P16/D128) | cudagraph | 212.64 | 175.11 | 37.53 / 17.6 | 9.48 | 28.05 | 53154 | 2341 | 30.4 |
| bs4_p16_d512 (BS4/P16/D512) | eager | 4844.60 | 734.95 | 4109.65 / 84.8 | 687.05 | 3422.60 | 206694 | 206441 | 15.5 |
| bs4_p16_d512 (BS4/P16/D512) | cudagraph | 856.16 | 730.01 | 126.16 / 14.7 | 35.02 | 91.14 | 211366 | 8873 | 29.3 |
| bs8_p16_d128 (BS8/P16/D128) | eager | 1688.11 | 175.59 | 1512.53 / 89.6 | 208.61 | 1303.91 | 49614 | 49389 | 13.2 |
| bs8_p16_d128 (BS8/P16/D128) | cudagraph | 216.47 | 178.82 | 37.65 / 17.4 | 9.68 | 27.97 | 53134 | 2349 | 30.6 |
| bs8_p16_d512 (BS8/P16/D512) | eager | 5212.93 | 759.48 | 4453.45 / 85.4 | 719.46 | 3733.99 | 206497 | 206372 | 15.1 |
| bs8_p16_d512 (BS8/P16/D512) | cudagraph | 878.36 | 756.75 | 121.61 / 13.8 | 31.87 | 89.74 | 211342 | 8877 | 28.2 |
| bs16_p16_d128 (BS16/P16/D128) | eager | 1249.70 | 183.84 | 1065.86 / 85.3 | 167.79 | 898.07 | 49622 | 49397 | 14.1 |
| bs16_p16_d128 (BS16/P16/D128) | cudagraph | 230.49 | 187.71 | 42.78 / 18.6 | 9.83 | 32.95 | 53142 | 2357 | 27.4 |
| bs16_p16_d512 (BS16/P16/D512) | eager | 4899.73 | 823.62 | 4076.11 / 83.2 | 683.49 | 3392.62 | 206678 | 206453 | 15.3 |
| bs16_p16_d512 (BS16/P16/D512) | cudagraph | 949.43 | 818.59 | 130.83 / 13.8 | 30.95 | 99.88 | 211350 | 8885 | 25.5 |

## Qwen3-1.7B

| Case | Mode | e2e (ms) | gpu_busy (ms) | bubble (ms/%) | unhidden launch (ms) | other host idle (ms) | Kernels | Launches | launch% |
|---|---|---|---|---|---|---|---|---|---|
| bs1_p16_d0 (BS1/P16/D1 effective) | eager | 22.92 | 4.15 | 18.77 / 81.9 | 2.31 | 16.46 | 770 | 517 | 11.4 |
| bs1_p16_d0 (BS1/P16/D1 effective) | cudagraph | 15.34 | 4.19 | 11.16 / 72.7 | 1.70 | 9.46 | 801 | 181 | 12.5 |
| bs1_p256_d0 (BS1/P256/D1 effective) | eager | 23.30 | 5.33 | 17.97 / 77.1 | 2.20 | 15.77 | 798 | 573 | 11.5 |
| bs1_p256_d0 (BS1/P256/D1 effective) | cudagraph | 15.59 | 5.35 | 10.23 / 65.6 | 1.66 | 8.57 | 801 | 209 | 12.8 |
| bs1_p1k_d0 (BS1/P1024/D1 effective) | eager | 26.12 | 9.93 | 16.19 / 62.0 | 1.44 | 14.74 | 798 | 573 | 11.0 |
| bs1_p1k_d0 (BS1/P1024/D1 effective) | cudagraph | 16.21 | 9.97 | 6.24 / 38.5 | 0.29 | 5.95 | 801 | 209 | 12.8 |
| bs1_p4k_d0 (BS1/P4096/D1 effective) | eager | 48.61 | 32.51 | 16.11 / 33.1 | 1.35 | 14.75 | 770 | 545 | 5.8 |
| bs1_p4k_d0 (BS1/P4096/D1 effective) | cudagraph | 39.46 | 32.63 | 6.83 / 17.3 | 0.20 | 6.64 | 773 | 181 | 5.2 |
| bs1_p8k_d0 (BS1/P8192/D1 effective) | eager | 100.87 | 76.63 | 24.24 / 24.0 | 2.00 | 22.24 | 770 | 545 | 4.2 |
| bs1_p8k_d0 (BS1/P8192/D1 effective) | cudagraph | 83.40 | 75.87 | 7.54 / 9.0 | 0.18 | 7.36 | 773 | 181 | 2.7 |
| bs1_p16_d128 (BS1/P16/D128) | eager | 1925.37 | 256.94 | 1668.43 / 86.7 | 223.90 | 1444.53 | 46038 | 45785 | 12.5 |
| bs1_p16_d128 (BS1/P16/D128) | cudagraph | 298.28 | 264.67 | 33.61 / 11.3 | 6.15 | 27.46 | 49570 | 2341 | 19.6 |
| bs1_p16_d512 (BS1/P16/D512) | eager | 4485.19 | 1063.26 | 3421.93 / 76.3 | 580.63 | 2841.30 | 192323 | 192083 | 14.7 |
| bs1_p16_d512 (BS1/P16/D512) | cudagraph | 1185.14 | 1075.78 | 109.36 / 9.2 | 19.72 | 89.63 | 197026 | 8869 | 19.3 |
| bs4_p16_d128 (BS4/P16/D128) | eager | 1188.29 | 270.62 | 917.67 / 77.2 | 157.86 | 759.80 | 49610 | 49385 | 14.3 |
| bs4_p16_d128 (BS4/P16/D128) | cudagraph | 316.13 | 275.57 | 40.56 / 12.8 | 7.43 | 33.13 | 53130 | 2345 | 20.1 |
| bs4_p16_d512 (BS4/P16/D512) | eager | 4868.91 | 1127.50 | 3741.41 / 76.8 | 644.28 | 3097.13 | 206666 | 206441 | 14.8 |
| bs4_p16_d512 (BS4/P16/D512) | cudagraph | 1252.86 | 1122.13 | 130.73 / 10.4 | 25.26 | 105.47 | 211338 | 8873 | 20.5 |
| bs8_p16_d128 (BS8/P16/D128) | eager | 1223.64 | 275.76 | 947.88 / 77.5 | 161.77 | 786.11 | 49614 | 49389 | 14.1 |
| bs8_p16_d128 (BS8/P16/D128) | cudagraph | 328.36 | 282.88 | 45.49 / 13.9 | 10.01 | 35.48 | 53134 | 2349 | 27.1 |
| bs8_p16_d512 (BS8/P16/D512) | eager | 4821.64 | 1157.49 | 3664.15 / 76.0 | 638.62 | 3025.53 | 206670 | 206445 | 14.8 |
| bs8_p16_d512 (BS8/P16/D512) | cudagraph | 1279.60 | 1158.38 | 121.21 / 9.5 | 21.53 | 99.68 | 211342 | 8877 | 19.6 |
| bs16_p16_d128 (BS16/P16/D128) | eager | 1227.06 | 284.50 | 942.56 / 76.8 | 170.28 | 772.28 | 49622 | 49397 | 15.1 |
| bs16_p16_d128 (BS16/P16/D128) | cudagraph | 333.80 | 292.44 | 41.35 / 12.4 | 6.96 | 34.39 | 53142 | 2357 | 19.3 |
| bs16_p16_d512 (BS16/P16/D512) | eager | 4803.92 | 1220.31 | 3583.61 / 74.6 | 638.44 | 2945.17 | 206677 | 206452 | 15.0 |
| bs16_p16_d512 (BS16/P16/D512) | cudagraph | 1350.18 | 1220.73 | 129.46 / 9.6 | 21.94 | 107.52 | 211350 | 8885 | 18.4 |

## Qwen3-8B

| Case | Mode | e2e (ms) | gpu_busy (ms) | bubble (ms/%) | unhidden launch (ms) | other host idle (ms) | Kernels | Launches | launch% |
|---|---|---|---|---|---|---|---|---|---|
| bs1_p16_d0 (BS1/P16/D1 effective) | eager | 27.80 | 12.66 | 15.14 / 54.5 | 1.47 | 13.67 | 1050 | 689 | 13.1 |
| bs1_p16_d0 (BS1/P16/D1 effective) | cudagraph | 19.29 | 12.85 | 6.43 / 33.4 | 0.83 | 5.60 | 1088 | 220 | 13.3 |
| bs1_p256_d0 (BS1/P256/D1 effective) | eager | 30.66 | 15.63 | 15.02 / 49.0 | 1.28 | 13.74 | 1050 | 761 | 12.0 |
| bs1_p256_d0 (BS1/P256/D1 effective) | cudagraph | 21.18 | 15.66 | 5.52 / 26.1 | 0.18 | 5.35 | 1041 | 245 | 10.8 |
| bs1_p1k_d0 (BS1/P1024/D1 effective) | eager | 47.79 | 32.81 | 14.99 / 31.4 | 1.21 | 13.77 | 1014 | 725 | 7.4 |
| bs1_p1k_d0 (BS1/P1024/D1 effective) | cudagraph | 38.91 | 32.25 | 6.66 / 17.1 | 0.15 | 6.51 | 984 | 209 | 6.1 |
| bs1_p4k_d0 (BS1/P4096/D1 effective) | eager | 143.62 | 127.02 | 16.59 / 11.6 | 1.18 | 15.42 | 1014 | 725 | 2.8 |
| bs1_p4k_d0 (BS1/P4096/D1 effective) | cudagraph | 130.59 | 125.23 | 5.36 / 4.1 | 0.22 | 5.14 | 987 | 221 | 2.6 |
| bs1_p8k_d0 (BS1/P8192/D1 effective) | eager | 302.92 | 287.03 | 15.89 / 5.2 | 1.09 | 14.81 | 1014 | 725 | 1.2 |
| bs1_p8k_d0 (BS1/P8192/D1 effective) | cudagraph | 283.69 | 279.43 | 4.26 / 1.5 | 0.15 | 4.12 | 970 | 221 | 0.9 |
| bs1_p16_d128 (BS1/P16/D128) | eager | 1417.47 | 793.70 | 623.77 / 44.0 | 109.55 | 514.22 | 63222 | 62861 | 15.4 |
| bs1_p16_d128 (BS1/P16/D128) | cudagraph | 843.04 | 809.15 | 33.89 / 4.0 | 4.95 | 28.93 | 67631 | 2381 | 9.5 |
| bs1_p16_d512 (BS1/P16/D512) | eager | 5852.37 | 3196.05 | 2656.32 / 45.4 | 475.95 | 2180.38 | 263286 | 262925 | 16.1 |
| bs1_p16_d512 (BS1/P16/D512) | cudagraph | 3316.94 | 3230.11 | 86.83 / 2.6 | 7.28 | 79.54 | 268833 | 8909 | 9.3 |
| bs4_p16_d128 (BS4/P16/D128) | eager | 1563.56 | 818.70 | 744.86 / 47.6 | 130.69 | 614.17 | 67798 | 67473 | 14.7 |
| bs4_p16_d128 (BS4/P16/D128) | cudagraph | 878.58 | 833.72 | 44.86 / 5.1 | 5.69 | 39.17 | 72212 | 2385 | 9.6 |
| bs4_p16_d512 (BS4/P16/D512) | eager | 6197.37 | 3316.31 | 2881.06 / 46.5 | 519.67 | 2361.39 | 281683 | 281358 | 15.5 |
| bs4_p16_d512 (BS4/P16/D512) | cudagraph | 3472.29 | 3340.53 | 131.76 / 3.8 | 11.86 | 119.90 | 287226 | 8906 | 9.5 |
| bs8_p16_d128 (BS8/P16/D128) | eager | 1523.34 | 827.66 | 695.68 / 45.7 | 129.27 | 566.41 | 67766 | 67477 | 15.8 |
| bs8_p16_d128 (BS8/P16/D128) | cudagraph | 879.80 | 841.08 | 38.72 / 4.4 | 3.15 | 35.58 | 72182 | 2389 | 9.4 |
| bs8_p16_d512 (BS8/P16/D512) | eager | 6153.62 | 3390.34 | 2763.27 / 44.9 | 513.73 | 2249.54 | 281654 | 281365 | 16.0 |
| bs8_p16_d512 (BS8/P16/D512) | cudagraph | 3516.50 | 3391.23 | 125.28 / 3.6 | 10.35 | 114.93 | 287222 | 8917 | 9.1 |
| bs16_p16_d128 (BS16/P16/D128) | eager | 1545.38 | 848.86 | 696.52 / 45.1 | 124.07 | 572.46 | 67774 | 67485 | 14.9 |
| bs16_p16_d128 (BS16/P16/D128) | cudagraph | 896.73 | 859.20 | 37.54 / 4.2 | 3.02 | 34.52 | 72190 | 2397 | 9.3 |
| bs16_p16_d512 (BS16/P16/D512) | eager | 6374.07 | 3534.81 | 2839.27 / 44.5 | 552.13 | 2287.14 | 281662 | 281373 | 16.3 |
| bs16_p16_d512 (BS16/P16/D512) | cudagraph | 3610.56 | 3501.48 | 109.08 / 3.0 | 9.04 | 100.04 | 287230 | 8925 | 9.1 |

## Qwen3-14B

| Case | Mode | e2e (ms) | gpu_busy (ms) | bubble (ms/%) | unhidden launch (ms) | other host idle (ms) | Kernels | Launches | launch% |
|---|---|---|---|---|---|---|---|---|---|
| bs1_p16_d0 (BS1/P16/D1 effective) | eager | 32.21 | 21.47 | 10.74 / 33.4 | 0.78 | 9.96 | 1122 | 761 | 11.8 |
| bs1_p16_d0 (BS1/P16/D1 effective) | cudagraph | 28.41 | 21.16 | 7.25 / 25.5 | 0.29 | 6.96 | 1162 | 241 | 9.1 |
| bs1_p256_d0 (BS1/P256/D1 effective) | eager | 38.51 | 26.44 | 12.08 / 31.4 | 0.70 | 11.38 | 1162 | 841 | 10.4 |
| bs1_p256_d0 (BS1/P256/D1 effective) | cudagraph | 25.13 | 21.32 | 3.81 / 15.1 | 0.30 | 3.50 | 906 | 281 | 11.2 |
| bs1_p1k_d0 (BS1/P1024/D1 effective) | eager | 72.66 | 60.84 | 11.81 / 16.3 | 0.56 | 11.26 | 1122 | 801 | 5.5 |
| bs1_p1k_d0 (BS1/P1024/D1 effective) | cudagraph | 56.03 | 52.94 | 3.09 / 5.5 | 0.20 | 2.89 | 770 | 241 | 4.6 |
| bs1_p4k_d0 (BS1/P4096/D1 effective) | eager | 241.94 | 229.56 | 12.38 / 5.1 | 0.46 | 11.92 | 1122 | 801 | 1.7 |
| bs1_p4k_d0 (BS1/P4096/D1 effective) | cudagraph | 229.23 | 224.98 | 4.26 / 1.9 | 0.16 | 4.09 | 882 | 241 | 1.2 |
| bs1_p8k_d0 (BS1/P8192/D1 effective) | eager | 501.25 | 488.33 | 12.92 / 2.6 | 0.45 | 12.46 | 1122 | 801 | 0.9 |
| bs1_p8k_d0 (BS1/P8192/D1 effective) | cudagraph | 488.71 | 484.70 | 4.01 / 0.8 | 0.14 | 3.87 | 882 | 241 | 0.6 |
| bs1_p16_d128 (BS1/P16/D128) | eager | 1597.92 | 1365.85 | 232.08 / 14.5 | 36.60 | 195.48 | 69962 | 69601 | 14.7 |
| bs1_p16_d128 (BS1/P16/D128) | cudagraph | 1409.08 | 1380.41 | 28.68 / 2.0 | 2.22 | 26.46 | 74719 | 2401 | 7.3 |
| bs1_p16_d512 (BS1/P16/D512) | eager | 7711.78 | 5478.94 | 2232.84 / 29.0 | 347.03 | 1885.81 | 291530 | 291169 | 15.0 |
| bs1_p16_d512 (BS1/P16/D512) | cudagraph | 5637.81 | 5517.67 | 120.14 / 2.1 | 7.22 | 112.93 | 297546 | 8929 | 6.6 |
| bs4_p16_d128 (BS4/P16/D128) | eager | 1690.84 | 1398.56 | 292.28 / 17.3 | 46.62 | 245.66 | 75086 | 74725 | 14.9 |
| bs4_p16_d128 (BS4/P16/D128) | cudagraph | 1441.17 | 1403.92 | 37.25 / 2.6 | 2.27 | 34.98 | 79658 | 2405 | 6.4 |
| bs4_p16_d512 (BS4/P16/D512) | eager | 7720.47 | 5631.80 | 2088.67 / 27.1 | 340.12 | 1748.55 | 312014 | 311653 | 15.8 |
| bs4_p16_d512 (BS4/P16/D512) | cudagraph | 5773.19 | 5630.28 | 142.91 / 2.5 | 8.00 | 134.91 | 318030 | 8933 | 6.1 |
| bs8_p16_d128 (BS8/P16/D128) | eager | 1713.88 | 1414.96 | 298.92 / 17.4 | 50.73 | 248.19 | 75090 | 74729 | 15.7 |
| bs8_p16_d128 (BS8/P16/D128) | cudagraph | 1469.44 | 1425.91 | 43.53 / 3.0 | 2.35 | 41.18 | 79954 | 2409 | 6.3 |
| bs8_p16_d512 (BS8/P16/D512) | eager | 6826.86 | 5740.24 | 1086.62 / 15.9 | 181.28 | 905.34 | 312018 | 311657 | 15.5 |
| bs8_p16_d512 (BS8/P16/D512) | cudagraph | 5860.65 | 5716.36 | 144.29 / 2.5 | 8.26 | 136.03 | 317859 | 8937 | 6.3 |
| bs16_p16_d128 (BS16/P16/D128) | eager | 1695.57 | 1428.15 | 267.42 / 15.8 | 41.77 | 225.64 | 69938 | 69617 | 14.7 |
| bs16_p16_d128 (BS16/P16/D128) | cudagraph | 1468.91 | 1429.07 | 39.84 / 2.7 | 2.39 | 37.45 | 74683 | 2417 | 6.2 |
| bs16_p16_d512 (BS16/P16/D512) | eager | 6745.21 | 5843.08 | 902.13 / 13.4 | 147.80 | 754.33 | 291506 | 291185 | 15.1 |
| bs16_p16_d512 (BS16/P16/D512) | cudagraph | 5898.31 | 5785.44 | 112.86 / 1.9 | 7.25 | 105.61 | 297522 | 8945 | 7.7 |

## Qwen3-30B-A3B

| Case | Mode | e2e (ms) | gpu_busy (ms) | bubble (ms/%) | unhidden launch (ms) | other host idle (ms) | Kernels | Launches | launch% |
|---|---|---|---|---|---|---|---|---|---|
| bs1_p16_d0 (BS1/P16/D1 effective) | eager | 55.72 | 14.10 | 41.62 / 74.7 | 3.37 | 38.26 | 1674 | 1001 | 9.9 |
| bs1_p16_d0 (BS1/P16/D1 effective) | cudagraph | 24.84 | 14.17 | 10.67 / 43.0 | 1.88 | 8.79 | 1708 | 220 | 17.4 |
| bs1_p256_d0 (BS1/P256/D1 effective) | eager | 63.90 | 21.41 | 42.49 / 66.5 | 3.41 | 39.08 | 1722 | 1097 | 9.3 |
| bs1_p256_d0 (BS1/P256/D1 effective) | cudagraph | 28.86 | 21.46 | 7.41 / 25.7 | 0.31 | 7.09 | 1725 | 281 | 12.3 |
| bs1_p1k_d0 (BS1/P1024/D1 effective) | eager | 71.54 | 29.72 | 41.82 / 58.5 | 3.13 | 38.69 | 1674 | 1049 | 7.5 |
| bs1_p1k_d0 (BS1/P1024/D1 effective) | cudagraph | 36.90 | 29.90 | 7.00 / 19.0 | 0.28 | 6.72 | 1677 | 233 | 9.3 |
| bs1_p4k_d0 (BS1/P4096/D1 effective) | eager | 135.36 | 92.66 | 42.70 / 31.5 | 3.20 | 39.50 | 1674 | 1049 | 4.3 |
| bs1_p4k_d0 (BS1/P4096/D1 effective) | cudagraph | 101.10 | 93.50 | 7.59 / 7.5 | 0.19 | 7.40 | 1677 | 233 | 3.5 |
| bs1_p8k_d0 (BS1/P8192/D1 effective) | eager | 261.98 | 218.43 | 43.55 / 16.6 | 3.25 | 40.31 | 1674 | 1049 | 2.3 |
| bs1_p8k_d0 (BS1/P8192/D1 effective) | cudagraph | 224.45 | 216.41 | 8.04 / 3.6 | 0.17 | 7.87 | 1677 | 233 | 1.6 |
| bs1_p16_d128 (BS1/P16/D128) | eager | 5467.08 | 538.67 | 4928.41 / 90.1 | 444.86 | 4483.55 | 102138 | 101465 | 9.0 |
| bs1_p16_d128 (BS1/P16/D128) | cudagraph | 606.35 | 549.42 | 56.92 / 9.4 | 10.08 | 46.84 | 107898 | 2393 | 20.2 |
| bs1_p16_d512 (BS1/P16/D512) | eager | 18252.04 | 2183.01 | 16069.03 / 88.0 | 1561.78 | 14507.25 | 422010 | 421337 | 9.8 |
| bs1_p16_d512 (BS1/P16/D512) | cudagraph | 2371.91 | 2181.74 | 190.17 / 8.0 | 34.21 | 155.96 | 428922 | 8921 | 19.0 |
| bs4_p16_d128 (BS4/P16/D128) | eager | 4902.08 | 587.46 | 4314.62 / 88.0 | 429.64 | 3884.99 | 108286 | 107613 | 9.6 |
| bs4_p16_d128 (BS4/P16/D128) | cudagraph | 663.67 | 601.30 | 62.37 / 9.4 | 10.77 | 51.61 | 114008 | 2377 | 19.2 |
| bs4_p16_d512 (BS4/P16/D512) | eager | 19228.19 | 2384.16 | 16844.04 / 87.6 | 1659.33 | 15184.70 | 446590 | 445917 | 9.7 |
| bs4_p16_d512 (BS4/P16/D512) | cudagraph | 2614.13 | 2378.98 | 235.16 / 9.0 | 43.44 | 191.71 | 453502 | 8925 | 19.4 |
| bs8_p16_d128 (BS8/P16/D128) | eager | 4851.63 | 595.22 | 4256.41 / 87.7 | 442.88 | 3813.53 | 108242 | 107617 | 10.0 |
| bs8_p16_d128 (BS8/P16/D128) | cudagraph | 671.40 | 605.38 | 66.03 / 9.8 | 11.14 | 54.88 | 114002 | 2401 | 18.7 |
| bs8_p16_d512 (BS8/P16/D512) | eager | 20019.29 | 2435.85 | 17583.43 / 87.8 | 1723.72 | 15859.72 | 446546 | 445921 | 9.7 |
| bs8_p16_d512 (BS8/P16/D512) | cudagraph | 2655.61 | 2423.23 | 232.38 / 8.8 | 41.19 | 191.19 | 453458 | 8929 | 18.4 |
| bs16_p16_d128 (BS16/P16/D128) | eager | 4960.55 | 629.93 | 4330.63 / 87.3 | 450.24 | 3880.38 | 108250 | 107625 | 9.9 |
| bs16_p16_d128 (BS16/P16/D128) | cudagraph | 705.50 | 637.95 | 67.56 / 9.6 | 12.26 | 55.29 | 114010 | 2409 | 20.1 |
| bs16_p16_d512 (BS16/P16/D512) | eager | 29188.10 | 2626.73 | 26561.37 / 91.0 | 2361.66 | 24199.71 | 446554 | 445929 | 9.2 |
| bs16_p16_d512 (BS16/P16/D512) | cudagraph | 2824.11 | 2578.64 | 245.46 / 8.7 | 42.02 | 203.44 | 453466 | 8937 | 18.1 |

## Qwen3.5-27B

| Case | Mode | e2e (ms) | gpu_busy (ms) | bubble (ms/%) | unhidden launch (ms) | other host idle (ms) | Kernels | Launches | launch% |
|---|---|---|---|---|---|---|---|---|---|
| bs1_p16_d0 (BS1/P16/D1 effective) | eager | 125.44 | 41.81 | 83.63 / 66.7 | 8.52 | 75.11 | 2758 | 2758 | 10.7 |
| bs1_p16_d0 (BS1/P16/D1 effective) | cudagraph | 90.67 | 25.57 | 65.10 / 71.8 | 7.71 | 57.39 | 1976 | 1813 | 11.3 |
| bs1_p256_d0 (BS1/P256/D1 effective) | eager | 127.73 | 47.98 | 79.75 / 62.4 | 7.82 | 71.93 | 2723 | 2723 | 10.3 |
| bs1_p256_d0 (BS1/P256/D1 effective) | cudagraph | 83.58 | 31.44 | 52.14 / 62.4 | 6.55 | 45.59 | 1920 | 1762 | 11.6 |
| bs1_p1k_d0 (BS1/P1024/D1 effective) | eager | 137.99 | 105.37 | 32.62 / 23.6 | 2.81 | 29.81 | 2707 | 2707 | 9.3 |
| bs1_p1k_d0 (BS1/P1024/D1 effective) | cudagraph | 107.06 | 88.64 | 18.42 / 17.2 | 2.81 | 15.61 | 1923 | 1746 | 11.5 |
| bs1_p4k_d0 (BS1/P4096/D1 effective) | eager | 405.75 | 372.26 | 33.49 / 8.3 | 3.11 | 30.38 | 2659 | 2659 | 20.9 |
| bs1_p4k_d0 (BS1/P4096/D1 effective) | cudagraph | 368.22 | 358.02 | 10.19 / 2.8 | 0.68 | 9.52 | 1858 | 1698 | 21.6 |
| bs1_p8k_d0 (BS1/P8192/D1 effective) | eager | 778.62 | 742.29 | 36.33 / 4.7 | 3.25 | 33.07 | 2659 | 2659 | 28.3 |
| bs1_p8k_d0 (BS1/P8192/D1 effective) | cudagraph | 736.52 | 726.95 | 9.57 / 1.3 | 0.98 | 8.59 | 1861 | 1698 | 31.0 |
| bs1_p16_d128 (BS1/P16/D128) | eager | 5555.66 | 2526.17 | 3029.49 / 54.5 | 291.83 | 2737.66 | 125316 | 125316 | 10.7 |
| bs1_p16_d128 (BS1/P16/D128) | cudagraph | 2577.79 | 2479.44 | 98.35 / 3.8 | 8.56 | 89.79 | 126537 | 4100 | 4.5 |
| bs1_p16_d512 (BS1/P16/D512) | eager | 22268.24 | 10057.71 | 12210.53 / 54.8 | 1155.02 | 11055.51 | 501252 | 501252 | 10.5 |
| bs1_p16_d512 (BS1/P16/D512) | cudagraph | 10129.42 | 9908.02 | 221.40 / 2.2 | 13.19 | 208.21 | 503235 | 11012 | 4.6 |
| bs4_p16_d128 (BS4/P16/D128) | eager | 6336.74 | 2640.44 | 3696.31 / 58.3 | 379.66 | 3316.65 | 145822 | 145822 | 10.9 |
| bs4_p16_d128 (BS4/P16/D128) | cudagraph | 2688.56 | 2581.25 | 107.31 / 4.0 | 9.36 | 97.94 | 146859 | 4083 | 5.2 |
| bs4_p16_d512 (BS4/P16/D512) | eager | 24395.69 | 10513.91 | 13881.78 / 56.9 | 1465.98 | 12415.80 | 583198 | 583198 | 11.4 |
| bs4_p16_d512 (BS4/P16/D512) | cudagraph | 10564.52 | 10315.95 | 248.57 / 2.4 | 15.61 | 232.96 | 585061 | 11038 | 4.9 |
| bs8_p16_d128 (BS8/P16/D128) | eager | 6360.73 | 2741.33 | 3619.40 / 56.9 | 387.49 | 3231.91 | 145858 | 145858 | 11.2 |
| bs8_p16_d128 (BS8/P16/D128) | cudagraph | 2788.76 | 2684.24 | 104.52 / 3.7 | 8.80 | 95.72 | 146959 | 4162 | 5.0 |
| bs8_p16_d512 (BS8/P16/D512) | eager | 24558.33 | 10914.72 | 13643.60 / 55.6 | 1429.07 | 12214.54 | 583234 | 583234 | 11.1 |
| bs8_p16_d512 (BS8/P16/D512) | cudagraph | 10811.44 | 10708.16 | 103.28 / 1.0 | 7.98 | 95.30 | 585133 | 11074 | 4.9 |
| bs16_p16_d128 (BS16/P16/D128) | eager | 6233.30 | 2868.42 | 3364.88 / 54.0 | 342.26 | 3022.62 | 145866 | 145866 | 11.3 |
| bs16_p16_d128 (BS16/P16/D128) | cudagraph | 2910.93 | 2814.46 | 96.47 / 3.3 | 7.98 | 88.49 | 146988 | 4170 | 4.7 |
| bs16_p16_d512 (BS16/P16/D512) | eager | 24535.77 | 11431.79 | 13103.99 / 53.4 | 1311.48 | 11792.50 | 583242 | 583242 | 11.2 |
| bs16_p16_d512 (BS16/P16/D512) | cudagraph | 11442.75 | 11239.14 | 203.60 / 1.8 | 12.33 | 191.27 | 585127 | 11082 | 4.9 |

## eager → cudagraph 对比（核心）

> Δbubble = eager 的 bubble% − cudagraph 的 bubble%（cudagraph 消掉的 GPU 空闲）；speedup = e2e(eager) / e2e(cudagraph)。Δ/speedup 越大 → 该场景越受 kernel launch/host 开销主导。

| Model | Case | bubble% eager→cg | Δbubble (pt) | e2e eager→cg (ms) | speedup |
|---|---|---|---|---|---|
| Qwen3-0.6B | bs1_p16_d0 (BS1/P16/D1 effective) | 92.1→86.6 | +5.5 | 33→20 | 1.65× |
| Qwen3-0.6B | bs1_p256_d0 (BS1/P256/D1 effective) | 84.0→77.2 | +6.7 | 22→16 | 1.43× |
| Qwen3-0.6B | bs1_p1k_d0 (BS1/P1024/D1 effective) | 74.1→56.7 | +17.4 | 24→15 | 1.66× |
| Qwen3-0.6B | bs1_p4k_d0 (BS1/P4096/D1 effective) | 47.1→26.5 | +20.6 | 36→26 | 1.39× |
| Qwen3-0.6B | bs1_p8k_d0 (BS1/P8192/D1 effective) | 30.5→16.8 | +13.7 | 65→54 | 1.21× |
| Qwen3-0.6B | bs1_p16_d128 (BS1/P16/D128) | 85.9→33.4 | +52.5 | 1131→247 | 4.57× |
| Qwen3-0.6B | bs1_p16_d512 (BS1/P16/D512) | 84.7→15.7 | +69.1 | 4462→807 | 5.53× |
| Qwen3-0.6B | bs4_p16_d128 (BS4/P16/D128) | 85.4→17.6 | +67.8 | 1181→213 | 5.55× |
| Qwen3-0.6B | bs4_p16_d512 (BS4/P16/D512) | 84.8→14.7 | +70.1 | 4845→856 | 5.66× |
| Qwen3-0.6B | bs8_p16_d128 (BS8/P16/D128) | 89.6→17.4 | +72.2 | 1688→216 | 7.80× |
| Qwen3-0.6B | bs8_p16_d512 (BS8/P16/D512) | 85.4→13.8 | +71.6 | 5213→878 | 5.93× |
| Qwen3-0.6B | bs16_p16_d128 (BS16/P16/D128) | 85.3→18.6 | +66.7 | 1250→230 | 5.42× |
| Qwen3-0.6B | bs16_p16_d512 (BS16/P16/D512) | 83.2→13.8 | +69.4 | 4900→949 | 5.16× |
| Qwen3-1.7B | bs1_p16_d0 (BS1/P16/D1 effective) | 81.9→72.7 | +9.2 | 23→15 | 1.49× |
| Qwen3-1.7B | bs1_p256_d0 (BS1/P256/D1 effective) | 77.1→65.6 | +11.5 | 23→16 | 1.49× |
| Qwen3-1.7B | bs1_p1k_d0 (BS1/P1024/D1 effective) | 62.0→38.5 | +23.5 | 26→16 | 1.61× |
| Qwen3-1.7B | bs1_p4k_d0 (BS1/P4096/D1 effective) | 33.1→17.3 | +15.8 | 49→39 | 1.23× |
| Qwen3-1.7B | bs1_p8k_d0 (BS1/P8192/D1 effective) | 24.0→9.0 | +15.0 | 101→83 | 1.21× |
| Qwen3-1.7B | bs1_p16_d128 (BS1/P16/D128) | 86.7→11.3 | +75.4 | 1925→298 | 6.45× |
| Qwen3-1.7B | bs1_p16_d512 (BS1/P16/D512) | 76.3→9.2 | +67.1 | 4485→1185 | 3.78× |
| Qwen3-1.7B | bs4_p16_d128 (BS4/P16/D128) | 77.2→12.8 | +64.4 | 1188→316 | 3.76× |
| Qwen3-1.7B | bs4_p16_d512 (BS4/P16/D512) | 76.8→10.4 | +66.4 | 4869→1253 | 3.89× |
| Qwen3-1.7B | bs8_p16_d128 (BS8/P16/D128) | 77.5→13.9 | +63.6 | 1224→328 | 3.73× |
| Qwen3-1.7B | bs8_p16_d512 (BS8/P16/D512) | 76.0→9.5 | +66.5 | 4822→1280 | 3.77× |
| Qwen3-1.7B | bs16_p16_d128 (BS16/P16/D128) | 76.8→12.4 | +64.4 | 1227→334 | 3.68× |
| Qwen3-1.7B | bs16_p16_d512 (BS16/P16/D512) | 74.6→9.6 | +65.0 | 4804→1350 | 3.56× |
| Qwen3-8B | bs1_p16_d0 (BS1/P16/D1 effective) | 54.5→33.4 | +21.1 | 28→19 | 1.44× |
| Qwen3-8B | bs1_p256_d0 (BS1/P256/D1 effective) | 49.0→26.1 | +22.9 | 31→21 | 1.45× |
| Qwen3-8B | bs1_p1k_d0 (BS1/P1024/D1 effective) | 31.4→17.1 | +14.2 | 48→39 | 1.23× |
| Qwen3-8B | bs1_p4k_d0 (BS1/P4096/D1 effective) | 11.6→4.1 | +7.4 | 144→131 | 1.10× |
| Qwen3-8B | bs1_p8k_d0 (BS1/P8192/D1 effective) | 5.2→1.5 | +3.7 | 303→284 | 1.07× |
| Qwen3-8B | bs1_p16_d128 (BS1/P16/D128) | 44.0→4.0 | +40.0 | 1417→843 | 1.68× |
| Qwen3-8B | bs1_p16_d512 (BS1/P16/D512) | 45.4→2.6 | +42.8 | 5852→3317 | 1.76× |
| Qwen3-8B | bs4_p16_d128 (BS4/P16/D128) | 47.6→5.1 | +42.5 | 1564→879 | 1.78× |
| Qwen3-8B | bs4_p16_d512 (BS4/P16/D512) | 46.5→3.8 | +42.7 | 6197→3472 | 1.78× |
| Qwen3-8B | bs8_p16_d128 (BS8/P16/D128) | 45.7→4.4 | +41.3 | 1523→880 | 1.73× |
| Qwen3-8B | bs8_p16_d512 (BS8/P16/D512) | 44.9→3.6 | +41.3 | 6154→3517 | 1.75× |
| Qwen3-8B | bs16_p16_d128 (BS16/P16/D128) | 45.1→4.2 | +40.9 | 1545→897 | 1.72× |
| Qwen3-8B | bs16_p16_d512 (BS16/P16/D512) | 44.5→3.0 | +41.5 | 6374→3611 | 1.77× |
| Qwen3-14B | bs1_p16_d0 (BS1/P16/D1 effective) | 33.4→25.5 | +7.8 | 32→28 | 1.13× |
| Qwen3-14B | bs1_p256_d0 (BS1/P256/D1 effective) | 31.4→15.1 | +16.2 | 39→25 | 1.53× |
| Qwen3-14B | bs1_p1k_d0 (BS1/P1024/D1 effective) | 16.3→5.5 | +10.8 | 73→56 | 1.30× |
| Qwen3-14B | bs1_p4k_d0 (BS1/P4096/D1 effective) | 5.1→1.9 | +3.3 | 242→229 | 1.06× |
| Qwen3-14B | bs1_p8k_d0 (BS1/P8192/D1 effective) | 2.6→0.8 | +1.8 | 501→489 | 1.03× |
| Qwen3-14B | bs1_p16_d128 (BS1/P16/D128) | 14.5→2.0 | +12.5 | 1598→1409 | 1.13× |
| Qwen3-14B | bs1_p16_d512 (BS1/P16/D512) | 29.0→2.1 | +26.8 | 7712→5638 | 1.37× |
| Qwen3-14B | bs4_p16_d128 (BS4/P16/D128) | 17.3→2.6 | +14.7 | 1691→1441 | 1.17× |
| Qwen3-14B | bs4_p16_d512 (BS4/P16/D512) | 27.1→2.5 | +24.6 | 7720→5773 | 1.34× |
| Qwen3-14B | bs8_p16_d128 (BS8/P16/D128) | 17.4→3.0 | +14.5 | 1714→1469 | 1.17× |
| Qwen3-14B | bs8_p16_d512 (BS8/P16/D512) | 15.9→2.5 | +13.5 | 6827→5861 | 1.16× |
| Qwen3-14B | bs16_p16_d128 (BS16/P16/D128) | 15.8→2.7 | +13.1 | 1696→1469 | 1.15× |
| Qwen3-14B | bs16_p16_d512 (BS16/P16/D512) | 13.4→1.9 | +11.5 | 6745→5898 | 1.14× |
| Qwen3-30B-A3B | bs1_p16_d0 (BS1/P16/D1 effective) | 74.7→43.0 | +31.7 | 56→25 | 2.24× |
| Qwen3-30B-A3B | bs1_p256_d0 (BS1/P256/D1 effective) | 66.5→25.7 | +40.8 | 64→29 | 2.21× |
| Qwen3-30B-A3B | bs1_p1k_d0 (BS1/P1024/D1 effective) | 58.5→19.0 | +39.5 | 72→37 | 1.94× |
| Qwen3-30B-A3B | bs1_p4k_d0 (BS1/P4096/D1 effective) | 31.5→7.5 | +24.0 | 135→101 | 1.34× |
| Qwen3-30B-A3B | bs1_p8k_d0 (BS1/P8192/D1 effective) | 16.6→3.6 | +13.0 | 262→224 | 1.17× |
| Qwen3-30B-A3B | bs1_p16_d128 (BS1/P16/D128) | 90.1→9.4 | +80.8 | 5467→606 | 9.02× |
| Qwen3-30B-A3B | bs1_p16_d512 (BS1/P16/D512) | 88.0→8.0 | +80.0 | 18252→2372 | 7.70× |
| Qwen3-30B-A3B | bs4_p16_d128 (BS4/P16/D128) | 88.0→9.4 | +78.6 | 4902→664 | 7.39× |
| Qwen3-30B-A3B | bs4_p16_d512 (BS4/P16/D512) | 87.6→9.0 | +78.6 | 19228→2614 | 7.36× |
| Qwen3-30B-A3B | bs8_p16_d128 (BS8/P16/D128) | 87.7→9.8 | +77.9 | 4852→671 | 7.23× |
| Qwen3-30B-A3B | bs8_p16_d512 (BS8/P16/D512) | 87.8→8.8 | +79.1 | 20019→2656 | 7.54× |
| Qwen3-30B-A3B | bs16_p16_d128 (BS16/P16/D128) | 87.3→9.6 | +77.7 | 4961→706 | 7.03× |
| Qwen3-30B-A3B | bs16_p16_d512 (BS16/P16/D512) | 91.0→8.7 | +82.3 | 29188→2824 | 10.34× |
| Qwen3.5-27B | bs1_p16_d0 (BS1/P16/D1 effective) | 66.7→71.8 | -5.1 | 125→91 | 1.38× |
| Qwen3.5-27B | bs1_p256_d0 (BS1/P256/D1 effective) | 62.4→62.4 | +0.0 | 128→84 | 1.53× |
| Qwen3.5-27B | bs1_p1k_d0 (BS1/P1024/D1 effective) | 23.6→17.2 | +6.4 | 138→107 | 1.29× |
| Qwen3.5-27B | bs1_p4k_d0 (BS1/P4096/D1 effective) | 8.3→2.8 | +5.5 | 406→368 | 1.10× |
| Qwen3.5-27B | bs1_p8k_d0 (BS1/P8192/D1 effective) | 4.7→1.3 | +3.4 | 779→737 | 1.06× |
| Qwen3.5-27B | bs1_p16_d128 (BS1/P16/D128) | 54.5→3.8 | +50.7 | 5556→2578 | 2.16× |
| Qwen3.5-27B | bs1_p16_d512 (BS1/P16/D512) | 54.8→2.2 | +52.6 | 22268→10129 | 2.20× |
| Qwen3.5-27B | bs4_p16_d128 (BS4/P16/D128) | 58.3→4.0 | +54.3 | 6337→2689 | 2.36× |
| Qwen3.5-27B | bs4_p16_d512 (BS4/P16/D512) | 56.9→2.4 | +54.5 | 24396→10565 | 2.31× |
| Qwen3.5-27B | bs8_p16_d128 (BS8/P16/D128) | 56.9→3.7 | +53.2 | 6361→2789 | 2.28× |
| Qwen3.5-27B | bs8_p16_d512 (BS8/P16/D512) | 55.6→1.0 | +54.6 | 24558→10811 | 2.27× |
| Qwen3.5-27B | bs16_p16_d128 (BS16/P16/D128) | 54.0→3.3 | +50.7 | 6233→2911 | 2.14× |
| Qwen3.5-27B | bs16_p16_d512 (BS16/P16/D512) | 53.4→1.8 | +51.6 | 24536→11443 | 2.14× |

## 各组合 Top-5 耗时 Kernel


### Qwen3-0.6B / cudagraph / bs16_p16_d128 (BS16/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 26.018 |
| 2 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 3584 | 25.668 |
| 3 | `nvjet_sm90_tst_64x8_64x16_1x2_h_bz_TNT` | 3584 | 20.297 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x2_h_bz_TNT` | 3584 | 19.514 |
| 5 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | 3584 | 19.130 |

### Qwen3-0.6B / cudagraph / bs16_p16_d512 (BS16/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 168.360 |
| 2 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 14336 | 103.679 |
| 3 | `nvjet_sm90_tst_64x8_64x16_1x2_h_bz_TNT` | 14336 | 80.744 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x2_h_bz_TNT` | 14336 | 78.052 |
| 5 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | 14336 | 75.744 |

### Qwen3-0.6B / cudagraph / bs1_p16_d0 (BS1/P16/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.487 |
| 2 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 2 | 0.207 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 28 | 0.197 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 0.169 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x2_h_bz_TNT` | 28 | 0.156 |

### Qwen3-0.6B / cudagraph / bs1_p16_d128 (BS1/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 10752 | 61.637 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 22.442 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 3584 | 18.334 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 13.267 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0` | 7168 | 11.154 |

### Qwen3-0.6B / cudagraph / bs1_p16_d512 (BS1/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 43008 | 245.221 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 113.228 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 14336 | 73.051 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 52.765 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0` | 28672 | 44.491 |

### Qwen3-0.6B / cudagraph / bs1_p1k_d0 (BS1/P1024/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `PersistentVariableLengthMergeStatesKernel` | 56 | 1.251 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 0.871 |
| 3 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 56 | 0.671 |
| 4 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 28 | 0.572 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.485 |

### Qwen3-0.6B / cudagraph / bs1_p256_d0 (BS1/P256/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.485 |
| 2 | `PersistentVariableLengthMergeStatesKernel` | 56 | 0.401 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 0.261 |
| 4 | `nvjet_sm90_tst_64x32_64x16_1x2_h_bz_TNT` | 28 | 0.244 |
| 5 | `BatchDecodeWithPagedKVCacheKernel` | 28 | 0.239 |

### Qwen3-0.6B / cudagraph / bs1_p4k_d0 (BS1/P4096/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 8.730 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 112 | 5.207 |
| 3 | `act_and_mul_kernel` | 56 | 0.687 |
| 4 | `FusedAddRMSNormKernel` | 56 | 0.558 |
| 5 | `fused_qknorm_warp` | 56 | 0.554 |

### Qwen3-0.6B / cudagraph / bs1_p8k_d0 (BS1/P8192/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 25.967 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 84 | 5.726 |
| 3 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 28 | 4.107 |
| 4 | `act_and_mul_kernel` | 56 | 1.424 |
| 5 | `elementwise_kernel` | 28 | 1.264 |

### Qwen3-0.6B / cudagraph / bs4_p16_d128 (BS4/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 10752 | 61.865 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 22.672 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 3584 | 18.442 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 13.419 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0` | 7168 | 12.888 |

### Qwen3-0.6B / cudagraph / bs4_p16_d512 (BS4/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 43008 | 246.427 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 117.150 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 14336 | 73.566 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 53.393 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0` | 28672 | 50.729 |

### Qwen3-0.6B / cudagraph / bs8_p16_d128 (BS8/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 10752 | 62.265 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 23.687 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 3584 | 18.668 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 13.606 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0` | 7168 | 12.966 |

### Qwen3-0.6B / cudagraph / bs8_p16_d512 (BS8/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 43008 | 249.468 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 132.466 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 14336 | 74.218 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 54.102 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0` | 28672 | 51.655 |

### Qwen3-0.6B / eager / bs16_p16_d128 (BS16/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 3584 | 26.184 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 24.826 |
| 3 | `nvjet_sm90_tst_64x8_64x16_1x2_h_bz_TNT` | 3584 | 20.510 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x2_h_bz_TNT` | 3584 | 19.864 |
| 5 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | 3584 | 19.523 |

### Qwen3-0.6B / eager / bs16_p16_d512 (BS16/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 161.784 |
| 2 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 14336 | 105.584 |
| 3 | `nvjet_sm90_tst_64x8_64x16_1x2_h_bz_TNT` | 14336 | 82.259 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x2_h_bz_TNT` | 14336 | 80.149 |
| 5 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | 14336 | 77.377 |

### Qwen3-0.6B / eager / bs1_p16_d0 (BS1/P16/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.488 |
| 2 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 2 | 0.207 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 28 | 0.196 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 0.173 |
| 5 | `nvjet_sm90_tst_64x8_64x16_1x2_h_bz_TNT` | 28 | 0.157 |

### Qwen3-0.6B / eager / bs1_p16_d128 (BS1/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 10752 | 62.404 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 20.550 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 3584 | 18.610 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 13.307 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0` | 7168 | 11.540 |

### Qwen3-0.6B / eager / bs1_p16_d512 (BS1/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 43008 | 249.506 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 107.705 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 14336 | 74.503 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 52.929 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0` | 28672 | 46.048 |

### Qwen3-0.6B / eager / bs1_p1k_d0 (BS1/P1024/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `PersistentVariableLengthMergeStatesKernel` | 56 | 1.254 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 0.875 |
| 3 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 56 | 0.670 |
| 4 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 28 | 0.572 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.486 |

### Qwen3-0.6B / eager / bs1_p256_d0 (BS1/P256/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.487 |
| 2 | `PersistentVariableLengthMergeStatesKernel` | 56 | 0.402 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 0.271 |
| 4 | `nvjet_sm90_tst_64x32_64x16_1x2_h_bz_TNT` | 28 | 0.245 |
| 5 | `nvjet_sm90_tst_96x128_64x7_2x1_v_bz_TNN` | 28 | 0.235 |

### Qwen3-0.6B / eager / bs1_p4k_d0 (BS1/P4096/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 8.748 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 112 | 5.189 |
| 3 | `act_and_mul_kernel` | 56 | 0.685 |
| 4 | `fused_qknorm_warp` | 56 | 0.553 |
| 5 | `FusedAddRMSNormKernel` | 56 | 0.552 |

### Qwen3-0.6B / eager / bs1_p8k_d0 (BS1/P8192/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 26.327 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 84 | 5.729 |
| 3 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 28 | 4.109 |
| 4 | `act_and_mul_kernel` | 56 | 1.422 |
| 5 | `elementwise_kernel` | 28 | 1.267 |

### Qwen3-0.6B / eager / bs4_p16_d128 (BS4/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 10752 | 62.583 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 21.562 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 3584 | 18.738 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 13.481 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0` | 7168 | 12.875 |

### Qwen3-0.6B / eager / bs4_p16_d512 (BS4/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 43008 | 250.512 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 113.212 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 14336 | 75.081 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 53.641 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0` | 28672 | 52.703 |

### Qwen3-0.6B / eager / bs8_p16_d128 (BS8/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 10752 | 63.195 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 22.599 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 3584 | 19.057 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 13.613 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0` | 7168 | 12.830 |

### Qwen3-0.6B / eager / bs8_p16_d512 (BS8/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 43008 | 253.678 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 127.742 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 14336 | 75.751 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 54.188 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0` | 28672 | 52.118 |

### Qwen3-1.7B / cudagraph / bs16_p16_d128 (BS16/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 3584 | 74.223 |
| 2 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 3584 | 40.066 |
| 3 | `nvjet_sm90_tst_64x8_64x16_1x2_h_bz_TNT` | 3584 | 31.062 |
| 4 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 27.961 |
| 5 | `nvjet_sm90_tst_384x16_64x4_2x1_v_bz_TNT` | 129 | 26.810 |

### Qwen3-1.7B / cudagraph / bs16_p16_d512 (BS16/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 14336 | 291.876 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 172.733 |
| 3 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 14336 | 159.567 |
| 4 | `nvjet_sm90_tst_64x8_64x16_1x2_h_bz_TNT` | 14336 | 123.604 |
| 5 | `nvjet_sm90_tst_384x16_64x4_2x1_v_bz_TNT` | 513 | 106.589 |

### Qwen3-1.7B / cudagraph / bs1_p16_d0 (BS1/P16/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.927 |
| 2 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 28 | 0.565 |
| 3 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 2 | 0.406 |
| 4 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 28 | 0.315 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 28 | 0.304 |

### Qwen3-1.7B / cudagraph / bs1_p16_d128 (BS1/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 10752 | 121.740 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 3584 | 38.471 |
| 3 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 26.092 |
| 4 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 24.947 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 7168 | 13.661 |

### Qwen3-1.7B / cudagraph / bs1_p16_d512 (BS1/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 43008 | 484.005 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 14336 | 153.950 |
| 3 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 121.098 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 103.755 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 28672 | 54.339 |

### Qwen3-1.7B / cudagraph / bs1_p1k_d0 (BS1/P1024/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 56 | 2.751 |
| 2 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 56 | 1.439 |
| 3 | `PersistentVariableLengthMergeStatesKernel` | 56 | 1.258 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.937 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 0.901 |

### Qwen3-1.7B / cudagraph / bs1_p256_d0 (BS1/P256/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.933 |
| 2 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 28 | 0.693 |
| 3 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | 56 | 0.692 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 2 | 0.407 |
| 5 | `PersistentVariableLengthMergeStatesKernel` | 56 | 0.406 |

### Qwen3-1.7B / cudagraph / bs1_p4k_d0 (BS1/P4096/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 112 | 16.640 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 8.860 |
| 3 | `act_and_mul_kernel` | 56 | 1.357 |
| 4 | `FusedAddRMSNormKernel` | 56 | 1.073 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.977 |

### Qwen3-1.7B / cudagraph / bs1_p8k_d0 (BS1/P8192/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 28.578 |
| 2 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 28 | 17.422 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 84 | 16.904 |
| 4 | `act_and_mul_kernel` | 56 | 2.715 |
| 5 | `FusedAddRMSNormKernel` | 56 | 2.475 |

### Qwen3-1.7B / cudagraph / bs4_p16_d128 (BS4/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 10752 | 120.073 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 3584 | 38.708 |
| 3 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 26.348 |
| 4 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 25.144 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 7168 | 17.032 |

### Qwen3-1.7B / cudagraph / bs4_p16_d512 (BS4/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 43008 | 478.874 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 14336 | 155.026 |
| 3 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 124.841 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 104.749 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 28672 | 67.886 |

### Qwen3-1.7B / cudagraph / bs8_p16_d128 (BS8/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 10752 | 122.946 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 3584 | 39.409 |
| 3 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 26.557 |
| 4 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 26.514 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 7168 | 17.489 |

### Qwen3-1.7B / cudagraph / bs8_p16_d512 (BS8/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 43008 | 490.242 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 14336 | 157.098 |
| 3 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 139.709 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 105.587 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 28672 | 69.858 |

### Qwen3-1.7B / eager / bs16_p16_d128 (BS16/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 3584 | 71.847 |
| 2 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 3584 | 40.366 |
| 3 | `nvjet_sm90_tst_64x8_64x16_1x2_h_bz_TNT` | 3584 | 31.543 |
| 4 | `nvjet_sm90_tst_384x16_64x4_2x1_v_bz_TNT` | 129 | 26.877 |
| 5 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 25.936 |

### Qwen3-1.7B / eager / bs16_p16_d512 (BS16/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 14336 | 286.398 |
| 2 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 166.023 |
| 3 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 14336 | 161.997 |
| 4 | `nvjet_sm90_tst_64x8_64x16_1x2_h_bz_TNT` | 14336 | 126.507 |
| 5 | `nvjet_sm90_tst_384x16_64x4_2x1_v_bz_TNT` | 513 | 106.890 |

### Qwen3-1.7B / eager / bs1_p16_d0 (BS1/P16/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.926 |
| 2 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 28 | 0.570 |
| 3 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 2 | 0.405 |
| 4 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 28 | 0.315 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 28 | 0.304 |

### Qwen3-1.7B / eager / bs1_p16_d128 (BS1/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 10752 | 118.852 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 3584 | 38.899 |
| 3 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 26.110 |
| 4 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 23.090 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 7168 | 14.665 |

### Qwen3-1.7B / eager / bs1_p16_d512 (BS1/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 43008 | 475.002 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 14336 | 156.117 |
| 3 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 115.128 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 103.910 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 28672 | 58.710 |

### Qwen3-1.7B / eager / bs1_p1k_d0 (BS1/P1024/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 56 | 2.737 |
| 2 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 56 | 1.436 |
| 3 | `PersistentVariableLengthMergeStatesKernel` | 56 | 1.253 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.927 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 0.899 |

### Qwen3-1.7B / eager / bs1_p256_d0 (BS1/P256/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.923 |
| 2 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | 56 | 0.698 |
| 3 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 28 | 0.685 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 2 | 0.407 |
| 5 | `PersistentVariableLengthMergeStatesKernel` | 56 | 0.402 |

### Qwen3-1.7B / eager / bs1_p4k_d0 (BS1/P4096/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 112 | 16.566 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 8.799 |
| 3 | `act_and_mul_kernel` | 56 | 1.362 |
| 4 | `FusedAddRMSNormKernel` | 56 | 1.064 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 84 | 0.971 |

### Qwen3-1.7B / eager / bs1_p8k_d0 (BS1/P8192/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchPrefillWithPagedKVCacheKernel` | 28 | 29.289 |
| 2 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 28 | 17.456 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 84 | 16.931 |
| 4 | `act_and_mul_kernel` | 56 | 2.715 |
| 5 | `FusedAddRMSNormKernel` | 56 | 2.480 |

### Qwen3-1.7B / eager / bs4_p16_d128 (BS4/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 10752 | 120.481 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 3584 | 39.042 |
| 3 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 26.389 |
| 4 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 23.369 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 7168 | 17.476 |

### Qwen3-1.7B / eager / bs4_p16_d512 (BS4/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 43008 | 482.756 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 14336 | 157.226 |
| 3 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 119.730 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 104.949 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 28672 | 70.274 |

### Qwen3-1.7B / eager / bs8_p16_d128 (BS8/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 10752 | 121.833 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 3584 | 39.636 |
| 3 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 26.584 |
| 4 | `BatchDecodeWithPagedKVCacheKernel` | 3584 | 24.465 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 7168 | 17.765 |

### Qwen3-1.7B / eager / bs8_p16_d512 (BS8/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 43008 | 487.838 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 14336 | 159.113 |
| 3 | `BatchDecodeWithPagedKVCacheKernel` | 14336 | 133.159 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 105.751 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 28672 | 72.102 |

### Qwen3-14B / cudagraph / bs16_p16_d128 (BS16/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 5112 | 624.403 |
| 2 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 5111 | 322.787 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 10224 | 246.556 |
| 4 | `nvjet_sm90_tst_384x16_64x4_2x1_v_bz_TNT` | 128 | 65.770 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 5152 | 53.597 |

### Qwen3-14B / cudagraph / bs16_p16_d512 (BS16/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 20480 | 2497.492 |
| 2 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 20480 | 1294.017 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 40960 | 991.080 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 20520 | 303.332 |
| 5 | `nvjet_sm90_tst_384x16_64x4_2x1_v_bz_TNT` | 513 | 263.621 |

### Qwen3-14B / cudagraph / bs1_p16_d0 (BS1/P16/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 40 | 4.883 |
| 2 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 40 | 4.861 |
| 3 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 40 | 2.527 |
| 4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 40 | 2.461 |
| 5 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 80 | 1.921 |

### Qwen3-14B / cudagraph / bs1_p16_d128 (BS1/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 5112 | 619.996 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 5112 | 314.376 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 5113 | 135.583 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 5113 | 102.136 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 128 | 64.957 |

### Qwen3-14B / cudagraph / bs1_p16_d512 (BS1/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 20480 | 2483.943 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 20480 | 1259.423 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 20480 | 543.197 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 20480 | 409.891 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 260.184 |

### Qwen3-14B / cudagraph / bs1_p1k_d0 (BS1/P1024/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 120 | 36.564 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 40 | 5.174 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 55 | 2.806 |
| 4 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 15 | 1.837 |
| 5 | `act_and_mul_kernel` | 55 | 1.294 |

### Qwen3-14B / cudagraph / bs1_p256_d0 (BS1/P256/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN` | 40 | 5.692 |
| 2 | `nvjet_sm90_tst_80x128_64x8_2x1_v_bz_TNN` | 80 | 4.686 |
| 3 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 22 | 2.681 |
| 4 | `nvjet_sm90_tst_112x128_64x7_2x1_v_bz_TNN` | 40 | 1.360 |
| 5 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 21 | 1.299 |

### Qwen3-14B / cudagraph / bs1_p4k_d0 (BS1/P4096/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 120 | 149.911 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 63 | 31.790 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 40 | 18.830 |
| 4 | `act_and_mul_kernel` | 63 | 5.475 |
| 5 | `FusedAddRMSNormKernel` | 80 | 4.538 |

### Qwen3-14B / cudagraph / bs1_p8k_d0 (BS1/P8192/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 120 | 293.076 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 63 | 113.005 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 40 | 36.858 |
| 4 | `act_and_mul_kernel` | 63 | 11.018 |
| 5 | `FusedAddRMSNormKernel` | 80 | 8.970 |

### Qwen3-14B / cudagraph / bs4_p16_d128 (BS4/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 5101 | 615.914 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 5100 | 317.478 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 5101 | 137.589 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 5101 | 105.052 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 128 | 65.126 |

### Qwen3-14B / cudagraph / bs4_p16_d512 (BS4/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 20480 | 2473.069 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 20480 | 1275.268 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 20480 | 553.049 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 20480 | 419.591 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 261.085 |

### Qwen3-14B / cudagraph / bs8_p16_d128 (BS8/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 5120 | 624.737 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 5120 | 321.642 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 5120 | 140.244 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 5120 | 105.957 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 65.935 |

### Qwen3-14B / cudagraph / bs8_p16_d512 (BS8/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 20468 | 2495.435 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 20468 | 1285.898 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 20469 | 560.638 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 20468 | 425.172 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 512 | 261.745 |

### Qwen3-14B / eager / bs16_p16_d128 (BS16/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 5120 | 628.146 |
| 2 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 5120 | 323.596 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 10240 | 247.109 |
| 4 | `nvjet_sm90_tst_384x16_64x4_2x1_v_bz_TNT` | 129 | 66.472 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 5160 | 44.922 |

### Qwen3-14B / eager / bs16_p16_d512 (BS16/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 20480 | 2517.222 |
| 2 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 20480 | 1295.296 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 40960 | 992.482 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 20520 | 296.500 |
| 5 | `nvjet_sm90_tst_384x16_64x4_2x1_v_bz_TNT` | 513 | 264.244 |

### Qwen3-14B / eager / bs1_p16_d0 (BS1/P16/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 40 | 4.865 |
| 2 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 40 | 4.817 |
| 3 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 40 | 2.531 |
| 4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 40 | 2.472 |
| 5 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 80 | 1.923 |

### Qwen3-14B / eager / bs1_p16_d128 (BS1/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 5120 | 616.598 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 5120 | 316.697 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 5120 | 136.957 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 5120 | 102.380 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 65.595 |

### Qwen3-14B / eager / bs1_p16_d512 (BS1/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 20480 | 2467.728 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 20480 | 1268.129 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 20480 | 548.228 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 20480 | 410.319 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 260.665 |

### Qwen3-14B / eager / bs1_p1k_d0 (BS1/P1024/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 120 | 36.972 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 40 | 5.209 |
| 3 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 40 | 4.898 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 80 | 3.173 |
| 5 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 40 | 2.529 |

### Qwen3-14B / eager / bs1_p256_d0 (BS1/P256/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN` | 40 | 5.694 |
| 2 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 40 | 4.813 |
| 3 | `nvjet_sm90_tst_80x128_64x8_2x1_v_bz_TNN` | 80 | 4.730 |
| 4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 40 | 2.481 |
| 5 | `nvjet_sm90_tst_112x128_64x7_2x1_v_bz_TNN` | 40 | 1.361 |

### Qwen3-14B / eager / bs1_p4k_d0 (BS1/P4096/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 120 | 149.393 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 80 | 32.101 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 40 | 18.672 |
| 4 | `act_and_mul_kernel` | 80 | 5.513 |
| 5 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 40 | 4.915 |

### Qwen3-14B / eager / bs1_p8k_d0 (BS1/P8192/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 120 | 292.818 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 80 | 111.958 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 40 | 36.691 |
| 4 | `act_and_mul_kernel` | 80 | 11.058 |
| 5 | `FusedAddRMSNormKernel` | 80 | 9.032 |

### Qwen3-14B / eager / bs4_p16_d128 (BS4/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 5120 | 617.776 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 5120 | 319.819 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 5120 | 139.369 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 5120 | 103.591 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 65.812 |

### Qwen3-14B / eager / bs4_p16_d512 (BS4/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 20480 | 2470.747 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 20480 | 1279.414 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 20480 | 557.902 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 20480 | 418.884 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 261.781 |

### Qwen3-14B / eager / bs8_p16_d128 (BS8/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 5120 | 619.055 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 5120 | 322.167 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 5120 | 141.578 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 5120 | 105.292 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 66.093 |

### Qwen3-14B / eager / bs8_p16_d512 (BS8/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 20480 | 2476.225 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 20480 | 1289.070 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 20480 | 565.672 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 20480 | 425.248 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 262.779 |

### Qwen3-30B-A3B / cudagraph / bs16_p16_d128 (BS16/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 12384 | 245.187 |
| 2 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 6144 | 60.563 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 6192 | 60.267 |
| 4 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | 6144 | 52.011 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x2_h_bz_TNT` | 6144 | 30.864 |

### Qwen3-30B-A3B / cudagraph / bs16_p16_d512 (BS16/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 49248 | 953.825 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 24624 | 292.415 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 24576 | 241.837 |
| 4 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | 24576 | 207.249 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 49152 | 122.703 |

### Qwen3-30B-A3B / cudagraph / bs1_p16_d0 (BS1/P16/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 192 | 8.552 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 0.748 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 48 | 0.452 |
| 4 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 47 | 0.447 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 2 | 0.405 |

### Qwen3-30B-A3B / cudagraph / bs1_p16_d128 (BS1/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 12384 | 198.466 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 6144 | 57.528 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 6192 | 55.379 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 6144 | 49.622 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 6144 | 29.934 |

### Qwen3-30B-A3B / cudagraph / bs1_p16_d512 (BS1/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 49248 | 772.386 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 24576 | 229.793 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 24624 | 228.853 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 24576 | 198.350 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 24576 | 119.417 |

### Qwen3-30B-A3B / cudagraph / bs1_p1k_d0 (BS1/P1024/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 192 | 17.352 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 3.201 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 48 | 1.539 |
| 4 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 48 | 1.162 |
| 5 | `moe_sum_reduce_warp_per_token_vec_kernel` | 48 | 0.612 |

### Qwen3-30B-A3B / cudagraph / bs1_p256_d0 (BS1/P256/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 192 | 13.711 |
| 2 | `PersistentVariableLengthMergeStatesKernel` | 96 | 1.163 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 0.936 |
| 4 | `nvjet_sm90_tst_80x128_64x8_2x1_v_bz_TNN` | 48 | 0.563 |
| 5 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | 48 | 0.541 |

### Qwen3-30B-A3B / cudagraph / bs1_p4k_d0 (BS1/P4096/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 192 | 38.027 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 26.836 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 48 | 6.165 |
| 4 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 4.821 |
| 5 | `moe_sum_reduce_warp_per_token_vec_kernel` | 48 | 2.502 |

### Qwen3-30B-A3B / cudagraph / bs1_p8k_d0 (BS1/P8192/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 92.635 |
| 2 | `fused_moe_kernel` | 192 | 68.913 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 48 | 12.449 |
| 4 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 9.609 |
| 5 | `moe_sum_reduce_warp_per_token_vec_kernel` | 48 | 4.892 |

### Qwen3-30B-A3B / cudagraph / bs4_p16_d128 (BS4/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 12382 | 225.237 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 6144 | 59.218 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 6190 | 56.019 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 6144 | 50.879 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 6144 | 30.000 |

### Qwen3-30B-A3B / cudagraph / bs4_p16_d512 (BS4/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 49248 | 861.051 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 24624 | 238.530 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 24576 | 236.113 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 24576 | 202.927 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 49152 | 121.552 |

### Qwen3-30B-A3B / cudagraph / bs8_p16_d128 (BS8/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 12384 | 223.450 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 6144 | 59.785 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 6192 | 57.467 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 6144 | 51.241 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 12288 | 31.261 |

### Qwen3-30B-A3B / cudagraph / bs8_p16_d512 (BS8/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 49248 | 871.339 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 24624 | 254.949 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 24576 | 238.299 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 24576 | 204.710 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 49152 | 124.964 |

### Qwen3-30B-A3B / eager / bs16_p16_d128 (BS16/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 12384 | 248.675 |
| 2 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 6144 | 61.695 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | 6144 | 52.174 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 6192 | 47.574 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 12288 | 31.930 |

### Qwen3-30B-A3B / eager / bs16_p16_d512 (BS16/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 49248 | 972.156 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 24624 | 268.735 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 24576 | 246.927 |
| 4 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | 24576 | 211.740 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x2_h_bz_TNT` | 24576 | 128.048 |

### Qwen3-30B-A3B / eager / bs1_p16_d0 (BS1/P16/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 192 | 8.563 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 0.645 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 48 | 0.463 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 48 | 0.459 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 2 | 0.406 |

### Qwen3-30B-A3B / eager / bs1_p16_d128 (BS1/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 12384 | 200.241 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 6144 | 58.619 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 6144 | 51.173 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 6192 | 41.954 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 6144 | 30.867 |

### Qwen3-30B-A3B / eager / bs1_p16_d512 (BS1/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 49248 | 779.888 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 24576 | 235.219 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 24576 | 204.633 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 24624 | 175.192 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 24576 | 124.148 |

### Qwen3-30B-A3B / eager / bs1_p1k_d0 (BS1/P1024/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 192 | 17.348 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 3.059 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 48 | 1.539 |
| 4 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 48 | 1.164 |
| 5 | `moe_sum_reduce_warp_per_token_vec_kernel` | 48 | 0.612 |

### Qwen3-30B-A3B / eager / bs1_p256_d0 (BS1/P256/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 192 | 13.706 |
| 2 | `PersistentVariableLengthMergeStatesKernel` | 96 | 1.181 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 0.826 |
| 4 | `nvjet_sm90_tst_80x128_64x8_2x1_v_bz_TNN` | 48 | 0.562 |
| 5 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | 48 | 0.546 |

### Qwen3-30B-A3B / eager / bs1_p4k_d0 (BS1/P4096/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 192 | 37.792 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 26.342 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 48 | 6.099 |
| 4 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 4.774 |
| 5 | `moe_sum_reduce_warp_per_token_vec_kernel` | 48 | 2.500 |

### Qwen3-30B-A3B / eager / bs1_p8k_d0 (BS1/P8192/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 94.474 |
| 2 | `fused_moe_kernel` | 192 | 68.880 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 48 | 12.531 |
| 4 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 9.673 |
| 5 | `moe_sum_reduce_warp_per_token_vec_kernel` | 48 | 4.893 |

### Qwen3-30B-A3B / eager / bs4_p16_d128 (BS4/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 12384 | 222.285 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 6144 | 60.271 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 6144 | 50.918 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 6192 | 43.459 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 12288 | 31.423 |

### Qwen3-30B-A3B / eager / bs4_p16_d512 (BS4/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 49248 | 871.902 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 24576 | 241.609 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 24576 | 206.830 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 24624 | 188.451 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 49152 | 126.235 |

### Qwen3-30B-A3B / eager / bs8_p16_d128 (BS8/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 12384 | 225.353 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 6144 | 61.055 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 6144 | 51.605 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 6192 | 45.159 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 12288 | 31.748 |

### Qwen3-30B-A3B / eager / bs8_p16_d512 (BS8/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 49248 | 884.239 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 24576 | 244.381 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 24576 | 208.439 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 24624 | 207.703 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 49152 | 127.508 |

### Qwen3-8B / cudagraph / bs16_p16_d128 (BS16/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x16_64x8_2x1_v_bz_TNT` | 4608 | 321.423 |
| 2 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 4608 | 169.403 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 4608 | 96.379 |
| 4 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 4608 | 63.803 |
| 5 | `nvjet_sm90_tst_384x16_64x4_2x1_v_bz_TNT` | 129 | 53.527 |

### Qwen3-8B / cudagraph / bs16_p16_d512 (BS16/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x16_64x8_2x1_v_bz_TNT` | 18432 | 1285.681 |
| 2 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 18432 | 677.419 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 18432 | 385.239 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 18468 | 272.511 |
| 5 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 18432 | 256.590 |

### Qwen3-8B / cudagraph / bs1_p16_d0 (BS1/P16/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x16_64x8_2x1_v_bz_TNT` | 36 | 2.519 |
| 2 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 36 | 2.456 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 72 | 1.776 |
| 4 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 36 | 1.327 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 2 | 0.820 |

### Qwen3-8B / cudagraph / bs1_p16_d128 (BS1/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 4608 | 314.038 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 9215 | 225.904 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 4608 | 92.981 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 128 | 52.347 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 4644 | 41.374 |

### Qwen3-8B / cudagraph / bs1_p16_d512 (BS1/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 18431 | 1256.166 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 36861 | 903.356 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 18431 | 371.314 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 512 | 209.304 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 18467 | 172.797 |

### Qwen3-8B / cudagraph / bs1_p1k_d0 (BS1/P1024/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 108 | 17.951 |
| 2 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 36 | 2.960 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 71 | 2.540 |
| 4 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 35 | 2.402 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 69 | 1.721 |

### Qwen3-8B / cudagraph / bs1_p256_d0 (BS1/P256/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 36 | 2.999 |
| 2 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 72 | 2.641 |
| 3 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 36 | 2.456 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 72 | 1.771 |
| 5 | `PersistentVariableLengthMergeStatesKernel` | 72 | 0.938 |

### Qwen3-8B / cudagraph / bs1_p4k_d0 (BS1/P4096/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 36 | 44.714 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 108 | 40.155 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 70 | 22.040 |
| 4 | `act_and_mul_kernel` | 70 | 3.465 |
| 5 | `FusedAddRMSNormKernel` | 72 | 3.121 |

### Qwen3-8B / cudagraph / bs1_p8k_d0 (BS1/P8192/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 72 | 112.504 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 69 | 77.656 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 72 | 58.885 |
| 4 | `act_and_mul_kernel` | 69 | 6.978 |
| 5 | `FusedAddRMSNormKernel` | 72 | 6.315 |

### Qwen3-8B / cudagraph / bs4_p16_d128 (BS4/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 4608 | 316.270 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 9216 | 227.088 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 4608 | 93.947 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 52.998 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 4644 | 42.611 |

### Qwen3-8B / cudagraph / bs4_p16_d512 (BS4/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 18431 | 1265.355 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 36862 | 910.146 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 18431 | 375.470 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 512 | 210.342 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 18467 | 187.671 |

### Qwen3-8B / cudagraph / bs8_p16_d128 (BS8/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 4608 | 318.252 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 9216 | 229.876 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 4608 | 94.590 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 53.224 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 4644 | 43.883 |

### Qwen3-8B / cudagraph / bs8_p16_d512 (BS8/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 18432 | 1273.639 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 36864 | 921.836 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 18432 | 378.137 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 18468 | 213.135 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 211.584 |

### Qwen3-8B / eager / bs16_p16_d128 (BS16/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x16_64x8_2x1_v_bz_TNT` | 4608 | 321.350 |
| 2 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 4608 | 170.090 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 4608 | 96.935 |
| 4 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 4608 | 63.744 |
| 5 | `nvjet_sm90_tst_384x16_64x4_2x1_v_bz_TNT` | 129 | 53.631 |

### Qwen3-8B / eager / bs16_p16_d512 (BS16/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x16_64x8_2x1_v_bz_TNT` | 18432 | 1286.084 |
| 2 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 18432 | 680.939 |
| 3 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 18432 | 387.833 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 18468 | 270.158 |
| 5 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 18432 | 257.960 |

### Qwen3-8B / eager / bs1_p16_d0 (BS1/P16/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x16_64x8_2x1_v_bz_TNT` | 36 | 2.519 |
| 2 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 36 | 2.448 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 72 | 1.759 |
| 4 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 36 | 1.328 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 2 | 0.822 |

### Qwen3-8B / eager / bs1_p16_d128 (BS1/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 4608 | 313.879 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 9216 | 225.071 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 4608 | 93.188 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 52.852 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 4644 | 31.802 |

### Qwen3-8B / eager / bs1_p16_d512 (BS1/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 18432 | 1256.111 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 36864 | 901.621 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 18432 | 372.851 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 210.172 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 18468 | 133.769 |

### Qwen3-8B / eager / bs1_p1k_d0 (BS1/P1024/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 108 | 18.004 |
| 2 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 36 | 2.971 |
| 3 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 36 | 2.471 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 72 | 2.446 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 72 | 1.791 |

### Qwen3-8B / eager / bs1_p256_d0 (BS1/P256/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 36 | 3.008 |
| 2 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 72 | 2.667 |
| 3 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 36 | 2.456 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 72 | 1.761 |
| 5 | `PersistentVariableLengthMergeStatesKernel` | 72 | 0.945 |

### Qwen3-8B / eager / bs1_p4k_d0 (BS1/P4096/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 36 | 45.216 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 108 | 40.478 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 72 | 22.243 |
| 4 | `act_and_mul_kernel` | 72 | 3.480 |
| 5 | `FusedAddRMSNormKernel` | 72 | 3.104 |

### Qwen3-8B / eager / bs1_p8k_d0 (BS1/P8192/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 72 | 114.095 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 72 | 81.469 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 72 | 60.111 |
| 4 | `act_and_mul_kernel` | 72 | 6.993 |
| 5 | `FusedAddRMSNormKernel` | 72 | 6.335 |

### Qwen3-8B / eager / bs4_p16_d128 (BS4/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 4608 | 316.053 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 9216 | 227.379 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 4608 | 94.513 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 53.041 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 4644 | 33.742 |

### Qwen3-8B / eager / bs4_p16_d512 (BS4/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 18432 | 1265.774 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 36864 | 913.342 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 18432 | 378.219 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 211.291 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 18468 | 155.178 |

### Qwen3-8B / eager / bs8_p16_d128 (BS8/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 4608 | 318.501 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 9216 | 230.142 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 4608 | 95.064 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 129 | 53.258 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 4644 | 35.512 |

### Qwen3-8B / eager / bs8_p16_d512 (BS8/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 18432 | 1274.901 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 36864 | 924.775 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 18432 | 380.691 |
| 4 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 513 | 211.925 |
| 5 | `BatchPrefillWithPagedKVCacheKernel` | 18468 | 199.511 |

### Qwen3.5-27B / cudagraph / bs16_p16_d128 (BS16/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 8139 | 1010.244 |
| 2 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 8139 | 514.936 |
| 3 | `nvjet_sm90_tst_128x16_64x11_2x1_v_bz_TNT` | 6105 | 354.216 |
| 4 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 6105 | 324.341 |
| 5 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 8139 | 237.220 |

### Qwen3.5-27B / cudagraph / bs16_p16_d512 (BS16/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 32715 | 4057.948 |
| 2 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 32715 | 2069.707 |
| 3 | `nvjet_sm90_tst_128x16_64x11_2x1_v_bz_TNT` | 24537 | 1423.894 |
| 4 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 24537 | 1273.235 |
| 5 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 32715 | 954.780 |

### Qwen3.5-27B / cudagraph / bs1_p16_d0 (BS1/P16/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 64 | 7.866 |
| 2 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 64 | 4.049 |
| 3 | `nvjet_sm90_tst_128x16_64x11_2x1_v_bz_TNT` | 48 | 2.770 |
| 4 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 64 | 1.654 |
| 5 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 11 | 1.350 |

### Qwen3.5-27B / cudagraph / bs1_p16_d128 (BS1/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 8137 | 1001.634 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 8137 | 502.238 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 6104 | 347.005 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 6104 | 315.452 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 8137 | 200.044 |

### Qwen3.5-27B / cudagraph / bs1_p16_d512 (BS1/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 32713 | 4027.010 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 32712 | 2018.853 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 24535 | 1394.756 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 24535 | 1267.156 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 32713 | 804.537 |

### Qwen3.5-27B / cudagraph / bs1_p1k_d0 (BS1/P1024/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 192 | 50.772 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 10.442 |
| 3 | `elementwise_kernel` | 343 | 3.218 |
| 4 | `nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN` | 16 | 3.023 |
| 5 | `act_and_mul_kernel` | 76 | 2.020 |

### Qwen3.5-27B / cudagraph / bs1_p256_d0 (BS1/P256/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN` | 64 | 8.761 |
| 2 | `nvjet_sm90_tst_80x128_64x8_2x1_v_bz_TNN` | 128 | 7.358 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 3.167 |
| 4 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 11 | 1.345 |
| 5 | `elementwise_kernel` | 341 | 1.279 |

### Qwen3.5-27B / cudagraph / bs1_p4k_d0 (BS1/P4096/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 256 | 266.662 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 18 | 15.482 |
| 3 | `elementwise_kernel` | 341 | 11.757 |
| 4 | `act_and_mul_kernel` | 75 | 8.649 |
| 5 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 48 | 7.988 |

### Qwen3.5-27B / cudagraph / bs1_p8k_d0 (BS1/P8192/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 208 | 437.097 |
| 2 | `nvjet_sm90_tst_192x192_64x4_1x2_h_bz_coopB_TNN` | 48 | 89.220 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 18 | 56.039 |
| 4 | `elementwise_kernel` | 341 | 22.625 |
| 5 | `act_and_mul_kernel` | 75 | 17.523 |

### Qwen3.5-27B / cudagraph / bs4_p16_d128 (BS4/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 8137 | 990.026 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 8137 | 507.478 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 6103 | 349.431 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 6103 | 325.896 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 8137 | 218.730 |

### Qwen3.5-27B / cudagraph / bs4_p16_d512 (BS4/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 32714 | 3979.166 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 32713 | 2040.320 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 24536 | 1404.872 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 24536 | 1309.296 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 32714 | 880.023 |

### Qwen3.5-27B / cudagraph / bs8_p16_d128 (BS8/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 8138 | 1005.597 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 8138 | 511.689 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 6105 | 351.986 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 6105 | 330.725 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 8138 | 235.564 |

### Qwen3.5-27B / cudagraph / bs8_p16_d512 (BS8/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 32716 | 4033.954 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 32715 | 2056.854 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 24537 | 1415.376 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 24537 | 1228.049 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 32716 | 948.713 |

### Qwen3.5-27B / eager / bs16_p16_d128 (BS16/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 8192 | 1009.160 |
| 2 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 8192 | 518.522 |
| 3 | `nvjet_sm90_tst_128x16_64x11_2x1_v_bz_TNT` | 6144 | 353.734 |
| 4 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 8192 | 238.784 |
| 5 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 6144 | 201.746 |

### Qwen3.5-27B / eager / bs16_p16_d512 (BS16/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 32768 | 4037.226 |
| 2 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 32768 | 2073.595 |
| 3 | `nvjet_sm90_tst_128x16_64x11_2x1_v_bz_TNT` | 24576 | 1414.813 |
| 4 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 32768 | 956.639 |
| 5 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 24576 | 806.989 |

### Qwen3.5-27B / eager / bs1_p16_d0 (BS1/P16/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 64 | 7.835 |
| 2 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 64 | 7.810 |
| 3 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 64 | 4.060 |
| 4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 64 | 3.960 |
| 5 | `nvjet_sm90_tst_128x16_64x11_2x1_v_bz_TNT` | 48 | 2.769 |

### Qwen3.5-27B / eager / bs1_p16_d128 (BS1/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 8192 | 1000.463 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 8192 | 508.007 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 6144 | 347.080 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 8192 | 201.862 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 129 | 108.480 |

### Qwen3.5-27B / eager / bs1_p16_d512 (BS1/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 32768 | 4001.925 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 32768 | 2031.878 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 24576 | 1387.961 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 32768 | 807.538 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 513 | 431.385 |

### Qwen3.5-27B / eager / bs1_p1k_d0 (BS1/P1024/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 192 | 50.806 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 10.463 |
| 3 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 64 | 7.835 |
| 4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 64 | 3.997 |
| 5 | `elementwise_kernel` | 368 | 3.349 |

### Qwen3.5-27B / eager / bs1_p256_d0 (BS1/P256/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN` | 64 | 8.777 |
| 2 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 64 | 7.802 |
| 3 | `nvjet_sm90_tst_80x128_64x8_2x1_v_bz_TNN` | 128 | 7.454 |
| 4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 64 | 3.955 |
| 5 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 3.173 |

### Qwen3.5-27B / eager / bs1_p4k_d0 (BS1/P4096/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 256 | 264.547 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 32 | 15.446 |
| 3 | `elementwise_kernel` | 368 | 11.785 |
| 4 | `act_and_mul_kernel` | 128 | 8.757 |
| 5 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 48 | 7.980 |

### Qwen3.5-27B / eager / bs1_p8k_d0 (BS1/P8192/D1 effective)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 208 | 435.904 |
| 2 | `nvjet_sm90_tst_192x192_64x4_1x2_h_bz_coopB_TNN` | 48 | 89.154 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 32 | 56.113 |
| 4 | `elementwise_kernel` | 368 | 22.729 |
| 5 | `act_and_mul_kernel` | 128 | 17.626 |

### Qwen3.5-27B / eager / bs4_p16_d128 (BS4/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 8192 | 996.836 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 8192 | 512.158 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 6144 | 349.705 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 8192 | 219.810 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 129 | 108.812 |

### Qwen3.5-27B / eager / bs4_p16_d512 (BS4/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 32768 | 3988.697 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 32768 | 2048.890 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 24576 | 1398.755 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 32768 | 880.125 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 513 | 432.669 |

### Qwen3.5-27B / eager / bs8_p16_d128 (BS8/P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 8192 | 1005.428 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 8192 | 516.291 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 6144 | 352.014 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 8192 | 237.018 |
| 5 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 6144 | 111.636 |

### Qwen3.5-27B / eager / bs8_p16_d512 (BS8/P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 32768 | 4022.432 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 32768 | 2064.943 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 24576 | 1407.837 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 32768 | 949.648 |
| 5 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 24576 | 447.295 |
