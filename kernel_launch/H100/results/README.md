# H100 Kernel Launch 实验结果

本文档由 `scripts/summarize.py` 自动生成，汇总 SGLang（FlashInfer 后端）在 Qwen3.5 系列上的 kernel launch 数量与开销。

## 指标说明

> 采集用 `cudaProfilerStart/Stop` + nsys `--capture-range=cudaProfilerApi`，只录被测的那一次 `generate()`；`--cuda-graph-trace=node` 记录 CUDA graph 内的每个 kernel。

- **e2e_ms**：被测 `generate()` 的端到端时间（采集区间内主机 API 的时间跨度）。
- **total_kernels**：GPU kernel 执行次数（含 CUDA graph 内节点）。
- **launch_count**：`cudaLaunchKernel*` / `cudaGraphLaunch*` / `cuLaunchKernel*` 主机侧 API 调用次数。
- **launch_overhead_ms / pct**：这些 launch 调用累计的主机耗时，以及其占端到端时间的比例。
- 口径提醒：launch 占比在 compute-bound 时会被 overlap/反压高估、launch-bound 时低估，衡量真实影响更宜看 GPU 空闲率或 eager→cudagraph 加速比；`total_kernel_gpu_ms` 是求和，kernel 并发时可能 > e2e。


## Qwen3.5-0.8B

| Case | Mode | e2e (ms) | Kernels | Launches | Launch 开销 (ms) | Launch 占比 (%) |
|---|---|---|---|---|---|---|
| case1 (P256/D0) | eager | 42.400 | 979 | 979 | 3.864 | 9.11 |
| case2 (P1024/D0) | eager | 43.666 | 979 | 979 | 3.802 | 8.71 |
| case3 (P8192/D0) | eager | 63.723 | 973 | 973 | 3.806 | 5.97 |
| case5 (P16/D128) | eager | 1748.633 | 46176 | 46176 | 175.886 | 10.06 |
| case6 (P16/D512) | eager | 6929.981 | 184800 | 184800 | 701.030 | 10.12 |
| case7 (P16/D1024) | eager | 14131.576 | 369632 | 369632 | 1435.952 | 10.16 |
| case1 (P256/D0) | cudagraph | 34.070 | 981 | 636 | 3.226 | 9.47 |
| case2 (P1024/D0) | cudagraph | 34.975 | 981 | 636 | 3.393 | 9.70 |
| case3 (P8192/D0) | cudagraph | 56.741 | 975 | 630 | 4.983 | 8.78 |
| case5 (P16/D128) | cudagraph | 344.273 | 47104 | 2944 | 106.539 | 30.95 |
| case6 (P16/D512) | cudagraph | 832.956 | 186496 | 9856 | 321.553 | 38.60 |
| case7 (P16/D1024) | cudagraph | 1560.370 | 372352 | 19072 | 505.572 | 32.40 |

## Qwen3.5-2B

| Case | Mode | e2e (ms) | Kernels | Launches | Launch 开销 (ms) | Launch 占比 (%) |
|---|---|---|---|---|---|---|
| case1 (P256/D0) | eager | 42.562 | 979 | 979 | 3.916 | 9.20 |
| case2 (P1024/D0) | eager | 67.599 | 979 | 979 | 5.423 | 8.02 |
| case3 (P8192/D0) | eager | 87.051 | 973 | 973 | 3.889 | 4.47 |
| case5 (P16/D128) | eager | 2171.313 | 46176 | 46176 | 209.690 | 9.66 |
| case6 (P16/D512) | eager | 7134.089 | 184800 | 184800 | 734.922 | 10.30 |
| case7 (P16/D1024) | eager | 13851.978 | 369632 | 369632 | 1431.722 | 10.34 |
| case1 (P256/D0) | cudagraph | 33.414 | 980 | 635 | 3.187 | 9.54 |
| case2 (P1024/D0) | cudagraph | 33.150 | 981 | 636 | 3.204 | 9.67 |
| case3 (P8192/D0) | cudagraph | 75.039 | 975 | 630 | 3.162 | 4.21 |
| case5 (P16/D128) | cudagraph | 316.246 | 47104 | 2944 | 61.792 | 19.54 |
| case6 (P16/D512) | cudagraph | 1242.849 | 186496 | 9856 | 342.223 | 27.54 |
| case7 (P16/D1024) | cudagraph | 2325.069 | 372352 | 19072 | 466.058 | 20.04 |

## Qwen3.5-4B

| Case | Mode | e2e (ms) | Kernels | Launches | Launch 开销 (ms) | Launch 占比 (%) |
|---|---|---|---|---|---|---|
| case1 (P256/D0) | eager | 102.871 | 1319 | 1319 | 8.969 | 8.72 |
| case2 (P1024/D0) | eager | 90.893 | 1311 | 1311 | 8.052 | 8.86 |
| case3 (P8192/D0) | eager | 193.417 | 1311 | 1311 | 7.574 | 3.92 |
| case5 (P16/D128) | eager | 2299.866 | 64872 | 64872 | 241.338 | 10.49 |
| case6 (P16/D512) | eager | 9406.627 | 259560 | 259560 | 991.086 | 10.54 |
| case7 (P16/D1024) | eager | 18185.307 | 519144 | 519144 | 1946.229 | 10.70 |
| case1 (P256/D0) | cudagraph | 40.012 | 1300 | 809 | 3.997 | 9.99 |
| case2 (P1024/D0) | cudagraph | 41.567 | 1313 | 822 | 4.109 | 9.89 |
| case3 (P8192/D0) | cudagraph | 161.837 | 1313 | 822 | 5.417 | 3.35 |
| case5 (P16/D128) | cudagraph | 570.911 | 66024 | 3176 | 122.126 | 21.39 |
| case6 (P16/D512) | cudagraph | 2154.231 | 261480 | 10088 | 434.154 | 20.15 |
| case7 (P16/D1024) | cudagraph | 4250.159 | 522088 | 19304 | 591.166 | 13.91 |

## Qwen3.5-9B

| Case | Mode | e2e (ms) | Kernels | Launches | Launch 开销 (ms) | Launch 占比 (%) |
|---|---|---|---|---|---|---|
| case1 (P256/D0) | eager | 53.816 | 1359 | 1359 | 5.208 | 9.68 |
| case2 (P1024/D0) | eager | 55.152 | 1351 | 1351 | 5.292 | 9.59 |
| case3 (P8192/D0) | eager | 267.487 | 1327 | 1327 | 5.132 | 1.92 |
| case5 (P16/D128) | eager | 2253.535 | 67960 | 67960 | 244.356 | 10.84 |
| case6 (P16/D512) | eager | 9544.776 | 271864 | 271864 | 1037.891 | 10.87 |
| case7 (P16/D1024) | eager | 18095.915 | 543736 | 543736 | 1985.371 | 10.97 |
| case1 (P256/D0) | cudagraph | 47.594 | 1359 | 846 | 5.423 | 11.39 |
| case2 (P1024/D0) | cudagraph | 47.948 | 1351 | 838 | 5.547 | 11.57 |
| case3 (P8192/D0) | cudagraph | 252.478 | 1327 | 814 | 5.252 | 2.08 |
| case5 (P16/D128) | cudagraph | 882.363 | 69078 | 3192 | 121.401 | 13.76 |
| case6 (P16/D512) | cudagraph | 3440.747 | 273681 | 10104 | 483.724 | 14.06 |
| case7 (P16/D1024) | cudagraph | 6815.098 | 546680 | 19320 | 608.999 | 8.94 |

## Qwen3.5-27B

| Case | Mode | e2e (ms) | Kernels | Launches | Launch 开销 (ms) | Launch 占比 (%) |
|---|---|---|---|---|---|---|
| case1 (P256/D0) | eager | 114.323 | 2723 | 2723 | 9.959 | 8.71 |
| case2 (P1024/D0) | eager | 146.071 | 2645 | 2645 | 10.716 | 7.34 |
| case3 (P8192/D0) | eager | 872.255 | 2659 | 2659 | 273.501 | 31.36 |
| case5 (P16/D128) | eager | 4971.475 | 125316 | 125316 | 537.341 | 10.81 |
| case6 (P16/D512) | eager | 17708.287 | 501252 | 501252 | 2126.185 | 12.01 |
| case7 (P16/D1024) | eager | 35211.169 | 1002500 | 1002500 | 3752.907 | 10.66 |
| case1 (P256/D0) | cudagraph | 76.744 | 2080 | 1760 | 8.628 | 11.24 |
| case2 (P1024/D0) | cudagraph | 122.471 | 2089 | 1746 | 13.917 | 11.36 |
| case3 (P8192/D0) | cudagraph | 839.369 | 2262 | 1698 | 251.715 | 29.99 |
| case5 (P16/D128) | cudagraph | 2580.637 | 127217 | 4100 | 164.251 | 6.36 |
| case6 (P16/D512) | cudagraph | 10049.251 | 503376 | 11003 | 563.940 | 5.61 |
| case7 (P16/D1024) | cudagraph | 20052.737 | 1005887 | 20228 | 1096.506 | 5.47 |

## Qwen3-30B-A3B

| Case | Mode | e2e (ms) | Kernels | Launches | Launch 开销 (ms) | Launch 占比 (%) |
|---|---|---|---|---|---|---|
| case1 (P256/D0) | eager | 86.749 | 1048 | 1097 | 7.128 | 8.22 |
| case2 (P1024/D0) | eager | 69.465 | 1000 | 1049 | 4.678 | 6.73 |
| case3 (P8192/D0) | eager | 273.497 | 1000 | 1049 | 7.211 | 2.64 |
| case5 (P16/D128) | eager | 7611.461 | 101416 | 101465 | 664.212 | 8.73 |
| case6 (P16/D512) | eager | 23448.371 | 421288 | 421337 | 2166.971 | 9.24 |
| case7 (P16/D1024) | eager | 42203.983 | 847784 | 847833 | 3862.782 | 9.15 |
| case1 (P256/D0) | cudagraph | 28.084 | 1725 | 281 | 3.527 | 12.56 |
| case2 (P1024/D0) | cudagraph | 36.112 | 1677 | 233 | 3.476 | 9.63 |
| case3 (P8192/D0) | cudagraph | 216.640 | 1677 | 233 | 3.859 | 1.78 |
| case5 (P16/D128) | cudagraph | 606.719 | 107898 | 2393 | 125.074 | 20.61 |
| case6 (P16/D512) | cudagraph | 2385.445 | 428922 | 8921 | 481.766 | 20.20 |
| case7 (P16/D1024) | cudagraph | 4778.854 | 856954 | 17625 | 956.016 | 20.01 |

## 各组合 Top-5 耗时 Kernel


### Qwen3-30B-A3B / cudagraph / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 192 | 13.701 |
| 2 | `PersistentVariableLengthMergeStatesKernel` | 96 | 1.163 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 0.935 |
| 4 | `nvjet_sm90_tst_80x128_64x8_2x1_v_bz_TNN` | 48 | 0.563 |
| 5 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | 48 | 0.538 |

### Qwen3-30B-A3B / cudagraph / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 192 | 17.157 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 3.103 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 48 | 1.475 |
| 4 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 48 | 1.161 |
| 5 | `moe_sum_reduce_warp_per_token_vec_kernel` | 48 | 0.610 |

### Qwen3-30B-A3B / cudagraph / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 89.585 |
| 2 | `fused_moe_kernel` | 192 | 65.560 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 48 | 11.895 |
| 4 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 9.213 |
| 5 | `moe_sum_reduce_warp_per_token_vec_kernel` | 48 | 4.895 |

### Qwen3-30B-A3B / cudagraph / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 12384 | 198.188 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 6144 | 57.492 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 6192 | 55.480 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 6144 | 49.682 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 6144 | 29.979 |

### Qwen3-30B-A3B / cudagraph / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 49248 | 771.632 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 24576 | 229.437 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 24624 | 228.939 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 24576 | 199.989 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 24576 | 119.378 |

### Qwen3-30B-A3B / cudagraph / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 98400 | 1537.058 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 49200 | 466.819 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 49152 | 459.356 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 49152 | 396.669 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 49152 | 238.563 |

### Qwen3-30B-A3B / eager / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 96 | 1.516 |
| 2 | `PersistentVariableLengthMergeStatesKernel` | 96 | 1.163 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 0.824 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 48 | 0.460 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 2 | 0.409 |

### Qwen3-30B-A3B / eager / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 3.012 |
| 2 | `fused_moe_kernel` | 96 | 1.512 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 48 | 0.465 |
| 4 | `elementwise_kernel` | 48 | 0.465 |
| 5 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 2 | 0.411 |

### Qwen3-30B-A3B / eager / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `BatchPrefillWithPagedKVCacheKernel` | 96 | 89.536 |
| 2 | `elementwise_kernel` | 48 | 4.302 |
| 3 | `fused_moe_kernel` | 96 | 1.561 |
| 4 | `vectorized_elementwise_kernel` | 54 | 1.018 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 48 | 0.470 |

### Qwen3-30B-A3B / eager / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 12288 | 193.440 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 6144 | 58.631 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 6144 | 51.460 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 6192 | 41.615 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 6144 | 31.025 |

### Qwen3-30B-A3B / eager / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 49152 | 771.982 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 24576 | 234.568 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 24576 | 205.636 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 24624 | 172.911 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 24576 | 123.479 |

### Qwen3-30B-A3B / eager / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `fused_moe_kernel` | 98304 | 1547.395 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 49152 | 470.034 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 49152 | 412.223 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 49200 | 357.075 |
| 5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 49152 | 248.395 |

### Qwen3.5-0.8B / cudagraph / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 54 | 0.354 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 2 | 0.337 |
| 3 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 18 | 0.306 |
| 4 | `nvjet_sm90_tst_64x32_64x16_1x2_h_bz_TNT` | 24 | 0.251 |
| 5 | `nvjet_sm90_tst_112x128_64x7_2x1_v_bz_TNN` | 24 | 0.243 |

### Qwen3.5-0.8B / cudagraph / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 42 | 1.101 |
| 2 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 48 | 0.611 |
| 3 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 18 | 0.453 |
| 4 | `elementwise_kernel` | 85 | 0.441 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 54 | 0.352 |

### Qwen3.5-0.8B / cudagraph / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 72 | 7.114 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 12 | 6.864 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 24 | 4.462 |
| 4 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 18 | 4.198 |
| 5 | `elementwise_kernel` | 85 | 3.057 |

### Qwen3.5-0.8B / cudagraph / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 6912 | 43.824 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 129 | 21.422 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 2304 | 19.353 |
| 4 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` | 2304 | 18.487 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 3072 | 17.026 |

### Qwen3.5-0.8B / cudagraph / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 27648 | 173.907 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 513 | 85.128 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 9216 | 77.098 |
| 4 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` | 9216 | 73.679 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 12288 | 67.575 |

### Qwen3.5-0.8B / cudagraph / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 55296 | 347.371 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1025 | 170.299 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 18432 | 154.268 |
| 4 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` | 18432 | 147.450 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 24576 | 134.909 |

### Qwen3.5-0.8B / eager / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 54 | 0.348 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 2 | 0.338 |
| 3 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 18 | 0.304 |
| 4 | `nvjet_sm90_tst_64x32_64x16_1x2_h_bz_TNT` | 24 | 0.252 |
| 5 | `nvjet_sm90_tst_112x128_64x7_2x1_v_bz_TNN` | 24 | 0.245 |

### Qwen3.5-0.8B / eager / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 42 | 1.099 |
| 2 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 48 | 0.609 |
| 3 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 18 | 0.458 |
| 4 | `elementwise_kernel` | 84 | 0.424 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 54 | 0.355 |

### Qwen3.5-0.8B / eager / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 72 | 7.107 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 12 | 6.793 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 24 | 4.485 |
| 4 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 18 | 4.176 |
| 5 | `elementwise_kernel` | 84 | 3.048 |

### Qwen3.5-0.8B / eager / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 6912 | 45.349 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 129 | 21.563 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 2304 | 19.474 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 3072 | 17.411 |
| 5 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 2304 | 11.487 |

### Qwen3.5-0.8B / eager / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 27648 | 180.713 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 513 | 85.791 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 9216 | 77.747 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 12288 | 69.029 |
| 5 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 9216 | 45.738 |

### Qwen3.5-0.8B / eager / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 55296 | 361.253 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1025 | 171.279 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 18432 | 155.621 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 24576 | 138.762 |
| 5 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 18432 | 91.286 |

### Qwen3.5-27B / cudagraph / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN` | 64 | 9.099 |
| 2 | `nvjet_sm90_tst_80x128_64x8_2x1_v_bz_TNN` | 128 | 7.537 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 3.542 |
| 4 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 21 | 2.567 |
| 5 | `elementwise_kernel` | 347 | 1.379 |

### Qwen3.5-27B / cudagraph / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 192 | 59.192 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 12.388 |
| 3 | `nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN` | 16 | 3.623 |
| 4 | `elementwise_kernel` | 347 | 3.431 |
| 5 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 23 | 2.852 |

### Qwen3.5-27B / cudagraph / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 208 | 502.169 |
| 2 | `nvjet_sm90_tst_192x192_64x4_1x2_h_bz_coopB_TNN` | 48 | 101.850 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 25 | 65.006 |
| 4 | `elementwise_kernel` | 355 | 24.893 |
| 5 | `act_and_mul_kernel` | 101 | 17.665 |

### Qwen3.5-27B / cudagraph / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 8182 | 1003.629 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 8182 | 505.535 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 6138 | 348.719 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 6138 | 321.357 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 8183 | 201.014 |

### Qwen3.5-27B / cudagraph / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 32723 | 3959.403 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 32723 | 2020.824 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 24543 | 1393.809 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 24543 | 1276.980 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 32723 | 804.168 |

### Qwen3.5-27B / cudagraph / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 65506 | 7926.619 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 65506 | 4045.557 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 49131 | 2790.119 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 49131 | 2555.498 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 65506 | 1610.305 |

### Qwen3.5-27B / eager / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN` | 64 | 9.092 |
| 2 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 64 | 7.874 |
| 3 | `nvjet_sm90_tst_80x128_64x8_2x1_v_bz_TNN` | 128 | 7.617 |
| 4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 64 | 3.949 |
| 5 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 3.550 |

### Qwen3.5-27B / eager / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 190 | 58.968 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 47 | 12.184 |
| 3 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 64 | 7.914 |
| 4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 64 | 4.056 |
| 5 | `nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN` | 16 | 3.621 |

### Qwen3.5-27B / eager / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 208 | 503.516 |
| 2 | `nvjet_sm90_tst_192x192_64x4_1x2_h_bz_coopB_TNN` | 48 | 102.237 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 32 | 65.231 |
| 4 | `elementwise_kernel` | 368 | 25.059 |
| 5 | `act_and_mul_kernel` | 128 | 17.770 |

### Qwen3.5-27B / eager / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 8192 | 1005.430 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 8192 | 508.077 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 6144 | 347.143 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 8192 | 202.086 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 129 | 108.300 |

### Qwen3.5-27B / eager / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 32768 | 3973.193 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 32768 | 2031.880 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 24576 | 1387.649 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 32768 | 807.803 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 513 | 430.996 |

### Qwen3.5-27B / eager / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 65536 | 7946.334 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 65536 | 4067.111 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 49152 | 2775.609 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 65536 | 1618.380 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1025 | 861.332 |

### Qwen3.5-2B / cudagraph / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 54 | 0.699 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 2 | 0.667 |
| 3 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 24 | 0.640 |
| 4 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | 48 | 0.533 |
| 5 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 18 | 0.314 |

### Qwen3.5-2B / cudagraph / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 42 | 2.576 |
| 2 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 48 | 1.225 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 54 | 0.696 |
| 4 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 2 | 0.669 |
| 5 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 18 | 0.444 |

### Qwen3.5-2B / cudagraph / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 24 | 14.122 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 8.972 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 24 | 8.719 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 12 | 7.346 |
| 5 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 18 | 4.315 |

### Qwen3.5-2B / cudagraph / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 6912 | 88.193 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 129 | 42.889 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 3072 | 33.505 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 2304 | 31.456 |
| 5 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` | 2304 | 30.765 |

### Qwen3.5-2B / cudagraph / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 27648 | 352.196 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 513 | 170.646 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 12288 | 133.691 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 9216 | 125.406 |
| 5 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` | 9216 | 122.546 |

### Qwen3.5-2B / cudagraph / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 55296 | 703.714 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1025 | 340.531 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 24576 | 267.158 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 18432 | 250.612 |
| 5 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` | 18432 | 244.836 |

### Qwen3.5-2B / eager / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 54 | 0.677 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 2 | 0.668 |
| 3 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 24 | 0.639 |
| 4 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | 48 | 0.577 |
| 5 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 18 | 0.316 |

### Qwen3.5-2B / eager / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 42 | 2.516 |
| 2 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 48 | 1.159 |
| 3 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 2 | 0.672 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 54 | 0.662 |
| 5 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 18 | 0.446 |

### Qwen3.5-2B / eager / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 24 | 14.162 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 8.963 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 24 | 8.735 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 12 | 7.118 |
| 5 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 18 | 4.254 |

### Qwen3.5-2B / eager / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 6912 | 85.559 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 129 | 42.829 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 3072 | 33.802 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 2304 | 31.287 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 6192 | 13.242 |

### Qwen3.5-2B / eager / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 27648 | 343.101 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 513 | 170.569 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 12288 | 135.229 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 9216 | 124.984 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 24624 | 52.617 |

### Qwen3.5-2B / eager / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 55296 | 685.870 |
| 2 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1025 | 340.727 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 24576 | 269.428 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 18432 | 250.147 |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0` | 49200 | 104.713 |

### Qwen3.5-4B / cudagraph / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_4x2_h_bz_coopB_TNT` | 32 | 1.486 |
| 2 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` | 32 | 1.116 |
| 3 | `nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN` | 32 | 0.937 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 64 | 0.882 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 2 | 0.839 |

### Qwen3.5-4B / cudagraph / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 32 | 4.401 |
| 2 | `nvjet_sm90_tst_168x128_64x5_1x2_h_bz_TNN` | 64 | 3.161 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 24 | 2.270 |
| 4 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` | 32 | 1.115 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 64 | 0.883 |

### Qwen3.5-4B / cudagraph / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 56 | 53.641 |
| 2 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 72 | 29.785 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 16 | 18.686 |
| 4 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 24 | 7.538 |
| 5 | `elementwise_kernel` | 113 | 6.137 |

### Qwen3.5-4B / cudagraph / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` | 4096 | 142.276 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 8192 | 112.414 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 3072 | 71.776 |
| 4 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` | 3072 | 55.867 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 129 | 53.861 |

### Qwen3.5-4B / cudagraph / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` | 16384 | 568.946 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 32768 | 448.641 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 12288 | 286.749 |
| 4 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` | 12288 | 227.741 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 513 | 214.231 |

### Qwen3.5-4B / cudagraph / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` | 32768 | 1138.013 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 65536 | 897.212 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 24576 | 573.045 |
| 4 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` | 24576 | 443.855 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1025 | 428.010 |

### Qwen3.5-4B / eager / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_4x2_h_bz_coopB_TNT` | 32 | 1.498 |
| 2 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` | 32 | 1.112 |
| 3 | `nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN` | 32 | 0.931 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 64 | 0.888 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 2 | 0.841 |

### Qwen3.5-4B / eager / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 32 | 4.575 |
| 2 | `nvjet_sm90_tst_168x128_64x5_1x2_h_bz_TNN` | 64 | 3.128 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 24 | 2.249 |
| 4 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` | 32 | 1.109 |
| 5 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 64 | 0.891 |

### Qwen3.5-4B / eager / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 56 | 53.517 |
| 2 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 72 | 29.826 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 16 | 18.839 |
| 4 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 24 | 7.519 |
| 5 | `elementwise_kernel` | 112 | 6.135 |

### Qwen3.5-4B / eager / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` | 4096 | 141.925 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 8192 | 113.618 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 3072 | 71.073 |
| 4 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 129 | 54.322 |
| 5 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT` | 1024 | 20.156 |

### Qwen3.5-4B / eager / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` | 16384 | 567.856 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 32768 | 453.633 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 12288 | 283.663 |
| 4 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 513 | 216.116 |
| 5 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT` | 4096 | 80.307 |

### Qwen3.5-4B / eager / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` | 32768 | 1136.843 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 65536 | 908.336 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 24576 | 568.463 |
| 4 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1025 | 431.602 |
| 5 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT` | 8192 | 160.695 |

### Qwen3.5-9B / cudagraph / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 56 | 3.738 |
| 2 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 64 | 2.436 |
| 3 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 32 | 2.202 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 64 | 1.594 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 2 | 1.365 |

### Qwen3.5-9B / cudagraph / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 120 | 18.574 |
| 2 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 32 | 2.204 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 64 | 1.596 |
| 4 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 2 | 1.372 |
| 5 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 8 | 1.006 |

### Qwen3.5-9B / cudagraph / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 56 | 104.563 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 64 | 50.335 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 16 | 20.375 |
| 4 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 8 | 8.035 |
| 5 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 24 | 7.885 |

### Qwen3.5-9B / cudagraph / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 4094 | 281.217 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 8188 | 202.859 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 3072 | 113.363 |
| 4 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 128 | 87.489 |
| 5 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_splitK_TNT` | 3072 | 59.735 |

### Qwen3.5-9B / cudagraph / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 16378 | 1124.903 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 32756 | 811.264 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 12284 | 454.181 |
| 4 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 512 | 349.842 |
| 5 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_splitK_TNT` | 12284 | 235.663 |

### Qwen3.5-9B / cudagraph / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 32768 | 2250.536 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 65536 | 1623.084 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 24576 | 910.080 |
| 4 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1025 | 700.452 |
| 5 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_splitK_TNT` | 24576 | 476.379 |

### Qwen3.5-9B / eager / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 56 | 3.718 |
| 2 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 64 | 2.421 |
| 3 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 32 | 2.204 |
| 4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 64 | 1.584 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 2 | 1.372 |

### Qwen3.5-9B / eager / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 120 | 18.641 |
| 2 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 32 | 2.217 |
| 3 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 64 | 1.609 |
| 4 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 2 | 1.381 |
| 5 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 8 | 1.007 |

### Qwen3.5-9B / eager / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 56 | 106.344 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 64 | 51.409 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 16 | 20.856 |
| 4 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 8 | 8.189 |
| 5 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 24 | 7.985 |

### Qwen3.5-9B / eager / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 4096 | 282.478 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 8192 | 202.797 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 3072 | 108.664 |
| 4 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 129 | 88.349 |
| 5 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT` | 1024 | 30.656 |

### Qwen3.5-9B / eager / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 16384 | 1130.367 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 32768 | 811.286 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 12288 | 435.015 |
| 4 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 513 | 351.496 |
| 5 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT` | 4096 | 123.142 |

### Qwen3.5-9B / eager / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` | 32768 | 2261.194 |
| 2 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 65536 | 1623.441 |
| 3 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 24576 | 867.164 |
| 4 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1025 | 702.231 |
| 5 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT` | 8192 | 245.913 |
