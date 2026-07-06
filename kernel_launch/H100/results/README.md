# H100 Kernel Launch 实验结果

本文档由 `scripts/summarize.py` 自动生成，汇总 SGLang（FlashInfer 后端）在 Qwen3.5 系列上的 kernel launch 数量与开销。

## 指标说明

- **e2e_ms**：被测 `generate()` 的端到端时间（NVTX `measure` 窗口时长）。
- **total_kernels**：窗口内 GPU kernel 执行次数。
- **launch_count**：窗口内 `cudaLaunchKernel*` 主机侧 API 调用次数。
- **launch_overhead_ms / pct**：这些 launch 调用累计的主机耗时，以及其占端到端时间的比例。


## Qwen3.5-0.8B

| Case | Mode | e2e (ms) | Kernels | Launches | Launch 开销 (ms) | Launch 占比 (%) |
|---|---|---|---|---|---|---|
| case1 (P256/D0) | eager | 42.400 | 979 | 979 | 3.864 | 9.11 |
| case2 (P1024/D0) | eager | 43.666 | 979 | 979 | 3.802 | 8.71 |
| case3 (P8192/D0) | eager | 63.723 | 973 | 973 | 3.806 | 5.97 |
| case5 (P16/D128) | eager | 1748.633 | 46176 | 46176 | 175.886 | 10.06 |
| case6 (P16/D512) | eager | 6929.981 | 184800 | 184800 | 701.030 | 10.12 |
| case7 (P16/D1024) | eager | 14131.576 | 369632 | 369632 | 1435.952 | 10.16 |
| case1 (P256/D0) | cudagraph | 30.936 | 635 | 636 | 2.791 | 9.02 |
| case2 (P1024/D0) | cudagraph | 29.545 | 635 | 636 | 2.811 | 9.51 |
| case3 (P8192/D0) | cudagraph | 49.626 | 629 | 630 | 2.835 | 5.71 |
| case5 (P16/D128) | cudagraph | 212.123 | 2816 | 2944 | 19.570 | 9.23 |
| case6 (P16/D512) | cudagraph | 765.697 | 9347 | 9856 | 72.906 | 9.52 |
| case7 (P16/D1024) | cudagraph | 1510.420 | 18050 | 19072 | 133.966 | 8.87 |

## Qwen3.5-2B

| Case | Mode | e2e (ms) | Kernels | Launches | Launch 开销 (ms) | Launch 占比 (%) |
|---|---|---|---|---|---|---|
| case1 (P256/D0) | eager | 42.562 | 979 | 979 | 3.916 | 9.20 |
| case2 (P1024/D0) | eager | 67.599 | 979 | 979 | 5.423 | 8.02 |
| case3 (P8192/D0) | eager | 87.051 | 973 | 973 | 3.889 | 4.47 |
| case5 (P16/D128) | eager | 2171.313 | 46176 | 46176 | 209.690 | 9.66 |
| case6 (P16/D512) | eager | 7134.089 | 184800 | 184800 | 734.922 | 10.30 |
| case7 (P16/D1024) | eager | 13851.978 | 369632 | 369632 | 1431.722 | 10.34 |
| case1 (P256/D0) | cudagraph | 30.749 | 635 | 636 | 2.854 | 9.28 |
| case2 (P1024/D0) | cudagraph | 35.999 | 635 | 636 | 3.324 | 9.23 |
| case3 (P8192/D0) | cudagraph | 77.870 | 629 | 630 | 4.046 | 5.20 |
| case5 (P16/D128) | cudagraph | 308.981 | 2818 | 2944 | 20.496 | 6.63 |
| case6 (P16/D512) | cudagraph | 1152.516 | 9343 | 9856 | 76.647 | 6.65 |
| case7 (P16/D1024) | cudagraph | 2282.541 | 18050 | 19072 | 126.947 | 5.56 |

## Qwen3.5-4B

| Case | Mode | e2e (ms) | Kernels | Launches | Launch 开销 (ms) | Launch 占比 (%) |
|---|---|---|---|---|---|---|
| case1 (P256/D0) | eager | 102.871 | 1319 | 1319 | 8.969 | 8.72 |
| case2 (P1024/D0) | eager | 90.893 | 1311 | 1311 | 8.052 | 8.86 |
| case3 (P8192/D0) | eager | 193.417 | 1311 | 1311 | 7.574 | 3.92 |
| case5 (P16/D128) | eager | 2299.866 | 64872 | 64872 | 241.338 | 10.49 |
| case6 (P16/D512) | eager | 9406.627 | 259560 | 259560 | 991.086 | 10.54 |
| case7 (P16/D1024) | eager | 18185.307 | 519144 | 519144 | 1946.229 | 10.70 |
| case1 (P256/D0) | cudagraph | 37.756 | 828 | 830 | 3.573 | 9.46 |
| case2 (P1024/D0) | cudagraph | 58.491 | 820 | 822 | 4.942 | 8.45 |
| case3 (P8192/D0) | cudagraph | 162.205 | 820 | 822 | 4.139 | 2.55 |
| case5 (P16/D128) | cudagraph | 557.648 | 3050 | 3176 | 22.293 | 4.00 |
| case6 (P16/D512) | cudagraph | 2129.919 | 9578 | 10088 | 78.895 | 3.70 |
| case7 (P16/D1024) | cudagraph | 4233.157 | 18282 | 19304 | 160.076 | 3.78 |

## Qwen3.5-9B

| Case | Mode | e2e (ms) | Kernels | Launches | Launch 开销 (ms) | Launch 占比 (%) |
|---|---|---|---|---|---|---|
| case1 (P256/D0) | eager | 53.816 | 1359 | 1359 | 5.208 | 9.68 |
| case2 (P1024/D0) | eager | 55.152 | 1351 | 1351 | 5.292 | 9.59 |
| case3 (P8192/D0) | eager | 267.487 | 1327 | 1327 | 5.132 | 1.92 |
| case5 (P16/D128) | eager | 2253.535 | 67960 | 67960 | 244.356 | 10.84 |
| case6 (P16/D512) | eager | 9544.776 | 271864 | 271864 | 1037.891 | 10.87 |
| case7 (P16/D1024) | eager | 18095.915 | 543736 | 543736 | 1985.371 | 10.97 |
| case1 (P256/D0) | cudagraph | 38.412 | 846 | 846 | 3.572 | 9.30 |
| case2 (P1024/D0) | cudagraph | 38.716 | 838 | 838 | 3.631 | 9.38 |
| case3 (P8192/D0) | cudagraph | 248.142 | 0 | 0 | 0.000 | 0.00 |
| case5 (P16/D128) | cudagraph | 872.708 | 3066 | 3192 | 22.131 | 2.54 |
| case6 (P16/D512) | cudagraph | 3387.858 | 9594 | 10104 | 77.591 | 2.29 |
| case7 (P16/D1024) | cudagraph | 6745.109 | 18298 | 19320 | 149.678 | 2.22 |

## Qwen3.5-27B

| Case | Mode | e2e (ms) | Kernels | Launches | Launch 开销 (ms) | Launch 占比 (%) |
|---|---|---|---|---|---|---|
| case1 (P256/D0) | eager | 104.294 | 2723 | 2723 | 10.280 | 9.86 |
| case2 (P1024/D0) | eager | 139.495 | 2707 | 2707 | 10.201 | 7.31 |
| case3 (P8192/D0) | eager | 846.603 | 2659 | 2659 | 267.477 | 31.59 |
| case5 (P16/D128) | eager | 4364.764 | 125316 | 125316 | 474.235 | 10.87 |
| case6 (P16/D512) | eager | 17708.287 | 501252 | 501252 | 2126.185 | 12.01 |
| case7 (P16/D1024) | eager | 35211.169 | 1002500 | 1002500 | 3752.907 | 10.66 |
| case1 (P256/D0) | cudagraph | 83.510 | 1762 | 1762 | 7.091 | 8.49 |
| case2 (P1024/D0) | cudagraph | 120.809 | 1746 | 1746 | 7.051 | 5.84 |
| case3 (P8192/D0) | cudagraph | 825.023 | 1698 | 1698 | 186.516 | 22.61 |
| case5 (P16/D128) | cudagraph | 2596.088 | 3974 | 4100 | 45.509 | 1.75 |
| case6 (P16/D512) | cudagraph | 10041.097 | 10502 | 11012 | 109.735 | 1.09 |
| case7 (P16/D1024) | cudagraph | 20022.028 | 19206 | 20228 | 204.371 | 1.02 |

## Qwen3-30B-A3B

| Case | Mode | e2e (ms) | Kernels | Launches | Launch 开销 (ms) | Launch 占比 (%) |
|---|---|---|---|---|---|---|
| case1 (P256/D0) | eager | 86.749 | 1048 | 1097 | 7.128 | 8.22 |
| case2 (P1024/D0) | eager | 69.465 | 1000 | 1049 | 4.678 | 6.73 |
| case3 (P8192/D0) | eager | 273.497 | 1000 | 1049 | 7.211 | 2.64 |
| case5 (P16/D128) | eager | 7611.461 | 101416 | 101465 | 664.212 | 8.73 |
| case6 (P16/D512) | eager | 23448.371 | 421288 | 421337 | 2166.971 | 9.24 |
| case7 (P16/D1024) | eager | 42203.983 | 847784 | 847833 | 3862.782 | 9.15 |
| case1 (P256/D0) | cudagraph | 26.472 | 0 | 0 | 0.000 | 0.00 |
| case2 (P1024/D0) | cudagraph | 34.558 | 0 | 0 | 0.000 | 0.00 |
| case3 (P8192/D0) | cudagraph | 215.816 | 0 | 0 | 0.000 | 0.00 |
| case5 (P16/D128) | cudagraph | 595.746 | 0 | 0 | 0.000 | 0.00 |
| case6 (P16/D512) | cudagraph | 2366.425 | 0 | 0 | 0.000 | 0.00 |
| case7 (P16/D1024) | cudagraph | 4745.042 | 0 | 0 | 0.000 | 0.00 |

## 各组合 Top-5 耗时 Kernel


### Qwen3-30B-A3B / cudagraph / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|

### Qwen3-30B-A3B / cudagraph / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|

### Qwen3-30B-A3B / cudagraph / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|

### Qwen3-30B-A3B / cudagraph / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|

### Qwen3-30B-A3B / cudagraph / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|

### Qwen3-30B-A3B / cudagraph / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|

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
| 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 18 | 0.306 |
| 2 | `nvjet_sm90_tst_64x32_64x16_1x2_h_bz_TNT` | 24 | 0.250 |
| 3 | `nvjet_sm90_tst_112x128_64x7_2x1_v_bz_TNN` | 24 | 0.242 |
| 4 | `elementwise_kernel` | 73 | 0.200 |
| 5 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 18 | 0.187 |

### Qwen3.5-0.8B / cudagraph / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 42 | 1.094 |
| 2 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 48 | 0.607 |
| 3 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 18 | 0.452 |
| 4 | `elementwise_kernel` | 73 | 0.415 |
| 5 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 18 | 0.313 |

### Qwen3.5-0.8B / cudagraph / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 72 | 7.089 |
| 2 | `BatchPrefillWithPagedKVCacheKernel` | 6 | 6.733 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 24 | 4.459 |
| 4 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 18 | 4.186 |
| 5 | `elementwise_kernel` | 73 | 3.023 |

### Qwen3.5-0.8B / cudagraph / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `reduce_kernel` | 129 | 1.056 |
| 2 | `vectorized_elementwise_kernel` | 471 | 0.818 |
| 3 | `index_elementwise_kernel` | 270 | 0.675 |
| 4 | `unrolled_elementwise_kernel` | 394 | 0.596 |
| 5 | `elementwise_kernel` | 200 | 0.357 |

### Qwen3.5-0.8B / cudagraph / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `reduce_kernel` | 514 | 4.155 |
| 2 | `vectorized_elementwise_kernel` | 1623 | 2.997 |
| 3 | `index_elementwise_kernel` | 1038 | 2.464 |
| 4 | `unrolled_elementwise_kernel` | 1547 | 2.442 |
| 5 | `create_flashinfer_kv_indices_triton` | 513 | 0.938 |

### Qwen3.5-0.8B / cudagraph / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `reduce_kernel` | 1025 | 8.295 |
| 2 | `vectorized_elementwise_kernel` | 3159 | 5.651 |
| 3 | `unrolled_elementwise_kernel` | 3083 | 5.001 |
| 4 | `index_elementwise_kernel` | 2062 | 4.880 |
| 5 | `create_flashinfer_kv_indices_triton` | 1025 | 2.177 |

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
| 1 | `nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN` | 64 | 9.005 |
| 2 | `nvjet_sm90_tst_80x128_64x8_2x1_v_bz_TNN` | 128 | 7.475 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 3.350 |
| 4 | `elementwise_kernel` | 337 | 1.335 |
| 5 | `nvjet_sm90_tst_112x128_64x7_2x1_v_bz_TNN` | 16 | 1.066 |

### Qwen3.5-27B / cudagraph / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 192 | 58.444 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 12.125 |
| 3 | `nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN` | 16 | 3.543 |
| 4 | `elementwise_kernel` | 337 | 3.460 |
| 5 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 48 | 2.189 |

### Qwen3.5-27B / cudagraph / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 208 | 485.222 |
| 2 | `nvjet_sm90_tst_192x192_64x4_1x2_h_bz_coopB_TNN` | 48 | 98.489 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 16 | 62.170 |
| 4 | `elementwise_kernel` | 337 | 24.202 |
| 5 | `act_and_mul_kernel` | 64 | 17.557 |

### Qwen3.5-27B / cudagraph / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 64 | 7.804 |
| 2 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | 64 | 4.049 |
| 3 | `vectorized_elementwise_kernel` | 591 | 2.938 |
| 4 | `nvjet_sm90_tst_128x16_64x11_2x1_v_bz_TNT` | 48 | 2.757 |
| 5 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 64 | 1.652 |

### Qwen3.5-27B / cudagraph / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `vectorized_elementwise_kernel` | 1743 | 13.692 |
| 2 | `nvjet_sm90_tst_384x16_64x4_4x1_v_bz_TNT` | 64 | 7.829 |
| 3 | `index_elementwise_kernel` | 1032 | 5.820 |
| 4 | `elementwise_kernel_with_index` | 513 | 4.513 |
| 5 | `unrolled_elementwise_kernel` | 1547 | 4.466 |

### Qwen3.5-27B / cudagraph / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `vectorized_elementwise_kernel` | 3279 | 30.485 |
| 2 | `index_elementwise_kernel` | 2056 | 9.343 |
| 3 | `unrolled_elementwise_kernel` | 3083 | 8.993 |
| 4 | `elementwise_kernel_with_index` | 1025 | 8.646 |
| 5 | `reduce_kernel` | 1025 | 8.307 |

### Qwen3.5-27B / eager / case1 (P256/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN` | 64 | 9.032 |
| 2 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 64 | 7.766 |
| 3 | `nvjet_sm90_tst_80x128_64x8_2x1_v_bz_TNN` | 128 | 7.506 |
| 4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 64 | 3.947 |
| 5 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 3.330 |

### Qwen3.5-27B / eager / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 192 | 57.978 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 12.069 |
| 3 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 64 | 7.812 |
| 4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 64 | 4.032 |
| 5 | `elementwise_kernel` | 368 | 3.551 |

### Qwen3.5-27B / eager / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 208 | 487.751 |
| 2 | `nvjet_sm90_tst_192x192_64x4_1x2_h_bz_coopB_TNN` | 48 | 99.450 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 32 | 62.594 |
| 4 | `elementwise_kernel` | 368 | 24.368 |
| 5 | `act_and_mul_kernel` | 128 | 17.647 |

### Qwen3.5-27B / eager / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` | 8192 | 993.292 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 8192 | 508.086 |
| 3 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | 6144 | 346.913 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 8192 | 201.897 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 129 | 108.375 |

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
| 1 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 24 | 0.638 |
| 2 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | 48 | 0.551 |
| 3 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1 | 0.334 |
| 4 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 18 | 0.319 |
| 5 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 18 | 0.306 |

### Qwen3.5-2B / cudagraph / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 42 | 2.545 |
| 2 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 48 | 1.220 |
| 3 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 18 | 0.454 |
| 4 | `elementwise_kernel` | 73 | 0.414 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1 | 0.338 |

### Qwen3.5-2B / cudagraph / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 24 | 14.221 |
| 2 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 48 | 9.037 |
| 3 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 24 | 8.783 |
| 4 | `BatchPrefillWithPagedKVCacheKernel` | 6 | 7.271 |
| 5 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 18 | 4.305 |

### Qwen3.5-2B / cudagraph / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `vectorized_elementwise_kernel` | 471 | 1.114 |
| 2 | `reduce_kernel` | 129 | 1.080 |
| 3 | `index_elementwise_kernel` | 270 | 0.773 |
| 4 | `unrolled_elementwise_kernel` | 395 | 0.714 |
| 5 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | 30 | 0.541 |

### Qwen3.5-2B / cudagraph / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `reduce_kernel` | 512 | 4.280 |
| 2 | `vectorized_elementwise_kernel` | 1623 | 3.873 |
| 3 | `index_elementwise_kernel` | 1038 | 3.024 |
| 4 | `unrolled_elementwise_kernel` | 1546 | 2.747 |
| 5 | `create_flashinfer_kv_indices_triton` | 513 | 0.998 |

### Qwen3.5-2B / cudagraph / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `vectorized_elementwise_kernel` | 3159 | 8.640 |
| 2 | `reduce_kernel` | 1025 | 8.593 |
| 3 | `unrolled_elementwise_kernel` | 3083 | 6.023 |
| 4 | `index_elementwise_kernel` | 2062 | 5.456 |
| 5 | `create_flashinfer_kv_indices_triton` | 1025 | 2.273 |

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
| 1 | `nvjet_sm90_tst_320x128_64x3_4x2_h_bz_coopB_TNT` | 32 | 1.481 |
| 2 | `nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN` | 32 | 0.936 |
| 3 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 24 | 0.669 |
| 4 | `nvjet_sm90_tst_80x64_64x11_1x2_h_bz_TNN` | 32 | 0.516 |
| 5 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1 | 0.421 |

### Qwen3.5-4B / cudagraph / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 32 | 4.493 |
| 2 | `nvjet_sm90_tst_168x128_64x5_1x2_h_bz_TNN` | 64 | 3.118 |
| 3 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 24 | 2.248 |
| 4 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 24 | 0.806 |
| 5 | `elementwise_kernel` | 97 | 0.797 |

### Qwen3.5-4B / cudagraph / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 56 | 53.809 |
| 2 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 72 | 30.157 |
| 3 | `BatchPrefillWithPagedKVCacheKernel` | 8 | 19.411 |
| 4 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 24 | 7.650 |
| 5 | `elementwise_kernel` | 97 | 6.191 |

### Qwen3.5-4B / cudagraph / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `vectorized_elementwise_kernel` | 495 | 1.414 |
| 2 | `nvjet_sm90_tst_256x16_64x6_4x1_v_bz_TNT` | 32 | 1.160 |
| 3 | `unrolled_elementwise_kernel` | 395 | 1.124 |
| 4 | `reduce_kernel` | 129 | 1.076 |
| 5 | `index_elementwise_kernel` | 276 | 1.072 |

### Qwen3.5-4B / cudagraph / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `vectorized_elementwise_kernel` | 1647 | 5.290 |
| 2 | `reduce_kernel` | 513 | 4.284 |
| 3 | `unrolled_elementwise_kernel` | 1547 | 4.212 |
| 4 | `index_elementwise_kernel` | 1044 | 4.112 |
| 5 | `elementwise_kernel_with_index` | 513 | 1.206 |

### Qwen3.5-4B / cudagraph / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `vectorized_elementwise_kernel` | 3183 | 10.860 |
| 2 | `reduce_kernel` | 1025 | 8.559 |
| 3 | `unrolled_elementwise_kernel` | 3083 | 8.311 |
| 4 | `index_elementwise_kernel` | 2068 | 8.195 |
| 5 | `elementwise_kernel_with_index` | 1025 | 2.344 |

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
| 1 | `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | 56 | 3.701 |
| 2 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 64 | 2.432 |
| 3 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 1 | 0.687 |
| 4 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 24 | 0.408 |
| 5 | `elementwise_kernel` | 97 | 0.312 |

### Qwen3.5-9B / cudagraph / case2 (P1024/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 120 | 18.612 |
| 2 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 8 | 1.004 |
| 3 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 24 | 0.821 |
| 4 | `elementwise_kernel` | 97 | 0.811 |
| 5 | `act_and_mul_kernel` | 32 | 0.703 |

### Qwen3.5-9B / cudagraph / case3 (P8192/D0)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|

### Qwen3.5-9B / cudagraph / case5 (P16/D128)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `nvjet_sm90_tst_192x16_64x8_2x1_v_bz_TNT` | 32 | 2.256 |
| 2 | `vectorized_elementwise_kernel` | 495 | 1.862 |
| 3 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | 32 | 1.181 |
| 4 | `index_elementwise_kernel` | 268 | 1.062 |
| 5 | `reduce_kernel` | 129 | 1.059 |

### Qwen3.5-9B / cudagraph / case6 (P16/D512)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `vectorized_elementwise_kernel` | 1647 | 6.994 |
| 2 | `index_elementwise_kernel` | 1036 | 4.922 |
| 3 | `reduce_kernel` | 513 | 4.178 |
| 4 | `unrolled_elementwise_kernel` | 1547 | 3.916 |
| 5 | `elementwise_kernel_with_index` | 513 | 2.564 |

### Qwen3.5-9B / cudagraph / case7 (P16/D1024)

| # | Kernel | 次数 | GPU 时间 (ms) |
|---|---|---|---|
| 1 | `vectorized_elementwise_kernel` | 3183 | 15.033 |
| 2 | `index_elementwise_kernel` | 2060 | 9.701 |
| 3 | `unrolled_elementwise_kernel` | 3083 | 8.886 |
| 4 | `reduce_kernel` | 1025 | 8.357 |
| 5 | `elementwise_kernel_with_index` | 1025 | 4.492 |

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
