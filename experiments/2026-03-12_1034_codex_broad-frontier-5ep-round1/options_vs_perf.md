# Broad Frontier 5-Epoch Results
## Summary
- Total conditions: 28
- Completed: 26
- Failed: 2
- OOM: 2
- Shared contract: 7 class-I alleles, broad numeric binding (`IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`), qualifiers `all`, warm start, no synthetics, no ranking, 5 epochs.
## Conditions
| design | family | condition | best val | SLL A02 / A24 | SLL ratio | FLR A02 / A24 | FLR ratio | NFL A02 / A24 | NFL ratio | all 3 probes correct | probe score |
|---|---|---|---:|---|---:|---|---:|---|---:|---|---:|
| PG01 | groove | Positional G01 end_only | 0.8974 | 13.8 / 29645.0 | 2148.4 | 16.9 / 9999.6 | 592.9 | 4551.4 / 28.7 | 158.6 | yes | 8.305 |
| DG1 | groove | Directness G1 groove transformer | 0.8000 | 31.0 / 31225.5 | 1007.8 | 9.5 / 1372.3 | 143.9 | 2931.9 / 7.0 | 418.4 | yes | 7.783 |
| PG07 | groove | Positional G07 triple_baseline | 0.8000 | 31.0 / 31225.5 | 1007.8 | 9.5 / 1372.3 | 143.9 | 2931.9 / 7.0 | 418.4 | yes | 7.783 |
| PG04 | groove | Positional G04 concat(start,end,frac) | 0.8084 | 17.3 / 23913.8 | 1382.2 | 89.7 / 5505.5 | 61.4 | 6610.2 / 13.1 | 503.4 | yes | 7.630 |
| PG03 | groove | Positional G03 concat(start,end) | 0.8047 | 44.5 / 23989.3 | 538.8 | 6.7 / 2310.7 | 347.2 | 737.3 / 6.0 | 123.8 | yes | 7.365 |
| PP00 | presto | Positional P00 start_only | 1.3795 | 37.9 / 35471.1 | 937.0 | 36.4 / 13790.7 | 379.4 | 20351.0 / 695.8 | 29.2 | yes | 7.017 |
| PG05 | groove | Positional G05 mlp(concat(start,end)) | 0.8156 | 49.7 / 6084.0 | 122.3 | 5.8 / 1619.0 | 278.7 | 4065.8 / 13.6 | 300.0 | yes | 7.010 |
| PG06 | groove | Positional G06 mlp(concat(start,end,frac)) | 0.7831 | 30.7 / 33670.0 | 1096.9 | 9.1 / 921.6 | 101.3 | 6458.0 / 75.7 | 85.3 | yes | 6.977 |
| A03 | presto | Assay A03 shared_base_segment_residual split_kd_proxy | 0.8177 | 50.8 / 24795.1 | 488.3 | 31.4 / 424.2 | 13.5 | 7026.1 / 49.7 | 141.4 | yes | 5.970 |
| PP01 | presto | Positional P01 end_only | 1.3051 | 49.3 / 11406.2 | 231.6 | 25.0 / 8755.8 | 350.7 | 991.0 / 94.6 | 10.5 | yes | 5.930 |
| PG00 | groove | Positional G00 start_only | 0.8125 | 45.0 / 18203.6 | 404.9 | 92.7 / 1058.7 | 11.4 | 2817.9 / 22.0 | 127.8 | yes | 5.772 |
| PG02 | groove | Positional G02 start_plus_end | 0.8393 | 91.8 / 17419.2 | 189.9 | 66.1 / 436.5 | 6.6 | 8141.5 / 30.6 | 265.8 | yes | 5.523 |
| PP07 | presto | Positional P07 triple_baseline | 1.7386 | 52.1 / 22169.6 | 425.9 | 62.9 / 1608.4 | 25.6 | 4747.0 / 207.9 | 22.8 | yes | 5.396 |
| PP03 | presto | Positional P03 concat(start,end) | 1.3866 | 138.9 / 16914.6 | 121.8 | 32.6 / 941.7 | 28.9 | 6392.0 / 299.2 | 21.4 | yes | 4.876 |
| A07 | presto | Assay A07 factorized_context_plus_segment split_kd_proxy | 0.7865 | 36.3 / 2926.7 | 80.6 | 105.4 / 1177.1 | 11.2 | 5180.5 / 69.0 | 75.1 | yes | 4.830 |
| PP02 | presto | Positional P02 start_plus_end | 1.2572 | 36.3 / 8864.1 | 244.3 | 72.3 / 775.0 | 10.7 | 3783.8 / 243.0 | 15.6 | yes | 4.610 |
| PP05 | presto | Positional P05 mlp(concat(start,end)) | 1.3039 | 107.6 / 2793.6 | 26.0 | 69.2 / 158.8 | 2.3 | 11317.3 / 41.0 | 276.4 | yes | 4.217 |
| A02 | presto | Assay A02 shared_base_segment_residual merged_kd | 1.1463 | 56.5 / 6368.8 | 112.7 | 86.0 / 622.4 | 7.2 | 20259.8 / 1188.0 | 17.1 | yes | 4.143 |
| A01 | presto | Assay A01 pooled_single_output split_kd_proxy | 1.0488 | 61.5 / 1007.3 | 16.4 | 9.9 / 861.5 | 86.7 | 5221.9 / 3453.0 | 1.5 | yes | 3.332 |
| A06 | presto | Assay A06 factorized_context_plus_segment merged_kd | 1.1112 | 292.7 / 1198.7 | 4.1 | 292.6 / 1944.6 | 6.6 | 2075.1 / 183.4 | 11.3 | yes | 2.489 |
| DP05 | presto | Directness P05 shared_base_segment_residual triple_plus_abs/triple | 1.9730 | 50.8 / 6759.8 | 133.2 | 44.7 / 6820.3 | 152.6 | 736.1 / 816.7 | 0.9 | no | 4.263 |
| PP04 | presto | Positional P04 concat(start,end,frac) | 1.5946 | 71.1 / 3011.3 | 42.4 | 28.2 / 618.2 | 21.9 | 931.2 / 2336.5 | 0.4 | no | 2.568 |
| PP06 | presto | Positional P06 mlp(concat(start,end,frac)) | 1.7554 | 19.3 / 1785.4 | 92.5 | 15.5 / 284.4 | 18.4 | 47.7 / 3654.9 | 0.0 | no | 1.345 |
| A05 | presto | Assay A05 factorized_context split_kd_proxy | 1.1343 | 193.5 / 324.2 | 1.7 | 287.3 / 255.3 | 0.9 | 643.2 / 431.2 | 1.5 | no | 0.346 |
| A04 | presto | Assay A04 factorized_context merged_kd | 2.1623 | 754.7 / 754.7 | 1.0 | 754.7 / 754.7 | 1.0 | 754.7 / 754.7 | 1.0 | no | -0.000 |
| A00 | presto | Assay A00 pooled_single_output merged_kd | 2.8662 | 1519.1 / 1519.1 | 1.0 | 1519.0 / 1519.0 | 1.0 | 1519.2 / 1519.2 | 1.0 | no | -0.000 |

## Failures
- `DP00`: status=`failed`, failure=`oom`
- `DP01`: status=`failed`, failure=`oom`
