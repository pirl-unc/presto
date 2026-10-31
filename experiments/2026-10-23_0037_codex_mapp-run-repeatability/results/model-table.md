| Model | Parameters | L2 / selected update | Validation KL | Development-test KL | Test log10 RMSE | Test Spearman | Change RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| length / exact | 1 | 0 / 25 | 0.1919 | 0.2972 | 0.3258 | -0.0663 | 0.1566 |
| sequence / exact | 92 | 1 / 875 | 0.1694 | 0.2928 | 0.3246 | -0.0225 | 0.1566 |
| interaction / exact | 184 | 1 / 25 | 0.1646 | 0.2908 | 0.3241 | 0.0500 | 0.1712 |
| interaction / reinforce64 / seed42 | 184 | 1 / 500 | 0.1601 | 0.2971 | 0.3290 | 0.0524 | 0.1749 |
| interaction / reinforce64 / seed43 | 184 | 1 / 475 | 0.1594 | 0.2940 | 0.3269 | -0.0042 | 0.1717 |
| interaction / reinforce64 / seed44 | 184 | 1 / 625 | 0.1533 | 0.3095 | 0.3358 | -0.0422 | 0.1768 |
| Uniform | 0 | — | 0.2031 | 0.2890 | 0.3213 | undefined | 0.1566 |
