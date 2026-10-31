# Data-curation retrain versus parent

Repeated fixed-holdout development comparison; no test-based selection.
Same held-out observations verified by ID and all observation fields.

| Seed | Split / era | Output / class / population | Exact n or binary n | Parent | Curated | Delta |
|---|---|---|---:|---:|---:|---:|
| 42 | test / 2026 | elution_observation / I / synthetic_reference_proteome (auprc) | 12 | 1.0000 | 1.0000 | 0.0000 |
| 42 | test / 2026 | elution_observation / I / synthetic_same_protein (auprc) | 31002 | 0.9043 | 0.9319 | 0.0276 |
| 42 | test / 2026 | elution_observation / II / no_background (auprc) | 2 | 1.0000 | 1.0000 | 0.0000 |
| 42 | test / 2026 | elution_observation / II / synthetic_reference_proteome (auprc) | 546 | 0.4890 | 0.6663 | 0.1773 |
| 42 | test / 2026 | elution_observation / II / synthetic_same_protein (auprc) | 39108 | 0.5412 | 0.6677 | 0.1265 |
| 42 | test / 2026 | kd_via_ic50 / I / reported_measurements (spearman) | 592 | 0.4648 | 0.6328 | 0.1681 |
| 42 | test / older_or_mixed | ec50 / I / reported_measurements (spearman) | 70 | 0.2998 | -0.0418 | -0.3416 |
| 42 | test / older_or_mixed | ec50 / II / reported_measurements (spearman) | 107 | 0.2027 | 0.4789 | 0.2762 |
| 42 | test / older_or_mixed | elution_observation / I / synthetic_same_protein (auprc) | 432 | 0.8649 | 0.9740 | 0.1092 |
| 42 | test / older_or_mixed | half_life / I / reported_measurements (spearman) | 62 | 0.2359 | 0.4530 | 0.2171 |
| 42 | test / older_or_mixed | half_life / II / reported_measurements (spearman) | 148 | -0.0751 | -0.1849 | -0.1097 |
| 42 | test / older_or_mixed | ic50 / I / reported_measurements (spearman) | 100 | 0.2392 | 0.3758 | 0.1366 |
| 42 | test / older_or_mixed | ic50 / II / reported_measurements (spearman) | 267 | 0.5199 | 0.5837 | 0.0638 |
| 42 | test / older_or_mixed | kd / I / reported_measurements (spearman) | 386 | 0.4465 | 0.5723 | 0.1258 |
| 42 | test / older_or_mixed | kd / II / reported_measurements (spearman) | 21 | 0.3987 | 0.6052 | 0.2065 |
| 42 | test / older_or_mixed | kd_via_ec50 / I / reported_measurements (spearman) | 66 | 0.4263 | 0.2463 | -0.1800 |
| 42 | test / older_or_mixed | kd_via_ec50 / II / reported_measurements (spearman) | 149 | 0.5462 | 0.4679 | -0.0783 |
| 42 | test / older_or_mixed | kd_via_ic50 / I / reported_measurements (spearman) | 12 | 0.0876 | -0.0245 | -0.1121 |
| 42 | test / older_or_mixed | kd_via_ic50 / II / reported_measurements (spearman) | 55 | 0.1777 | 0.2572 | 0.0795 |
| 42 | test / older_or_mixed | qualitative_binding / I / reported_measurements (auprc) | 290 | 0.5385 | 0.7001 | 0.1616 |
| 42 | test / older_or_mixed | qualitative_binding / II / reported_measurements (auprc) | 151 | 0.4483 | 0.7572 | 0.3088 |
| 42 | test / older_or_mixed | td50 / I / reported_measurements (spearman) | 140 | 0.4776 | 0.5071 | 0.0295 |
| 42 | validation / older_or_mixed | ec50 / I / reported_measurements (spearman) | 69 | 0.5977 | 0.3510 | -0.2466 |
| 42 | validation / older_or_mixed | elution_observation / I / synthetic_reference_proteome (auprc) | 180 | 0.9152 | 0.9330 | 0.0179 |
| 42 | validation / older_or_mixed | elution_observation / I / synthetic_same_protein (auprc) | 10116 | 0.8982 | 0.9716 | 0.0734 |
| 42 | validation / older_or_mixed | elution_observation / II / synthetic_reference_proteome (auprc) | 12 | 0.8333 | 0.7500 | -0.0833 |
| 42 | validation / older_or_mixed | elution_observation / II / synthetic_same_protein (auprc) | 4152 | 0.5049 | 0.6543 | 0.1494 |
| 42 | validation / older_or_mixed | half_life / I / reported_measurements (spearman) | 71 | 0.3928 | 0.4948 | 0.1019 |
| 42 | validation / older_or_mixed | ic50 / I / reported_measurements (spearman) | 141 | 0.4995 | 0.5707 | 0.0711 |
| 42 | validation / older_or_mixed | ic50 / II / reported_measurements (spearman) | 1129 | 0.4248 | 0.5306 | 0.1058 |
| 42 | validation / older_or_mixed | kd / I / reported_measurements (spearman) | 151 | 0.4821 | 0.6573 | 0.1753 |
| 42 | validation / older_or_mixed | kd_via_ec50 / I / reported_measurements (spearman) | 106 | 0.3869 | 0.6411 | 0.2541 |
| 42 | validation / older_or_mixed | kd_via_ec50 / II / reported_measurements (spearman) | 352 | 0.6270 | 0.6056 | -0.0214 |
| 42 | validation / older_or_mixed | kd_via_ic50 / I / reported_measurements (spearman) | 83 | 0.5316 | 0.5120 | -0.0196 |
| 42 | validation / older_or_mixed | kd_via_ic50 / II / reported_measurements (spearman) | 246 | 0.1511 | 0.0900 | -0.0611 |
| 42 | validation / older_or_mixed | qualitative_binding / I / reported_measurements (auprc) | 689 | 0.9318 | 0.9566 | 0.0249 |
| 42 | validation / older_or_mixed | qualitative_binding / II / reported_measurements (auprc) | 446 | 0.3790 | 0.5148 | 0.1359 |
| 42 | validation / older_or_mixed | td50 / I / reported_measurements (spearman) | 145 | 0.4765 | 0.4455 | -0.0310 |
| 43 | test / 2026 | elution_observation / I / synthetic_reference_proteome (auprc) | 12 | 1.0000 | 1.0000 | 0.0000 |
| 43 | test / 2026 | elution_observation / I / synthetic_same_protein (auprc) | 31002 | 0.9236 | 0.9293 | 0.0057 |
| 43 | test / 2026 | elution_observation / II / no_background (auprc) | 2 | 1.0000 | 1.0000 | 0.0000 |
| 43 | test / 2026 | elution_observation / II / synthetic_reference_proteome (auprc) | 546 | 0.6752 | 0.6518 | -0.0235 |
| 43 | test / 2026 | elution_observation / II / synthetic_same_protein (auprc) | 39108 | 0.6312 | 0.6738 | 0.0427 |
| 43 | test / 2026 | kd_via_ic50 / I / reported_measurements (spearman) | 592 | 0.7250 | 0.6996 | -0.0253 |
| 43 | test / older_or_mixed | ec50 / I / reported_measurements (spearman) | 70 | -0.0346 | 0.1661 | 0.2006 |
| 43 | test / older_or_mixed | ec50 / II / reported_measurements (spearman) | 107 | 0.3109 | 0.2408 | -0.0701 |
| 43 | test / older_or_mixed | elution_observation / I / synthetic_same_protein (auprc) | 432 | 0.9335 | 0.9805 | 0.0470 |
| 43 | test / older_or_mixed | half_life / I / reported_measurements (spearman) | 62 | 0.5028 | 0.4053 | -0.0975 |
| 43 | test / older_or_mixed | half_life / II / reported_measurements (spearman) | 148 | -0.1715 | -0.0953 | 0.0762 |
| 43 | test / older_or_mixed | ic50 / I / reported_measurements (spearman) | 100 | 0.2629 | 0.3023 | 0.0394 |
| 43 | test / older_or_mixed | ic50 / II / reported_measurements (spearman) | 267 | 0.6034 | 0.5662 | -0.0372 |
| 43 | test / older_or_mixed | kd / I / reported_measurements (spearman) | 386 | 0.5436 | 0.5877 | 0.0441 |
| 43 | test / older_or_mixed | kd / II / reported_measurements (spearman) | 21 | 0.5909 | 0.5961 | 0.0052 |
| 43 | test / older_or_mixed | kd_via_ec50 / I / reported_measurements (spearman) | 66 | 0.4544 | 0.2114 | -0.2431 |
| 43 | test / older_or_mixed | kd_via_ec50 / II / reported_measurements (spearman) | 149 | 0.6083 | 0.5595 | -0.0489 |
| 43 | test / older_or_mixed | kd_via_ic50 / I / reported_measurements (spearman) | 12 | 0.1716 | 0.0946 | -0.0771 |
| 43 | test / older_or_mixed | kd_via_ic50 / II / reported_measurements (spearman) | 55 | 0.2390 | 0.2434 | 0.0044 |
| 43 | test / older_or_mixed | qualitative_binding / I / reported_measurements (auprc) | 290 | 0.6927 | 0.5821 | -0.1106 |
| 43 | test / older_or_mixed | qualitative_binding / II / reported_measurements (auprc) | 151 | 0.7889 | 0.8644 | 0.0754 |
| 43 | test / older_or_mixed | td50 / I / reported_measurements (spearman) | 140 | 0.3198 | 0.4787 | 0.1589 |
| 43 | validation / older_or_mixed | ec50 / I / reported_measurements (spearman) | 69 | 0.2210 | 0.2240 | 0.0030 |
| 43 | validation / older_or_mixed | elution_observation / I / synthetic_reference_proteome (auprc) | 180 | 0.9081 | 0.9296 | 0.0215 |
| 43 | validation / older_or_mixed | elution_observation / I / synthetic_same_protein (auprc) | 10116 | 0.9565 | 0.9658 | 0.0094 |
| 43 | validation / older_or_mixed | elution_observation / II / synthetic_reference_proteome (auprc) | 12 | 0.8333 | 0.8333 | 0.0000 |
| 43 | validation / older_or_mixed | elution_observation / II / synthetic_same_protein (auprc) | 4152 | 0.5887 | 0.6575 | 0.0688 |
| 43 | validation / older_or_mixed | half_life / I / reported_measurements (spearman) | 71 | 0.4176 | 0.5047 | 0.0871 |
| 43 | validation / older_or_mixed | ic50 / I / reported_measurements (spearman) | 141 | 0.5344 | 0.5491 | 0.0147 |
| 43 | validation / older_or_mixed | ic50 / II / reported_measurements (spearman) | 1129 | 0.4580 | 0.4928 | 0.0348 |
| 43 | validation / older_or_mixed | kd / I / reported_measurements (spearman) | 151 | 0.5626 | 0.5829 | 0.0203 |
| 43 | validation / older_or_mixed | kd_via_ec50 / I / reported_measurements (spearman) | 106 | 0.5577 | 0.5924 | 0.0347 |
| 43 | validation / older_or_mixed | kd_via_ec50 / II / reported_measurements (spearman) | 352 | 0.6247 | 0.6008 | -0.0239 |
| 43 | validation / older_or_mixed | kd_via_ic50 / I / reported_measurements (spearman) | 83 | 0.5694 | 0.6381 | 0.0686 |
| 43 | validation / older_or_mixed | kd_via_ic50 / II / reported_measurements (spearman) | 246 | 0.1184 | 0.1372 | 0.0188 |
| 43 | validation / older_or_mixed | qualitative_binding / I / reported_measurements (auprc) | 689 | 0.9411 | 0.9773 | 0.0362 |
| 43 | validation / older_or_mixed | qualitative_binding / II / reported_measurements (auprc) | 446 | 0.4592 | 0.5575 | 0.0983 |
| 43 | validation / older_or_mixed | td50 / I / reported_measurements (spearman) | 145 | 0.4986 | 0.4360 | -0.0626 |

All assay, class and study strata, including RMSE, Pearson, AUROC and censor satisfaction, are in `parent-comparison.csv`/`.json`. Three source-identified suspect generated test negatives are preserved in the main comparison and excluded in the separately recorded sensitivity analysis for both model generations.
