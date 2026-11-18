# Working checkpoint decision

Use **expanded-state-sqrt-s42-n6000**, selected at update **6000** by the original equal-class/equal-study presentation BCE criterion. Both classes use the same end-to-end model, 64-dimensional APC state, centralized adapter and public predictor. The optional pathway projection is left off because its two-seed comparison gives no meaningful predictive advantage.

All 12 fits and their selected/fixed-terminal comparisons are complete and independently verified. This is an ordinary-presentation working model, not a claim of a win on every processing benchmark or independent TEST confirmation.

| cohort | class | rows | previous_AUROC | shared_AUROC | delta | paired_95CI | cluster | previous_pair_win | shared_pair_win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expanded | I | 780 | 0.8503 | 0.8759 | 0.0255 | [+0.0124, +0.0375] | protein | 0.8564 | 0.8744 |
| expanded | II | 1194 | 0.6872 | 0.8373 | 0.1501 | [+0.0768, +0.1997] | protein | 0.6918 | 0.8157 |
| apc | I | 1860 | 0.9239 | 0.9257 | 0.0018 | [-0.0026, +0.0064] | protein | 0.9118 | 0.9215 |
| apc | II | 372 | 0.7627 | 0.7742 | 0.0115 | [-0.0173, +0.0373] | protein | 0.7634 | 0.7581 |
| broad | I | 4830 | 0.9500 | 0.9532 | 0.0032 | [+0.0008, +0.0055] | condition | — | — |
| matched | I | 680 | 0.6342 | 0.6142 | -0.0201 | [-0.0323, -0.0080] | protein | 0.6735 | 0.6588 |
| matched | II | 316 | 0.5763 | 0.5692 | -0.0071 | [-0.0379, +0.0252] | protein | 0.5696 | 0.6076 |

Intervals use 1,000 paired resamples and condition on these fitted models and development cohorts. They exclude model-selection and seed uncertainty. Sampled backgrounds are unobserved source windows, not measured biological nonbinders. Pooled AUROC and within-pair ordering answer different questions; both remain reported.

## Measured affinity retention

| model | population | class | exact_n | spearman | rmse_log10_nm | delta_spearman | delta_rmse | operational_guard_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expanded-state-sqrt-s42-n6000 | affinity | I | 169 | 0.6705 | 1.0480 | 0.0007 | -0.0006 | True |
| expanded-state-sqrt-s42-n6000 | affinity | II | 1170 | 0.4980 | 0.8285 | 0.0050 | -0.0167 | True |
| expanded-state-sqrt-s42-n6000 | affinity_ancestry_screened | I | 108 | 0.6814 | 1.0426 | -0.0003 | -0.0009 | True |
| expanded-state-sqrt-s42-n6000 | affinity_ancestry_screened | II | 1113 | 0.4900 | 0.8360 | 0.0028 | -0.0162 | True |

The original and ancestry-screened IC50 cohorts both pass the registered operational screen (no Spearman drop above .02 or log10(nM) RMSE rise above .03). Intrinsic affinity takes no APC input; its parameters were jointly trained, not frozen.

## Attribution and remaining limits

Broader native data and repaired study allocation account for most of the class-II improvement. Compare shared-state against no-state within each seed and at the same update endpoint to assess the APC increment. Older APC class-I gains replicate at 6,000 updates: +.00512 and +.00524 AUROC in seeds 42/43, with positive protein- and condition-cluster intervals. New class-II gains are +.01373 and +.01247. The primary seed's protein-cluster 95% interval is [.00027, .02760], but its condition-cluster interval crosses zero; both replication intervals cross zero. This is suggestive incremental class-II evidence, not a robustly resolved effect. Unknown/shuffled controls show predictive state associations, while little within-study signature variation limits intervention identification. Gene/drug effects and absolute KO severity remain unvalidated.

The affinity-matched class-I regression is a real limitation of this default. Keep the previous checkpoint as an explicit matched-benchmark reference. The new model is chosen for ordinary presentation across both classes and retained measured affinity.

## Reproduce and use

Checkpoint SHA256:`1b6011396a563f52ea1d6da4bb1d4e69b8dbf0103148ba829225581c4d312f68`. Local:`results/fits/expanded-state-sqrt-s42-n6000/raw/selected.pt`. Remote:`presto-checkpoints:/2026-11-17_1155_codex_shared-presentation-expansion/fits/expanded-state-sqrt-s42-n6000/raw/selected.pt`.

Run `python code/predict_example.py` from this experiment directory using the shared Python environment; the helper validates the chosen checkpoint hash. See [USAGE.md](USAGE.md), [ARCHITECTURE.md](ARCHITECTURE.md), [data support](results/analysis/apc-training-support.md) and [complete comparison](results/analysis/report.md). TEST remains sealed.
