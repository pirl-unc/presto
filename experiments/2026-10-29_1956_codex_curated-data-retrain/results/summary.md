# Shared multi-output results

Both fits completed and their saved validation/test targets, losses and metrics passed independent checks. Public serving replay passed on TRAIN probes. These checks establish reproducibility; they do not independently qualify biological mechanisms or future-study accuracy.

Validation-selected condition: **curated-multi-s43**. Selection used the minimum equal-endpoint validation loss; test outcomes were excluded from selection.

## Exact quantitative measurements

Bounds contribute one-sided training/evaluation loss, but are excluded from Spearman, Pearson and RMSE here. Affinity/EC50/proxy RMSE is log10(nM); half-life is log10(minutes), koff is log10(1/s), and td50 is degrees Celsius. N/A denotes unavailable or undefined metrics. Recent and older study cohorts remain separate.

| split | era | endpoint | mhc_class | n | n_exact | curated-multi-s42 · spearman | curated-multi-s42 · pearson | curated-multi-s42 · rmse | curated-multi-s43 · spearman | curated-multi-s43 · pearson | curated-multi-s43 · rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 2026 | kd_via_ic50 | I | 592 | 592 | 0.6328 | 0.6937 | 1.3605 | 0.6996 | 0.7244 | 1.0878 |
| test | older_or_mixed | ec50 | I | 70 | 70 | -0.0418 | 0.0282 | 1.8131 | 0.1661 | 0.1471 | 1.8114 |
| test | older_or_mixed | ec50 | II | 107 | 107 | 0.4789 | 0.4598 | 0.8659 | 0.2408 | 0.2377 | 1.0277 |
| test | older_or_mixed | half_life | I | 62 | 62 | 0.4530 | 0.4985 | 1.0831 | 0.4053 | 0.4190 | 1.2031 |
| test | older_or_mixed | half_life | II | 148 | 148 | -0.1849 | -0.1882 | 1.0090 | -0.0953 | -0.0886 | 0.6887 |
| test | older_or_mixed | ic50 | I | 100 | 100 | 0.3758 | 0.3466 | 1.0880 | 0.3023 | 0.2866 | 1.0443 |
| test | older_or_mixed | ic50 | II | 379 | 267 | 0.5837 | 0.5908 | 0.9823 | 0.5662 | 0.5872 | 1.0001 |
| test | older_or_mixed | kd | I | 449 | 386 | 0.5723 | 0.6136 | 1.1523 | 0.5877 | 0.6267 | 1.1465 |
| test | older_or_mixed | kd | II | 30 | 21 | 0.6052 | 0.5917 | 0.8556 | 0.5961 | 0.5532 | 0.7614 |
| test | older_or_mixed | kd_via_ec50 | I | 66 | 66 | 0.2463 | 0.2238 | 1.0617 | 0.2114 | 0.2453 | 1.0641 |
| test | older_or_mixed | kd_via_ec50 | II | 339 | 149 | 0.4679 | 0.4810 | 0.6043 | 0.5595 | 0.5542 | 0.5662 |
| test | older_or_mixed | kd_via_ic50 | I | 12 | 12 | -0.0245 | 0.1037 | 0.9597 | 0.0946 | 0.0752 | 1.0135 |
| test | older_or_mixed | kd_via_ic50 | II | 55 | 55 | 0.2572 | 0.2807 | 1.0670 | 0.2434 | 0.2380 | 1.0101 |
| test | older_or_mixed | td50 | I | 140 | 140 | 0.5071 | 0.5153 | 5.8592 | 0.4787 | 0.4648 | 6.3162 |
| validation | older_or_mixed | ec50 | I | 69 | 69 | 0.3510 | 0.4080 | 1.7246 | 0.2240 | 0.2367 | 1.7758 |
| validation | older_or_mixed | half_life | I | 71 | 71 | 0.4948 | 0.5823 | 1.3527 | 0.5047 | 0.5406 | 1.2284 |
| validation | older_or_mixed | ic50 | I | 141 | 141 | 0.5707 | 0.5797 | 1.1341 | 0.5491 | 0.5718 | 1.2805 |
| validation | older_or_mixed | ic50 | II | 1,129 | 1,129 | 0.5306 | 0.5753 | 0.8551 | 0.4928 | 0.5441 | 0.8342 |
| validation | older_or_mixed | kd | I | 532 | 151 | 0.6573 | 0.6776 | 1.2780 | 0.5829 | 0.6087 | 1.4096 |
| validation | older_or_mixed | kd_via_ec50 | I | 106 | 106 | 0.6411 | 0.6058 | 0.6854 | 0.5924 | 0.5451 | 0.7905 |
| validation | older_or_mixed | kd_via_ec50 | II | 422 | 352 | 0.6056 | 0.6168 | 0.6275 | 0.6008 | 0.6154 | 0.5830 |
| validation | older_or_mixed | kd_via_ic50 | I | 96 | 83 | 0.5120 | 0.4722 | 1.0338 | 0.6381 | 0.5955 | 0.8767 |
| validation | older_or_mixed | kd_via_ic50 | II | 246 | 246 | 0.0900 | 0.0758 | 1.2266 | 0.1372 | 0.1181 | 1.1552 |
| validation | older_or_mixed | td50 | I | 145 | 145 | 0.4455 | 0.5088 | 7.5206 | 0.4360 | 0.3875 | 6.7385 |

## Ligands versus generated unobserved comparators

Same-protein and reference-proteome comparisons are distinct populations. Comparators are length-matched, sampled with replacement, and excluded from the known-positive pool. They are not measured non-binders. AUPRC depends on the five-comparators-per-parent evaluation contract.

| split | era | mhc_class | population | n | positives | negatives | curated-multi-s42 · auroc | curated-multi-s42 · auprc | curated-multi-s43 · auroc | curated-multi-s43 · auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 2026 | I | synthetic_reference_proteome | 12 | 2 | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | 2026 | I | synthetic_same_protein | 31,002 | 5,167 | 25,835 | 0.9726 | 0.9319 | 0.9712 | 0.9293 |
| test | 2026 | II | synthetic_reference_proteome | 546 | 91 | 455 | 0.8471 | 0.6663 | 0.8424 | 0.6518 |
| test | 2026 | II | synthetic_same_protein | 39,108 | 6,518 | 32,590 | 0.8282 | 0.6677 | 0.8337 | 0.6738 |
| test | older_or_mixed | I | synthetic_same_protein | 432 | 72 | 360 | 0.9959 | 0.9740 | 0.9966 | 0.9805 |
| validation | older_or_mixed | I | synthetic_reference_proteome | 180 | 30 | 150 | 0.9842 | 0.9330 | 0.9791 | 0.9296 |
| validation | older_or_mixed | I | synthetic_same_protein | 10,116 | 1,686 | 8,430 | 0.9945 | 0.9716 | 0.9929 | 0.9658 |
| validation | older_or_mixed | II | synthetic_reference_proteome | 12 | 2 | 10 | 0.9000 | 0.7500 | 0.9500 | 0.8333 |
| validation | older_or_mixed | II | synthetic_same_protein | 4,152 | 692 | 3,460 | 0.8675 | 0.6543 | 0.8638 | 0.6575 |

## Source-reported ligand outcomes

Each study and assay method is shown separately. Single-class outcome sets cannot define an AUROC; generated comparators are absent from this table.

| split | era | mhc_class | study | assay_method | n | positives | negatives | curated-multi-s42 · auroc | curated-multi-s42 · auprc | curated-multi-s43 · auroc | curated-multi-s43 · auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 2026 | I | pmid:42069060 | cellular MHC/mass spectrometry | 5,169 | 5,169 | 0 | N/A | 1.0000 | N/A | 1.0000 |
| test | 2026 | II | pmid:41534530 | cellular MHC/mass spectrometry | 6,611 | 6,611 | 0 | N/A | 1.0000 | N/A | 1.0000 |
| test | older_or_mixed | I | pmid:29308320 | cellular MHC/mass spectrometry | 136 | 72 | 64 | 0.2428 | 0.4741 | 0.1012 | 0.4159 |
| validation | older_or_mixed | I | pmid:35474673 | mass spectrometry | 53 | 31 | 22 | 0.7639 | 0.8525 | 0.7610 | 0.8564 |
| validation | older_or_mixed | I | pmid:35561310 | cellular MHC/mass spectrometry | 1,643 | 1,643 | 0 | N/A | 1.0000 | N/A | 1.0000 |
| validation | older_or_mixed | I | pmid:40138458 | cellular MHC/mass spectrometry | 52 | 42 | 10 | 0.4381 | 0.7894 | 0.4381 | 0.7894 |
| validation | older_or_mixed | II | pmid:35561310 | cellular MHC/mass spectrometry | 694 | 694 | 0 | N/A | 1.0000 | N/A | 1.0000 |

## Qualitative binding

These are the source assay outcomes; their labels do not imply a universal 500 nM cutoff.

| split | era | endpoint | mhc_class | n | positives | negatives | curated-multi-s42 · auroc | curated-multi-s42 · auprc | curated-multi-s43 · auroc | curated-multi-s43 · auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | older_or_mixed | qualitative_binding | I | 290 | 36 | 254 | 0.9085 | 0.7001 | 0.9003 | 0.5821 |
| test | older_or_mixed | qualitative_binding | II | 151 | 86 | 65 | 0.6395 | 0.7572 | 0.7877 | 0.8644 |
| validation | older_or_mixed | qualitative_binding | I | 689 | 29 | 660 | 0.9895 | 0.9566 | 0.9985 | 0.9773 |
| validation | older_or_mixed | qualitative_binding | II | 446 | 53 | 393 | 0.8759 | 0.5148 | 0.8806 | 0.5575 |

## Quantitative affinity at 500 nM

Only exact concentration measurements with an eligible endpoint enter this thresholded comparison. Proxy assays keep their original output names.

| split | era | endpoint | mhc_class | n | positives | negatives | curated-multi-s42 · auroc | curated-multi-s42 · auprc | curated-multi-s42 · balanced_accuracy | curated-multi-s43 · auroc | curated-multi-s43 · auprc | curated-multi-s43 · balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 2026 | kd_via_ic50 | I | 592 | 132 | 460 | 0.8802 | 0.6683 | 0.7833 | 0.8676 | 0.6251 | 0.7846 |
| test | older_or_mixed | ec50 | I | 70 | 15 | 55 | 0.5709 | 0.2632 | 0.5182 | 0.6218 | 0.2967 | 0.5576 |
| test | older_or_mixed | ec50 | II | 107 | 95 | 12 | 0.7439 | 0.9528 | 0.6193 | 0.6930 | 0.9478 | 0.5307 |
| test | older_or_mixed | ic50 | I | 100 | 93 | 7 | 0.8264 | 0.9845 | 0.7765 | 0.7650 | 0.9794 | 0.5783 |
| test | older_or_mixed | ic50 | II | 267 | 133 | 134 | 0.7725 | 0.7821 | 0.7148 | 0.7664 | 0.8067 | 0.7033 |
| test | older_or_mixed | kd | I | 386 | 307 | 79 | 0.7422 | 0.9176 | 0.6457 | 0.7591 | 0.9243 | 0.6694 |
| test | older_or_mixed | kd | II | 21 | 8 | 13 | 0.8365 | 0.8530 | 0.6827 | 0.8269 | 0.7993 | 0.6971 |
| test | older_or_mixed | kd_via_ec50 | I | 66 | 64 | 2 | 0.6016 | 0.9840 | 0.4766 | 0.8438 | 0.9948 | 0.4766 |
| test | older_or_mixed | kd_via_ec50 | II | 149 | 50 | 99 | 0.7699 | 0.6105 | 0.6791 | 0.7879 | 0.6417 | 0.6293 |
| test | older_or_mixed | kd_via_ic50 | I | 12 | 0 | 12 | N/A | N/A | N/A | N/A | N/A | N/A |
| test | older_or_mixed | kd_via_ic50 | II | 55 | 0 | 55 | N/A | N/A | N/A | N/A | N/A | N/A |
| validation | older_or_mixed | ec50 | I | 69 | 8 | 61 | 0.5041 | 0.2310 | 0.5932 | 0.6189 | 0.3109 | 0.5277 |
| validation | older_or_mixed | ic50 | I | 141 | 141 | 0 | N/A | 1.0000 | N/A | N/A | 1.0000 | N/A |
| validation | older_or_mixed | ic50 | II | 1,129 | 446 | 683 | 0.7689 | 0.6976 | 0.7074 | 0.7522 | 0.6753 | 0.6715 |
| validation | older_or_mixed | kd | I | 151 | 114 | 37 | 0.8381 | 0.9428 | 0.7245 | 0.7909 | 0.9014 | 0.7559 |
| validation | older_or_mixed | kd_via_ec50 | I | 106 | 106 | 0 | N/A | 1.0000 | N/A | N/A | 1.0000 | N/A |
| validation | older_or_mixed | kd_via_ec50 | II | 352 | 150 | 202 | 0.8235 | 0.7892 | 0.6936 | 0.8339 | 0.8000 | 0.6527 |
| validation | older_or_mixed | kd_via_ic50 | I | 83 | 15 | 68 | 0.7931 | 0.3518 | 0.7642 | 0.8775 | 0.4913 | 0.8230 |
| validation | older_or_mixed | kd_via_ic50 | II | 246 | 0 | 246 | N/A | N/A | N/A | N/A | N/A | N/A |

## Output support and actual optimizer exposures

Original measurements and generated training comparators are counted separately. Repeated exposures are not additional measurements. A trained output with no held-out observations has no measured generalization result.

| endpoint | mhc_class | train_original_n | train_generated_n | validation_n | test_n | curated-multi-s42 · optimizer_exposures | curated-multi-s43 · optimizer_exposures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ec50 | I | 115 | 0 | 69 | 70 | 11,353 | 11,353 |
| ec50 | II | 97 | 0 | 0 | 107 | 10,426 | 10,426 |
| elution_observation | I | 1,348,309 | 1,324,944 | 10,328 | 31,510 | 2,673,253 | 2,673,253 |
| elution_observation | II | 306,303 | 306,302 | 4,164 | 39,656 | 1,279,707 | 1,279,707 |
| half_life | I | 8,592 | 0 | 71 | 62 | 98,124 | 98,124 |
| half_life | II | 137 | 0 | 0 | 148 | 12,391 | 12,391 |
| ic50 | I | 18,526 | 0 | 141 | 100 | 549,575 | 549,575 |
| ic50 | II | 36,186 | 0 | 1,129 | 379 | 768,079 | 768,079 |
| kd | I | 16,330 | 0 | 532 | 449 | 135,276 | 135,276 |
| kd | II | 1,483 | 0 | 0 | 30 | 40,766 | 40,766 |
| kd_via_ec50 | I | 36,079 | 0 | 106 | 66 | 201,073 | 201,073 |
| kd_via_ec50 | II | 2,590 | 0 | 422 | 339 | 53,874 | 53,874 |
| kd_via_ic50 | I | 51,152 | 0 | 96 | 604 | 239,418 | 239,418 |
| kd_via_ic50 | II | 21,461 | 0 | 246 | 55 | 155,078 | 155,078 |
| koff | I | 6 | 0 | 0 | 0 | 2,593 | 2,593 |
| koff | II | 2 | 0 | 0 | 0 | 1,498 | 1,498 |
| qualitative_binding | I | 22,181 | 0 | 689 | 290 | 172,668 | 172,668 |
| qualitative_binding | II | 18,279 | 0 | 446 | 151 | 156,746 | 156,746 |
| td50 | I | 496 | 0 | 145 | 140 | 23,576 | 23,576 |
| td50 | II | 7 | 0 | 0 | 0 | 2,801 | 2,801 |

## Checkpoint coverage and runtime

Each completed fit traversed the full weighted training epoch. Validation may select an earlier checkpoint with lower unique-row exposure; both counts are explicit below. Wall time includes data admission and evaluation.

| condition | selected steps | completed steps | selected unique rows | completed unique rows | wall hours | optimizer hours | peak GPU GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| curated-multi-s42 | 51,471 | 51,471 | 3,519,577 | 3,519,577 | 11.6384 | 8.2053 | 3.5116 |
| curated-multi-s43 | 51,471 | 51,471 | 3,519,577 | 3,519,577 | 15.4157 | 9.0626 | 3.5503 |

| condition | validation overall loss | test overall loss | serving probes |
| --- | --- | --- | --- |
| curated-multi-s42 | 0.2962 | 0.3654 | 78 |
| curated-multi-s43 | 0.3080 | 0.3412 | 78 |

## Evidence and limitations

These whole-study holdouts differ from earlier inspected development splits; scores cannot establish improvement over the earlier IC50 models without a matched comparison. Some cells contain only one study. Peptide-bootstrap intervals are conditional on the fitted model and sampled studies; they do not measure generalization to new studies or training seeds.

- [All assay/study and threshold classification metrics](stratified-metrics.csv)
- [TRAIN-mean and constant-score comparisons](baseline-comparison.csv)
- [Study concentration](study-concentration.csv)
- [Cluster uncertainty and unavailable-interval reasons](cluster-uncertainty.json)
- [Length, candidate and allele failure strata](failure-strata.csv)
- [Register diagnostics](register-diagnostics.csv)
- [Parameter coordinate coverage](parameter-coordinate-coverage.csv)
- [Input feature coverage](input-feature-coverage.csv)
- [Comparison plot](performance-comparison.png)
- [Independent prediction verification](prediction-verification.json)
- [Experiment contract, exclusions and reproduction](../README.md)

Register weights have no measured core-localization truth here. Coordinate updates do not establish identified biological effects. Ligand scores combine factors of presentation and measurement; they do not separately qualify antigen processing or MS detectability.

Per-example prediction dumps and checkpoint hashes are located in each condition’s collection record:

- [curated-multi-s42](curated-multi-s42/collection.json)
- [curated-multi-s43](curated-multi-s43/collection.json)
