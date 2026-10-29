# Shared multi-output results

Both fits completed and their saved validation/test targets, losses and metrics passed independent checks. Public serving replay passed on TRAIN probes. These checks establish reproducibility; they do not independently qualify biological mechanisms or future-study accuracy.

Validation-selected condition: **wide-multi-s42**. Selection used the minimum equal-endpoint validation loss; test outcomes were excluded from selection.

## Exact quantitative measurements

Bounds contribute one-sided training/evaluation loss, but are excluded from Spearman, Pearson and RMSE here. Affinity/EC50/proxy RMSE is log10(nM); half-life is log10(minutes), koff is log10(1/s), and td50 is degrees Celsius. N/A denotes unavailable or undefined metrics. Recent and older study cohorts remain separate.

| split | era | endpoint | mhc_class | n | n_exact | wide-multi-s42 · spearman | wide-multi-s42 · pearson | wide-multi-s42 · rmse | wide-multi-s43 · spearman | wide-multi-s43 · pearson | wide-multi-s43 · rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 2026 | kd_via_ic50 | I | 592 | 592 | 0.4648 | 0.5191 | 1.3274 | 0.7250 | 0.7363 | 1.1612 |
| test | older_or_mixed | ec50 | I | 70 | 70 | 0.2998 | 0.3830 | 1.2979 | -0.0346 | 0.1401 | 1.7978 |
| test | older_or_mixed | ec50 | II | 107 | 107 | 0.2027 | 0.1822 | 1.0501 | 0.3109 | 0.3002 | 0.9850 |
| test | older_or_mixed | half_life | I | 62 | 62 | 0.2359 | 0.3171 | 1.4647 | 0.5028 | 0.5263 | 1.1045 |
| test | older_or_mixed | half_life | II | 148 | 148 | -0.0751 | -0.0468 | 0.9033 | -0.1715 | -0.1942 | 0.8770 |
| test | older_or_mixed | ic50 | I | 100 | 100 | 0.2392 | 0.2374 | 1.0231 | 0.2629 | 0.2425 | 1.0778 |
| test | older_or_mixed | ic50 | II | 379 | 267 | 0.5199 | 0.5265 | 1.0122 | 0.6034 | 0.6038 | 0.9658 |
| test | older_or_mixed | kd | I | 449 | 386 | 0.4465 | 0.4547 | 1.4574 | 0.5436 | 0.5687 | 1.1710 |
| test | older_or_mixed | kd | II | 30 | 21 | 0.3987 | 0.4023 | 0.8634 | 0.5909 | 0.5409 | 1.0238 |
| test | older_or_mixed | kd_via_ec50 | I | 66 | 66 | 0.4263 | 0.3698 | 1.1976 | 0.4544 | 0.4464 | 0.8779 |
| test | older_or_mixed | kd_via_ec50 | II | 339 | 149 | 0.5462 | 0.5506 | 0.5688 | 0.6083 | 0.5999 | 0.5381 |
| test | older_or_mixed | kd_via_ic50 | I | 12 | 12 | 0.0876 | 0.1197 | 1.0527 | 0.1716 | 0.2531 | 1.0831 |
| test | older_or_mixed | kd_via_ic50 | II | 55 | 55 | 0.1777 | 0.1486 | 0.9720 | 0.2390 | 0.2196 | 1.0622 |
| test | older_or_mixed | td50 | I | 140 | 140 | 0.4776 | 0.5067 | 5.8446 | 0.3198 | 0.3135 | 7.0567 |
| validation | older_or_mixed | ec50 | I | 69 | 69 | 0.5977 | 0.5496 | 1.3914 | 0.2210 | 0.2794 | 1.9701 |
| validation | older_or_mixed | half_life | I | 71 | 71 | 0.3928 | 0.5202 | 1.4362 | 0.4176 | 0.5473 | 1.2231 |
| validation | older_or_mixed | ic50 | I | 141 | 141 | 0.4995 | 0.5271 | 1.3178 | 0.5344 | 0.5438 | 1.2182 |
| validation | older_or_mixed | ic50 | II | 1,129 | 1,129 | 0.4248 | 0.4408 | 0.8463 | 0.4580 | 0.4920 | 0.8264 |
| validation | older_or_mixed | kd | I | 532 | 151 | 0.4821 | 0.4937 | 1.5521 | 0.5626 | 0.6024 | 1.2098 |
| validation | older_or_mixed | kd_via_ec50 | I | 106 | 106 | 0.3869 | 0.3993 | 1.1354 | 0.5577 | 0.5432 | 0.7082 |
| validation | older_or_mixed | kd_via_ec50 | II | 422 | 352 | 0.6270 | 0.6360 | 0.5019 | 0.6247 | 0.6182 | 0.5905 |
| validation | older_or_mixed | kd_via_ic50 | I | 96 | 83 | 0.5316 | 0.5408 | 0.9790 | 0.5694 | 0.6347 | 0.8846 |
| validation | older_or_mixed | kd_via_ic50 | II | 246 | 246 | 0.1511 | 0.1110 | 1.2099 | 0.1184 | 0.0491 | 1.2251 |
| validation | older_or_mixed | td50 | I | 145 | 145 | 0.4765 | 0.5055 | 6.6408 | 0.4986 | 0.5139 | 6.1852 |

## Ligands versus generated unobserved comparators

Same-protein and reference-proteome comparisons are distinct populations. Comparators are length-matched, sampled with replacement, and excluded from the known-positive pool. They are not measured non-binders. AUPRC depends on the five-comparators-per-parent evaluation contract.

| split | era | mhc_class | population | n | positives | negatives | wide-multi-s42 · auroc | wide-multi-s42 · auprc | wide-multi-s43 · auroc | wide-multi-s43 · auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 2026 | I | synthetic_reference_proteome | 12 | 2 | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | 2026 | I | synthetic_same_protein | 31,002 | 5,167 | 25,835 | 0.9625 | 0.9043 | 0.9687 | 0.9236 |
| test | 2026 | II | synthetic_reference_proteome | 546 | 91 | 455 | 0.8203 | 0.4890 | 0.8448 | 0.6752 |
| test | 2026 | II | synthetic_same_protein | 39,108 | 6,518 | 32,590 | 0.8061 | 0.5412 | 0.8188 | 0.6312 |
| test | older_or_mixed | I | synthetic_same_protein | 432 | 72 | 360 | 0.9748 | 0.8649 | 0.9873 | 0.9335 |
| validation | older_or_mixed | I | synthetic_reference_proteome | 180 | 30 | 150 | 0.9593 | 0.9152 | 0.9656 | 0.9081 |
| validation | older_or_mixed | I | synthetic_same_protein | 10,116 | 1,686 | 8,430 | 0.9810 | 0.8982 | 0.9911 | 0.9565 |
| validation | older_or_mixed | II | synthetic_reference_proteome | 12 | 2 | 10 | 0.9500 | 0.8333 | 0.9500 | 0.8333 |
| validation | older_or_mixed | II | synthetic_same_protein | 4,152 | 692 | 3,460 | 0.7739 | 0.5049 | 0.8434 | 0.5887 |

## Source-reported ligand outcomes

Each study and assay method is shown separately. Single-class outcome sets cannot define an AUROC; generated comparators are absent from this table.

| split | era | mhc_class | study | assay_method | n | positives | negatives | wide-multi-s42 · auroc | wide-multi-s42 · auprc | wide-multi-s43 · auroc | wide-multi-s43 · auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 2026 | I | pmid:42069060 | cellular MHC/mass spectrometry | 5,169 | 5,169 | 0 | N/A | 1.0000 | N/A | 1.0000 |
| test | 2026 | II | pmid:41534530 | cellular MHC/mass spectrometry | 6,611 | 6,611 | 0 | N/A | 1.0000 | N/A | 1.0000 |
| test | older_or_mixed | I | pmid:29308320 | cellular MHC/mass spectrometry | 136 | 72 | 64 | 0.1012 | 0.4159 | 0.2119 | 0.4741 |
| validation | older_or_mixed | I | pmid:35474673 | mass spectrometry | 53 | 31 | 22 | 0.6804 | 0.7941 | 0.7991 | 0.8889 |
| validation | older_or_mixed | I | pmid:35561310 | cellular MHC/mass spectrometry | 1,643 | 1,643 | 0 | N/A | 1.0000 | N/A | 1.0000 |
| validation | older_or_mixed | I | pmid:40138458 | cellular MHC/mass spectrometry | 52 | 42 | 10 | 0.4381 | 0.7894 | 0.4381 | 0.7894 |
| validation | older_or_mixed | II | pmid:35561310 | cellular MHC/mass spectrometry | 694 | 694 | 0 | N/A | 1.0000 | N/A | 1.0000 |

## Qualitative binding

These are the source assay outcomes; their labels do not imply a universal 500 nM cutoff.

| split | era | endpoint | mhc_class | n | positives | negatives | wide-multi-s42 · auroc | wide-multi-s42 · auprc | wide-multi-s43 · auroc | wide-multi-s43 · auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | older_or_mixed | qualitative_binding | I | 290 | 36 | 254 | 0.8727 | 0.5385 | 0.9033 | 0.6927 |
| test | older_or_mixed | qualitative_binding | II | 151 | 86 | 65 | 0.3102 | 0.4483 | 0.6891 | 0.7889 |
| validation | older_or_mixed | qualitative_binding | I | 689 | 29 | 660 | 0.9931 | 0.9318 | 0.9845 | 0.9411 |
| validation | older_or_mixed | qualitative_binding | II | 446 | 53 | 393 | 0.8005 | 0.3790 | 0.8381 | 0.4592 |

## Quantitative affinity at 500 nM

Only exact concentration measurements with an eligible endpoint enter this thresholded comparison. Proxy assays keep their original output names.

| split | era | endpoint | mhc_class | n | positives | negatives | wide-multi-s42 · auroc | wide-multi-s42 · auprc | wide-multi-s42 · balanced_accuracy | wide-multi-s43 · auroc | wide-multi-s43 · auprc | wide-multi-s43 · balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 2026 | kd_via_ic50 | I | 592 | 132 | 460 | 0.7643 | 0.5802 | 0.6612 | 0.9032 | 0.6762 | 0.8307 |
| test | older_or_mixed | ec50 | I | 70 | 15 | 55 | 0.7079 | 0.4531 | 0.6788 | 0.5406 | 0.3825 | 0.4545 |
| test | older_or_mixed | ec50 | II | 107 | 95 | 12 | 0.6351 | 0.9394 | 0.5566 | 0.6447 | 0.9281 | 0.5671 |
| test | older_or_mixed | ic50 | I | 100 | 93 | 7 | 0.7435 | 0.9652 | 0.6605 | 0.7081 | 0.9718 | 0.6605 |
| test | older_or_mixed | ic50 | II | 267 | 133 | 134 | 0.7523 | 0.7437 | 0.6473 | 0.7905 | 0.8199 | 0.7036 |
| test | older_or_mixed | kd | I | 386 | 307 | 79 | 0.6702 | 0.8799 | 0.6293 | 0.7213 | 0.9110 | 0.5920 |
| test | older_or_mixed | kd | II | 21 | 8 | 13 | 0.7115 | 0.6507 | 0.5865 | 0.8558 | 0.8337 | 0.7067 |
| test | older_or_mixed | kd_via_ec50 | I | 66 | 64 | 2 | 0.6484 | 0.9869 | 0.4609 | 0.8281 | 0.9942 | 0.4922 |
| test | older_or_mixed | kd_via_ec50 | II | 149 | 50 | 99 | 0.8303 | 0.6696 | 0.7185 | 0.8368 | 0.7197 | 0.6446 |
| test | older_or_mixed | kd_via_ic50 | I | 12 | 0 | 12 | N/A | N/A | N/A | N/A | N/A | N/A |
| test | older_or_mixed | kd_via_ic50 | II | 55 | 0 | 55 | N/A | N/A | N/A | N/A | N/A | N/A |
| validation | older_or_mixed | ec50 | I | 69 | 8 | 61 | 0.7766 | 0.2565 | 0.6670 | 0.6168 | 0.2410 | 0.5359 |
| validation | older_or_mixed | ic50 | I | 141 | 141 | 0 | N/A | 1.0000 | N/A | N/A | 1.0000 | N/A |
| validation | older_or_mixed | ic50 | II | 1,129 | 446 | 683 | 0.7070 | 0.6086 | 0.6268 | 0.7357 | 0.6587 | 0.6776 |
| validation | older_or_mixed | kd | I | 151 | 114 | 37 | 0.7698 | 0.9061 | 0.6854 | 0.7954 | 0.9050 | 0.7366 |
| validation | older_or_mixed | kd_via_ec50 | I | 106 | 106 | 0 | N/A | 1.0000 | N/A | N/A | 1.0000 | N/A |
| validation | older_or_mixed | kd_via_ec50 | II | 352 | 150 | 202 | 0.8440 | 0.8182 | 0.7178 | 0.8370 | 0.7823 | 0.6493 |
| validation | older_or_mixed | kd_via_ic50 | I | 83 | 15 | 68 | 0.8814 | 0.5087 | 0.8162 | 0.9284 | 0.7890 | 0.8343 |
| validation | older_or_mixed | kd_via_ic50 | II | 246 | 0 | 246 | N/A | N/A | N/A | N/A | N/A | N/A |

## Output support and actual optimizer exposures

Original measurements and generated training comparators are counted separately. Repeated exposures are not additional measurements. A trained output with no held-out observations has no measured generalization result.

| endpoint | mhc_class | train_original_n | train_generated_n | validation_n | test_n | wide-multi-s42 · optimizer_exposures | wide-multi-s43 · optimizer_exposures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ec50 | I | 115 | 0 | 69 | 70 | 11,253 | 11,253 |
| ec50 | II | 97 | 0 | 0 | 107 | 10,334 | 10,334 |
| elution_observation | I | 1,348,309 | 1,324,979 | 10,328 | 31,510 | 2,673,288 | 2,673,288 |
| elution_observation | II | 292,754 | 292,753 | 4,164 | 39,656 | 1,251,092 | 1,251,092 |
| half_life | I | 8,592 | 0 | 71 | 62 | 97,259 | 97,259 |
| half_life | II | 137 | 0 | 0 | 148 | 12,282 | 12,282 |
| ic50 | I | 18,531 | 0 | 141 | 100 | 545,561 | 545,561 |
| ic50 | II | 36,205 | 0 | 1,129 | 379 | 762,567 | 762,567 |
| kd | I | 16,349 | 0 | 532 | 449 | 134,162 | 134,162 |
| kd | II | 1,483 | 0 | 0 | 30 | 40,407 | 40,407 |
| kd_via_ec50 | I | 36,342 | 0 | 106 | 66 | 200,027 | 200,027 |
| kd_via_ec50 | II | 2,590 | 0 | 422 | 339 | 53,399 | 53,399 |
| kd_via_ic50 | I | 51,437 | 0 | 96 | 604 | 237,969 | 237,969 |
| kd_via_ic50 | II | 21,488 | 0 | 246 | 55 | 153,809 | 153,809 |
| koff | I | 6 | 0 | 0 | 0 | 2,571 | 2,571 |
| koff | II | 2 | 0 | 0 | 0 | 1,484 | 1,484 |
| qualitative_binding | I | 22,181 | 0 | 689 | 290 | 171,420 | 171,420 |
| qualitative_binding | II | 18,279 | 0 | 446 | 151 | 155,613 | 155,613 |
| td50 | I | 496 | 0 | 145 | 140 | 23,369 | 23,369 |
| td50 | II | 7 | 0 | 0 | 0 | 2,777 | 2,777 |

## Checkpoint coverage and runtime

Each completed fit traversed the full weighted training epoch. Validation may select an earlier checkpoint with lower unique-row exposure; both counts are explicit below. Wall time includes data admission and evaluation.

| condition | selected steps | completed steps | selected unique rows | completed unique rows | wall hours | optimizer hours | peak GPU GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wide-multi-s42 | 7,000 | 51,099 | 687,255 | 3,493,132 | 9.8931 | 6.8297 | 3.4660 |
| wide-multi-s43 | 32,000 | 51,099 | 2,418,755 | 3,493,132 | 13.0892 | 8.9389 | 3.5882 |

| condition | validation overall loss | test overall loss | serving probes |
| --- | --- | --- | --- |
| wide-multi-s42 | 0.3815 | 0.3919 | 78 |
| wide-multi-s43 | 0.3273 | 0.3694 | 78 |

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

- [wide-multi-s42](wide-multi-s42/collection.json)
- [wide-multi-s43](wide-multi-s43/collection.json)
