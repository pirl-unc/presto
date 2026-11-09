# Presto performance report — 2026-11-08_221000_4935bd7f

The current assay-aware unified development checkpoint decisively improves operational processing versus MHCflurry, is statistically similar for presentation, remains behind both NetMHC affinity tools, and has no PRIME-qualified T-cell result.

Status: **development**  
Generated at: `2026-11-08T22:10:00Z`  
Git: `4935bd7f6676a6a3fa003b8411062eb3c70ae010` (dirty: `false`)  
Model: `stage_b_plus_binary_only_tcell`

## Qualified claims

- On the exact same 48,530 development rows and biological inputs, Stage B operational-processing AUROC is 0.90578 versus 0.80567 for MHCflurry; the paired difference interval is strictly positive.
- Stage B presentation AUROC/AUPRC is 0.95205/0.95267 versus 0.94843/0.95428 for MHCflurry, so performance is similar rather than superior.
- The binary-only assay-aware T-cell head is preferred to the ordinal arm within the exposed development comparison, while neither is promoted.
- All exact training/development row bundles and prediction dumps in this report are recoverable from hash-pinned Modal-volume paths.

## Blocked claims

- Presto does not yet beat NetMHCpan on matched MHC-I exact-affinity development correlation.
- Presto does not yet beat NetMHCIIpan on matched MHC-II exact-affinity development correlation.
- Presentation performance is similar on three exposed studies, but the paired intervals do not support a superiority claim.
- No PRIME or independent T-cell TEST result exists, and the current binary T-cell head is threshold-degenerate at 0.5.

## Primary matched-cohort metrics

| Task / comparison output | Split | Metric | Method / source output | Value | 95% interval | Rows |
|---|---|---|---|---:|---:|---:|
| mhc_i_affinity / matched_exact_ic50 | development | spearman | NetMHCpan_4.2c / archive_unspecified_assay_ic50_BA | 0.758445 | not available | 169 |
| mhc_i_affinity / matched_exact_ic50 | development | spearman | Presto_Stage_B / assay_aware_ic50_panel | 0.668853 | not available | 169 |
| mhc_ii_affinity / matched_exact_ic50 | development | spearman | NetMHCIIpan_4.3e / archive_unspecified_assay_ic50_BA | 0.777863 | not available | 1,170 |
| mhc_ii_affinity / matched_exact_ic50 | development | spearman | Presto_Stage_B / assay_aware_ic50_panel | 0.499492 | not available | 1,170 |
| presentation / presentation | development | auprc | MHCflurry_2.3.16 / presentation_score | 0.954281 | not available | 48,530 |
| presentation / presentation | development | auprc | Presto_Stage_B / presentation | 0.952668 | not available | 48,530 |
| presentation / presentation | development | auprc_difference | Presto_minus_MHCflurry / presentation_minus_presentation_score | -0.00161304 | [-0.0160511, 0.0147989] | 48,530 |
| presentation / presentation | development | auroc | MHCflurry_2.3.16 / presentation_score | 0.948433 | not available | 48,530 |
| presentation / presentation | development | auroc | Presto_Stage_B / presentation | 0.952049 | not available | 48,530 |
| presentation / presentation | development | auroc_difference | Presto_minus_MHCflurry / presentation_minus_presentation_score | 0.00361576 | [-0.0106392, 0.0223006] | 48,530 |
| processing / operational_processing | development | auroc | MHCflurry_2.3.16 / processing_AP | 0.805672 | not available | 48,530 |
| processing / operational_processing | development | auroc | Presto_Stage_B / operational_processing | 0.905777 | not available | 48,530 |
| processing / operational_processing | development | auroc_difference | Presto_minus_MHCflurry / operational_processing_minus_processing_AP | 0.100104 | [0.070592, 0.14577] | 48,530 |
| tcell_immunogenicity / functional_assay_aware_binary | development | equal_track_equal_study_auroc | binary_only / functional_assay_aware_binary | 0.630998 | not available | 861 |
| tcell_immunogenicity / functional_assay_aware_binary | development | equal_track_equal_study_auroc | binary_ordinal / functional_assay_aware_binary | 0.491024 | not available | 861 |

## Promotion gates

- **fail — mhc_i_affinity_vs_netmhcpan**: Presto Stage B exact-IC50 Spearman 0.66885 is below NetMHCpan 4.2c at 0.75845 on 169 matched development rows.
- **fail — mhc_ii_affinity_vs_netmhciipan**: Presto Stage B exact-IC50 Spearman 0.49949 is below NetMHCIIpan 4.3e at 0.77786 on 1,170 matched development rows.
- **fail — presentation_vs_mhcflurry**: AUROC difference is positive but both AUROC and AUPRC paired 95% intervals cross zero.
- **pass — processing_vs_mhcflurry**: Paired development AUROC difference is 0.10010 with 95% interval [0.07059, 0.14577].
- **not_evaluated — tcell_vs_prime**: PRIME was not run: PRIME 2.1 requires licensed MixMHCpred 3.x, while only 2.0.2 is installed; no independent T-cell TEST exists.
- **not_evaluated — independent_all_task_test**: No untouched all-task TEST population has been scored by the frozen unified checkpoint.

## Limitations

- All headline metrics are development-only; no current cohort is relabeled as TEST.
- Presentation has only three independent studies and participated in checkpoint selection.
- NetMHC comparator versions are installed 4.2c/4.3e rather than the newest available patches.
- MHCflurry historical training lineage cannot be reconstructed completely.
- T-cell functional discrimination rests on few two-class studies per assay track and every row is positive at the fixed 0.5 threshold.

## Provenance

See `report.json`, `data_manifest.json`, `predictions/index.json`, `tables/metrics.csv`, `reproduce/`, and `checksums.sha256`. Artifact references are valid only when their recorded byte counts and SHA-256 hashes verify.
