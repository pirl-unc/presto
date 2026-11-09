# Presto performance report — 2026-11-09_150000_1cca535e

The retained assay-aware reference checkpoint preserves decisive processing performance, remains similar for presentation, and rejects a comparator-derived BA expansion that improved MHC-I but failed to improve MHC-II materially.

Status: **development**  
Generated at: `2026-11-09T15:00:00Z`  
Git: `1cca535e99a20e58c0ce00bc27edcbefbba3ad7d` (dirty: `false`)  
Model: `unified_reference_2026_11_08`

## Qualified claims

- Both matched H100 fits completed 5,000 updates with passed coverage and zero TEST/locked access.
- Operational-processing development AUROC is 0.90576 for the retained reference versus 0.80567 for MHCflurry; the near-identical parity arm has a paired +0.10114 interval wholly above zero.
- Presentation AUROC/AUPRC is 0.95203/0.95265 for the retained reference versus 0.94843/0.95428 for MHCflurry, which supports similarity rather than superiority.
- The parity arm improved MHC-I exact-IC50 Spearman but failed the preregistered MHC-II parent gate, so the reference checkpoint is retained.
- The evaluator preserves assay-aware physical endpoints, and independent replay matches 110 estimates to maximum absolute delta 3.33e-16.
- PRIME 2.1 through mhctools/MixMHCpred 3.0 is operational, but accuracy remains unevaluated.

## Blocked claims

- The selected reference checkpoint does not beat NetMHCpan on matched MHC-I development affinity.
- The selected reference checkpoint remains substantially behind NetMHCIIpan on matched MHC-II development affinity.
- Presentation is similar on three exposed studies but does not support a superiority claim.
- No newly disjoint T-cell/PRIME accuracy comparison or independent all-task TEST exists.

## Primary matched-cohort metrics

| Task / comparison output | Split | Metric | Method / source output | Value | 95% interval | Rows |
|---|---|---|---|---:|---:|---:|
| mhc_i_affinity / exact_ic50_log10_nm | development | spearman | NetMHCpan_4.2c / archive_ba_ic50 | 0.68338 | [0.634921, 0.731839] | 169 |
| mhc_i_affinity / exact_ic50_log10_nm | development | spearman | Presto_parity_experimental / assay_aware_ic50 | 0.621077 | [0.591907, 0.650246] | 169 |
| mhc_i_affinity / exact_ic50_log10_nm | development | spearman | Presto_reference / assay_aware_ic50 | 0.585005 | [0.559173, 0.610837] | 169 |
| mhc_i_affinity / exact_ic50_log10_nm | development | spearman_difference | Presto_parity_minus_NetMHCpan_4.2c / assay_aware_ic50_difference | -0.0623031 | [-0.139932, 0.0153257] | 169 |
| mhc_i_affinity / exact_ic50_log10_nm | development | spearman_difference | Presto_parity_minus_Presto_reference / assay_aware_ic50_difference | 0.0360713 | [0.0327338, 0.0394089] | 169 |
| mhc_ii_affinity / exact_ic50_log10_nm | development | spearman | NetMHCIIpan_4.3e / archive_ba_ic50 | 0.807017 | [0.772315, 0.841718] | 1,170 |
| mhc_ii_affinity / exact_ic50_log10_nm | development | spearman | Presto_parity_experimental / assay_aware_ic50 | 0.53999 | [0.497505, 0.582476] | 1,170 |
| mhc_ii_affinity / exact_ic50_log10_nm | development | spearman | Presto_reference / assay_aware_ic50 | 0.576466 | [0.487896, 0.665036] | 1,170 |
| mhc_ii_affinity / exact_ic50_log10_nm | development | spearman_difference | Presto_parity_minus_NetMHCIIpan_4.3e / assay_aware_ic50_difference | -0.267026 | [-0.274811, -0.259242] | 1,170 |
| mhc_ii_affinity / exact_ic50_log10_nm | development | spearman_difference | Presto_parity_minus_Presto_reference / assay_aware_ic50_difference | -0.0364762 | [-0.0825606, 0.00960818] | 1,170 |
| presentation / presentation | development | auroc | MHCflurry_2.3.16_presentation / presentation_score | 0.948433 | [0.922121, 0.964942] | 48,530 |
| presentation / presentation | development | auroc | Presto_Stage_B / presentation | 0.952049 | [0.943476, 0.965051] | 48,530 |
| presentation / presentation | development | auroc | Presto_parity_experimental / presentation | 0.952255 | [0.94282, 0.96764] | 48,530 |
| presentation / presentation | development | auroc | Presto_reference / presentation | 0.952032 | [0.943427, 0.965042] | 48,530 |
| presentation / presentation | development | auroc_difference | Presto_parity_minus_MHCflurry_2.3.16_presentation / presentation_difference | 0.0038215 | [-0.0119323, 0.0206988] | 48,530 |
| presentation / presentation | development | auroc_difference | Presto_parity_minus_Presto_reference / presentation_difference | 0.000223045 | [-0.00132165, 0.00259781] | 48,530 |
| processing / operational_processing | development | auroc | MHCflurry_2.3.16_processing / processing_AP | 0.805672 | [0.761039, 0.838284] | 48,530 |
| processing / operational_processing | development | auroc | Presto_Stage_B / operational_processing | 0.905777 | [0.894103, 0.920879] | 48,530 |
| processing / operational_processing | development | auroc | Presto_parity_experimental / operational_processing | 0.906811 | [0.895989, 0.918206] | 48,530 |
| processing / operational_processing | development | auroc | Presto_reference / operational_processing | 0.905762 | [0.894095, 0.920904] | 48,530 |
| processing / operational_processing | development | auroc_difference | Presto_parity_minus_MHCflurry_2.3.16_processing / operational_processing_difference | 0.101139 | [0.0782947, 0.1452] | 48,530 |
| processing / operational_processing | development | auroc_difference | Presto_parity_minus_Presto_reference / operational_processing_difference | 0.00104882 | [-0.00269805, 0.00395068] | 48,530 |

## Promotion gates

- **fail — data_only_parent**: Parity MHC-II pooled exact-IC50 Spearman improved only +0.00636 versus required +0.05, and equal-study delta was -0.03648 with CI crossing zero.
- **fail — mhc_i_affinity_vs_netmhcpan**: Selected reference equal-study Spearman is 0.58501 versus NetMHCpan 4.2c at 0.68338; pooled values are 0.66971 versus 0.75845.
- **fail — mhc_ii_affinity_vs_netmhciipan**: Selected reference equal-study Spearman is 0.57647 versus NetMHCIIpan 4.3e at 0.80702; pooled values are 0.49965 versus 0.77786.
- **fail — presentation_vs_mhcflurry**: Selected reference AUROC/AUPRC is 0.95203/0.95265 versus MHCflurry 0.94843/0.95428; the parity paired intervals cross zero, so this remains similarity rather than superiority.
- **pass — processing_vs_mhcflurry**: Selected reference AUROC is 0.90576 versus MHCflurry 0.80567; parity's paired delta is +0.10114 with 95% CI [+0.07829,+0.14520], and parity-reference preservation interval crosses zero near zero.
- **pass — assay_aware_output_preservation**: Final evaluation preserves all four physical-affinity endpoint identities plus separate presentation and operational-processing outputs.
- **pass — prime_software_integration**: PRIME 2.1 ran through mhctools with the licensed MixMHCpred 3.0 backend in a two-peptide smoke test.
- **not_evaluated — tcell_vs_prime**: No newly disjoint T-cell accuracy cohort has been scored by the selected checkpoint or PRIME; the smoke is not an accuracy benchmark.
- **not_evaluated — independent_all_task_test**: No untouched all-task TEST population has been scored by a frozen unified checkpoint.

## Limitations

- All headline metrics are development-only; no current cohort is relabeled as TEST.
- Affinity equal-study estimates contain only two studies, and presentation/processing contain only three.
- The experimental BA rows are comparator training archives and cannot establish independent superiority over those tools.
- The selected reference checkpoint does not contain a promoted T-cell extension; historical T-cell results are context only.
- NetMHC comparator versions are 4.2c/4.3e rather than newer patches, and MHCflurry lineage is only partially reconstructable.

## Provenance

See `report.json`, `data_manifest.json`, `predictions/index.json`, `tables/metrics.csv`, `reproduce/`, and `checksums.sha256`. Artifact references are valid only when their recorded byte counts and SHA-256 hashes verify.
