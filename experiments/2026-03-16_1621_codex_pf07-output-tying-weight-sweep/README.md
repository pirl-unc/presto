# PF07 Output-Tying Weight Sweep

- Agent: `codex`
- Source script: `experiments/2026-03-16_1621_codex_pf07-output-tying-weight-sweep/code/launch.py`
- Created: `2026-03-16T16:23:56.334758`

## Dataset Contract

```json
{
  "alleles": [
    "HLA-A*02:01",
    "HLA-A*24:02"
  ],
  "assay_families_supervised": [
    "IC50",
    "KD",
    "KD(~IC50)",
    "KD(~EC50)",
    "EC50"
  ],
  "assay_selector_inputs_forbidden": true,
  "input_fields": [
    "nflank",
    "peptide",
    "cflank",
    "mhc_a",
    "mhc_b"
  ],
  "measurement_profile": "numeric_no_qualitative",
  "probe_peptides": [
    "SLLQHLIGL",
    "FLRYLLFGI",
    "NFLIKFLLI",
    "IMLEGETKL"
  ],
  "qualifier_filter": "all",
  "source": "data/merged_deduped.tsv",
  "split_policy": "peptide_group_80_10_10",
  "split_seed": 42,
  "train_seed": 43
}
```

## Training

```json
{
  "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
  "affinity_loss_mode": "full",
  "affinity_target_encoding": "mhcflurry",
  "batch_size": 256,
  "binding_core_lengths": [
    8,
    9,
    10,
    11
  ],
  "binding_core_refinement": "shared",
  "binding_direct_segment_mode": "off",
  "binding_kinetic_input_mode": "affinity_vec",
  "binding_output_consistency_beta": 0.25,
  "d_model": 32,
  "epochs": 50,
  "groove_pos_mode": "concat_start_end_frac",
  "kd_grouping_mode": "split_kd_proxy",
  "max_affinity_nM": 100000,
  "n_heads": 4,
  "n_layers": 2,
  "optimizer": "AdamW",
  "peptide_pos_mode": "concat_start_end_frac",
  "probe_artifact_schema": [
    "KD_nM",
    "IC50_nM",
    "EC50_nM",
    "KD_proxy_ic50_nM",
    "KD_proxy_ec50_nM",
    "binding_affinity_probe_kd"
  ],
  "ranking_losses": false,
  "requested_gpu": "H100!",
  "synthetic_negatives": false,
  "warm_start": "",
  "weight_decay": 0.01
}
```

## Tested Conditions

```json
[
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "binding_kd_family_consistency_weight": 0.0,
    "binding_output_consistency_beta": 0.25,
    "binding_proxy_cross_consistency_weight": 0.0,
    "condition_key": "pf07_kd0_cross0",
    "description": "PF07 control with output tying (kd_family=0, proxy_cross=0, beta=0.25)",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "binding_kd_family_consistency_weight": 0.0,
    "binding_output_consistency_beta": 0.25,
    "binding_proxy_cross_consistency_weight": 0.001,
    "condition_key": "pf07_kd0_cross0p001",
    "description": "PF07 control with output tying (kd_family=0, proxy_cross=0.001, beta=0.25)",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "binding_kd_family_consistency_weight": 0.0,
    "binding_output_consistency_beta": 0.25,
    "binding_proxy_cross_consistency_weight": 0.004,
    "condition_key": "pf07_kd0_cross0p004",
    "description": "PF07 control with output tying (kd_family=0, proxy_cross=0.004, beta=0.25)",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "binding_kd_family_consistency_weight": 0.0025,
    "binding_output_consistency_beta": 0.25,
    "binding_proxy_cross_consistency_weight": 0.0,
    "condition_key": "pf07_kd0p0025_cross0",
    "description": "PF07 control with output tying (kd_family=0.0025, proxy_cross=0, beta=0.25)",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "binding_kd_family_consistency_weight": 0.0025,
    "binding_output_consistency_beta": 0.25,
    "binding_proxy_cross_consistency_weight": 0.001,
    "condition_key": "pf07_kd0p0025_cross0p001",
    "description": "PF07 control with output tying (kd_family=0.0025, proxy_cross=0.001, beta=0.25)",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "binding_kd_family_consistency_weight": 0.0025,
    "binding_output_consistency_beta": 0.25,
    "binding_proxy_cross_consistency_weight": 0.004,
    "condition_key": "pf07_kd0p0025_cross0p004",
    "description": "PF07 control with output tying (kd_family=0.0025, proxy_cross=0.004, beta=0.25)",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "binding_kd_family_consistency_weight": 0.01,
    "binding_output_consistency_beta": 0.25,
    "binding_proxy_cross_consistency_weight": 0.0,
    "condition_key": "pf07_kd0p01_cross0",
    "description": "PF07 control with output tying (kd_family=0.01, proxy_cross=0, beta=0.25)",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "binding_kd_family_consistency_weight": 0.01,
    "binding_output_consistency_beta": 0.25,
    "binding_proxy_cross_consistency_weight": 0.001,
    "condition_key": "pf07_kd0p01_cross0p001",
    "description": "PF07 control with output tying (kd_family=0.01, proxy_cross=0.001, beta=0.25)",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "binding_kd_family_consistency_weight": 0.01,
    "binding_output_consistency_beta": 0.25,
    "binding_proxy_cross_consistency_weight": 0.004,
    "condition_key": "pf07_kd0p01_cross0p004",
    "description": "PF07 control with output tying (kd_family=0.01, proxy_cross=0.004, beta=0.25)",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "binding_kd_family_consistency_weight": 0.04,
    "binding_output_consistency_beta": 0.25,
    "binding_proxy_cross_consistency_weight": 0.0,
    "condition_key": "pf07_kd0p04_cross0",
    "description": "PF07 control with output tying (kd_family=0.04, proxy_cross=0, beta=0.25)",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "binding_kd_family_consistency_weight": 0.04,
    "binding_output_consistency_beta": 0.25,
    "binding_proxy_cross_consistency_weight": 0.001,
    "condition_key": "pf07_kd0p04_cross0p001",
    "description": "PF07 control with output tying (kd_family=0.04, proxy_cross=0.001, beta=0.25)",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "binding_kd_family_consistency_weight": 0.04,
    "binding_output_consistency_beta": 0.25,
    "binding_proxy_cross_consistency_weight": 0.004,
    "condition_key": "pf07_kd0p04_cross0p004",
    "description": "PF07 control with output tying (kd_family=0.04, proxy_cross=0.004, beta=0.25)",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  }
]
```

## Results

This sweep is now fully collected. All `12 / 12` runs completed and were aggregated from `results/runs/` into the summary tables below.

| Condition | KD tie | Proxy tie | Test Spearman | Test AUROC | Test AUPRC | Test RMSE log10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pf07_kd0_cross0-e050-s43` | 0.0 | 0.0 | 0.8196 | 0.9288 | 0.8852 | 0.9208 |
| `pf07_kd0p01_cross0p001-e050-s43` | 0.01 | 0.001 | 0.8192 | 0.9288 | 0.8850 | 0.9319 |
| `pf07_kd0p0025_cross0p001-e050-s43` | 0.0025 | 0.001 | 0.8123 | 0.9258 | 0.8863 | 0.9463 |
| `pf07_kd0p01_cross0-e050-s43` | 0.01 | 0.0 | 0.8120 | 0.9277 | 0.8787 | 0.9244 |
| `pf07_kd0p0025_cross0-e050-s43` | 0.0025 | 0.0 | 0.8113 | 0.9232 | 0.8715 | 0.9394 |
| `pf07_kd0p04_cross0-e050-s43` | 0.04 | 0.0 | 0.8095 | 0.9246 | 0.8693 | 0.9554 |
| `pf07_kd0p01_cross0p004-e050-s43` | 0.01 | 0.004 | 0.8070 | 0.9220 | 0.8741 | 0.9431 |
| `pf07_kd0_cross0p001-e050-s43` | 0.0 | 0.001 | 0.8051 | 0.9164 | 0.8615 | 0.9471 |
| `pf07_kd0_cross0p004-e050-s43` | 0.0 | 0.004 | 0.8044 | 0.9210 | 0.8626 | 0.9843 |
| `pf07_kd0p04_cross0p004-e050-s43` | 0.04 | 0.004 | 0.8015 | 0.9190 | 0.8638 | 0.9796 |
| `pf07_kd0p04_cross0p001-e050-s43` | 0.04 | 0.001 | 0.8011 | 0.9191 | 0.8603 | 0.9815 |
| `pf07_kd0p0025_cross0p004-e050-s43` | 0.0025 | 0.004 | 0.8004 | 0.9182 | 0.8660 | 0.9567 |

## Winner

- Best overall condition: `kd=0.0`, `cross=0.0`
- Test Spearman: `0.8196399`
- Test AUROC: `0.9288369`
- Test AUPRC: `0.8851608`
- Test RMSE log10: `0.9208454`

The best regularized condition was `kd=0.01`, `cross=0.001`, but it did not beat the untied control:

- Test Spearman: `0.8192348`
- Test AUROC: `0.9287891`
- Test AUPRC: `0.8849943`
- Test RMSE log10: `0.9318614`

## Takeaway

- Weak output-side tying did not improve this corrected sequence-only PF07 contract.
- The no-regularization control remained best on the primary metric.
- Very small tying (`kd=0.01`, `cross=0.001`) came close on test Spearman, but still lost and had worse RMSE.
- Stronger regularization, especially `cross=0.004` and the `kd=0.04` settings, clearly hurt.
- This experiment does not change the practical baseline. No update is needed to `model_to_beat.md`.

## Operational Notes

- The first detached-launch pass hit a Modal image-build race for `11 / 12` runs. Those launches failed before training started.
- The failed conditions were relaunched without changing the model or dataset contract, and all `12 / 12` runs then completed successfully.
- Requested GPU: `H100!`
- Observed peak reserved GPU memory from per-epoch logs was approximately `23.5 GiB`.

## Artifacts

- Aggregated condition table: [`results/condition_summary.csv`](./results/condition_summary.csv)
- Aggregated condition JSON: [`results/condition_summary.json`](./results/condition_summary.json)
- Combined summary bundle: [`results/summary_bundle.json`](./results/summary_bundle.json)
- Final probe panel dump: [`results/final_probe_predictions.csv`](./results/final_probe_predictions.csv)
- Ranking plot: [`results/test_spearman_ranking.png`](./results/test_spearman_ranking.png)
- Metric grid plot: [`results/test_metric_grid.png`](./results/test_metric_grid.png)
- Raw fetched runs: [`results/runs/`](./results/runs/)
- Reproducibility bundle: [`reproduce/`](./reproduce/)
