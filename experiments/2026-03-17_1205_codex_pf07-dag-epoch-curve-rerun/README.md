# PF07 DAG Epoch-Curve Rerun

- Agent: `codex`
- Source script: `experiments/2026-03-17_1205_codex_pf07-dag-epoch-curve-rerun/code/launch.py`
- Created: `2026-03-17T10:54:53.427587`

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
  "d_model": 32,
  "epoch_val_metrics_frequency": 1,
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
  "probe_plot_frequency": "final",
  "requested_gpu": "H100!",
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
    "condition_key": "pf07_control_constant",
    "description": "Flat honest PF07 control rerun: 1e-3 constant",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_family",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_family_constant",
    "description": "Family-anchor DAG rerun: 1e-3 constant",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_method_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_method_leaf_constant",
    "description": "Method-leaf DAG rerun: 1e-3 constant",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_prep_readout_leaf_constant",
    "description": "Prep/readout-leaf DAG rerun: 1e-3 constant",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_method_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_method_leaf_warmup_cosine",
    "description": "Method-leaf DAG schedule variant: 3e-4 warmup_cosine",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.0003,
    "lr_schedule": "warmup_cosine",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_prep_readout_leaf_warmup_cosine",
    "description": "Prep/readout-leaf DAG schedule variant: 3e-4 warmup_cosine",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.0003,
    "lr_schedule": "warmup_cosine",
    "max_affinity_nM": 100000
  }
]
```

## Notes

## Results

All `6 / 6` runs were collected locally under [`results/runs/`](./results/runs/), aggregated with the shared summary tool plus the experiment-local curve aggregator, and reproduced the earlier ranking cleanly.

### Terminal Test Metrics

| Condition | Residual mode | LR / schedule | Test Spearman | Test AUROC | Test AUPRC | Test RMSE log10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `pf07_dag_prep_readout_leaf_constant` | `dag_prep_readout_leaf` | `1e-3 constant` | `0.8379` | `0.9367` | `0.8907` | `0.8778` |
| `pf07_dag_method_leaf_constant` | `dag_method_leaf` | `1e-3 constant` | `0.8339` | `0.9336` | `0.8854` | `0.8819` |
| `pf07_dag_method_leaf_warmup_cosine` | `dag_method_leaf` | `3e-4 warmup_cosine` | `0.8313` | `0.9307` | `0.8812` | `0.8649` |
| `pf07_dag_prep_readout_leaf_warmup_cosine` | `dag_prep_readout_leaf` | `3e-4 warmup_cosine` | `0.8312` | `0.9325` | `0.8822` | `0.8740` |
| `pf07_dag_family_constant` | `dag_family` | `1e-3 constant` | `0.8206` | `0.9270` | `0.8765` | `0.9203` |
| `pf07_control_constant` | `shared_base_factorized_context_plus_segment_residual` | `1e-3 constant` | `0.8160` | `0.9281` | `0.8826` | `0.9225` |

Winner: `pf07_dag_prep_readout_leaf_constant`.

### Validation-Curve Summary

The new per-epoch metric logging shows that the prep/readout-leaf DAG is also the strongest validation-curve model.

| Condition | Final epoch | Final val Spearman | Best val Spearman | Best epoch | Best val AUROC | Best val AUPRC | Best val RMSE log10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pf07_dag_prep_readout_leaf_constant` | `50` | `0.8277` | `0.8338` | `18` | `0.9341` | `0.8931` | `0.8573` |
| `pf07_dag_prep_readout_leaf_warmup_cosine` | `50` | `0.8268` | `0.8303` | `38` | `0.9366` | `0.8943` | `0.8595` |
| `pf07_dag_method_leaf_constant` | `50` | `0.8204` | `0.8274` | `21` | `0.9317` | `0.8920` | `0.8839` |
| `pf07_dag_method_leaf_warmup_cosine` | `50` | `0.8190` | `0.8231` | `28` | `0.9338` | `0.9005` | `0.8752` |

Interpretation:
- the same `dag_prep_readout_leaf` model still wins after rerun
- constant `1e-3` remains slightly better on the primary test metric than the warmup-cosine variant
- all four leaf-DAG runs peak on validation before epoch `50`, but the ranking stays stable

### Probe Artifacts

Per-run probe trajectories now cover all affinity outputs:
- `KD_nM`
- `IC50_nM`
- `EC50_nM`
- `KD_proxy_ic50_nM`
- `KD_proxy_ec50_nM`
- `binding_affinity_probe_kd`

The experiment-level local artifacts needed to recreate the plots are:
- [`results/epoch_metrics_by_condition.csv`](./results/epoch_metrics_by_condition.csv)
- [`results/epoch_metrics_by_condition.json`](./results/epoch_metrics_by_condition.json)
- [`results/probe_affinity_by_condition.csv`](./results/probe_affinity_by_condition.csv)
- [`results/probe_affinity_by_condition_long.csv`](./results/probe_affinity_by_condition_long.csv)
- [`results/final_probe_predictions.csv`](./results/final_probe_predictions.csv)
- [`results/val_spearman_over_epochs.png`](./results/val_spearman_over_epochs.png)
- [`results/val_auroc_over_epochs.png`](./results/val_auroc_over_epochs.png)
- [`results/val_auprc_over_epochs.png`](./results/val_auprc_over_epochs.png)
- [`results/val_rmse_log10_over_epochs.png`](./results/val_rmse_log10_over_epochs.png)
- [`results/val_metric_curves_over_epochs.png`](./results/val_metric_curves_over_epochs.png)

Each run directory also contains its own:
- `epoch_metrics.csv` / `epoch_metrics.json`
- `probe_affinity_over_epochs.csv` / `probe_affinity_over_epochs.json`
- `probe_affinity_over_epochs.png`
- `probe_affinity_all_outputs_over_epochs.png`
- `val_predictions.csv`
- `test_predictions.csv`

## Takeaway

This rerun validates the earlier DAG result under a stronger local artifact contract. The best honest architecture does not change: `PF07` with `dag_prep_readout_leaf`, `mhcflurry`, `split_kd_proxy`, `50` epochs, and sequence-only inputs remains the model to beat for the 2-allele broad-numeric affinity contract. Relative to the original DAG sweep winner, the rerun is effectively the same on Spearman (`-0.00028`) while improving AUROC (`+0.00082`) and AUPRC (`+0.01263`) at slightly worse RMSE (`+0.00741`).

Reproducibility bundle: [`reproduce/`](./reproduce/)
