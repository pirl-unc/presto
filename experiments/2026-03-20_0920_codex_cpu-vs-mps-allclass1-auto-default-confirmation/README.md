# CPU vs MPS All-Class-I Auto-Default Confirmation

- Agent: `codex`
- Source script: `experiments/2026-03-20_0920_codex_cpu-vs-mps-allclass1-auto-default-confirmation/code/launch.py`
- Created: `2026-03-20T13:25:15.809857`

## Dataset Contract

```json
{
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
  "max_records": 5000,
  "measurement_profile": "numeric_no_qualitative",
  "probe_alleles": [
    "HLA-A*02:01",
    "HLA-A*24:02"
  ],
  "probe_peptides": [
    "SLLQHLIGL",
    "FLRYLLFGI",
    "NFLIKFLLI",
    "IMLEGETKL"
  ],
  "qualifier_filter": "all",
  "sequence_resolution": "mhcseqs_first_with_index_fallback",
  "source": "data/merged_deduped.tsv",
  "source_filter": "iedb",
  "source_refresh": "canonical rebuild 2026-03-17",
  "split_policy": "peptide_group_80_10_10",
  "split_seed": 42,
  "train_all_alleles": true,
  "train_mhc_class_filter": "I",
  "train_seed": 43,
  "validation_purpose": "cpu_vs_mps_all_class_i_auto_default_confirmation"
}
```

## Training

```json
{
  "downstream": {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "affinity_target_encoding": "mhcflurry",
    "batch_size": 256,
    "d_model": 32,
    "design_id": "presto_pf07_dag_prep_readout_leaf_constant",
    "devices": [
      "cpu",
      "mps"
    ],
    "epochs": 3,
    "kd_grouping_mode": "split_kd_proxy",
    "mps_safe_mode": "auto",
    "n_heads": 4,
    "n_layers": 2
  },
  "pretraining": {
    "mode": "mhc_pretrain",
    "warm_start_checkpoint_local": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt"
  }
}
```

## Tested Conditions

```json
[
  {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "batch_size": 256,
    "condition_key": "cpu",
    "description": "Matched all-class-I reduced PF07 validation on CPU under default auto mode",
    "device": "cpu",
    "epochs": 3,
    "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "max_records": 5000,
    "mps_safe_mode": "auto",
    "seed": 43,
    "split_seed": 42
  },
  {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "batch_size": 256,
    "condition_key": "mps",
    "description": "Matched all-class-I reduced PF07 validation on Apple Silicon MPS under default auto mode",
    "device": "mps",
    "epochs": 3,
    "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "max_records": 5000,
    "mps_safe_mode": "auto",
    "seed": 43,
    "split_seed": 42
  }
]
```

## Notes

## Results

| Device | Test Spearman | Test AUROC | Test AUPRC | Test RMSE log10 | Test Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cpu` | `0.20254` | `0.59127` | `0.43533` | `1.38687` | `0.06661` |
| `mps` | `0.30026` | `0.65745` | `0.48325` | `1.36865` | `0.06579` |

MPS minus CPU:
- Spearman: `+0.09772`
- AUROC: `+0.06618`
- AUPRC: `+0.04793`
- RMSE log10: `-0.01823`
- Loss: `-0.00083`

## Runtime Notes

- Both runs were launched with `--mps-safe-mode auto`.
- Aggregated runtime config confirms the new default semantics on both backends:
  - `mps_safe_mode_requested = auto`
  - `mps_safe_mode_applied = manual_dropout`
  - `mps_safe_dropout_modules_replaced = 20`
  - `mps_safe_dropout_modules_zeroed = 0`
- No launch-log warnings, fallback messages, NaNs, or `non_finite_train_loss` events were observed.

## Takeaway

- The default path now really uses seeded manual dropout on both CPU and MPS; this is no longer a special MPS-only workaround.
- The default-path confirmation reproduced the same pattern as the explicit-manual run from `2026-03-19_1505_codex_cpu-vs-mps-allclass1-manualdropout-validation`.
- So the focused PF07 default contract is now:
  - `--mps-safe-mode auto`
  - internally applied as `manual_dropout` on all devices
  - use `--mps-safe-mode off` only for explicit native-dropout benchmarking

## Artifacts

- Summary table: [`results/condition_summary.csv`](./results/condition_summary.csv)
- Epoch summary: [`results/epoch_summary.csv`](./results/epoch_summary.csv)
- Final probes: [`results/final_probe_predictions.csv`](./results/final_probe_predictions.csv)
- Reproducibility bundle: [`reproduce/`](./reproduce/)
