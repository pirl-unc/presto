# CPU vs MPS All-Class-I Manual-Dropout Validation

- Agent: `codex`
- Source script: `experiments/2026-03-19_1505_codex_cpu-vs-mps-allclass1-manualdropout-validation/code/launch.py`
- Created: `2026-03-19T14:22:40.568812`

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
  "validation_purpose": "cpu_vs_mps_all_class_i_manual_dropout"
}
```

## Training

```json
{
  "downstream": {
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "batch_size": 256,
    "devices": [
      "cpu",
      "mps"
    ],
    "d_model": 32,
    "epochs": 3,
    "kd_grouping_mode": "split_kd_proxy",
    "mps_safe_mode": "manual_dropout",
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
    "description": "Matched all-class-I reduced PF07 validation on CPU",
    "device": "cpu",
    "epochs": 3,
    "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "max_records": 5000,
    "mps_safe_mode": "manual_dropout",
    "seed": 43,
    "split_seed": 42
  },
  {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "batch_size": 256,
    "condition_key": "mps",
    "description": "Matched all-class-I reduced PF07 validation on Apple Silicon MPS",
    "device": "mps",
    "epochs": 3,
    "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "max_records": 5000,
    "mps_safe_mode": "manual_dropout",
    "seed": 43,
    "split_seed": 42
  }
]
```

## Results

| Device | Test Spearman | Test AUROC | Test AUPRC | Test RMSE log10 | Test Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cpu` | `0.20249` | `0.59125` | `0.43531` | `1.38687` | `0.06661` |
| `mps` | `0.30026` | `0.65745` | `0.48325` | `1.36865` | `0.06579` |

MPS minus CPU:
- Spearman: `+0.09777`
- AUROC: `+0.06620`
- AUPRC: `+0.04794`
- RMSE log10: `-0.01823`
- Loss: `-0.00083`

## Runtime Notes

- Both backends used the same seeded manual-dropout implementation:
  - `mps_safe_mode_requested = manual_dropout`
  - `mps_safe_mode_applied = manual_dropout`
  - `mps_safe_dropout_modules_replaced = 20`
  - `mps_safe_dropout_modules_zeroed = 0`
  - `mps_safe_multihead_attention_modules_zeroed = 0`
- No launch-log warnings, fallback messages, NaNs, or `non_finite_train_loss` events were observed.
- This was still a reduced validation contract:
  - current best honest PF07 downstream head
  - all-class-I numeric binding slice
  - `max_records = 5000`
  - `epochs = 3`

## Takeaway

- With seeded manual dropout forced on both backends, Apple Silicon `mps` completed the same honest all-class-I PF07 contract cleanly and did not underperform CPU on this slice.
- This does not prove exact backend equivalence, but it is now strong enough evidence to treat `mps` as usable for local focused training on the parity-preserving manual-dropout contract.
- For local continuation work, the safest shared contract is now:
  - `--device cpu|mps`
  - `--mps-safe-mode manual_dropout`

## Artifacts

- Summary table: [`results/condition_summary.csv`](./results/condition_summary.csv)
- Epoch summary: [`results/epoch_summary.csv`](./results/epoch_summary.csv)
- Final probes: [`results/final_probe_predictions.csv`](./results/final_probe_predictions.csv)
- Reproducibility bundle: [`reproduce/`](./reproduce/)
