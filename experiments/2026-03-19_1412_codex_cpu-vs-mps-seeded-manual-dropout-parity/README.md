# CPU vs MPS Focused PF07 Seeded Manual-Dropout Parity Compare

- Agent: `codex`
- Source script: `experiments/2026-03-19_1412_codex_cpu-vs-mps-seeded-manual-dropout-parity/code/launch.py`
- Created: `2026-03-19T14:11:35.947357`

## Dataset Contract

```json
{
  "assay_selector_inputs_forbidden": true,
  "input_fields": [
    "nflank",
    "peptide",
    "cflank",
    "mhc_a",
    "mhc_b"
  ],
  "max_records": 200,
  "measurement_profile": "numeric_no_qualitative",
  "probe_alleles": [
    "HLA-A*02:01",
    "HLA-A*24:02"
  ],
  "probe_peptides": [
    "SLLQHLIGL",
    "FLRYLLFGI"
  ],
  "qualifier_filter": "all",
  "source": "data/merged_deduped.tsv",
  "source_filter": "iedb",
  "split_policy": "peptide_group_80_10_10",
  "split_seed": 42,
  "train_mhc_class_filter": "I",
  "train_seed": 43
}
```

## Training

```json
{
  "downstream": {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "affinity_target_encoding": "mhcflurry",
    "batch_size": 8,
    "d_model": 32,
    "devices": [
      "cpu",
      "mps"
    ],
    "epochs": 3,
    "kd_grouping_mode": "split_kd_proxy",
    "matmul_precision": "default",
    "n_heads": 4,
    "n_layers": 2
  },
  "pretraining": {
    "warm_start_checkpoint_local": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt"
  }
}
```

## Tested Conditions

```json
[
  {
    "batch_size": 8,
    "condition_key": "cpu",
    "description": "Matched tiny focused PF07 smoke run on CPU",
    "device": "cpu",
    "epochs": 3,
    "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "max_records": 200,
    "seed": 43,
    "split_seed": 42
  },
  {
    "batch_size": 8,
    "condition_key": "mps",
    "description": "Matched tiny focused PF07 smoke run on Apple Silicon MPS",
    "device": "mps",
    "epochs": 3,
    "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "max_records": 200,
    "seed": 43,
    "split_seed": 42
  }
]
```

## Results

| Device | Test Spearman | Test AUROC | Test AUPRC | Test RMSE log10 | Test Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cpu` | `-0.22238` | `0.39286` | `0.22614` | `1.37729` | `0.05504` |
| `mps` | `-0.28853` | `0.38889` | `0.22469` | `1.40880` | `0.05685` |

CPU-vs-MPS deltas:
- Spearman: `-0.06615`
- AUROC: `-0.00397`
- AUPRC: `-0.00145`
- RMSE log10: `+0.03151`

## Runtime Notes

- This follow-up forced `manual_dropout` on both CPU and MPS and seeded each dropout module from the training seed using CPU-generated Bernoulli masks.
- Both runs completed cleanly with:
  - `mps_safe_mode_applied = manual_dropout`
  - `mps_safe_dropout_modules_replaced = 20`
  - `mps_safe_dropout_modules_zeroed = 0`
  - `mps_safe_multihead_attention_modules_zeroed = 0`
- No launch-log fallback warnings or `non_finite_train_loss` events occurred.

## Takeaway

- The dropout implementation is now genuinely hardware-independent when `--mps-safe-mode manual_dropout` is used on both backends.
- That solved the contract mismatch problem, but it did not produce exact metric parity: AUROC/AUPRC stayed very close, while Spearman on this tiny 3-epoch contract still drifted more than expected.
- So the remaining CPU-vs-MPS gap is no longer “because dropout is different.” It is backend math / optimization drift elsewhere in the stack.

## Artifacts

- Summary table: [`results/condition_summary.csv`](./results/condition_summary.csv)
- Aggregated bundle: [`results/summary_bundle.json`](./results/summary_bundle.json)
- Reproducibility bundle: [`reproduce/`](./reproduce/)
