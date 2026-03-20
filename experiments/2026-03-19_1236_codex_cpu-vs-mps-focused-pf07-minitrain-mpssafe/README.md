# CPU vs MPS Focused PF07 Mini-Train Compare

- Agent: `codex`
- Source script: `experiments/2026-03-19_1236_codex_cpu-vs-mps-focused-pf07-minitrain-mpssafe/code/launch.py`
- Created: `2026-03-19T10:58:43.707908`

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
| `cpu` | `-0.29232` | `0.40873` | `0.22934` | `1.41349` | `0.05629` |
| `mps` | `-0.27928` | `0.40476` | `0.22817` | `1.40135` | `0.05595` |

CPU-vs-MPS deltas:
- Spearman: `+0.01304`
- AUROC: `-0.00397`
- AUPRC: `-0.00117`
- RMSE log10: `-0.01214`

## Runtime Notes

- This follow-up extends the same matched tiny contract from `1` epoch to `3` epochs.
- Both runs completed cleanly with finite summaries.
- The MPS runtime again applied the same historical safeguard:
  - `mps_safe_mode_applied = zero_dropout`
  - `mps_safe_dropout_modules_zeroed = 20`
  - `mps_safe_multihead_attention_modules_zeroed = 17`
- No launch-log fallback warnings were observed.

## Takeaway

- The MPS-safe path remains stable beyond startup and through a short multi-epoch training run.
- CPU and MPS stayed closely matched on all terminal metrics, so Apple Silicon is now usable for local focused PF07 training on this codepath.
- The main caveat for this historical run remains unchanged: it used a slightly different backend contract from CPU/CUDA because dropout was intentionally disabled on `mps` for stability.
- It has since been superseded by the later manual-dropout and seeded-manual-dropout follow-ups.

## Artifacts

- Summary table: [`results/condition_summary.csv`](./results/condition_summary.csv)
- Aggregated bundle: [`results/summary_bundle.json`](./results/summary_bundle.json)
- CPU summary: [`results/runs/presto-focused-device-minitrain-20260319a-cpu-e003-s43/summary.json`](./results/runs/presto-focused-device-minitrain-20260319a-cpu-e003-s43/summary.json)
- MPS summary: [`results/runs/presto-focused-device-minitrain-20260319a-mps-e003-s43/summary.json`](./results/runs/presto-focused-device-minitrain-20260319a-mps-e003-s43/summary.json)
- Reproducibility bundle: [`reproduce/`](./reproduce/)
