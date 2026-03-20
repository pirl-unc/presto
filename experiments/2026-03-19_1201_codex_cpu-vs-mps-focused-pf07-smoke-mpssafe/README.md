# CPU vs MPS Focused PF07 Smoke Compare

- Agent: `codex`
- Source script: `experiments/2026-03-19_1201_codex_cpu-vs-mps-focused-pf07-smoke-mpssafe/code/launch.py`
- Created: `2026-03-19T10:55:59.191799`

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
    "epochs": 1,
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
    "epochs": 1,
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
    "epochs": 1,
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
| `cpu` | `-0.10479` | `0.58333` | `0.31422` | `1.48043` | `0.06303` |
| `mps` | `-0.10811` | `0.57540` | `0.30583` | `1.47338` | `0.06224` |

CPU-vs-MPS deltas:
- Spearman: `-0.00332`
- AUROC: `-0.00794`
- AUPRC: `-0.00839`
- RMSE log10: `-0.00705`

## Runtime Notes

- This is the first post-fix validation of the new MPS-safe path in `scripts/focused_binding_probe.py`.
- `mps` completed cleanly instead of diverging with `non_finite_train_loss`.
- The recorded runtime config shows the safeguard that was actually applied on `mps` in this historical run:
  - `mps_safe_mode_applied = zero_dropout`
  - `mps_safe_dropout_modules_zeroed = 20`
  - `mps_safe_multihead_attention_modules_zeroed = 17`
- The matched `cpu` run left the safeguard off.
- The launch logs did not contain fallback warnings such as the earlier `nonzero` warning.

## Takeaway

- The explicit MPS-safe runtime fix is sufficient to make the matched 1-epoch focused PF07 smoke run complete on Apple Silicon.
- Metrics stayed very close to CPU on the same tiny contract, so the fix is not just avoiding a crash.
- This was still an MPS-specific training contract because dropout was disabled on `mps` for stability.
- It has since been superseded by the later manual-dropout and seeded-manual-dropout follow-ups.

## Artifacts

- Summary table: [`results/condition_summary.csv`](./results/condition_summary.csv)
- Aggregated bundle: [`results/summary_bundle.json`](./results/summary_bundle.json)
- CPU summary: [`results/runs/presto-focused-device-smoke-20260319b-cpu-e001-s43/summary.json`](./results/runs/presto-focused-device-smoke-20260319b-cpu-e001-s43/summary.json)
- MPS summary: [`results/runs/presto-focused-device-smoke-20260319b-mps-e001-s43/summary.json`](./results/runs/presto-focused-device-smoke-20260319b-mps-e001-s43/summary.json)
- Reproducibility bundle: [`reproduce/`](./reproduce/)
