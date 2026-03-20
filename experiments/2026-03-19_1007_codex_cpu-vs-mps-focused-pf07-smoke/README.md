# CPU vs MPS Focused PF07 Smoke Compare

- Agent: `codex`
- Source script: `experiments/2026-03-19_1007_codex_cpu-vs-mps-focused-pf07-smoke/code/launch.py`
- Created: `2026-03-19T10:08:45.204557`

## Question

Can the local Apple Silicon `mps` path run the same tiny focused PF07 training contract as `cpu` without diverging, and does it behave similarly enough to use for local experiment continuation?

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

- `cpu`
  - Run: `presto-focused-device-smoke-20260319a-cpu-e001-s43`
  - Status: completed
  - Validation: Spearman `-0.0591`, AUROC `0.5469`, AUPRC `0.3866`, RMSE log10 `1.8723`, loss `0.1179`
  - Test: Spearman `-0.1048`, AUROC `0.5833`, AUPRC `0.3142`, RMSE log10 `1.4804`, loss `0.0630`
- `mps`
  - Run: `presto-focused-device-smoke-20260319a-mps-e001-s43`
  - Status: diverged
  - Divergence: epoch `1`, reason `non_finite_train_loss`
  - Held-out metrics: none produced because training diverged before final evaluation
  - Launch log also shows MPS fallback behavior on `nonzero`, which indicates partial CPU fallback during execution

## Artifacts

- Aggregate summaries:
  - [`results/condition_summary.csv`](./results/condition_summary.csv)
  - [`results/condition_summary.json`](./results/condition_summary.json)
  - [`results/summary_bundle.json`](./results/summary_bundle.json)
- Plots:
  - [`results/test_metric_grid.png`](./results/test_metric_grid.png)
  - [`results/test_spearman_ranking.png`](./results/test_spearman_ranking.png)
- Per-run raw summaries:
  - [`results/runs/presto-focused-device-smoke-20260319a-cpu-e001-s43/summary.json`](./results/runs/presto-focused-device-smoke-20260319a-cpu-e001-s43/summary.json)
  - [`results/runs/presto-focused-device-smoke-20260319a-mps-e001-s43/summary.json`](./results/runs/presto-focused-device-smoke-20260319a-mps-e001-s43/summary.json)
- Launch logs:
  - [`launch_logs/presto-focused-device-smoke-20260319a-cpu-e001-s43.log`](./launch_logs/presto-focused-device-smoke-20260319a-cpu-e001-s43.log)
  - [`launch_logs/presto-focused-device-smoke-20260319a-mps-e001-s43.log`](./launch_logs/presto-focused-device-smoke-20260319a-mps-e001-s43.log)

## Takeaway

This smoke test does not support using `mps` as a drop-in local continuation backend for focused PF07 training. The matched `cpu` run completed, while the matched `mps` run diverged on epoch `1` with `non_finite_train_loss`. For local continuation on Apple Silicon, the current safe default remains `cpu`; `mps` needs numeric stabilization work before it can be trusted for real experiment reruns.

Reproducibility bundle: [`reproduce/`](./reproduce/)
