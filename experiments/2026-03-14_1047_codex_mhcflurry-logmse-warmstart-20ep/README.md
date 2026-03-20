# MHCflurry / LogMSE Cold vs Warm Start

Status: completed

## Goal

Test whether partial warm-starting the fixed self-contained backbone from `mhc-pretrain-20260308b` improves the two viable regression heads from the clean 12-condition benchmark.

## Fixed Contract

- Source: `data/merged_deduped.tsv`
- Panel: `HLA-A*02:01`, `HLA-A*24:02`, `HLA-A*03:01`, `HLA-A*11:01`, `HLA-A*01:01`, `HLA-B*07:02`, `HLA-B*44:02`
- Assay families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- Qualifier policy: `all`
- Split: deterministic peptide-group `80/10/10`, seed `42`
- Backbone: `FixedBackbone(embed=128,layers=2,heads=4,ff=128)`
- Training: batch `256`, epochs `20`, `AdamW`, `lr=1e-4`, `warmup_cosine`, `weight_decay=0.01`
- GPU: `H100!`

## Conditions

- `c01_mhcflurry_additive_max200k_cold`
- `c02_mhcflurry_additive_max200k_warm`
- `c03_log_mse_additive_max200k_cold`
- `c04_log_mse_additive_max200k_warm`

## Warm Start

- Source checkpoint on Modal: `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
- Load policy: shape-compatible encoder weights only
- Intended compatible subset:
  - `aa_embedding.weight`
  - attention / norm blocks from `stream_encoder.layers.*`
- Intentionally incompatible subset stays random:
  - feed-forward sublayers
  - positional embeddings
  - all non-backbone heads

## Runtime Boundary

- Self-contained package: `code/distributional_ba/`
- Self-contained launcher: `code/launch.py`
- Modal shim only: `scripts/train_modal.py::distributional_ba_regwarm_run`

## Results

| condition | warm_start | test_spearman | test_auroc | test_auprc | test_f1 | test_rmse_log10 | mean_epoch_s |
|-----------|------------|---------------|------------|------------|---------|-----------------|--------------|
| `c01_mhcflurry_additive_max200k_cold` | no | **0.6865** | **0.8523** | **0.8387** | **0.7435** | **1.0746** | 4.60 |
| `c03_log_mse_additive_max200k_cold` | no | 0.6462 | 0.8298 | 0.8130 | 0.7150 | 1.1371 | 5.24 |
| `c02_mhcflurry_additive_max200k_warm` | yes | 0.5945 | 0.7952 | 0.7747 | 0.6858 | 1.1816 | 4.12 |
| `c04_log_mse_additive_max200k_warm` | yes | 0.5865 | 0.7910 | 0.7751 | 0.6790 | 1.2002 | 3.72 |

## Takeaways

1. Partial encoder warm-start from `mhc-pretrain-20260308b` hurt both viable regression heads under this fixed-backbone broad benchmark.
2. The best condition is the cold-start `mhcflurry` run `c01`, not either warm-start condition.
3. Relative to the 10-epoch clean self-contained benchmark winner from EXP-17 (`test Spearman 0.6209`), the same backbone/head family improved materially here (`0.6865`) by training longer, not by warm-starting.

## Artifacts

- Raw run artifacts were pulled locally under `results/runs/`
- Summary tables:
  - `results/condition_summary.csv`
  - `results/family_summary.csv`
  - `results/warm_start_summary.csv`
  - `results/per_allele_metrics.csv`
  - `results/metric_verification.csv`
- Plots:
  - `results/test_spearman_ranking.png`
  - `results/val_spearman_curves.png`
  - `results/final_probe_heatmap.png`
- Recomputed bundle:
  - `results/summary_bundle.json`
