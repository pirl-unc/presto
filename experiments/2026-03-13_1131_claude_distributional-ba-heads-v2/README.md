# Distributional BA Heads V2: Bin Sweep + Uncertainty Heads (28 conditions)

- Agent: `claude`
- Source script: `scripts/benchmark_distributional_ba_heads_v2.py`
- Created: `2026-03-13T11:31:30.447784`

## Dataset Contract

```json
{
  "measurement_profile": "numeric_no_qualitative",
  "panel": [
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*01:01",
    "HLA-B*07:02",
    "HLA-B*44:02"
  ],
  "qualifier_filter": "all",
  "split": "peptide-stratified 80/10/10"
}
```

## Training

```json
{
  "batch_size": 256,
  "encoder": "AblationEncoder(embed=128, layers=2, heads=4)",
  "epochs": 20,
  "lr": "1e-3",
  "seed": 42,
  "weight_decay": 0.01
}
```

## Tested Conditions

```json
[
  {
    "assay_mode": "affine",
    "cond_id": 5,
    "head_type": "gaussian",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "affine",
    "cond_id": 6,
    "head_type": "gaussian",
    "max_nM": 250000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "additive",
    "cond_id": 7,
    "head_type": "gaussian",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "additive",
    "cond_id": 8,
    "head_type": "gaussian",
    "max_nM": 250000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "affine",
    "cond_id": 9,
    "head_type": "quantile",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "affine",
    "cond_id": 10,
    "head_type": "quantile",
    "max_nM": 250000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "additive",
    "cond_id": 11,
    "head_type": "quantile",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "additive",
    "cond_id": 12,
    "head_type": "quantile",
    "max_nM": 250000,
    "n_bins": 128,
    "sigma_mult": 0.75
  }
]
```

## Results

| condition | head | assay_mode | max_nM | test_spearman | test_auroc | test_rmse_log10 |
|-----------|------|------------|--------|---------------|------------|-----------------|
| `c05_gaussian_affine_max50k` | gaussian | affine | 50k | **0.7949** | 0.9308 | **0.8731** |
| `c12_quantile_additive_max250k` | quantile | additive | 250k | 0.7920 | 0.9277 | 0.9613 |
| `c09_quantile_affine_max50k` | quantile | affine | 50k | 0.7912 | 0.9254 | 0.9124 |
| `c10_quantile_affine_max250k` | quantile | affine | 250k | 0.7888 | 0.9305 | 0.9803 |
| `c08_gaussian_additive_max250k` | gaussian | additive | 250k | 0.7879 | 0.9297 | 0.9751 |
| `c06_gaussian_affine_max250k` | gaussian | affine | 250k | 0.7871 | **0.9377** | 0.9637 |
| `c07_gaussian_additive_max50k` | gaussian | additive | 50k | 0.7866 | 0.9264 | 0.8859 |
| `c11_quantile_additive_max50k` | quantile | additive | 50k | 0.7832 | 0.9252 | 0.9043 |

## Takeaways

1. This leftover v2 subset never changed the broader head-ranking story: the best condition here (`c05_gaussian_affine_max50k`) is competitive but not strong enough to displace the later `mhcflurry`-family winners.
2. Gaussian and quantile heads cluster tightly in the high-`0.78` / low-`0.79` Spearman range on this ablation backbone subset.

## Artifacts

- Raw run artifacts were pulled locally under `results/runs/`
- Summary tables:
  - `results/condition_summary.csv`
  - `results/epoch_summary.csv`
  - `results/final_probe_predictions.csv`
- Plots:
  - `results/test_spearman_ranking.png`
  - `results/test_metric_grid.png`
  - `results/training_curves.png`
  - `results/final_probe_heatmap.png`
- Reproducibility bundle: [`reproduce/`](./reproduce/)
