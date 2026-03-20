# Distributional vs Regression BA Heads (32 conditions)

- Agent: `claude`
- Source script: `scripts/benchmark_distributional_ba_heads.py`
- Created: `2026-03-13T07:19:59.854088`

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
    "cond_id": 1,
    "head_type": "mhcflurry",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "affine",
    "cond_id": 2,
    "head_type": "mhcflurry",
    "max_nM": 100000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "affine",
    "cond_id": 3,
    "head_type": "log_mse",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "affine",
    "cond_id": 4,
    "head_type": "log_mse",
    "max_nM": 100000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d1_affine",
    "cond_id": 5,
    "head_type": "twohot",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d1_affine",
    "cond_id": 6,
    "head_type": "twohot",
    "max_nM": 100000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d1_affine",
    "cond_id": 7,
    "head_type": "hlgauss",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d1_affine",
    "cond_id": 8,
    "head_type": "hlgauss",
    "max_nM": 100000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d1_affine",
    "cond_id": 9,
    "head_type": "twohot",
    "max_nM": 50000,
    "n_bins": 64,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d1_affine",
    "cond_id": 10,
    "head_type": "twohot",
    "max_nM": 100000,
    "n_bins": 64,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d1_affine",
    "cond_id": 11,
    "head_type": "hlgauss",
    "max_nM": 50000,
    "n_bins": 64,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d1_affine",
    "cond_id": 12,
    "head_type": "hlgauss",
    "max_nM": 100000,
    "n_bins": 64,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d2_logit",
    "cond_id": 13,
    "head_type": "twohot",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d2_logit",
    "cond_id": 14,
    "head_type": "twohot",
    "max_nM": 100000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d2_logit",
    "cond_id": 15,
    "head_type": "twohot",
    "max_nM": 50000,
    "n_bins": 64,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d2_logit",
    "cond_id": 16,
    "head_type": "twohot",
    "max_nM": 100000,
    "n_bins": 64,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d2_logit",
    "cond_id": 17,
    "head_type": "hlgauss",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d2_logit",
    "cond_id": 18,
    "head_type": "hlgauss",
    "max_nM": 100000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d2_logit",
    "cond_id": 19,
    "head_type": "hlgauss",
    "max_nM": 50000,
    "n_bins": 64,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d2_logit",
    "cond_id": 20,
    "head_type": "hlgauss",
    "max_nM": 100000,
    "n_bins": 64,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "d1_affine",
    "cond_id": 21,
    "head_type": "hlgauss",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.5
  },
  {
    "assay_mode": "d1_affine",
    "cond_id": 22,
    "head_type": "hlgauss",
    "max_nM": 100000,
    "n_bins": 128,
    "sigma_mult": 0.5
  },
  {
    "assay_mode": "d1_affine",
    "cond_id": 23,
    "head_type": "hlgauss",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 1.5
  },
  {
    "assay_mode": "d1_affine",
    "cond_id": 24,
    "head_type": "hlgauss",
    "max_nM": 100000,
    "n_bins": 128,
    "sigma_mult": 1.5
  },
  {
    "assay_mode": "d2_logit",
    "cond_id": 25,
    "head_type": "hlgauss",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.5
  },
  {
    "assay_mode": "d2_logit",
    "cond_id": 26,
    "head_type": "hlgauss",
    "max_nM": 100000,
    "n_bins": 128,
    "sigma_mult": 0.5
  },
  {
    "assay_mode": "d2_logit",
    "cond_id": 27,
    "head_type": "hlgauss",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 1.5
  },
  {
    "assay_mode": "d2_logit",
    "cond_id": 28,
    "head_type": "hlgauss",
    "max_nM": 100000,
    "n_bins": 128,
    "sigma_mult": 1.5
  },
  {
    "assay_mode": "additive",
    "cond_id": 29,
    "head_type": "mhcflurry",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "additive",
    "cond_id": 30,
    "head_type": "mhcflurry",
    "max_nM": 100000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "additive",
    "cond_id": 31,
    "head_type": "log_mse",
    "max_nM": 50000,
    "n_bins": 128,
    "sigma_mult": 0.75
  },
  {
    "assay_mode": "additive",
    "cond_id": 32,
    "head_type": "log_mse",
    "max_nM": 100000,
    "n_bins": 128,
    "sigma_mult": 0.75
  }
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
- Reproducibility bundle: [`reproduce/`](./reproduce/)
