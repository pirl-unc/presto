# MHCflurry MAX × Encoder Dimension Sweep (24 conditions)

- Agent: `claude`
- Source script: `scripts/benchmark_distributional_ba_heads_v3.py`
- Created: `2026-03-13T12:18:49.176078`

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
  "encoder": "AblationEncoder(n_layers=2, n_heads=4)",
  "epochs": 20,
  "head": "MHCflurry additive",
  "lr": "1e-3",
  "seed": 42,
  "weight_decay": 0.01
}
```

## Tested Conditions

```json
[
  {
    "cond_id": 1,
    "embed_dim": 128,
    "max_nM": 25000
  },
  {
    "cond_id": 2,
    "embed_dim": 128,
    "max_nM": 50000
  },
  {
    "cond_id": 3,
    "embed_dim": 128,
    "max_nM": 75000
  },
  {
    "cond_id": 4,
    "embed_dim": 128,
    "max_nM": 100000
  },
  {
    "cond_id": 5,
    "embed_dim": 128,
    "max_nM": 125000
  },
  {
    "cond_id": 6,
    "embed_dim": 128,
    "max_nM": 150000
  },
  {
    "cond_id": 7,
    "embed_dim": 256,
    "max_nM": 25000
  },
  {
    "cond_id": 8,
    "embed_dim": 256,
    "max_nM": 50000
  },
  {
    "cond_id": 9,
    "embed_dim": 256,
    "max_nM": 75000
  },
  {
    "cond_id": 10,
    "embed_dim": 256,
    "max_nM": 100000
  },
  {
    "cond_id": 11,
    "embed_dim": 256,
    "max_nM": 125000
  },
  {
    "cond_id": 12,
    "embed_dim": 256,
    "max_nM": 150000
  },
  {
    "cond_id": 13,
    "embed_dim": 384,
    "max_nM": 25000
  },
  {
    "cond_id": 14,
    "embed_dim": 384,
    "max_nM": 50000
  },
  {
    "cond_id": 15,
    "embed_dim": 384,
    "max_nM": 75000
  },
  {
    "cond_id": 16,
    "embed_dim": 384,
    "max_nM": 100000
  },
  {
    "cond_id": 17,
    "embed_dim": 384,
    "max_nM": 125000
  },
  {
    "cond_id": 18,
    "embed_dim": 384,
    "max_nM": 150000
  },
  {
    "cond_id": 19,
    "embed_dim": 512,
    "max_nM": 25000
  },
  {
    "cond_id": 20,
    "embed_dim": 512,
    "max_nM": 50000
  },
  {
    "cond_id": 21,
    "embed_dim": 512,
    "max_nM": 75000
  },
  {
    "cond_id": 22,
    "embed_dim": 512,
    "max_nM": 100000
  },
  {
    "cond_id": 23,
    "embed_dim": 512,
    "max_nM": 125000
  },
  {
    "cond_id": 24,
    "embed_dim": 512,
    "max_nM": 150000
  }
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
- Reproducibility bundle: [`reproduce/`](./reproduce/)
