# Fine-Grained Encoder Dimension Sweep (6 conditions, 50 epochs)

- Agent: `codex`
- Source script: `scripts/benchmark_distributional_ba_heads_v4.py`
- Created: `2026-03-13T13:13:19.208800`

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
  "epochs": 50,
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
    "embed_dim": 32,
    "max_nM": 50000
  },
  {
    "cond_id": 2,
    "embed_dim": 64,
    "max_nM": 50000
  },
  {
    "cond_id": 3,
    "embed_dim": 96,
    "max_nM": 50000
  },
  {
    "cond_id": 4,
    "embed_dim": 128,
    "max_nM": 50000
  },
  {
    "cond_id": 5,
    "embed_dim": 192,
    "max_nM": 50000
  },
  {
    "cond_id": 6,
    "embed_dim": 256,
    "max_nM": 50000
  }
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
- Reproducibility bundle: [`reproduce/`](./reproduce/)
