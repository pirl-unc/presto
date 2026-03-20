# Broad Frontier 5-Epoch Bakeoff

- Agent: `codex`
- Source script: `scripts/benchmark_broad_frontier_5ep.py`
- Created: `2026-03-12T08:50:37.700971`

## Dataset Contract

```json
{
  "alleles": [
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*01:01",
    "HLA-B*07:02",
    "HLA-B*44:02"
  ],
  "broad_numeric_families": [
    "IC50",
    "KD",
    "KD (~IC50)",
    "KD (~EC50)",
    "EC50"
  ],
  "measurement_profile": "numeric_no_qualitative",
  "probe_peptides": [
    "SLLQHLIGL",
    "FLRYLLFGI",
    "NFLIKFLLI"
  ],
  "qualifier_filter": "all"
}
```

## Training

```json
{
  "epochs": 5,
  "ranking_losses": false,
  "synthetic_negatives": false,
  "warm_start": "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"
}
```

## Tested Conditions

```json
[
  {
    "batch_size": 140,
    "description": "Directness P00 legacy triple/triple",
    "design_id": "DP00",
    "family": "presto"
  }
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
