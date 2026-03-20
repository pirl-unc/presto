# EXP-16 Main-Path Baseline Rebuild

- Agent: `codex`
- Source script: `scripts/benchmark_distributional_ba_v6_backbone_compare.py`
- Created: `2026-03-15T10:05:23.705092`

## Dataset Contract

```json
{
  "measurement_profile": "numeric",
  "panel": [
    "HLA-A*02:01",
    "HLA-A*24:02"
  ],
  "qualifier_filter": "all",
  "source": "data/merged_deduped.tsv",
  "split": "peptide_group_80_10_10_seed42"
}
```

## Training

```json
{
  "batch_size": 256,
  "config_version": "v6",
  "epochs": 50,
  "gpu": "H100!",
  "lr": "1e-3",
  "seed": 42,
  "warm_start": false,
  "weight_decay": 0.01
}
```

## Tested Conditions

```json
[
  {
    "cond_id": 1,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 2,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 3,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 4,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 5,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 6,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 7,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 8,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 9,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 10,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 11,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 12,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 13,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 14,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 15,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 16,
    "content_conditioned": false,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 1,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 2,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 3,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 4,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 5,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 6,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 7,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 8,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 9,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 10,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 11,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 12,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 13,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 14,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 15,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 16,
    "content_conditioned": true,
    "encoder_backbone": "historical_ablation"
  },
  {
    "cond_id": 1,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 2,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 3,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 4,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 5,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 6,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 7,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 8,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 9,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 10,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 11,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 12,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 13,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 14,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 15,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 16,
    "content_conditioned": false,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 1,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 2,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 3,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 4,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 5,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 6,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 7,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 8,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 9,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 10,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 11,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 12,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 13,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 14,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 15,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  },
  {
    "cond_id": 16,
    "content_conditioned": true,
    "encoder_backbone": "groove"
  }
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
- Reproducibility bundle: [`reproduce/`](./reproduce/)
