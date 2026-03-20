# Presto vs Groove A2 Head-to-Head Sweep (v3)

- Agent: `codex`
- Source script: `scripts/benchmark_presto_bakeoff.py`
- Created: `2026-03-12T13:00:22.444709`

## Dataset Contract

```json
{
  "measurement_profile": "numeric_no_qualitative",
  "panel": [
    "HLA-A*02:01",
    "HLA-A*24:02"
  ],
  "qualifier_filter": "all",
  "synthetics": false
}
```

## Training

```json
{
  "batch_size": 128,
  "epochs": 20,
  "groove_lr": "1e-3",
  "presto_lr": "varies (1e-3 or 2.8e-4)",
  "seed": 42,
  "warm_start_checkpoint": "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"
}
```

## Tested Conditions

```json
[
  "P5",
  "P6",
  "P7",
  "P8"
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
