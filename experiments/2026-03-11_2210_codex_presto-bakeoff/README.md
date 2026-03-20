# Presto vs Groove Transformer Bakeoff

- Agent: `codex`
- Source script: `scripts/benchmark_presto_bakeoff.py`
- Created: `2026-03-11T22:10:47.794365`

## Dataset Contract

```json
{
  "measurement_profile": "numeric_no_qualitative",
  "panel": [
    "HLA-A*02:01",
    "HLA-A*24:02"
  ],
  "qualifier_filter": "all"
}
```

## Training

```json
{
  "batch_size": 256,
  "epochs": 20,
  "lr": "1e-3",
  "seed": 42,
  "synthetics": false,
  "warm_start": null
}
```

## Tested Conditions

```json
[
  "C1",
  "C2",
  "C3",
  "C4",
  "C5"
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
