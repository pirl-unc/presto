# Presto vs Groove Transformer Bakeoff

- Agent: `claude`
- Source script: `scripts/benchmark_presto_bakeoff.py`
- Created: `2026-03-11T12:44:33.596856`

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
  "C5",
  "C6"
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
