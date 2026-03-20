# Clean Distributional vs Regression BA Heads Round 1

- Agent: `codex`
- Source script: `experiments/2026-03-13_1305_codex_clean-distributional-ba-heads-round1/code/run_all.py`
- Created: `2026-03-13T12:57:58.672316`

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
  "assay_families": [
    "IC50",
    "KD",
    "KD (~IC50)",
    "KD (~EC50)",
    "EC50"
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
  "epochs": 10,
  "gpu": "H100!",
  "lr": 0.0001,
  "optimizer": "AdamW",
  "schedule": "warmup_cosine",
  "weight_decay": 0.01
}
```

## Tested Conditions

```json
[
  "mhcflurry_{50k,200k}",
  "log_mse_{50k,200k}",
  "twohot_d2_logit_{50k,200k}_K{64,128}",
  "hlgauss_d2_logit_{50k,200k}_K{64,128}"
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
- Reproducibility bundle: [`reproduce/`](./reproduce/)
