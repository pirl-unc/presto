# Clean Distributional BA Heads Round 2

- Agent: `codex`
- Source script: `experiments/2026-03-13_1530_codex_clean-distributional-ba-heads-round2/run_all.py`
- Created: `2026-03-13T13:43:20.647476`

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
  "assays": [
    "IC50",
    "direct KD",
    "KD (~IC50)",
    "KD (~EC50)",
    "EC50"
  ],
  "qualifiers": "all",
  "source": "data/merged_deduped.tsv",
  "split": "peptide-group 80/10/10 seed=42"
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
  "mhcflurry_50k",
  "mhcflurry_200k",
  "log_mse_50k",
  "log_mse_200k",
  "twohot_d2_logit_50k_K64",
  "twohot_d2_logit_50k_K128",
  "twohot_d2_logit_200k_K64",
  "twohot_d2_logit_200k_K128",
  "hlgauss_d2_logit_50k_K64",
  "hlgauss_d2_logit_50k_K128",
  "hlgauss_d2_logit_200k_K64",
  "hlgauss_d2_logit_200k_K128"
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
- Reproducibility bundle: [`reproduce/`](./reproduce/)
