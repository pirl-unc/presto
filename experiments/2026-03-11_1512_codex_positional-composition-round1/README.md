# Positional Composition Round 1

- Agent: `codex`
- Source script: `scripts/benchmark_positional_composition.py`
- Created: `2026-03-11T15:12:19.885573`

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
  "excluded_assays": [
    "qualitative binding"
  ],
  "expected_split": {
    "train_rows": 32855,
    "val_rows": 8194
  },
  "included_assays": [
    "IC50",
    "KD",
    "KD (~IC50)",
    "KD (~EC50)",
    "EC50"
  ],
  "measurement_profile": "numeric_no_qualitative",
  "qualifier_filter": "all",
  "source": "data/merged_deduped.tsv"
}
```

## Training

```json
{
  "epochs": 3,
  "groove_batch_size": 256,
  "presto_batch_size": 140,
  "ranking": false,
  "synthetics": false,
  "warm_start": "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"
}
```

## Tested Conditions

```json
[
  {
    "description": "Presto P03-style, peptide=start_only, groove=start_only",
    "design_id": "P00",
    "family": "presto"
  },
  {
    "description": "G1-style groove transformer, peptide=start_only, groove=start_only",
    "design_id": "G00",
    "family": "groove_transformer"
  },
  {
    "description": "Presto P03-style, peptide=end_only, groove=end_only",
    "design_id": "P01",
    "family": "presto"
  },
  {
    "description": "G1-style groove transformer, peptide=end_only, groove=end_only",
    "design_id": "G01",
    "family": "groove_transformer"
  },
  {
    "description": "Presto P03-style, peptide=start_plus_end, groove=start_plus_end",
    "design_id": "P02",
    "family": "presto"
  },
  {
    "description": "G1-style groove transformer, peptide=start_plus_end, groove=start_plus_end",
    "design_id": "G02",
    "family": "groove_transformer"
  },
  {
    "description": "Presto P03-style, peptide=concat_start_end, groove=concat_start_end",
    "design_id": "P03",
    "family": "presto"
  },
  {
    "description": "G1-style groove transformer, peptide=concat_start_end, groove=concat_start_end",
    "design_id": "G03",
    "family": "groove_transformer"
  },
  {
    "description": "Presto P03-style, peptide=concat_start_end_frac, groove=concat_start_end_frac",
    "design_id": "P04",
    "family": "presto"
  },
  {
    "description": "G1-style groove transformer, peptide=concat_start_end_frac, groove=concat_start_end_frac",
    "design_id": "G04",
    "family": "groove_transformer"
  },
  {
    "description": "Presto P03-style, peptide=mlp_start_end, groove=mlp_start_end",
    "design_id": "P05",
    "family": "presto"
  },
  {
    "description": "G1-style groove transformer, peptide=mlp_start_end, groove=mlp_start_end",
    "design_id": "G05",
    "family": "groove_transformer"
  },
  {
    "description": "Presto P03-style, peptide=mlp_start_end_frac, groove=mlp_start_end_frac",
    "design_id": "P06",
    "family": "presto"
  },
  {
    "description": "G1-style groove transformer, peptide=mlp_start_end_frac, groove=mlp_start_end_frac",
    "design_id": "G06",
    "family": "groove_transformer"
  },
  {
    "description": "Presto P03-style, peptide=triple_baseline, groove=triple_baseline",
    "design_id": "P07",
    "family": "presto"
  },
  {
    "description": "G1-style groove transformer, peptide=triple_baseline, groove=triple_baseline",
    "design_id": "G07",
    "family": "groove_transformer"
  }
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
