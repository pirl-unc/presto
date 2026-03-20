# Broad Frontier 5-Epoch Bakeoff

- Agent: `codex`
- Source script: `scripts/benchmark_broad_frontier_5ep.py`
- Created: `2026-03-12T08:49:38.438686`

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
  },
  {
    "batch_size": 140,
    "description": "Directness P01 shared_base_segment_residual triple/triple",
    "design_id": "DP01",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Directness P05 shared_base_segment_residual triple_plus_abs/triple",
    "design_id": "DP05",
    "family": "presto"
  },
  {
    "batch_size": 256,
    "description": "Directness G1 groove transformer",
    "design_id": "DG1",
    "family": "groove"
  },
  {
    "batch_size": 140,
    "description": "Positional P00 start_only",
    "design_id": "PP00",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Positional P01 end_only",
    "design_id": "PP01",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Positional P02 start_plus_end",
    "design_id": "PP02",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Positional P03 concat(start,end)",
    "design_id": "PP03",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Positional P04 concat(start,end,frac)",
    "design_id": "PP04",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Positional P05 mlp(concat(start,end))",
    "design_id": "PP05",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Positional P06 mlp(concat(start,end,frac))",
    "design_id": "PP06",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Positional P07 triple_baseline",
    "design_id": "PP07",
    "family": "presto"
  },
  {
    "batch_size": 256,
    "description": "Positional G00 start_only",
    "design_id": "PG00",
    "family": "groove"
  },
  {
    "batch_size": 256,
    "description": "Positional G01 end_only",
    "design_id": "PG01",
    "family": "groove"
  },
  {
    "batch_size": 256,
    "description": "Positional G02 start_plus_end",
    "design_id": "PG02",
    "family": "groove"
  },
  {
    "batch_size": 256,
    "description": "Positional G03 concat(start,end)",
    "design_id": "PG03",
    "family": "groove"
  },
  {
    "batch_size": 256,
    "description": "Positional G04 concat(start,end,frac)",
    "design_id": "PG04",
    "family": "groove"
  },
  {
    "batch_size": 256,
    "description": "Positional G05 mlp(concat(start,end))",
    "design_id": "PG05",
    "family": "groove"
  },
  {
    "batch_size": 256,
    "description": "Positional G06 mlp(concat(start,end,frac))",
    "design_id": "PG06",
    "family": "groove"
  },
  {
    "batch_size": 256,
    "description": "Positional G07 triple_baseline",
    "design_id": "PG07",
    "family": "groove"
  },
  {
    "batch_size": 140,
    "description": "Assay A00 pooled_single_output merged_kd",
    "design_id": "A00",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Assay A01 pooled_single_output split_kd_proxy",
    "design_id": "A01",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Assay A02 shared_base_segment_residual merged_kd",
    "design_id": "A02",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Assay A03 shared_base_segment_residual split_kd_proxy",
    "design_id": "A03",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Assay A04 factorized_context merged_kd",
    "design_id": "A04",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Assay A05 factorized_context split_kd_proxy",
    "design_id": "A05",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Assay A06 factorized_context_plus_segment merged_kd",
    "design_id": "A06",
    "family": "presto"
  },
  {
    "batch_size": 140,
    "description": "Assay A07 factorized_context_plus_segment split_kd_proxy",
    "design_id": "A07",
    "family": "presto"
  }
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
