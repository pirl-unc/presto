# Assay Head / KD Grouping Bakeoff Round 1

- Agent: `codex`
- Source script: `scripts/benchmark_assay_head_bakeoff.py`
- Created: `2026-03-11T22:44:21.528010`

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
  "affinity_loss_mode": "assay_heads_only",
  "batch_size": 140,
  "epochs": 3,
  "ranking_losses": false,
  "synthetic_negatives": false,
  "warm_start": "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"
}
```

## Tested Conditions

```json
[
  {
    "description": "P04 positional base + pooled_single_output + merged_kd",
    "design_id": "A00",
    "extra_args": [
      "--d-model",
      "128",
      "--peptide-pos-mode",
      "concat_start_end_frac",
      "--groove-pos-mode",
      "concat_start_end_frac",
      "--binding-core-lengths",
      "8,9,10,11",
      "--binding-core-refinement",
      "shared",
      "--affinity-assay-mode",
      "legacy",
      "--affinity-assay-residual-mode",
      "pooled_single_output",
      "--kd-grouping-mode",
      "merged_kd",
      "--affinity-target-encoding",
      "log10",
      "--max-affinity-nm",
      "50000"
    ]
  },
  {
    "description": "P04 positional base + pooled_single_output + split_kd_proxy",
    "design_id": "A01",
    "extra_args": [
      "--d-model",
      "128",
      "--peptide-pos-mode",
      "concat_start_end_frac",
      "--groove-pos-mode",
      "concat_start_end_frac",
      "--binding-core-lengths",
      "8,9,10,11",
      "--binding-core-refinement",
      "shared",
      "--affinity-assay-mode",
      "legacy",
      "--affinity-assay-residual-mode",
      "pooled_single_output",
      "--kd-grouping-mode",
      "split_kd_proxy",
      "--affinity-target-encoding",
      "log10",
      "--max-affinity-nm",
      "50000"
    ]
  },
  {
    "description": "P04 positional base + shared_base_segment_residual + merged_kd",
    "design_id": "A02",
    "extra_args": [
      "--d-model",
      "128",
      "--peptide-pos-mode",
      "concat_start_end_frac",
      "--groove-pos-mode",
      "concat_start_end_frac",
      "--binding-core-lengths",
      "8,9,10,11",
      "--binding-core-refinement",
      "shared",
      "--affinity-assay-mode",
      "legacy",
      "--affinity-assay-residual-mode",
      "shared_base_segment_residual",
      "--kd-grouping-mode",
      "merged_kd",
      "--affinity-target-encoding",
      "log10",
      "--max-affinity-nm",
      "50000"
    ]
  },
  {
    "description": "P04 positional base + shared_base_segment_residual + split_kd_proxy",
    "design_id": "A03",
    "extra_args": [
      "--d-model",
      "128",
      "--peptide-pos-mode",
      "concat_start_end_frac",
      "--groove-pos-mode",
      "concat_start_end_frac",
      "--binding-core-lengths",
      "8,9,10,11",
      "--binding-core-refinement",
      "shared",
      "--affinity-assay-mode",
      "legacy",
      "--affinity-assay-residual-mode",
      "shared_base_segment_residual",
      "--kd-grouping-mode",
      "split_kd_proxy",
      "--affinity-target-encoding",
      "log10",
      "--max-affinity-nm",
      "50000"
    ]
  },
  {
    "description": "P04 positional base + factorized_context_residual + merged_kd",
    "design_id": "A04",
    "extra_args": [
      "--d-model",
      "128",
      "--peptide-pos-mode",
      "concat_start_end_frac",
      "--groove-pos-mode",
      "concat_start_end_frac",
      "--binding-core-lengths",
      "8,9,10,11",
      "--binding-core-refinement",
      "shared",
      "--affinity-assay-mode",
      "legacy",
      "--affinity-assay-residual-mode",
      "shared_base_factorized_context_residual",
      "--kd-grouping-mode",
      "merged_kd",
      "--affinity-target-encoding",
      "log10",
      "--max-affinity-nm",
      "50000"
    ]
  },
  {
    "description": "P04 positional base + factorized_context_residual + split_kd_proxy",
    "design_id": "A05",
    "extra_args": [
      "--d-model",
      "128",
      "--peptide-pos-mode",
      "concat_start_end_frac",
      "--groove-pos-mode",
      "concat_start_end_frac",
      "--binding-core-lengths",
      "8,9,10,11",
      "--binding-core-refinement",
      "shared",
      "--affinity-assay-mode",
      "legacy",
      "--affinity-assay-residual-mode",
      "shared_base_factorized_context_residual",
      "--kd-grouping-mode",
      "split_kd_proxy",
      "--affinity-target-encoding",
      "log10",
      "--max-affinity-nm",
      "50000"
    ]
  },
  {
    "description": "P04 positional base + factorized_context_plus_segment_residual + merged_kd",
    "design_id": "A06",
    "extra_args": [
      "--d-model",
      "128",
      "--peptide-pos-mode",
      "concat_start_end_frac",
      "--groove-pos-mode",
      "concat_start_end_frac",
      "--binding-core-lengths",
      "8,9,10,11",
      "--binding-core-refinement",
      "shared",
      "--affinity-assay-mode",
      "legacy",
      "--affinity-assay-residual-mode",
      "shared_base_factorized_context_plus_segment_residual",
      "--kd-grouping-mode",
      "merged_kd",
      "--affinity-target-encoding",
      "log10",
      "--max-affinity-nm",
      "50000"
    ]
  },
  {
    "description": "P04 positional base + factorized_context_plus_segment_residual + split_kd_proxy",
    "design_id": "A07",
    "extra_args": [
      "--d-model",
      "128",
      "--peptide-pos-mode",
      "concat_start_end_frac",
      "--groove-pos-mode",
      "concat_start_end_frac",
      "--binding-core-lengths",
      "8,9,10,11",
      "--binding-core-refinement",
      "shared",
      "--affinity-assay-mode",
      "legacy",
      "--affinity-assay-residual-mode",
      "shared_base_factorized_context_plus_segment_residual",
      "--kd-grouping-mode",
      "split_kd_proxy",
      "--affinity-target-encoding",
      "log10",
      "--max-affinity-nm",
      "50000"
    ]
  }
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
