# Full Class I Best Hits (6 conditions x 3 seeds, mhcseqs grooves)

- Agent: `claude`
- Source script: `experiments/2026-03-28_claude_class1-best-hits/launch.py`
- Created: `2026-03-28T09:05:16.186798`

## Dataset Contract

```json
{
  "measurement_profile": "numeric_no_qualitative",
  "probe_alleles": [
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*01:01",
    "HLA-B*07:02",
    "HLA-B*44:02"
  ],
  "probe_peptides": [
    "SLLQHLIGL",
    "FLRYLLFGI",
    "NFLIKFLLI"
  ],
  "qualifier_filter": "all",
  "sequence_resolution": "mhcseqs_first_with_index_fallback",
  "split_seed": 42,
  "train_all_alleles": true,
  "train_mhc_class_filter": "I"
}
```

## Training

```json
{
  "affinity_target_encoding": "mhcflurry",
  "batch_size": 256,
  "binding_core_lengths": [
    8,
    9,
    10,
    11
  ],
  "binding_core_refinement": "shared",
  "contrastive": false,
  "d_model": 128,
  "epochs": 50,
  "kd_grouping_mode": "split_kd_proxy",
  "lr": "3e-4",
  "lr_schedule": "warmup_cosine",
  "max_affinity_nM": 100000,
  "n_heads": 4,
  "n_layers": 2,
  "seeds": [
    42,
    43,
    44
  ],
  "split_seed": 42,
  "synthetic_negatives": false,
  "warm_start": "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt",
  "weight_decay": 0.01
}
```

## Tested Conditions

```json
[
  {
    "description": "d128 DAG heads_only pretrain lr=3e-4 warmup [ctrl=L2]",
    "design_id": "F1",
    "extra_args": [
      "--affinity-assay-residual-mode",
      "dag_prep_readout_leaf",
      "--affinity-loss-mode",
      "assay_heads_only"
    ],
    "uses_pretrain": true
  },
  {
    "description": "d128 A07 heads_only pretrain lr=3e-4 warmup",
    "design_id": "F2",
    "extra_args": [
      "--affinity-assay-residual-mode",
      "shared_base_factorized_context_plus_segment_residual",
      "--affinity-loss-mode",
      "assay_heads_only"
    ],
    "uses_pretrain": true
  },
  {
    "description": "d128 DAG full pretrain lr=3e-4 warmup",
    "design_id": "F3",
    "extra_args": [
      "--affinity-assay-residual-mode",
      "dag_prep_readout_leaf",
      "--affinity-loss-mode",
      "full"
    ],
    "uses_pretrain": true
  },
  {
    "description": "d128 DAG heads_only no-pretrain lr=3e-4 warmup",
    "design_id": "F4",
    "extra_args": [
      "--affinity-assay-residual-mode",
      "dag_prep_readout_leaf",
      "--affinity-loss-mode",
      "assay_heads_only"
    ],
    "uses_pretrain": false
  },
  {
    "description": "d128 DAG full no-pretrain lr=3e-4 warmup",
    "design_id": "F5",
    "extra_args": [
      "--affinity-assay-residual-mode",
      "dag_prep_readout_leaf",
      "--affinity-loss-mode",
      "full"
    ],
    "uses_pretrain": false
  },
  {
    "description": "d128 A03 heads_only pretrain lr=3e-4 warmup",
    "design_id": "F6",
    "extra_args": [
      "--affinity-assay-residual-mode",
      "shared_base_segment_residual",
      "--affinity-loss-mode",
      "assay_heads_only"
    ],
    "uses_pretrain": true
  }
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
- Reproducibility bundle: [`reproduce/`](./reproduce/)
