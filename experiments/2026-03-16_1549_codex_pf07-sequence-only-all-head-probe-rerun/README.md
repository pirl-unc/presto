# PF07 Sequence-Only All-Head Probe Rerun

- Agent: `codex`
- Source script: `experiments/2026-03-16_1549_codex_pf07-sequence-only-all-head-probe-rerun/code/launch.py`
- Created: `2026-03-16T15:50:55.544896`

## Dataset Contract

```json
{
  "alleles": [
    "HLA-A*02:01",
    "HLA-A*24:02"
  ],
  "assay_families_supervised": [
    "IC50",
    "KD",
    "KD(~IC50)",
    "KD(~EC50)",
    "EC50"
  ],
  "assay_selector_inputs_forbidden": true,
  "input_fields": [
    "nflank",
    "peptide",
    "cflank",
    "mhc_a",
    "mhc_b"
  ],
  "measurement_profile": "numeric_no_qualitative",
  "probe_peptides": [
    "SLLQHLIGL",
    "FLRYLLFGI",
    "NFLIKFLLI",
    "IMLEGETKL"
  ],
  "qualifier_filter": "all",
  "source": "data/merged_deduped.tsv",
  "split_policy": "peptide_group_80_10_10",
  "split_seed": 42,
  "train_seed": 43
}
```

## Training

```json
{
  "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
  "affinity_loss_mode": "full",
  "affinity_target_encoding": "mhcflurry",
  "batch_size": 256,
  "binding_core_lengths": [
    8,
    9,
    10,
    11
  ],
  "binding_core_refinement": "shared",
  "binding_direct_segment_mode": "off",
  "binding_kinetic_input_mode": "affinity_vec",
  "d_model": 32,
  "epochs": 50,
  "groove_pos_mode": "concat_start_end_frac",
  "kd_grouping_mode": "split_kd_proxy",
  "max_affinity_nM": 100000,
  "n_heads": 4,
  "n_layers": 2,
  "optimizer": "AdamW",
  "peptide_pos_mode": "concat_start_end_frac",
  "probe_artifact_schema": [
    "KD_nM",
    "IC50_nM",
    "EC50_nM",
    "KD_proxy_ic50_nM",
    "KD_proxy_ec50_nM",
    "binding_affinity_probe_kd"
  ],
  "ranking_losses": false,
  "requested_gpu": "H100!",
  "synthetic_negatives": false,
  "warm_start": "",
  "weight_decay": 0.01
}
```

## Tested Conditions

```json
[
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "PF07_ctrl_lr1e3_constant",
    "description": "Current PF07 positive control: lr=1e-3, constant schedule",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "PF07_lr2p8e4_warmup_cosine",
    "description": "Historical A07 validation winner: lr=2.8e-4, warmup_cosine",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.00028,
    "lr_schedule": "warmup_cosine",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "PF07_lr2p8e4_onecycle",
    "description": "Historical A07 near-tie: lr=2.8e-4, onecycle",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.00028,
    "lr_schedule": "onecycle",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "PF07_lr1e4_warmup_cosine",
    "description": "Historical A07 lower-LR warmup comparator: lr=1e-4, warmup_cosine",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.0001,
    "lr_schedule": "warmup_cosine",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "PF07_lr1e4_constant",
    "description": "Historical A07 lower-LR constant comparator: lr=1e-4, constant schedule",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.0001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  }
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
- Reproducibility bundle: [`reproduce/`](./reproduce/)
