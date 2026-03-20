# PF07 All-Class-I 1ep-Pretrain Epoch Sweep

- Agent: `codex`
- Source script: `experiments/2026-03-17_1404_codex_pf07-all-classI-1ep-pretrain-epoch-sweep/code/launch.py`
- Created: `2026-03-17T17:12:20.809837`

## Dataset Contract

```json
{
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
  "probe_alleles": [
    "HLA-A*02:01",
    "HLA-A*24:02"
  ],
  "probe_peptides": [
    "SLLQHLIGL",
    "FLRYLLFGI",
    "NFLIKFLLI",
    "IMLEGETKL"
  ],
  "qualifier_filter": "all",
  "sequence_resolution": "mhcseqs_first_with_index_fallback",
  "source": "data/merged_deduped.tsv",
  "source_filter": "iedb",
  "source_refresh": "canonical rebuild 2026-03-17",
  "split_policy": "peptide_group_80_10_10",
  "split_seed": 42,
  "train_all_alleles": true,
  "train_mhc_class_filter": "I",
  "train_seed": 43
}
```

## Training

```json
{
  "downstream": {
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
    "epoch_grid": [
      10,
      25,
      50
    ],
    "groove_pos_mode": "concat_start_end_frac",
    "kd_grouping_mode": "split_kd_proxy",
    "max_affinity_nM": 100000,
    "n_heads": 4,
    "n_layers": 2,
    "optimizer": "AdamW",
    "peptide_pos_mode": "concat_start_end_frac",
    "requested_gpu": "H100!",
    "synthetic_negatives": false,
    "weight_decay": 0.01
  },
  "pretraining": {
    "d_model": 32,
    "mode": "mhc_pretrain",
    "n_heads": 4,
    "n_layers": 2,
    "warm_start_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "warm_start_epochs": 1
  }
}
```

## Tested Conditions

```json
[
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_control_constant",
    "description": "Flat honest PF07 control on rebuilt all-class-I numeric data",
    "epoch_budget": 10,
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_control_constant",
    "description": "Flat honest PF07 control on rebuilt all-class-I numeric data",
    "epoch_budget": 25,
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_control_constant",
    "description": "Flat honest PF07 control on rebuilt all-class-I numeric data",
    "epoch_budget": 50,
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_method_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_method_leaf_constant",
    "description": "Method-leaf DAG on rebuilt all-class-I numeric data",
    "epoch_budget": 10,
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_method_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_method_leaf_constant",
    "description": "Method-leaf DAG on rebuilt all-class-I numeric data",
    "epoch_budget": 25,
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_method_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_method_leaf_constant",
    "description": "Method-leaf DAG on rebuilt all-class-I numeric data",
    "epoch_budget": 50,
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_prep_readout_leaf_constant",
    "description": "Prep/readout-leaf DAG on rebuilt all-class-I numeric data",
    "epoch_budget": 10,
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_prep_readout_leaf_constant",
    "description": "Prep/readout-leaf DAG on rebuilt all-class-I numeric data",
    "epoch_budget": 25,
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_prep_readout_leaf_constant",
    "description": "Prep/readout-leaf DAG on rebuilt all-class-I numeric data",
    "epoch_budget": 50,
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  }
]
```

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
- Reproducibility bundle: [`reproduce/`](./reproduce/)
