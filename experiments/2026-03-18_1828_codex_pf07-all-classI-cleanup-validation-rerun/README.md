# PF07 All-Class-I Cleanup Validation Rerun

- Agent: `codex`
- Source script: `experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun/code/launch.py`
- Created: `2026-03-18T18:29:50.218374`

## Status

- Modal rerun is not closed.
- The original remote launch was interrupted by exhausted Modal billing-cycle credits.
- A local continuation path is now supported through the same experiment-local launcher:
  - `python experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun/code/launch.py --backend local --device mps`
- Runtime note:
  - explicit `mps` selection is now supported for Apple Silicon
  - `scripts/focused_binding_probe.py` now uses seeded `manual_dropout` by default via `--mps-safe-mode auto`, so the dropout implementation itself is aligned across CPU and MPS unless you explicitly opt out with `off`
  - the later reduced all-class-I validation in `2026-03-19_1505_codex_cpu-vs-mps-allclass1-manualdropout-validation` completed cleanly on both backends with no fallback warnings
  - the checked-in `local_resume.sh` now defaults to `auto`, which means manual dropout on all devices under the current contract
  - `PRESTO_LOCAL_DEVICE=mps` is now a real opt-in path for this larger unfinished sweep
- The canonical repo-local warm-start checkpoint for local resume is:
  - `experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt`
- Local runs write directly into:
  - `results/runs/<run_id>`
- Local aggregation is automatic after a fully successful local launcher pass.

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
  "comparison_target": "2026-03-17_1404_codex_pf07-all-classI-1ep-pretrain-epoch-sweep",
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
  "train_seed": 43,
  "validation_purpose": "post_mhcseqs_cleanup_rerun"
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
    "warm_start_checkpoint_local": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "warm_start_checkpoint_modal": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
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

## Resume

Another machine can continue this family without Modal from the checked-in experiment directory.

Apple Silicon example:

```bash
python experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun/code/launch.py \
  --out-dir experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun \
  --backend local \
  --device mps
```

CPU example:

```bash
python experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun/code/launch.py \
  --out-dir experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun \
  --backend local \
  --device cpu
```

Notes:

- The launcher preserves the same condition matrix and writes the same artifact contract expected by `scripts/aggregate_summary_runs.py`.
- Local resume uses the repo-local 1-epoch MHC pretrain checkpoint, not the Modal volume path.
- `local_resume.sh` now also forwards `PRESTO_LOCAL_MPS_SAFE_MODE`, so you can choose:
  - `PRESTO_LOCAL_MPS_SAFE_MODE=auto` for the default MPS-safe manual dropout on Apple Silicon
  - `PRESTO_LOCAL_MPS_SAFE_MODE=manual_dropout` if you want the same dropout implementation on CPU and MPS for parity checks
- The original Modal reproducibility bundle remains under [`reproduce/`](./reproduce/); use [`reproduce/local_resume.sh`](./reproduce/local_resume.sh) for the portable local-resume wrapper.
- Verified locally in this pass:
  - `--backend local --device mps` dry-run succeeds and emits the correct local commands
  - `--backend local --device cpu` dry-run succeeds
  - post-fix tiny CPU and MPS validations both completed
  - seeded manual-dropout parity still shows some metric drift, but the remaining gap is no longer caused by a different dropout contract

## Notes

- Launcher-created stub. Add results, plots, and takeaways here after collection.
- Reproducibility bundle: [`reproduce/`](./reproduce/)
