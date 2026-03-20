# PF07 All-Class-I Cleanup Validation Rerun

## Objective

Rerun the latest all-class-I PF07 sweep on the cleaned `mhcseqs`-owned exact-MHC input path and verify that the post-cleanup code is not worse than the pre-cleanup path under the same experimental contract.

## Why This Rerun Exists

- The shared code now routes exact class-I groove inputs through `mhcseqs` more cleanly.
- The most recent experiment family affected by that change was:
  - `2026-03-17_1404_codex_pf07-all-classI-1ep-pretrain-epoch-sweep`
- That earlier family never fully closed locally:
  - only the three `10`-epoch runs were fetched
  - no final aggregated summary bundle exists
- A clean rerun is therefore the best regression check and the cleanest replacement baseline for this contract.

## Fixed Contract

- Main `Presto` path via `focused_binding_run`
- Inputs only:
  - `nflank`
  - `peptide`
  - `cflank`
  - `mhc_a`
  - `mhc_b`
- Dataset:
  - `data/merged_deduped.tsv`
  - rebuilt canonical dataset with bag-aware unification
  - `source_filter=iedb`
  - `train_mhc_class_filter=I`
  - `train_all_alleles=true`
  - `measurement_profile=numeric_no_qualitative`
  - `qualifier_filter=all`
  - peptide-group `80/10/10` split
  - split seed `42`
  - train seed `43`
- Sequence resolution:
  - `mhcseqs` exact groove ownership first
  - local fallback only for non-catalog/runtime cases
- Probes:
  - alleles: `HLA-A*02:01`, `HLA-A*24:02`
  - peptides: `SLLQHLIGL`, `FLRYLLFGI`, `NFLIKFLLI`, `IMLEGETKL`

## Pretraining Contract

- Warm start:
  - `/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt`
- Pretrain history:
  - 1 epoch of MHC class/species pretraining
  - `d_model=32`
  - `n_layers=2`
  - `n_heads=4`

## Downstream Model Grid

Three variants, each at `10 / 25 / 50` epochs:

1. `pf07_control_constant`
   - `affinity_assay_residual_mode=shared_base_factorized_context_plus_segment_residual`
2. `pf07_dag_method_leaf_constant`
   - `affinity_assay_residual_mode=dag_method_leaf`
3. `pf07_dag_prep_readout_leaf_constant`
   - `affinity_assay_residual_mode=dag_prep_readout_leaf`

Shared downstream settings:

- `d_model=32`
- `n_layers=2`
- `n_heads=4`
- `batch_size=256`
- `lr=1e-3`
- `lr_schedule=constant`
- `weight_decay=0.01`
- `affinity_loss_mode=full`
- `affinity_target_encoding=mhcflurry`
- `kd_grouping_mode=split_kd_proxy`
- `binding_kinetic_input_mode=affinity_vec`
- `binding_direct_segment_mode=off`
- `binding_core_lengths=8,9,10,11`
- `binding_core_refinement=shared`
- `peptide_pos_mode=concat_start_end_frac`
- `groove_pos_mode=concat_start_end_frac`
- `max_affinity_nM=100000`
- no synthetic negatives
- requested GPU `H100!`

## Expected Outputs

- experiment dir:
  - `experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun`
- canonical launcher:
  - `code/launch.py`
- run metadata:
  - `manifest.json`
- fetched raw runs:
  - `results/runs/`
- derived summaries:
  - `results/condition_summary.csv`
  - `results/model_summary.csv`
  - `results/epoch_metrics_by_condition.csv` if available
- reproducibility bundle:
  - `reproduce/`

## Comparison Plan

Primary comparison target:

- overlapping `10`-epoch runs from `2026-03-17_1404_codex_pf07-all-classI-1ep-pretrain-epoch-sweep`

Metrics to compare:

- test Spearman
- test AUROC
- test AUPRC
- test RMSE log10

Interpretation rule:

- If rerun metrics are equal or better within normal run variance, treat the cleanup as non-regressive.
- If rerun metrics are worse across the board, treat the cleanup as suspect and inspect whether:
  - the new exact-input path changed effective MHC tokenization
  - the rebuilt data contract drifted
  - the warm-start checkpoint or launch contract changed

## Closure Requirements

- Fetch all `9 / 9` runs locally
- Aggregate summaries locally
- Update the new experiment `README.md`
- Update `experiments/experiment_log.md`
- Explicitly state whether the cleanup validation passed
