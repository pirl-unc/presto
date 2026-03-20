# PF07 Output-Tying Weight Sweep (2026-03-16)

## Goal

Measure whether weak output-side consistency regularization improves the main-path PF07 affinity model on the fixed EXP-21-style 2-allele numeric affinity contract.

## Fixed Contract

- Model path: main `Presto` affinity-only runner
- Base architecture: `PF07`
- Inputs only: `nflank`, `peptide`, `cflank`, `mhc_a`, `mhc_b`
- Assay-selector inputs: forbidden
- Dataset: `data/merged_deduped.tsv`
- Alleles: `HLA-A*02:01`, `HLA-A*24:02`
- Measurement profile: `numeric_no_qualitative`
- Qualifier filter: `all`
- Split policy: peptide-group `80/10/10`
- Split seed: `42`
- Train seed: `43`
- Epochs: `50`
- Batch size: `256`
- Optimizer: `AdamW`
- LR schedule: `constant`
- LR: `1e-3`
- Weight decay: `0.01`
- Affinity target encoding: `mhcflurry`
- Affinity loss mode: `full`
- Affinity assay residual mode: `shared_base_factorized_context_plus_segment_residual`
- KD grouping mode: `split_kd_proxy`
- Binding kinetic input mode: `affinity_vec`
- Binding direct segment mode: `off`
- Synthetic negatives: off
- Contrastive losses: off

## Regularizers Under Test

### KD family tie

Weak tie among:
- `KD_nM`
- `KD_proxy_ic50_nM`
- `KD_proxy_ec50_nM`

Implementation:
- anchor-style smooth-L1 in `log10(nM)` space
- off by default

### Proxy cross-family tie

Weaker tie between:
- `IC50_nM <-> KD_proxy_ic50_nM`
- `EC50_nM <-> KD_proxy_ec50_nM`

Implementation:
- smooth-L1 in `log10(nM)` space
- lower intended scale than the KD-family tie

## Grid

- `binding_kd_family_consistency_weight`: `0.0`, `0.0025`, `0.01`, `0.04`
- `binding_proxy_cross_consistency_weight`: `0.0`, `0.001`, `0.004`
- `binding_output_consistency_beta`: fixed at `0.25`

Total runs:
- `12`

## Why This Grid

- Includes a clean no-regularization baseline.
- Keeps all added weights materially below the supervised loss scale.
- Tests cross-family ties at weaker values than KD-family ties.
- Covers a small-to-moderate range without turning this into a large sweep.

## Metrics To Compare

Primary:
- held-out test Spearman on the aggregate numeric affinity contract

Secondary:
- Pearson
- RMSE log10
- AUROC
- AUPRC
- thresholded `<=500 nM` metrics
- probe-head agreement on the four canonical probe peptides

Additional diagnostics:
- `reg_binding_kd_family_consistency_raw`
- `reg_binding_proxy_cross_consistency_raw`
- head-gap summaries written by the runner

## Acceptance Criteria

- Any promoted setting must beat or at least match the PF07 control on held-out test Spearman without a clear RMSE collapse.
- If multiple settings tie on Spearman, prefer the one with better RMSE and more stable head agreement.
- If all regularized settings regress, keep the control and conclude that simple weak tying is not enough on this contract.

## Expected Artifacts

- experiment-local launcher at `code/launch.py`
- `manifest.json`
- `launch_logs/`
- fetched raw runs under `results/runs/`
- aggregated `results/*.csv` and `results/*.json`
- per-example validation and test predictions for each run
- updated experiment `README.md` and canonical log entry after collection
