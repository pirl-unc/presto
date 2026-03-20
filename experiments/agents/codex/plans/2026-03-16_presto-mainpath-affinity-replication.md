# Presto Main-Path Affinity Replication

- Date: `2026-03-16`
- Agent: `codex`
- Goal: reproduce the current best broad-numeric binding configuration on the main Presto model path while enforcing a strict input contract of `peptide`, `nflank`, `cflank`, `mhc_a`, `mhc_b` only.

## Why This Exists

Recent broad-numeric binding winners were established on the shared `distributional_ba` benchmark path, not on the canonical main Presto model. The user now wants the best result rebuilt on top of the normal Presto codebase and input surface, without assay metadata entering as model input.

That means:
- use `models/presto.py`, not the experiment-local distributional encoder/head wrapper
- keep affinity assay identities as outputs only
- make any needed architectural/training changes explicit and reusable in the main code path

## Target Contract To Reproduce

Source winner to beat:
- EXP-21 winner: `groove c02`, 50 epochs

Key hyperparameters to carry over:
- `d_model=32`
- `n_layers=2`
- `n_heads=4`
- `affinity_target_encoding=mhcflurry`
- `max_affinity_nM=100000`
- `batch_size=256`
- `lr=1e-3`
- `weight_decay=0.01`
- `epochs=50`
- no warm start

Dataset / evaluation contract:
- source: `data/merged_deduped.tsv`
- alleles: `HLA-A*02:01`, `HLA-A*24:02`
- measurement profile: `numeric_no_qualitative`
- included assay families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- qualifier filter: `all`
- split: same peptide-group 80/10/10 policy used by the recent EXP-20/21 binding benchmarks

## Non-Negotiable Input Contract

Allowed model inputs for this path:
- `peptide`
- `nflank`
- `cflank`
- `mhc_a`
- `mhc_b`

Forbidden model inputs for this path:
- binding assay type
- binding assay method
- binding assay prep
- binding assay geometry
- binding assay readout
- any other assay identity feature used to influence affinity outputs

Those fields may still exist in the dataset and may still define which output target is supervised, but they must not be consumed as model inputs.

## Current Code Reality

Observed from current code:
- `Presto` already uses the correct five-segment input path.
- `Presto.forward(..., binding_context=...)` and `AffinityPredictor` still expose assay-context machinery.
- If `binding_context` is omitted today, the affinity path still passes through default zero/unknown assay ids rather than an explicit "no assay input" mode.
- The full `train_iedb.py` path is broader than this task and couples many unrelated tasks/regularizers.

Implication:
- the main model can host this experiment, but it needs an explicit assay-free affinity mode and a dedicated trainer/eval path for this binding contract

## Proposed Implementation

### 1. Main-model change

Add an explicit no-assay-input affinity mode to the main Presto affinity predictor.

Requirements:
- no assay embeddings or assay-context projections should influence affinity outputs in this mode
- output heads for KD / IC50 / EC50 should still exist and remain outputs
- the mode should be expressed as an explicit config/CLI option, not inferred from missing context

Likely shape:
- extend `affinity_assay_mode` with a mode like `none`
- ensure `AffinityPredictor.forward()` bypasses assay context encoding entirely in that mode
- ensure any residual path that currently expects assay context gets a clean no-context path

### 2. Trainer

Add a dedicated main-path affinity trainer that:
- instantiates `Presto`
- uses `PrestoDataset` / `PrestoCollator`
- filters to the exact binding contract above
- trains only the binding-affinity outputs relevant to this benchmark
- calls `forward_affinity_only(...)` without assay inputs
- writes held-out validation/test predictions and summary metrics into an experiment directory

This trainer should be narrow and benchmark-oriented, not a partial fork of the full multitask `train_iedb.py`.

### 3. Evaluation

Track at minimum:
- validation/test loss
- Spearman
- Pearson
- RMSE log10
- `<=500 nM` accuracy / balanced accuracy / precision / recall / F1 / AUROC / AUPRC

Preserve:
- per-example val/test predictions
- split artifacts
- raw per-epoch logs

## Validation Plan

Before launching a real run:
- smoke-test dataset loading and splitting on the exact 2-allele contract
- smoke-test `Presto.forward_affinity_only()` with assay-free mode enabled
- run one short local training/eval pass and confirm:
  - only the allowed inputs are consumed
  - no assay context tensors are required
  - held-out metrics and prediction dumps are produced

After launch:
- collect all raw outputs locally into the experiment directory
- compare directly against EXP-21 `groove c02` 50-epoch result

## Success Criteria

Minimum success:
- the new main-path Presto affinity run trains and evaluates cleanly with assay inputs disabled
- metrics are reproducible from local artifacts

Stronger success:
- the main-path run reaches the same performance class as the current benchmark winner

Failure cases to watch:
- accidental leakage of assay metadata through `binding_context`
- hidden dependence on the broad multitask trainer
- a nominal "main path" that is actually an experiment-only side model again
