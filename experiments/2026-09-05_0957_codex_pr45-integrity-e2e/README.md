# PR #45 data-integrity end-to-end smoke

- Date: 2026-09-05
- Agent/model: Codex / GPT-5
- Purpose: verify the reviewed canonical Hitlist path loads real data, resolves
  MHC inputs from `mhcseqs` without an index CSV, trains, evaluates validation
  and test splits, and emits traceable per-example prediction dumps.
- Status: pending the clean-commit launch.

## Contract

This is a functional smoke, not a model-quality comparison. It first preflights
the default merged-TSV path with small per-modality reservoir caps and complete
funnel/split artifacts. It then trains on the HLA-A*02:01 Hitlist slice with
reservoir caps of 96 binding, 2 kinetics, 24 stability, and 96 elution
observations. `kon`, `koff`, and `tm` are excluded from training; all
synthetic-data ratios and MHC augmentation are zero. Both conditions use a
peptide-grouped 60/20/20 split with fixed data and split seed 42.

The model is a small one-layer, 32-dimensional CPU configuration trained for
one epoch with two training batches and one in-loop validation batch. The
post-training held-out pass still scores every validation and test example.
No pretraining checkpoint is used. Active supervised outputs are determined by
the retained real assay rows: affinity-family binding outputs, half-life, and
elution/presentation outputs. Standard unified loss terms are used with learned
uncertainty weighting disabled; no synthetic-data or augmentation losses are
introduced by the data contract.

## Acceptance criteria

- The source-specific path reads Hitlist only even though the repository also
  contains a merged TSV.
- The default merged path emits pre-cap candidates, post-cap rows, cap losses,
  and ingest-drop reasons in its machine-readable funnel.
- MHC resolution reports `mhcseqs` as its source with no `--index-csv`.
- Split audit reports no lineage, duplicate-ID, or fake-null-sequence issues.
- Training completes and emits nonempty `val_predictions.csv`,
  `test_predictions.csv`, `val_summary.json`, and `test_summary.json`.
- No `holdout_error.json` is present.

## Reproduction

Run `reproduce/launch.sh` from a clean checkout at the recorded launch commit.
The exact commands are in `reproduce/source/run_merged_preflight.sh` and
`reproduce/source/run_smoke.sh`; structured contract metadata is in
`reproduce/launch.json`.

## Results

Pending.
