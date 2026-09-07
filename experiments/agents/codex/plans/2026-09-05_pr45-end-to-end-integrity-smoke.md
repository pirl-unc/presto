# PR #45 end-to-end integrity smoke

## Objective

Prove the final PR #45 code can load a real canonical data path, resolve MHC
inputs through `mhcseqs`, construct and split the dataset, train a model, run
held-out evaluation, and emit prediction artifacts that the experiment
aggregator accepts.

## Contract

- Agent/model: Codex / GPT-5 family.
- Hardware: local CPU; no Modal/GPU launch.
- Data: a capped real repository corpus, selected after implementation by the
  smallest canonical loader path that contains quantitative binding and
  supports non-empty train/validation/test splits.
- MHC: `mhcseqs` canonical resolution; no required CSV fallback.
- Training: one epoch, small model/batch, explicit batch cap if supported; this
  is a functional smoke, not a quality comparison.
- Splits: non-empty train/validation/test. Preserve per-example validation and
  test predictions when the trainer supports them.
- Synthetic data: disabled unless a binary target is intentionally included;
  any enabled synthetic family must be stated explicitly in the final README.
- Qualifiers/censoring: preserve the loader's real qualifier contract.
- Success: process exits zero after optimization and held-out evaluation;
  run artifacts include config, metrics, split/data audits, and available
  validation/test predictions. The experiment-specific aggregator must also
  pass its schema/parity integration verification independently.

## Required bundle

Create `experiments/2026-09-05_HHMM_codex_pr45-integrity-smoke/` at launch with:

- `README.md`
- `reproduce/launch.sh`
- `reproduce/launch.json`
- `reproduce/source/` launcher or command snapshot
- copied summary JSON/CSV and validation/test prediction dumps
- explicit links to any raw run directory outside the experiment folder

After completion, record exact argv/environment, git hash and dirty state,
dataset/curation/split contract, losses and output mapping, runtime, available
held-out metrics, and the fact that this smoke is not a predictive benchmark.
Add the completed family to `experiments/experiment_log.md`; do not update
`model_to_beat.md`.
