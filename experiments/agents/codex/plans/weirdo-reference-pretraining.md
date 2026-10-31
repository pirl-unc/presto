# Bounded peptide-source reference pretraining

Prospective plan, 2026-10-21 UTC; Codex / GPT-6. This is one real auxiliary
subset of the full model goal. It does not qualify MHC/TCR annotation, affinity,
processing, MS response or biological species-of-origin inference.

## Frozen data and admission

Use the actual local Weirdo Swiss-Prot reference and category-split archive
pinned in `tasks/modular_model/auxiliary-source-inventory.json`. The reference
SHA256 is `e9c44a34fcd3f8fb1f4151d489985dd5578a947eda3e62213d5109ace36e115d`,
archive `99b21eb93196cf2b0a0b1ae4bbb63066224a6864e48dd3a8d47fe67bd34fcf79`,
metadata `571356e92fd1da71cf173be9f5e59c6aa128467f347de37aa30815d78923fc1f`.
Select 5,000/1,000/1,000 train/validation/test rows uniformly without replacement
using the existing adapter's Python seeds 42/43/44, independent of their labels.
Retain the archive's eight-residue length and ten declared categories.

Audit every selected peptide against every reference protein before fitting:
reconstruct its exact membership mask, all matching protein identifiers and
positions, mnemonic-prefix grouping, fold and minimum bucket. Recompute folds
using CRC32 of `category-predictor-v1:<prefix>` with 70/15/15 boundaries and
buckets using the source `:bucket:` salt modulo 20. Stop this recipe on any
missing peptide, mismatched membership/fold/bucket, cross-fold occurrence or
duplicate selected peptide. Preserve the discrepancies rather than removing
them or changing labels after inspection. The earlier table-verification metric
compares another table; document this separately from this complete selected-row
audit. Pin source-building code as provenance, while independently reconstructing
the selected memberships.

Labels mean presence in a pinned reference division, not exact organism origin
or absence in the biological world. Categories can overlap. Record all source
families; connected components of selected peptides linked through shared family
keys define resampling groups. These mnemonic keys do not certify homology-based
independence. No new synthetic labels or generated sequences are introduced.

## Conditions, training and budget

- Prior baseline: independent smoothed training prevalence per category
  `(positives + 0.5)/(n + 1)`, plus majority-decision accuracy.
- Composition baseline: one independent logistic regression per category, 20
  residue frequencies, `C=1`, L2, `lbfgs`, at most 1,000 iterations. No tuning or
  class balancing; fit only training observations. Preserve convergence status.
- Network: current modular shared encoder (128 dimensions, two layers/four
  heads), peptide-source readout only, no assay/pairing/species heads or initial
  weights. Ten independent binary CE outcomes per peptide, common masked loss.
  AdamW `lr=3e-4`, weight decay `1e-4`, dropout .1, seed 42, batch 64; complete
  epochs, maximum 20, validation patience five after at least five epochs.
  Select by validation mean auxiliary CE; no test-based choices. Snapshot every
  100 updates. Required tensor/feature/output coverage.
- Local CPU, one Torch/BLAS thread, no GPU/Modal. Optimization/final-evaluation
  subprocess wall-time ceiling 15 minutes; preserve an unfinished/budget-limited
  result without silently extending it. Source audit has a separate 15-minute
  ceiling. No model code changes during an executed condition.

## Evaluation and fixed screen

Save per-example validation/test probabilities and targets for all conditions.
Report each category's prevalence/support, CE, accuracy, balanced accuracy,
precision/recall/F1, AUROC/AUPRC, Brier score, ten equal-width-bin calibration
error and probability spread. Report macro mean CE and training/validation/test
loss. AUPRC with constant scores equals prevalence; do not rank arbitrary ties.

Primary validation screen: the upper bound of each paired 95% interval for
network-minus-baseline macro CE must be below zero, versus both prior and
composition. Use 2,000 connected-source-family-component bootstrap resamples,
seed 20261021. If validation has fewer than 20 components, or a category has
fewer than 20 positives or negatives, report the relevant screen as underpowered;
do not resample a more favorable cohort. Keep per-category results even if the
macro screen passes. Inspect test once after selection, with the same reports;
it cannot change the recipe or winning checkpoint.

This screen is a bounded reference-membership diagnostic, not an automatic
serving qualification or evidence of useful affinity transfer. Report source
grouping and eight-residue applicability limits. Larger peptide lengths,
unrepresented taxa and an exact species-of-origin interpretation are unsupported.

## Reproduction and closure

Register the family before preparation. Preserve selected membership, full
source-match audit/dispositions, pinned config, source versions, actual git state,
launcher/env, runtime/RSS, all metrics and support, val/test prediction dumps,
and raw checkpoint/prediction paths and hashes. Verify the selected checkpoint
in a fresh process on the first training batch; do not rescore test for that
check. Close the family and canonical experiment log on success, failed admission,
failed screen or budget failure. The full MHC/TCR/APC/MS/affinity requirements
remain open regardless of this result.

## Amendment before fitting: absent archive group arrays

The first frozen preparation attempt (`42689fd`) stopped before source scanning
or fitting: the actual pinned NPZ has k-mer and mask arrays, but no `*_groups`
arrays. Array headers and the failed invocation/log/source snapshot are retained.
The original adapter incorrectly assumed the newer grouping-array layout.

Keep the same input hashes, 7,000 selected rows, seeds, labels and model/baseline
recipe. A separate recorded preparation attempt may explicitly read missing
groups as unknown, with no admitted group ID. It must reconstruct actual protein
occurrences, source mnemonic families/folds and minimum reference buckets by the
original full-reference audit before admission. Supplied buckets, when present,
still must match. Absent supplied buckets cannot count as a successful supplied-
bucket comparison; record their absence and reconstructed values separately.
Use the connected source-family components for admitted grouping as planned.
Any label/fold/reference discrepancy still stops preparation. No fitting or test
prediction has occurred, and no cohort is reselected in response to this failure.
