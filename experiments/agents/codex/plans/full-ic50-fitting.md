# Full-cohort assay-only IC50 comparison

Prospective specification, 2026-10-21; Codex / GPT-6. This is the first complete
exact-IC50 training comparison for the modular architecture. It does not replace
the auxiliary/APC/MS/TCR objective, prove auxiliary benefit or qualify unsupported
species. The censored-training manifest is preserved for a subsequent controlled
condition; this family first establishes the exact-only baseline.

## Frozen observations and models

Use the admitted exact-only manifest SHA256
`914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be` and its individually
pinned source files under `data/modular_cohorts/2026-10-21-full-ic50/`.
Train on all21,636 exact measurements. Validation/test are the unchanged
2,925/2,803 observations, including bounds. Preserve all prior source/partition
limitations: peptide-disjoint, shared studies, unknown donor/specimen/acquisition
identity, previously inspected historical held-outs. No cohort caps, clipping,
independent-measurement averaging, fabricated partners or source reassignments.

Fixed conditions:

1. Training mean in raw log10(nM), fitted on exact train only.
2. Ridge with peptide length,20 residue fractions and two class indicators;
   train-only feature mean/std (constant scales set1), unpenalized intercept and
   fixed alpha1. No hyperparameter tuning or held-out transformation fitting.
3. Three fresh modular fits, seeds42/43/44. Shared d128/l2/h4 encoder, full-peptide
   class I, all explicit nine-residue class-II registers/flanks and learned
   pooling; direct IC50 scalar. Supplied human molecular species/class, no
   auxiliary heads/pretraining/replay, APC/MS/TCR branches or variance penalty.
4. One negative control: the same exact-only network/seed42, with training values
   permuted without replacement within class using NumPy seed424242. Original
   source values/parent identities remain in provenance; generated training
   labels and their assignment are explicitly marked as a permutation control.
   Validation/test remain unchanged. The control is not a biological checkpoint.

Network recipe: AdamW3e-4, weight decay1e-4, dropout.1, constant learning rate,
batch64, all complete epochs, maximum100; validation patience15 after at least10
epochs. No arbitrary step cap. Fit the target scaler on each condition's exact
training labels; permutation preserves the classwise multiset. Validation exact
IC50 RMSE alone selects the checkpoint; test is evaluated once after selection.
No best-seed selection, test-tuned recipe or transfer from the memorized weights.

## Execution and recording

Register the family and freeze source/manifests/baseline definitions before any
fitting. Use local CPU/one thread for the small closed-form baselines, and Modal
`H100!` for neural runs. Record actual GPU name/memory, Torch/CUDA/Python/package
versions, source/manifest hashes, float32 backend settings and runtime. Use no
mixed precision, TF32 or compilation change in this first contract. Allow at
most two GPU containers; initially submit seed42 to establish infrastructure,
then submit the remaining prespecified runs without tuning from its metrics.
Function calls have a finite timeout and persistent output paths. Preserve failure
state; resume only after an authoritative terminal status and an exact matching
canonical resume contract. Observation timeouts never authorize restarts.

Each call must preserve initial/selected/last/resume checkpoints, complete history,
training/validation/test predictions, tensor/feature coverage, actual manifests
and execution metadata. Checkpoint/summary persistence must survive normal exit
and handled failure. Before fitting, extend canonical prediction records to retain
JSON-safe resolved metadata/register diagnostics and source/input provenance,
so saved predictions support class/allele/study analysis without a test rescore.
Test that these additions preserve numerical predictions/losses and serialize
actual multi-register outputs. They change reporting, not the optimization.

## Prespecified reporting and screen

Report mean standardized loss, raw log10(nM) exact RMSE/Pearson/Spearman, censored
one-sided squared error/satisfaction, prediction SD/target-SD/range/quantiles and
500 nM confusion/discrimination for train/validation/test, overall and by class.
For thresholded censored labels use only bounds that establish the class; retain
strict/inclusive distinctions at500 nM. Also report per-allele/study support and
metrics where at least20 exact measurements and nonzero target spread exist;
other strata remain visible as unsupported. Retain full per-example dumps.

Each neural seed must improve exact training RMSE over the training-mean fit and
pass coverage. For a positive validation screen, require positive exact Spearman
and paired peptide-bootstrap upper95% RMSE-difference bounds below0 against BOTH
mean and ridge, overall AND separately for both classes. Use2,000 component
resamples, NumPy seed20261021, preserving all measurements of each peptide;
report all three seeds without choosing a winner from test. Require each real
seed to pass for a reproducible family-level claim. SD ratio<0.1 triggers an
explicit contraction investigation, never artificial variance inflation.

The permutation control must not reproduce that real-label validation advantage;
report its exact result even if it unexpectedly passes. These are diagnostic
screens within the declared cohort, not study/donor independence or an all-species
serving certificate. No neural checkpoint is promoted solely because it varies.
Failure preserves evidence and motivates a separately registered diagnosis.

After all terminal runs, independently recompute saved metrics/baseline coefficients,
verify original source/target joins, replay a fixed training batch from the selected
checkpoint in a fresh process (no test rescore), record complete artifacts and
close the README/canonical log. A running or stopped Modal app alone is not closure.
