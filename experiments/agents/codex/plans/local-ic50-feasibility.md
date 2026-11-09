# Local IC50 training feasibility

Prospective specification, 2026-10-22. Codex / GPT-6. User requests local training
instead of Modal. This bounded diagnostic supports local execution of the ongoing
architecture/availability program; it does not replace biological availability
qualification or complete the full IC50 fitting plan.

## Data and model

Use the exact-only admitted manifest at data/modular_cohorts/2026-10-21-full-ic50/
manifest-exact.json, SHA256 914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be.
All 21,636 exact training measurements; unchanged 2,925 validation and 2,803 test
observations including bounds. Preserve original labels, scales, groups, and
sequence scope. Peptide-disjoint historical partitions have shared studies and
unknown donor/specimen independence. No new source admission, synthetic examples,
or generated negatives. Inputs peptide and two MHC groove sequences, supplied
human species/class; shared d128/l2/h4 encoder, explicit class-II 9mer registers
and flanks, learned register pooling, direct IC50 head. No pretraining, identity
auxiliaries, APC, TCR, or MS branches; all allocated parameters remain trainable.
Raw physical IC50 becomes log10(nM), standardized with exact training-only
mean/SD; squared standardized error weight 1. Bounds in evaluation use one-sided
squared error. No clipping or prediction-variance penalty.

## Local comparison and training

Record M1 Pro CPU/GPU details, memory, PyTorch and all source hashes. MPS requires
an unsandboxed local process; do not enable silent CPU fallback. No cloud jobs,
package changes, mixed precision, compilation, or architecture changes.

1. Fixed training-only timing comparison: CPU with four threads versus MPS with
   four host threads; seed42, AdamW lr3e-4/wd1e-4, batch64, unchanged dropout0.1.
   Same shuffled first12 batches (768 observations); first4 updates warmup,
   remaining8 measured. Synchronize MPS before/after timings, retain losses and
   coverage and initial/final checkpoints. This measures the canonical audited
   training step including its per-parameter synchronization overhead. No held-out
   data or predictive claims from this timing check. Device selection uses only
   finite training and median measured step time, not predictive metrics.
2. Run one complete epoch (339 updates) from a fresh seed42 model on the faster
   supported device using the canonical fit_index trainer. Batch64, lr3e-4,
   weight_decay1e-4, dropout0.1, no replay, no scheduler. No sample cap or early
   metric-based stop. Snapshot every50updates. The single epoch is selected;
   evaluation on train/validation/test then runs once and preserves per-example
   predictions. This is a fixed one-epoch feasibility diagnostic, not a completed
   convergence or multi-seed comparison. Stop with preserved state on nonfinite
   arithmetic or resource failure; do not silently reduce the cohort/model.
3. Fit fixed training-mean and ridge baselines on the same training rows. Ridge
   uses length,20 residue fractions,two class indicators, train-only standardization,
   unpenalized intercept,alpha1. Save coefficients and all split predictions.

## Verification and report

Use canonical checkpoint/coverage/feature reports, retain initial/selected/last/
resume snapshots and histories. All allocated trainable tensors must receive real
loss gradients and updates. Save manifest, source provenance, launch metadata and
frozen code; raw checkpoints under artifacts/<family> and compact outputs under
experiments/<family>/results. Independently recompute metrics from prediction dumps,
check source IDs/targets/qualifiers and reload selected checkpoint in a fresh
process for a fixed training-batch replay. Report wall time, process peak RSS
(unified GPU allocator memory separately), update timing, training/validation/test
loss, exact RMSE/Pearson/Spearman, prediction SD/range, and <=500nM discrimination.
No claim of improved generalization or biological concentration recovery based
on finite loss, nonconstant output, or a successful local run. Update README and
canonical registry at closure. Full IC50 convergence plan remains open.

## Verification finding, 17:53 UTC

The independent verifier rejected a reported Pearson correlation for a mathematically
constant mean baseline. Floating-point reduction produced SD about4e-16, so the
canonical std==0 guard emitted a meaningless tiny coefficient. Preserve the original
fit artifacts and failed verification attempt. Change the canonical correlation
guard to test zero range, add a regression case that fails against the old code,
and report undefined Pearson as null in independently derived summaries. This is
a metric-only correction: no change to the fit, held-out predictions, selection,
or runtime. The verifier must explicitly record which original metrics were
corrected rather than silently accept arbitrary mismatches.
