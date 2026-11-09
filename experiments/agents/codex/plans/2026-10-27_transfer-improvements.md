# Shared transfer improvements — prospective plan

Owner: Codex / GPT-6. Created 2026-10-27 before implementation or new outcomes.
User requests practical experimental validation and applying demonstrated wins to
one streamlined observation-manifest training path. Isolate changes from the dirty
workspace; base is shared-v4 commit 4615573598f24dc1e1f05ec1f8b130d0a90c64a9.

## Questions and fixed comparisons

1. Does more per-class exposure explain the weak fixed-budget joint result?
2. Does balancing class exposure help at equal optimizer updates?
3. Does a shared residue-level peptide/groove interaction before pooling improve
   within-class affinity or cross-class transfer?

Twelve new fits: seeds 42,43 × six conditions:
- pooled-natural-joint: current v4 architecture, natural sampling, 2000 updates.
- pooled-balanced-joint: v4, balanced class sampling, 2000 updates.
- interaction-natural-joint: shared residue cross-attention, natural, 2000 updates.
- interaction-balanced-joint: same interaction, balanced, 2000 updates.
- interaction-I: interaction, I-only, natural, 1000 updates.
- interaction-II: interaction, II-only, natural, 1000 updates.
Reuse the prior frozen H100 v4 single-class1000/joint1000 results as historical
controls, with identical source/data/hardware settings documented. New paired
joint comparisons have identical training budgets; balanced joint exposure is
approximately equal to 1000 source-only updates per class. Record actual counts.
Epoch lengths and therefore validation cadence may differ under balancing; report
this limitation rather than presenting sampling as a perfectly isolated cause.
No unplanned parameter sweep or test-driven follow-up fits in this family.

## Data and optimization

Reuse admitted purified human direct IC50 cohort and exact original splits:
train21636 (I14375/II7261), validation2925, test2803. Source manifest SHA256
914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be.
Input hashes inherited from 2026-10-27_1930_codex_modal-cross-class. Physical
unclipped log10(nM), training-only target scaling; exact training labels, original
one-sided bounds retained in validation/test loss. Preserve missing-qualifier
provenance. No synthetic data, pretraining, new biological data, affinity proxies,
MS or TCR labels. mhcseqs-derived paired groove sequences remain unchanged.
AdamW lr3e-4, wd1e-4, batch64, dropout.1, min/maxepochs100, patience100,
identityweight0; no AMP/compile/TF32. Existing common manifest trainer and step.
Source-only checkpoints selected only by source validation. Opposite-class labels
are evaluation-only, expanded routing preserves every tensor. Same-study, already
inspected development holdouts; no claim of independent biological qualification.

## Implementation hypothesis

One shared 4-head128-wide cross-attention maps candidate peptide token states to
both groove token sequences before residue pooling, with residual normalization.
Candidate tokens average core-relative states and the corresponding full-peptide
states. All interaction weights are exercised by either class; no class-specific
heads, flank-only weights, extra biological labels, or learned class gates.
Keep one current production architecture if promoted; freeze previous checkpoint
implementation for inference/reproduction. Experimental negative results stay in
the reproducibility bundle, not as a new supported training sidecar.

Balance observations by declared metadata via the canonical trainer, deterministic
seeded shuffled cycles within strata, no held-out resampling or label alteration.
Keep natural sampling as the generic default unless validation supports a scoped
joint-IC50 recipe. Persist complete sampler order/RNG for exact resume. Validate
ambiguous/unsupported strata before training. Count actual class examples.

## Selection and promotion, fixed before results

Primary joint statistic: mean of class-I and class-II exact validation Spearman.
A candidate can replace its matched control only if two-seed mean improves by
at least .02, both seeds improve the macro statistic, neither class mean drops
more than .01 rho, and neither class mean RMSE worsens more than2%. Source-only
and transfer metrics are separate secondary questions. No class-only validation
result may be used to select an opposite-class-only checkpoint. Apply a winning
sampling recipe only to the relevant declared training population. Apply architecture
only if the joint gate passes and source-only results reveal no material regression
(>.02 rho within-class mean). No win is an acceptable informative result.
Freeze decisions using validation before reading new test scores; test dumps are
retained and independently summarized only after the decision receipt is sealed.

## Execution and closure

Requested PRESTO_MODAL_GPU=H100!, max2concurrent workers,4CPU,16GiBhost,7200s
per fit, no automatic retry. Upload only source snapshots and the already approved
cohort to the user's Modal account. Record requested/observed GPU, library versions,
actual wall time, GPU allocated/reserved memory, hostRSS, parameter count and class
exposures. Run unit/resume/gradient/serialization checks before launch. Publish live
training and validation results; keep test metrics hidden until selection is frozen.

Every fit retains selected checkpoint, original manifest, config, losses, source
hashes and per-example validation/test dumps. Independently recompute exact Spearman,
Pearson, RMSE, all-label loss and <=500nM accuracy/balanced accuracy/precision/recall/
F1/AUROC/AUPRC. Preserve all failures. Add paired deltas/bootstrap intervals and
plots when useful. Same-H100 training-probe replay checks serialization independently.
Close README, reproduce/launch.sh/.json/source, summary tables and canonical log.
Integrate validated winners in the clean shared branch, test all affected endpoint
and legacy checkpoint contracts, and deliver a durable patch/source/evidence bundle
without overwriting unrelated dirty workspace files. PR131 scientific gates remain.
