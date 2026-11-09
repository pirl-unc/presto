# Frozen temporal-model transfer to organ-derived 20S measurements

2026-10-25, Codex/GPT-6. Previous goal turn: progress (original ESP contract
qualified and precise precursor-scale metadata request published). The full goal
is unchanged: precursor-relative APC peptide quantities, including short/unobserved
products and machinery-conditioned processing, with matched exact/RL validation.
No concentration claim follows from the conditional test proposed here.

## Why this can proceed independently of absolute calibration

The canonical PXD058121 source includes original F3 organ time courses that have
not been used by the F2 temporal models. Within each acquisition, a product/intact
signal ratio cancels shared run gain; a change in that ratio over time cancels
fixed product/parent response coefficients. Time-dependent matrix/response effects
remain assumptions. This is a defensible relative endpoint to test while raw-area
and precursor-scale dependencies remain unresolved, not a substitute concentration
endpoint. No precursor-intensity decline is treated as a measured amount.

This tests transfer of frozen F2 sequence/physical predictions to organ-derived
enzyme preparations, rather than another feature/optimizer sweep of inspected
native response or F2 late-time partitions. F3 source cells/metadata were previously
audited. It is not an untouched-data, independent-animal, new-protein or APC test:
all organs come from one source animal and the substrate is shared alpha-synuclein.
No decision uses new F3 effects or model scores before registration.

## Stage A: metadata-only feasibility before implementation

Register one experiment family. Open the existing independently verified source
through Hitlist EvidenceStore. Query only source views, acquisition/material links,
assignment identities, sequence spans and confidence/linkage fields; no quantitative
values or original F2 validation/test labels. Pin source manifest, shared package
commit and these queries. Inventory F3 organ/time/acquisition coverage, raw-file
identity overlap with F2 and reused-source assertions. Exclude source-backed
acquisition reuse; unknown preparation independence stays unknown. Never infer
three animals from replicate labels. Require each Brain/Liver/Spleen series to have
source-backed nominal0,30,60-minute acquisition coverage before numeric admission.
If this fails, preserve scope and stop without selecting substitute timepoints.

## Stage B: shared endpoint admission

If metadata supports the question, extend the existing Hitlist intact-kinetic
recipe machinery with a separately registered F3 zero-baseline change contract.
Do not change the existing F2 recipe output or train through a Presto workbook
adapter. Freeze the shared implementation/spec before reading F3 target values.
Use canonical fields and preserve all decisions, raw quantities and provenance.
Only original `organs` views, never publisher-selected S5 or reused F1/F2 views.
Filter quantities by selected F3 measurement IDs before interpreting their values.

Use the existing positivity/finite, valid span, single reported/replayed assignment,
absolute error<=10ppm, explicit-unmodified and exact input-ID/intensity-link gates.
Exclude all eligible multiple-mass sequence/acquisition groups. Require one
qualified intact normalizer in the same acquisition. Do not fill absent masses,
invent ID links, sum uncertain forms or reinterpret missing measurements as zeros.

For each organ/peptide with an eligible nominal-zero observation, compute the mean
of zero-time log10(product/intact) across its available replicate cells. This is
supplied baseline conditioning, not a predicted outcome. Preserve baseline IDs,
values, replicate counts and reuse dependencies. A peptide absent at zero is
excluded from this endpoint and retained in dispositions; no future support can
retroactively create baseline eligibility. Every eligible later replicate is a
separate observed target: its log ratio minus the baseline mean. Preserve the
missing Spleen30minR3 and Brain0minR2 input-ID limitation; no repairs or invented data.

Primary times30/60minutes. Secondary90/120minutes describe extrapolation; neither
can select a model or rescue primary failure. Each organ must have>=20qualified
baseline-supported sequences and>=40primary target cells; support selection never
uses response magnitude beyond original positivity or model predictions. If any
organ fails, stop scoring the entire primary comparison; report support only.
Record shared peptide overlap with F2 and no independent-protein interpretation.

## Frozen predictors, no fitting or condition adaptation

Before numeric admission hash the complete stored checkpoints and selection files.
No additional optimizer, hyperparameter choice, affine recalibration or organ
intercept is fitted. Organ labels are evaluation strata, not trained features.
Use the following fixed arms, preserving distinct historical training contracts:

1. Zero change.
2. Shared slope, length slope, sequence slope from0143: equal averaging of the five
   stored outer-fold predictors; keep each saved training-only transform and its
   inner-selected alpha. Prediction is elapsed minutes/80 times its fixed slope.
3. Three0143 physical variants (uniform, residue penalty.01, residue penalty0):
   each averages the five stored training-objective-selected initializations.
4. Original0046 matched reference pairs: uniform/residue exact fits and their
   16/64-draw conditional-score fits, seeds42/43/44, all fixed final checkpoints.
   Evaluate their physical component at t minus its component at nominal0, so
   peptide offsets cancel. These are exact-forward/REINFORCE-gradient hybrids,
   not simulator-only RL or GRPO. No new RL training is claimed.

The21fixed conditions are not selected using F3. Unknown/unsupported feature
coordinates follow the stored transform, with support flags reported and no
prediction-based exclusions. Inputs are peptide sequence/span, parent sequence,
time and supplied zero-time baseline. No allele/class/TCR/APC-state/absolute-MS
head is trained. The shared physical process retains all spans; this test only
scores observed supported products and does not validate unobserved mass.

## Qualification, metrics and decisions

Independently replay source eligibility, baseline/normalizer joins, arithmetic,
all dispositions and counts before scoring. Verify zero-time cancellation and
stored direct/physical prediction reconstruction on fixed fixtures; replay frozen
checkpoint hashes. No original F2 held-out targets/predictions are read. Preserve
all F3 per-example outcomes/predictions after admission, including IDs/source links.

Primary weightedMSE gives equal mass to organs, then peptide curves, then times,
then replicate cells. Report overall and organ/time RMSE, MAE, R², Pearson/Spearman,
prediction/target SD and gains against zero and length. Report unweighted rank
correlations separately. Validation split:none, because every model is already
frozen and there is no F3 selection; this is one frozen condition-transfer evaluation.
Historical0143development and0046future-time metrics retain their original meanings.
IC50/500nM metrics are inapplicable.

The preselected sequence ensemble must improve primaryRMSE>=5% over both zero and
length, have positive pooled Pearson/Spearman, SD>=.1targetSD and no organRMSE>10%
worse than length. Preserve every gate and result. Paired peptide bootstrap1000
resamples/seed31415 resamples peptide identities jointly across organs; this is
within-substrate uncertainty, not animal replication. No threshold changes.
Report frozen exact/score-pair differences and inference runtime/memory separately;
original training score-call/runtime comparisons remain0046results, not rerun claims.

If sequence transfers, freeze it as a relative temporal comparator for a separately
planned shared mechanistic representation; it still does not identify concentrations.
If it fails, stop adapting this F3 evaluation and seek broader independently measured
substrates/contexts. Neither outcome removes the calibration/data requests or closes
the full availability goal. No new feature search or F3 refit within this family.

## Execution, publication and preservation

LocalCPU/float64, one BLAS thread;3GiB process-tree RSS and startup free+speculative,
20min/stage,1GiB disk reserve+32MiB margin;50ms polling plus OS peak. No Modal,
shared-environment modifications, raw-MS downloads or unrelated process changes.
Freeze per-stage source/invocations and explicit allowlisted input hashes. Keep
raw source once under data; use shared recipe snapshots by existing provenance API.
New Hitlist work uses an isolated branch/worktree, requiredformat/lint/test gates,
PR/version/release and independent source replay; no edits to main or other work.
Close with README, summary/coverage tables, predictions or exact support-failure
reason, replay evidence, canonical registry and precise upstream issue ownership.
