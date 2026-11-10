# Shared condition-dependent pMHC selection: implementation and fit gates

Date: 2026-11-10. Codex / GPT-6. Status: next work, not implemented or launched.
Owners: Presto157 (capacity), Presto155 (qualified source fit), Hitlist898/897
(source release). Parent family:2026-11-10_1607_codex_dp-dm-cohort-qualification.

## Problem established by evidence

After complete known embargoes,10,034 source rows/4,871 peptides retain two
paired DP allotypes. The current staged model can fit a peptide-by-context shift
but cannot condition that shift on MHC identity; its register weights ignore
context. Artificial probes verify the limitation in both class branches while
purified affinity remains invariant. This is capacity evidence, not a trained
biological performance result.

## Architecture implementation gate

Keep the existing shared encoder, sequence interaction and pMHC candidate latent.
Preserve the MHC-independent precursor/supply function and the quantitative
assay decoder. Use a single common contextual pMHC selection function for both
classes, downstream of measured affinity, rather than a class-II/DM-specific
model. Prefer reuse of existing context-fusion machinery. A neutral-start
selection correction is a candidate design, not yet an adopted architecture.
Its initial outputs must reproduce the frozen staged parent; frozen affinity,
source-independent detector and out-of-scope outputs must remain unchanged.
Do not feed sample/study IDs or assay labels into a biological selection latent.
Keep unknown condition different from explicit absence. Determine whether the
correction operates on pooled pMHC or each register; only the latter can express
context-dependent register selection, which must be tested as its own claim.

Before fitting: run nonlinear MHC-by-condition fixtures for both classes,
active-column/parameter-update audits, exact neutral-start and affinity/detector
preservation, old-checkpoint replay, and canonical manifest admission. Defaults
for new training should converge on the qualified shared implementation; old
modes remain only as frozen reproduction/control contracts. Do not change the
other worker's active Atlas/T-cell runtime or select its checkpoints from this
source audit.

## Data and evaluation gate

Use only identities clear under the complete seventeen-namespace embargo. The
frozen per-row disposition artifact and its hashes are canonical; no changing
split membership after inspecting model scores. Physical specimen independence
remains unknown; A/B labels are source samples, not donors.

Build a source-qualified detection panel for DP04:02 and DP10:01 with explicit
sample acquisition support, observed positives and separately marked unobserved
comparisons. Preserve all original positives even if a robust contrast subset
requires repeated detection. DP04:01's present-only data may support detection
but cannot supply an absent-DM contrast. Define missing-acquisition/nondetection
semantics, detection-depth confounding and sample weighting before materializing
fit labels. Never turn the panel into affinity or abundance supervision.

Partition peptide components, connecting exact/core8/Hamming1 identities across
all allotypes and conditions. Freeze TRAIN/internal-validation and a sealed
internal test before training. Report component counts and refuse a component
allocation that leaves a paired condition empty. These splits test within-study
sequence generalization; they are not independent-study pathway qualification.
Source release and all biological metadata must pass their acceptance gates.

## Matched comparison gate

Freeze exact parent checkpoint/config/scales/runtime and load identical shared
weights for every arm. The retained Stage-B parent remains the production
reference; the separately qualified class-II IC50 adapter is evidence for a
later unified retrain, not a silently promoted checkpoint here.

Candidate arms: condition-masked control, current staged context response,
shared MHC-conditioned selection, and a condition-label-shuffled control with
matched parameters/exposure. Resolve the exact architecture/control matrix and
parameter counts before launch; a random new head versus a pretrained head is
not a clean mechanism comparison. Use two fixed seeds and one common observation
manifest/trainer. Freeze batch composition, updates, optimizer schedule, selection
metric and non-regression gates in the timestamped fit launch bundle.

Report detection AUROC/AUPRC/log loss by allotype and DM state, and a paired
within-pMHC condition-contrast metric on the prespecified informative subset.
Include prevalence and comparison construction. A global DM offset can improve
pooled detection without learning allele-specific editing; require improvement
in the corresponding interaction/contrast test. Preserve per-example validation
and final test predictions, uncertainty clustered by peptide component, and all
source/condition exclusions. An isolated capacity result cannot select a winner.

Use H100! explicitly on Modal only after CPU/admission gates pass, respect the
existing two-GPU total concurrency limit, and check authoritative live jobs before
allocating. No GPU is launched by this plan. Report requested and observed hardware,
runtime and memory. Failed scientific gates mean retain the parent; do not refill
or loosen evaluation cohorts in response to poor performance.
