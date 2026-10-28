# Scale the common Presto architecture and audit support

Owner: Codex / GPT-6; 2026-10-27. User explicitly selects the validated shared
interaction architecture and requests more inputs, outputs and data, with a deep
failure/weight audit. Base: 2f5705184f6997d50883639f5d38f6c133b81995.

## Scope and sequence

1. Make shared-v5 the named current development architecture and baseline in the
   common docs/config/CLI; continue in the durable clean shared-training checkout.
   Preserve the original dirty worktree. Do not merge scientific PR131 early.
2. Audit saved v5 checkpoints: allocated parameters versus element updates,
   selected versus final checkpoint evidence, unavailable outputs versus allocated
   untrained outputs, input/category exposure, and structural zero-gradient slices.
   Tensor movement alone is not proof of learning; separate decay-only movement,
   inherited evidence, non-identifiable attention biases and unobserved inputs.
3. Audit predictive failures from existing heldout dumps without refitting those
   holdouts: class, groove-pair, peptide length, sequence novelty, repeated-study
   effects, numeric range, censor direction, calibration, weak/strong binding and
   register count. Preserve row-level residuals and bootstrap uncertainty. Do not
   claim pooled correlation measures per-allele discrimination.
4. Inventory existing source-qualified data and schema coverage. Prefer larger
   direct measured affinity/stability and original bounds, plus genuinely admitted
   cellular/MS/receptor records with explicit provenance. Keep assay endpoints
   distinct and retain verified MHC partners, actual flanks and context missingness.
   No guessed negatives, no fabricated crop labels and no relabeling observational
   MS scores as independently measured processing or abundance. Unavailable TCRvdb
   remains a data gap, not a reason to block independent scale-up.
5. Fix proven common-core/reporting/scalability defects, with a spec amendment for
   any new architecture change. Keep one architecture and one observation-manifest
   trainer. Parameterize width/depth only if the runtime/coverage audit supports it;
   do not add independent class or dataset backends.
6. Before any new predictive fit, freeze a concrete condition matrix, dataset
   hashes, leakage controls, optimization budget, selection rule and GPU contract
   in a timestamped experiment directory. Compare at least an IC50 control with
   admitted enlarged/multi-output training, using both-class validation/test and
   per-output metrics. Previously inspected holdouts are development-only; freeze
   novel source-qualified evaluation where available and protect original heldouts
   from cross-endpoint duplication. Include a negative control where meaningful.
7. Execute on Modal H100!, maximum two concurrent GPU workers, explicit CPU/host
   memory, no arbitrary laptop cap. Bound the first family to at most eight fits,
   12,000 updates per fit and 4h worker timeout; choose smaller scientifically
   adequate contracts after inventory and freeze before launch. Publish actual
   validation progress and keep tests out of condition selection.
8. Close every family with all per-example predictions, independent loss/exact
   Spearman/Pearson/RMSE and binding-threshold metrics, coverage evidence, runtime,
   canonical log and reproducibility bundle. Apply supported improvements to the
   same source path, test historical readers and preserve all negative findings.

## Verification and acceptance

Meaningful regression tests for any discovered support-accounting defect, actual
mixed-output/common-model training, selected-checkpoint coverage and exact resume.
Unknown/unobserved inputs must remain distinguishable from trained zero values.
Gradient magnitude and parameter movement do not establish identifiability or
scientific qualification. Do not loosen coverage gates merely to make a run pass.
Use frozen pre-change controls to demonstrate behavior and prediction parity when
changes should be software-only. Report actual data and output counts, not schema
capacity as trained support. Record an explicit supported/missing matrix at closure.

## Initial experiment records

`2026-10-27_2235_codex_shared-support-audit` records read-only checkpoint, failure and
source analyses (no fitting; existing validation/test predictions only). A separate
prospective scaling family will record its final admitted inputs before fitting.
This is a staged specification: concrete scientific fit contracts are deliberately
not invented before source qualification. The user has authorized implementation
and Modal use; proceed within scope without another generic approval loop.

## Implementation amendment A — qualified quantitative source expansion

Audit found six existing v5 fits with all non-padding weights moved; tensor gates
miss the distinction between structural redundancy and identified effects. Retain
key-bias parameters for checkpoint/prediction compatibility, report their softmax
invariance and numerical gradient scale; do not count tiny gradients as evidence
of identifiability. Add elementwise cumulative gradient-and-update evidence for
future fits, preserving unknown historical element coverage rather than inventing
it from tensor metadata. Exact same-element evidence must define tensor support.

Generalize the existing strict purified IC50 adapter into one literal assay adapter
for IC50, directKD, EC50, the two separately named KD proxies, half-life and off-rate.
Keep the existing IC50 wrapper and byte-equivalent observation contract. Preserve
source/method/peptidoform/numeric-text/qualifier/lineage/verified-pair checks.
Concentration units remain the existing compatible units; half-life supports only
minutes and off-rate only1/s in this first source contract. No mapping of50percent
dissociation temperature to melting temperature, no on-rate/qualitative proxies.
Every output remains separately declared; no physical relation or label equivalence
is imposed. No parameterized concentration conversion between different assays.

Partition all new outcomes using peptide plus original-measurement connected
components, inheriting existing IC50 and source-historical assignments. Components
with conflicting frozen splits are deferred, not resolved by label or prediction.
Deduplicate original outcomes and preserve independent replicate disagreements.
Audit cross-endpoint assay IDs as well as the existing endpoint-specific lineage
keys to avoid related readouts crossing splits. Retain original IC50 rows exactly.
The source inventory precedes the final predictive fit specification.

## Implementation amendment B — capacity and element evidence

Expose width, encoder depth, attention heads and feedforward width in the existing
PrestoConfig (defaults128/2/4/512 preserve v5 computation and checkpoint shapes).
Use the same dimensions in every reachable identity, pMHC, context, digest and MS
module and in input-feature labels. Validate positive integer dimensions, even
positional width and head divisibility. Reject incompatible warm starts; never
silently project learned latent dimensions. This is one configurable architecture.

Persist cumulative Boolean masks for coordinates with nonzero gradient and actual
movement in the same step. Record complete-history provenance separately: old
checkpoints have unknown historical element coverage. Transfer masks with exactly
the same named feature projection as weights. Keep tensor gates, but fix their
same-coordinate criterion. Coverage reports separate fixed padding and redundant
attention K biases from active weights; these are diagnostics, not identifiability
or biological qualification. Verify save/load/resume and default prediction parity.

## Implementation amendment C — comparable validation cadence at larger scale

The larger corpus makes epoch-only validation both slow to display and unequal
between conditions. Add optional `validation_every_steps` to the common trainer.
When set, evaluate/select at those update boundaries and at the final budget,
not additionally at every data epoch. Preserve partial-epoch order and remainder;
never skip rows to reach a validation boundary. Persist exact sampler, replay,
loss/exposure accumulation and coordinate coverage through interruption/resume.
Unset keeps the historical epoch behavior. In interval mode, patience counts
validation events; minimum-epoch gating remains unchanged. The scaling comparison
will use a fixed update interval and disabled early stopping across all conditions.

## Post-freeze correction — sampler implementation receipt

Review found that the generic resume implementation fingerprint omitted the shared
sampler module. Add it and verify that changing a copied sampler source invalidates
resume before any update. This is a provenance-only correction after the predictive
runtime was frozen at0c4c984b; do not mutate or restart the running comparison.
Its Modal worker already pins and verifies the sampler in its full runtime bundle.

## Post-freeze correction — explicit generic checkpoint selection

The common selector implicitly prioritizes IC50 whenever present, so adding outputs
still selects solely for IC50. Make `training.selection_endpoint` explicit: when
named, select that active endpoint's exact RMSE if present, otherwise its validation
loss. With null, use the observation-weighted mean validation assay loss (identity
loss for an identity-only run). Reject absent/frozen primary endpoints or missing
validation outcomes before fitting. No endpoint-specific implementation branch.
The carried-forward IC50 recipe explicitly names IC50 to retain its existing
selection semantics. The current eight-fit benchmark remains on frozen0c4c984b
with its predeclared IC50 selection policy; no models or validation gates change.
Do not claim this software generalization improves predictive quality. Document
that pooled objectives can trade one class against another; preserve that finding.

## Packaging closure

The installed-wheel smoke currently exercises historical loaders only. Extend that
same smoke to instantiate the canonical model at nondefault width, execute both
MHC candidate rules on explicit artificial observations, and check that public
prediction rejects untrained outputs. No fitting or scientific evaluation is added.
