# Shared supervision for coverage, MIL and held-out artifacts

Status: partially implemented in PR #56 after the merged #49/#52 repairs.
Related issues: #48, #50, #51; experimental consumer: #53.
Author: Codex / GPT-6, 2026-09-07. Baseline: merged #47 (`b2e939e`).

PR #56 implements shared bag targets/predictions, selected T-cell bag-panel
objectives, complete/chunked final evaluation and bag identity/export. Its
split-support census consumes the same target views but remains separate from
the legacy row-mask table. The general row/derived/panel registry, endpoint
gates, CE/vector identity, real-source census and experiment closure remain
future steps below. Full issue acceptance is not claimed by the bag repair.

## Problem and boundary

The canonical trainer has three descriptions of supervision: task specs plus
derived targets, extra binding/excision panel objectives, and MIL channels.
Support counting walks batch masks; held-out extraction walks task specs and a
separate row forward. Each is locally plausible but refers to different
observations. An output registry alone will not repair that divergence.

Define effective supervised observations once, then let loss, support counting
and prediction export consume them. Preserve biological meaning and predictive
inputs from the authoritative I/O contract. Role-specific repertoire/context
schema changes remain #46. Internal diagnostic tensors do not become endpoints
merely because they exist in a forward dictionary.

## Proposed implementation contract

Place the model-independent specification and target resolution in a training
module rather than importing trainer scripts from evaluation/support modules.
Move the existing task definitions and pure target transforms into that layer
incrementally, with compatibility re-exports from train_synthetic during the
migration. Avoid a circular import between collator, trainer and evaluation.

A declared output specification identifies:

- Canonical endpoint and aliases; model configuration in which it exists.
- Row, bag, categorical or vector-component observation shape.
- Prediction path, target resolver, target transform/inverse, unit and qualifier
  policy, mask, selected axis/column and allowed source families.
- Direct measurement, proxy, generated-label, auxiliary or diagnostic status.
- Base loss weight, existing duplicate/alias objective weighting and stage.
- Vocabulary and explicit unsupported columns, including unknown behavior.

A resolved target view carries tensors plus immutable observation identity:
sample/bag ID, source row and source lineage, instance membership, task/column,
raw and transformed target, unit, qualifier, selection mask and source-kind.
Target views can be created without a model forward for support counting.
Categorical targets retain their single observation/class identity. Vector BCE
retains a component axis until explicit expansion repeats the correct sample
metadata for each component.

A resolved prediction pairs one target view with the output that actually
receives loss. It has a single final observation axis and retains any class
probability vector. Shape or selector mismatches must produce a descriptive
error rather than a silent skip. Numeric identifiers alone are insufficient
when a vocabulary name can be exported.

## Bags and panel supervision

Use one bag materialization/forward/aggregation helper for training and final
evaluation. It preserves boundary flags, biological context and source identity.
Assay selectors remain output-side.

For each observed T-cell axis on a pathway bag, gather that fixed panel column
for every candidate molecule, then apply the declared bag response aggregation
against the measured bag response. Keep ordinary row masks off; a positive bag
must not assign a positive response to every candidate. Unknown axis selectors
retain the current explicit unsupported policy. Scalar response/immunogenicity
objectives remain declared separately with their actual proxy relationships.

Elution, presentation and the ms alias use the same effective bag prediction
in their declared losses and artifact records. Represent aliases once for
endpoint coverage, while preserving existing loss multiplicity until an
explicitly reviewed weighting change removes or changes it.

Training can sample candidate instances under an explicit training cap.
Selected-checkpoint evaluation traverses complete bags deterministically.
Chunk instance forwards for memory control; aggregate stable sufficient
statistics across chunks so the result matches the uncapped bag. Record bag
size, evaluated count and any explicit evaluation exclusion.

## Loss reduction and export

Separate per-observation loss values and support/weight sums from the final
task aggregation. Preserve current training reduction and learned-uncertainty
semantics in the first migration step; use parity tests to expose any existing
batch-dependent reduction before proposing a new global metric definition.
Do not equate a sum of CSV row losses with total loss without including task
normalization, stage/uncertainty weights and regularization.

Export every effective supported observation from the selected checkpoint:
split, canonical output and alias metadata, sample/bag/component/column ID,
source lineage, raw/transformed target, unit, qualifier and prediction.
Categorical records include class identities/probabilities. Keep panel selector
names and vocab version. Save complete loss numerators, denominators and
weight/regularizer metadata needed to reconstruct the declared final loss.

Coverage and export must reconcile counts by task, axis/column, source kind and
split. Output artifacts should fail closure on missing expected rows while
preserving checkpoint and errors for recovery.

## Coverage gates and corpus closure

Build the support matrix from declared outputs and resolved target views,
including zero-support entries. Separate source observations from expanded
instances, unique peptides/alleles and vector components. Count exact/censored,
positive/negative/graded, real/generated/proxy, class/species and output column.
Derived MHC auxiliaries and MIL observations participate in the same census.

Use a supported-endpoint manifest to require selected endpoints prospectively.
No opt-in gate may silently drop an entirely absent required output.
Nonzero count is not adequate scientific support; report sparse, one-class and
unmeasured columns. Measure actual selected-row gradients/update counts on real
batches, including frozen stages and initially zero-coupled parameters, without
treating nonzero gradients as evidence of generalization.

Run the census through each explicitly supported source mode after curation
and splitting. Preserve data/source/hash and split-seed provenance. The #49
before-cap adapter census helps ingestion diagnosis but does not replace this
post-curation matrix.

## Reviewable implementation sequence

1. Extract output specs and target views with behavior-preserving row/derived
   target loss parity. Enumerate extras/MIL/aliases and explicit zero support.
   This establishes the shared dependency for #48/#51.
2. Wire shared effective bag predictions into loss and export, adding selected
   T-cell bag-panel objectives. This addresses #50 and the bag part of #51
   through the same implementation.
3. Complete CE/vector/panel artifact identity and loss/count reconciliation;
   enforce endpoint manifests and run the real-source/split coverage census.
   Close #48/#51 only when their full acceptance criteria are satisfied.
4. Register #53's fresh real-data fitting and held-out experiment using those
   repaired contracts and #52's corrected metric estimator.

These steps may be combined when the extraction and integration remain easy
to review. Each PR must say which acceptance criteria it completes and which
remain open; a scaffolding PR must not close a scientific-coverage issue.

## Required behavioral evidence

- Mixed ordinary/bag batches, unequal bag sizes, class I/II candidate bags,
  positive/negative labels, unknown selectors, censored binding and empty masks.
- Exact Noisy-OR fixture: two 0.5 instances yield 0.75 in both loss and dumps.
- Selected T-cell panel column receives bag response gradient; unselected
  columns and row-level response masks behave as declared.
- Multiple samples with vector targets never exchange source identities.
- CE targets produce evaluable records and task-appropriate metrics.
- Full/chunked evaluation agrees, independent of training instance caps.
- Declared zero-support and one-class selected columns fail requested gates.
- Recomputed per-task losses/counts from artifacts match the canonical
  selected-checkpoint evaluation under the declared weighting contract.
- Source observation and synthetic-parent overlap checks precede #53 fitting;
  validation/test membership is never used to tune the fitting check.

No architecture or model-quality result is claimed by this design document.
