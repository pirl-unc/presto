# Separate processing pathways from cellular interventions

Codex / GPT-6, 2026-10-26. Parent objective: `mhc2_improvement_spec.md`.
This implements a missing input distinction for the eventual biological fits;
software checks alone cannot establish learned pathways or improved prediction.

## Evidence and problem

The current model has cell types, cytokines, machinery states and one delivery
category. It cannot separately represent a reported intervention such as added
HLA-DM, without calling it a cytokine or inventing measured machinery activity.
Hitlist2.0.42 already distinguishes transfection/overexpression and other concurrent
conditions; MAPTAC loses observation-level DM membership (#853), while Stražar's
reported added-DM context is uniform. Neither supplies a valid DM contrast by
itself. Current `delivery=protein` also does not assert an observed intracellular
processing pathway. Keep these concepts separate.

## Contract and implementation

1. Add a typed categorical `ContextAssertion` with present/absent/unknown state
   and retained evidence. A supplied state describes a scoped source/user claim,
   not an inferred measurement, enzyme activity or causal pathway contribution.
2. Add independent named `processing_pathways` and `interventions` dictionaries,
   each with explicit overrides, to Context. Concurrent claims remain concurrent;
   missing values and explicit unknown overrides stay unknown. No MHC-class,
   gene-expression, cell-type or species default supplies these assertions.
3. Allocate separate declared pathway/intervention vocabularies in ModularConfig,
   default empty. Encode each independently in the cellular context input; no
   mutually-exclusive route selection and no sum of invented physical yields.
   Preserve input evidence, supplied/override origin and ignored fields in the
   diagnostic record. Do not feed evidence IDs or free text into predictions.
4. Mask these fields for purified affinity, molecular TCR binding, laboratory
   digestion and isolated MS response. Changes may affect the cellular part of
   elution/T-cell endpoints but must leave the separate MS component invariant.
5. Training preflight requires actual present AND absent training support for
   every allocated trainable factor. Unknown is not a control. This is necessary
   coverage only: it does not establish matched arms or causal identifiability.
   Qualified frozen inherited factors retain the existing freeze semantics.
6. Integrate feature support, untrained-state rejection, checkpoint save/load and
   transfer contracts. Empty/default additions preserve prior numerical behavior
   and legacy semantic fingerprints; no active old runner is edited or restarted.

## Validation and remaining science

Use artificial fixtures explicitly labeled as such to test independent/concurrent
states, override provenance, serialization, unsupported vocabularies, noncellular
invariance, MS-component invariance, gradients, feature coverage, checkpoint
round-trip and unsupported-state rejection. Test preflight missing/unknown-only
contrasts and held-out-only support. Run relevant existing model/context/training
tests and repository checks; no fixture accuracy is a biological result.

Real-source admission and prediction comparisons remain mandatory after this
change: acquire the canonical corrected class-II cohort/context export, preserve
source-backed assertions at their supported scope, and qualify paired molecular
inputs, context contrasts and independent study/sample evaluation. Distinct
processing pathways need source-supported contrasts; do not activate guessed
pathways simply to exercise this representation. Binding and MS performance gates
and upstream missing-source work remain open.

## Software verification

The focused model/context/training/Hitlist suite passes 106 tests, including five
new artificial-fixture tests. The complete repository suite passes 2,629 tests,
four skips, in 597.22 seconds. Pinned Ruff and strict MkDocs build pass.
`tasks/verification/` preserves the logs and an independent old/new runtime check:
default predictions, all 40 parameter tensors, feature names and semantic
fingerprints match exactly. Production source was unchanged during that run.
The extension is ready for source-backed experiments; no new biological accuracy,
mechanistic identification or causal intervention effect is established.
