# Subset transfer, function freezing and exact continuation

The full goal remains active. This slice implements task subsetting and true
dependency freezing first; exact interrupted-run resume is the following part
of the same requirement, not satisfied by optimizer serialization alone.

## Transfer and scalers

- Extend the existing checkpoint initializer with explicit task-removal consent.
  Copy retained modules and compatible task-dependent feature columns by their
  canonical names. Removed endpoints/domains must not leave stray heads, scaler
  entries or support claims. Newly allocated columns/heads stay unsupported.
- Preserve meanings/order of shared categorical outputs, units, defaults and
  numeric features; reject incompatible retained contracts. Task-driven removal
  of masked receptor/cellular/coreceptor columns is a structural projection, not
  permission to reinterpret a feature or change its vocabulary.
- Keep inherited scalers for retained endpoints. Fitting another cohort must
  not silently reinterpret already learned standardized output values. Fit
  training-only scalers only for newly added regression endpoints.

## Freezing

- A manifest's `pretrained.freeze` names output tasks or explicit component
  targets (`component:ms_response`, or `component:ms_response:<domain>` for a
  quantity readout). Compile their full upstream parameter dependency union.
- Freezing an output includes its encoder, fusion/pooling, relevant inferred
  categorical dependencies and readout. Freezing MS includes the shared encoder,
  detector features and its readout. This intentionally limits learning in
  sibling paths sharing those parameters. Never call a head-only freeze a
  frozen function.
- Require inherited gradient/update evidence before freezing a dependency.
  Freeze parameters and stochastic module behavior (encoder dropout stays in
  evaluation mode). Optimizers contain exactly trainable parameters. Reject
  manually inconsistent freeze flags and frozen features without prior support.
- Persist target declarations and resolved parameter sets; reload recomputes and
  verifies them. `pretrained_frozen` describes training support, separately from
  validation status; freezing never invents scientific qualification.
- Admitted frozen endpoints can be retained for evaluation without new labels.
  Exclude fully frozen observations from optimization sampling while retaining
  their audit/evaluation records. Encoder-frozen auxiliary paths need no replay
  to preserve that function. Unfrozen inherited paths still require supervision.
- Coverage closure excludes verified frozen dependencies from current-update
  requirements but requires inherited feature evidence. Report inherited,
  frozen and newly trained parameter evidence separately.

## Exact resume (subsequent implementation)

Resume must restore optimizer/RNG/sampler position, partial epoch accounting,
selection/patience/coverage, source pins, scales and freeze contracts, and must
not rescore test data after a completed run. A new-stage initializer is not exact
resume. Do not mark this item complete until interrupted-versus-uninterrupted
tests prove equivalence and the registered CLI can resume safely.

## Verification

Use meaningful artificial tests for retained-output prediction equivalence,
task-driven feature projection, rejection of changed meanings, frozen-function
invariance despite sibling updates/dropout, supported-state enforcement,
optimizer membership, checkpoint reload and streamed subset fitting/closure.
Existing real IC50 fitting evidence remains pinned to its executed commit.
Run focused modular, lint/docs and relevant broader regression before publishing.
