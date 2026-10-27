# Source-relative ordered cellular context

Codex / GPT-6, 2026-10-27. The previous goal turn was progress: DO:DM source
measurements and exact biological-sample joins were independently verified in
experiment0400. All known fit handles are terminal. Current RAM15.61GiB is below
the unchanged pending MS/censor20GiB/60s admission gate; do not launch them.

## Problem and required behavior

The qualified DO:DM source distinguishes low/medium/high ratios within a defined
T2-derived series. Current component states and interventions cannot retain that
ordering as a distinct typed observation without inventing activity values or
mislabeling a state as an intervention. Add general, source-relative ordinal
context, with an explicit scale reference. It must not create absolute activity,
equal spacing, monotonic biological response, cell-ID embeddings or a DO-only
causal interpretation. The exact DRA remains unresolved; do not admit fabricated
paired-MHC observations or call this software work biological improvement.

## Runtime isolation

Pending native MS scoring imports the evaluation checkout. Preserve that runtime.
Create /private/tmp/presto-mhc2-context-20261027/presto from committed3260488 in a
separate local clone/branch. Its origin is local; no push. Keep original dirt,
all experiment/source artifacts, shared environment and old runtimes untouched.
Record a patch/branch handoff back to the goal workspace, not an early merge into
the pending evaluation runtime. Use the shared Python with the new parent on
PYTHONPATH and the existing dependency overlay. No installation/cloud/GPU.

## Data and model contract

- Add immutable OrdinalScale(levels, reference) and OrdinalValue(level, scale,
  evidence). A scale has at least two distinct, nonempty ordered string labels
  and a nonempty reference describing its calibration/comparison frame. Unknown
  is represented by None, never by the first/lowest level. Known values require
  a scale reference; unknown values cannot carry a claimed known scale.
- Context receives named ordinal_values and ordinal_overrides. Explicit unknown
  overrides clear supplied values. Preserve supplied/override/effective records,
  their origin and evidence in diagnostics. No species/cell-based defaults.
- ModularConfig receives ordinal_scales keyed by biological observable. Require
  cellular outcomes, typed scales and valid feature names. Encode one column per
  ordered label plus unknown in deterministic sorted feature-name order. Ordering
  is retained as metadata; one-hot encoding does not impose equal spacing or a
  monotonic response. Reference strings and evidence are never tensor features.
- On cellular routes, reject unknown allocated names, unsupported levels and
  mismatched scale references. Unallocated families are reported as ignored,
  following existing context behavior. Noncellular requests preserve diagnostics
  while masking these inputs from purified binding, fixed-pMHC TCR binding,
  digestion, bulk MS and the independent MS-response component.
- Default empty fields leave tensor allocation, random initialization and
  predictions unchanged. JSON/config/checkpoint round trips retain the contract.
  Empty optional fields do not change historical semantic fingerprints; evidence
  text is excluded but level/scale changes remain semantic. Checkpoint transfer
  must reject changed scale order/reference/name even when shapes match.
- Preflight requires all declared levels on training cellular observations unless
  the contextual weights are frozen under the existing contract. Held-out rows
  and unknown values cannot satisfy coverage. Feature update tracking and strict
  prediction reject untrained levels, including unknown when untrained.

## Verification

Use artificial software fixtures only. Tests cover malformed/ambiguous values,
three-level distinction and unknown, override precedence, reference mismatches,
missing/unallocated context, serialization, stable historical fingerprints,
deterministic feature order, cellular-only effects, endpoint/component masking,
actual gradient/weight updates, unseen-level rejection, checkpoint reload and
same-shape incompatible transfer, and train-only preflight coverage.

Run focused tests, then the existing modular/context/checkpoint/manifest tests
affected by the new optional fields; lint/format changed files. Compare no-ordinal
model state/predictions/fingerprints against the immutable base in separate
processes. Record exact commands/results, source hashes and the isolated patch.
No predictive experiment or source/model score is run here, so no synthetic
biology, validation/test model metrics or experimental winner is claimed. Update
docs and task review; leave the full binding/MS/fitted-pathway goal active.

## Review checklist

- [x] Isolated checkout and frozen before-change comparison inputs.
- [x] Schema, resolution, cellular features and diagnostics implemented.
- [x] Lineage, preflight and checkpoint compatibility integrated.
- [x] Meaningful tests and old/new parity verified.
- [x] Documentation and patch handoff recorded without altering pending runtime.

## Review

All219modular tests pass in15.53s, including the new ordinal and old-checkpoint
cases. Nine changed/new code and test files pass lint/format. The isolated
before/after artificial-fixture comparison is exactly equal for47state tensors,
five endpoint predictions/component outputs, feature layout and five historical
semantic fingerprints. Supplied levels and scale references stay in input
identity; evidence text and absent optional fields do not alter it. All scientific
evaluation runtimes remain unchanged. Local implementation is ready for the next
registered condition comparison; no biological improvement or full-goal completion
is claimed. `tasks/mhc2_ordinal_context_20261027/verification.json` records exact
commands, source hashes, runtime paths and original source evidence.
