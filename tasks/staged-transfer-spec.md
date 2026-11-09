# Add presentation without changing retained affinity functions

The current transfer guard compares elution staging fields whenever the destination
allocates staged elution. This prevents an affinity-only checkpoint from gaining
its first presentation output. No presentation function exists in that checkpoint
whose behavior needs preservation. This blocks the specified affinity-first,
unified-model training workflow.

## Scope and invariants

- Allow adding staged elution only when the source has no elution endpoint.
- Keep all existing shared-feature, architecture, assay-track, scaler and task
  removal checks. Copy every compatible trained parameter and coverage record.
- Keep rejecting staging mode, affinity dependency or detach changes when the
  source already has an elution output. Do not reinterpret learned presentation.
- Default initialization, historical checkpoint hashes and forward math stay fixed.

## Verification

1. A trained affinity-only fixture must currently fail to initialize the expanded
   model. Preserve that failure before changing the guard.
2. After transfer, both MHC classes and both assay-specific affinity outputs must
   match the original exactly. Retained scales and update evidence must transfer;
   new presentation parameters must remain unqualified until trained.
3. Freeze affinity dependencies and train the new presentation path; demonstrate
   that affinity tensors and predictions remain identical, then round-trip the
   checkpoint. Detach alone is not an encoder-freezing guarantee.
4. Existing elution mode/dependency/detach changes must still fail. Run staged,
   continuation, assay, quantitative, resume and common-trainer tests; hosted CI
   covers the integrated branch.

No active Modal runtime is modified. A subsequent fit must freeze a new source
bundle containing this reviewed change.

## Before-fix evidence

The new transfer fixture failed at `initialize_from_checkpoint` with
`ValueError: incompatible pretrained feature contract: elution_affinity_endpoint`
(1 failed, 12 deselected, 1.32s). Both the source and destination were valid models;
the source had only IC50 and the destination added staged elution.

## Verified result

The narrowed guard passes96 staged/continuation/assay/quantitative/common-trainer/
resume tests in21.25s; Ruff and diff whitespace checks pass. The regression covers
both classes and two assay readouts, transferred scalers/update evidence, exact
retained tensors and predictions after freezing plus presentation training, and
checkpoint replay. Mode/dependency/detach changes to existing elution remain
rejected. Inference comparisons use the same no-grad kernel path on both sides;
comparing grad-enabled and frozen attention kernels produces floating-point
rounding differences even when every parameter tensor is identical. No tolerance
was loosened. Logs: `/private/tmp/presto-staged-transfer-tests-20261107.log` and
`/private/tmp/presto-staged-transfer-tests-fixed-20261107.log`.
