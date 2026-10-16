# Assay-context applicability — Presto #126

The user prioritizes source lineage and assay-context applicability together with
donor/specimen independence. Add explicit inspection metadata distinguishing
not-applicable cellular processing from missing or unresolved cellular context.
This change must not alter model features, labels, optimization or historical data.

## Policy

- Preserve the original `assay_method` as `source_assay_method` in shared lineage
  for every native record/sample. Keep it outside model tensors.
- Interpret an explicit purified-MHC preparation as cellular processing not
  applicable; explicit cellular-MHC preparation as applicable. Native MS/elution
  observations establish cellular presentation applicability even when their
  method is only "mass spectrometry". Do not infer preparation from IC50/KD.
- Contradictory preparations or method/evidence contradictions have unknown
  applicability with a conflicting-evidence basis; missing methods remain unknown
  unless explicit elution evidence supplies the basis. Preserve raw text.
- Report applicability and context status separately: not_applicable,
  applicable_resolved, applicable_unresolved, or applicability_unknown. Never
  invalidate a purified-MHC affinity label for lack of presenting-cell identity.
- Generated records have no measured method. Preserve parent preparation/method
  provenance, inherit only unanimous applicability, and label the inherited basis.
  Mixed/unknown parents cannot become a known preparation.
- Thread raw and derived metadata through frozen final membership, prediction
  exports and coverage facets/fingerprints. Fail on mutated final provenance.
  Preserve original frozen files; declare a new membership policy/version.
- Bind the reporting policy in the output contract so old/new support receipts
  cannot silently compare equal. No assay text becomes a new predictive input.

## Validation and evidence

Write real classifier/contradiction/unknown/generated-parent regressions and
integration checks across loading, collation, frozen membership and coverage.
Verify identical predictive tensors and targets when only applicability metadata
changes. Audit all 9,053 published rows and all 2,614 IC50 measurements from the
prior native-presenter audit through datacache; append fields and reproduce every
prior field unchanged. Expected explicit IC50 source preparations: 2,533 purified
MHC and 81 cellular MHC. These are source facts, not a new fitting claim.

Register a new clean experiment recipe, publish readable joined rows, per-split
coverage, source/label policy and independent verification. Preserve the original
executed recipes, fitting results and historical contracts. Run formatting, lint,
full tests and docs; review, merge, release exact successful clean-main artifacts,
and download/hash/test the public wheel. #115/#48/#53/#678 remain broader work.
