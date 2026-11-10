# Preserve source-qualified processing conditions in the shared MS adapter

Date: 2026-11-10. Agent: Codex. Base: e554e7d3. Isolated branch:
`codex/mhc2-conditions-20261110`; concurrent Atlas work is out of scope.

## Problem and decision

`data/modular_hitlist_ms.py` admits six intervention families and cytokines, but
drops Hitlist's explicit `condition_genetic_variants`, `condition_infection`,
`condition_stimulation`, and `condition_antigen_exposure` from model context.
The original records survive in provenance, so this is a downstream feature
gap rather than absent upstream annotation. Use existing `Context.interventions`
and the common cellular encoder for both MHC classes; add no model or task path.

## Data contract and audit

Pin Hitlist checkout 43e2266 and SHA-256 of `hitlist/data/pmid_overrides.yaml`,
`conditions.py`, and `condition_vocabulary.yaml`. Inventory only curated arm
metadata. Count arms and studies by declared MHC class and condition field;
never call an arm a peptide, independent specimen, or admitted training example.
Do not read outcome tables, TEST, confirmation, or comparator predictions.
Register the source audit under `experiments/2026-11-10_1420_codex_processing-conditions/`.
No training, GPU, loss, synthetic labels, validation/test metrics or prediction
dumps apply to this metadata-only audit; biological improvement remains untested.

## Implementation contract

1. Add stable namespaced factors `genetic_variant:`, `infection:`, `stimulation:`,
   and `antigen_exposure:` to the shared source-condition mapping.
2. Keep the existing unique-arm, annotated-status, evidence and reference gates.
   Pooled, mixed and partial annotations remain provenance-only in this change.
3. Preserve simultaneous tokens, original source field/value and evidence.
   Blanks/unknown tokens remain unknown. Only source-permitted explicit `none`
   supplies absence in the same condition family and source study vocabulary.
   Antigen exposure does not permit `none` in Hitlist; reject it rather than
   inventing an absent exposure claim.
4. Never map source exposure labels to measured pathway activity or component
   status. In particular a `cross_presentation` source label is a categorical
   exposure annotation, not evidence that a specific pathway has been identified.
5. Version the adapter condition policy. Existing frozen manifests/checkpoints
   are unchanged. A new cohort must explicitly qualify vocabularies and contrasts
   through the standard preflight before training new condition weights.

## Verification and publication

- Regress retained/dropped fields, concurrent factors, explicit absence versus
  unknown, disallowed `none`, attribution/evidence gates and provenance roundtrip.
- Exercise admitted class-I and class-II observations through the actual common
  model: condition inputs affect cellular output and receive gradients while
  purified IC50 stays invariant. These are software fixtures, not performance.
- Run existing MS/context/support/preflight tests and Ruff on the frozen diff.
- Record audit counts, tests, limitations, and a prospective condition-contrast
  evaluation follow-up. Search current issues before filing; reuse existing
  Hitlist DM/DO and arm-identity issues rather than duplicating them.
- Commit all audit/fix artifacts and publish a reviewable PR without merging
  scientific milestones or unrelated concurrent work.

## Publication-base amendment

The later experimental base has an unrelated stale default-config test for
`tcell_direct_peptide`. Publish only this scoped change onto current PR131 head
`189fe276`, where that experimental flag is absent, and repeat verification.
Do not import the intervening Atlas/T-cell experimental commits or edit their
active checkout. The MS adapter and its existing test file are byte-identical
between these two bases. Keep #155 open for scientific qualification.
