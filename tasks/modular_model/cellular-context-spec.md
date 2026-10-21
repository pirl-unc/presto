# APC cell type, cytokines and sourced machinery priors

Part of the full completion goal; this change supplies missing typed/model paths.
It does not claim curated biological cohorts, validated profiles or measured gains.

## Data and resolution

- Add independent APC `cell_type` and `cell_type_override` fields. Names come from
  an explicitly declared cell-type vocabulary, never cell-line identifiers.
- Add per-cytokine `Exposure` records and independent per-cytokine overrides.
  Each record distinguishes present, absent and unknown, with optional finite
  nonnegative dose/unit and duration in hours plus source evidence. Missing is
  unknown, not an untreated control. Concurrent exposures remain separate.
- A `MachineryProfile` declares an ID, species, optional cell type, component
  priors, source reference and version. Resolve defaults from a species-wide
  profile then a matching cell-specific profile, then actual sample component
  annotations, then explicit overrides. Reject ambiguous duplicate scopes.
  Preserve the selected profile/assumption in component evidence. Retain the
  earlier species-name-list profiles as explicitly unverified reference priors;
  they do not become source-qualified facts or measured perturbations.
- Explicit unknown annotations override a default just as intact/lost annotations
  do. Unsupported species/cell types must not borrow another profile. Declaring a
  profile is an explicit caller assumption, not automated biological curation.

## Model and training

- Cell type uses a configured one-hot vocabulary plus unknown. Each cytokine has
  three state columns. Allocate dose/value-known columns only for cytokines with
  declared dose units; allocate duration/value-known columns only where declared.
  Numeric inputs use fixed `log1p(value)` transforms, retaining explicit units.
- Use the existing contextual fusion; add no new neural stage. Only cellular and
  paired functional-response routes consume these fields. Purified affinity and
  stability, fixed supplied-pMHC receptor binding, digestion and the shared MS
  detector must remain invariant to them.
- Keep raw annotations, effective overrides, reference provenance and ignored
  context families visible in diagnostics. An unallocated family is intentionally
  withheld; an unsupported value inside an allocated family is an error.
- Extend feature-column support, checkpoint vocabulary compatibility and preflight.
  Training must observe every configured cell category and contrasted present/
  absent cytokine states; unknown and reference states cannot fake a contrast.
  Declared numeric channels need varied measured training values. Profile priors
  do not substitute for sample-attested component contrasts.
- Preserve schema/checkpoint round trips. Existing configurations with no added
  fields keep their behavior and dimensions. No random weights are authorized
  merely because new context fields exist.

## Verification

1. Resolution precedence, concurrent exposures, unknown versus absent, dose/unit
   and duration validation, explicit unknown overriding a reference.
2. Endpoint masks and MS-component invariance; relevant cellular sensitivity and
   nonzero gradients in newly active columns.
3. Preflight contrast/coverage failures and unknown/untrained-category serving
   rejection; source normalization and checkpoint round trips.
4. Modular regression, pinned Ruff and documentation checks. Real matched-cohort
   fitting remains a separate registered experiment, with no claim from fixtures.
