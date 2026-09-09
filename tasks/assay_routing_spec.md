# Explicit assay routing for merged observations — Presto #59

## Problem and scope

The registered real-source inventory found 492,858 non-MS-method observations
routed as elution and two structural measurements interpreted as nM affinity
within the complete MS-excluded-study subset. The current classifier uses scalar
presence/absence as its fallback assay definition. Correct classification must
precede the full #48 census and #53 quality experiment. Publication lineage is
separately tracked in #60; Hitlist's study exclusions remain upstream in #444.

This PR changes only assay classification and explicit ingestion accounting,
plus necessary documentation/tests and a registered before/after source audit.
It must not add a biological target, invent affinities for qualitative outcomes,
change model/loss equations or silently apply a study blacklist to shared data.

## Required behavior

1. Inspect all actual source assay descriptors before finalizing the routing
   vocabulary. Stream the complete frozen merged file through the real loader,
   observing every classification and emitted record before a one-record head
   cap bounds retained memory. Record pre-cap counts and ordered record hashes
   per descriptor group; the retained cap is never reported as an uncapped
   training population. This pass performs no MHC resolution or training.
2. Define supported quantitative affinity, kinetics and stability measurements
   using explicit measurement labels. Preserve valid values, original units and
   qualifiers. Exact/token-boundary matching must prevent accidental matches of
   short tokens inside unrelated method names or of association within
   dissociation. Inspect quantitative method labels and assay-type-only cases.
3. Establish elution from explicit elution/MS evidence or a documented existing
   presentation observation contract, never from an absent number alone. Keep
   acquisition subtypes conditional on presentation/MS semantics; substrings
   such as `dia` or `targeted` in unrelated assays are insufficient.
4. Represent qualitative binding, structural and unknown measurements as
   distinct unsupported training buckets unless an existing suitable objective
   and measurement contract exists. Preserve them in source/export inventory;
   expose their exclusion from typed training targets in detailed, disjoint
   source-ingest reasons. Do not label their unsupported status as zero evidence
   for all biological questions or manufacture an nM threshold.
5. Ensure the canonical loader, focused binding selectors, dedup statistics and
   assay CSV exports share classification semantics. Verify that unsupported
   source rows remain exportable and cannot feed elution-derived cell-HLA joins.
6. Preserve all unaffected emitted records exactly. The registered before/after
   audit must reconcile every route, pre-cap record count and skip reason and
   compare ordered payload hashes for unchanged descriptor groups. Any changed
   valid quantitative family must be explained, not accepted as incidental.

## Validation and acceptance

- Semantic fixtures cover positive/negative microarray and fluorescence binding,
  structural scalars, unknown values/methods, supported quantitative aliases,
  association versus dissociation, assay-type-only metadata and MS acquisitions.
- Loader/selector/export/funnel integration proves accurate unsupported counts,
  unchanged retained labels/units/qualifiers and explicit supported routing.
- Full-source before/after classification and typed-record evidence is frozen in
  `experiments/2026-09-09_1133_codex_assay-routing/`, with exact source hash,
  invocation, production code state and launcher snapshots. No validation/test
  metrics are appropriate: this is an ingestion audit with no fitted model.
- Review the complete diff, run affected tests and pinned Ruff, then full CI.
  The PR closes #59 only after demonstrated source correction. #48/#50/#53 and
  lineage #60 remain open for their separate evidence/work requirements.

## Sequence

- [x] Register and execute full-source baseline before production edits.
- [x] Finalize descriptor policy against baseline groups and implement it.
- [x] Verify semantic tests and execute the registered after condition.
- [x] Reconcile source transitions and unaffected payload hashes; close artifacts.
- [ ] Review, publish, verify CI and merge; start lineage #60 next.

## Baseline-informed policy

The baseline at clean `1c5d1ea` completed in 100.868 seconds. All 3,423,737 input
rows are accounted for: 156,765 invalid peptides and 3,266,972 classifications
in 927 descriptor groups. Every constructed record reconciles with pre-cap
loader counts. Structural scalars (1,239), qualitative-binding scalars (6,250)
and equilibrium association constants (4) currently enter inappropriate
quantitative objectives. The five supported concentration response types total
241,803 rows. Explicit MS-method presentation totals 2,073,797 rows; 1,166 other
presentation rows have Edman degradation, T-cell recognition or coelution
methods and cannot be called MS observations by this adapter.

Use explicit normalized controlled measurement labels and compact aliases.
Prefer `value_type`, then `assay_type`, then an exact measurement label in
`assay_method` when higher-priority fields are absent. Do not guess a unit or
family from an arbitrary scalar or a short substring. Distinguish qualitative
binding, structure, equilibrium association constant, non-MS/unknown-method
presentation and otherwise unknown binding buckets. Missing numeric labels in
supported quantitative families retain their family but get a missing-label
skip reason. Explicit `record_type=elution` is already an elution declaration;
binding-format rows require an explicit MS method or a presentation label plus
a recognized MS acquisition term. Presentation alone with an absent method is
not an MS declaration. Valid unsupported source observations remain exportable.

Persist mutually exclusive skip reasons and assert that they sum to the existing
aggregate unroutable/missing-label counter. The normalized funnel substitutes
the detailed reasons for that aggregate when available, so it cannot double
count the same omissions. Binding selectors must use the same fallback
measurement label when `value_type` is absent; preserve all observed descriptors.
