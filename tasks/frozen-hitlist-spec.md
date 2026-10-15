# Frozen Hitlist inputs and retained lineage — Presto #71

## Scope

Consume verified frozen Hitlist training bundles in the canonical Presto trainer.
The active repositories are Hitlist and Presto; MHCflurry is explicitly deferred.
Hitlist #645 / PR #686 must ship before this consumer series is released.
Implementation tests establish the adapter contract; real corpus qualification,
scoped detectability and model evaluation remain subsequent stages of #678.

## Frozen acquisition and admission

Accept a bundle directory or a pinned downloadable artifact definition. Verify
Hitlist's manifest, payload hashes, identities, contributor coverage and lineage
registry before creating records. Install/reuse the exact artifact set with
public datacache bundle APIs under the shared Hitlist data root, keyed by the
manifest SHA-256. The shared generation is immutable and usable offline. Do not
copy raw source datasets into experiments or introduce another download cache.
Record manifest identity, source location, all sidecar hashes and label policy
in the run's data-funnel/receipt, with readable source and artifact references.

The canonical `--data-source hitlist` training path must select a frozen bundle
explicitly. Preserve any necessary live-loader compatibility as an explicit
legacy API path, without presenting it as qualified/reproducible training.
Never silently regenerate a bundle, accept stale inputs or change cohort scope.
Read only needed parquet columns; source mapping alternatives remain one assay
observation. Apply requested filtering/caps deterministically and record the
actual selected observation IDs. Reject missing required columns or conflicting
contexts rather than filling fabricated metadata.

## Lineage and label semantics

Carry the bundle identity and stable original observation/contributor lookup
through mapping collapse, record conversion, generated controls, dataset samples,
batching and held-out prediction exports. Raw contributor payloads remain in
shared sidecars, rather than being duplicated into each training record.
Generated records have explicit linkage to every parent observation (including
the donor of a mismatched peptide) and generation kind;
they are never counted as additional measured observations. Preserve publication,
assay, source record, observation, origin, specimen, donor and acquisition roles
and unknown/pooled/ambiguous resolution. A sample display label, cell-line name,
PMID or genotype never substitutes for reviewed biological identity.

Keep original assay endpoint, measurement units and censoring recoverable and
explicit in the run policy/prediction metadata. Record target transformations
(such as minutes to hours) separately from the original measurement. Distinguish
measured labels, generated controls and proxies. Preserve both cellular typing
and the experimental MHC restriction/candidate set, and cell type, line,
condition and assay metadata in the frozen source artifact for human inspection.

## Actual partitions and audit

Write the selected train/validation/test membership after every filter, mapping
collapse, sequence-resolution step and generated-negative transformation. Accept
frozen observation partition assignments, with generated descendants inheriting
their parents' membership. Generate controls within the assigned partition;
creating random cross-parent controls before splitting can join unrelated
partitions or erase the origin of the donor peptide. Preserve every parent,
and fail if a generated record would span partitions. Where a seeded split is requested, record that actual
assignment as an artifact and audit it; a policy string alone is not a split.

Use Hitlist's frozen lineage registry and split-audit semantics to report peptide,
pMHC, observation, source reuse, reviewed experimental origins, specimens and
donors separately. Under a declared independence policy, keep known reused
experiments together and reject overlap; unresolved required identity produces
inconclusive coverage or an explicitly restricted cohort, never a pass. Preserve
selected source IDs even when no biological-independence claim is requested.
Include policy, assignment hashes, row/observation/parent counts, unknown counts
and readable conflict examples in the run artifacts.

## Verification and release

Regressions cover a verified frozen bundle, missing/corrupt files and columns,
repeated mappings, cross-PMID reuse, unknown specimens, generated-parent lineage,
actual partition assignment, collator host metadata and prediction export parity.
Use tiny real-schema fixtures; do not rely only on mocked dictionaries. Compare
labels and filtering against the current adapter for a fixed input, recording
any deliberate policy change. Run the pinned formatter/linter and full test,
docs, package and installed-wheel checks in an isolated environment/CI.

Land two bounded PRs if needed: frozen ingestion and metadata propagation first,
then generated-parent/partition integration and actual cohort audits. #71 remains
open until both contracts are satisfied. Each PR bumps the version, merges only its tested head,
then publishes verified clean-main wheel/source artifacts to the established
GitHub release destination. PyPI `presto` belongs to an unrelated project.
Do not modify the user's dirty working tree or shared editable dependencies.

## Remaining series

## Full-suite review adjustment — #77

The first complete CI run found seven integration failures. Keep legacy live
adapter compatibility at 1.53 while enforcing a separate 1.74 frozen-input
runtime floor matching packaging and remote workers. Update the competing-input
fixture to supply a bundle and explicitly expect rejection of live bulk union.

Real-corpus qualification must select a frozen bundle and its SHA-256 through
`PRESTO_TEST_HITLIST_BUNDLE` and `PRESTO_TEST_HITLIST_BUNDLE_SHA256`. With neither,
report these three checks as unavailable; with either alone or a broken supplied
bundle, fail. Preserve both class-specific flank coverage floors and the complete
observed condition-vocabulary check. Ordinary CI still exercises real Hitlist
generation, export, verification and ingestion on a small fixture. Document and
run the separate qualification invocation when #680 publishes the actual corpus.

Verify the changed contracts with the failing test files and frozen integration
tests, then rerun formatting, lint, docs and the complete suite on a fixed head.

After #71, implement verified search-scoped detectability (#70/Hitlist #681),
then build/publish the actual pinned corpus and frozen cohorts (#680, including
CI publication fix #685), then support census and bounded real-data qualification
(#48/#53). Before experiments, follow experiments/EXPERIMENT_WORKFLOW.md and its
canonical registry/reproduction/prediction-dump requirements. Sources and large
datasets stay in datacache; experiment records point to their exact identities.
