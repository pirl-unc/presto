# Verified scoped proteomics consumer — Presto #70

## Delivery boundary

First PR: verified source admission, one-vote record projection, retained host
lineage and explicit output-support classification. Follow with canonical trainer
selection, frozen within-search partitions and a registered bounded real-data
preflight. Keep #70 open until that end-to-end demonstration ships. MHCflurry is
out of scope. Existing user checkouts and shared editable environments stay intact.

## Source and label contract

Use Hitlist >=1.75.0 and its published PXD013455 archive by default. An explicit
local bundle requires its manifest SHA-256. Verify the full inventory, then retain
a datacache-owned immutable copy; never inspect mutable legacy bulk indexes.
The exact manifest and candidate/source/policy/reference hashes travel in receipts.
Preserve original and reanalysis citations, cell type, conditions, MHC-not-applicable
status and unresolved donor/specimen/replicate identities in the source sidecars.

The target is binary observed_in_selected_search_scope, conditional on a parent
supported in the same search. It is proxy evidence for the model's detectability
latent, not measured intrinsic detectability or a depth score. Preserve By MS/MS,
By matching and not-observed categories separately. Biological replicate counts
and first-seen depth remain None, including for unobserved candidates (the source's
aggregate zero must not become an established biological replicate count).
Do not create new excision targets or wrong-enzyme controls in this adapter; the
legacy path retains its separately identified rule-derived controls and no
qualified detectability labels. No cross-protocol mixing is implied by this API.

## One vote and occurrence lookup

Collapse every candidate occurrence of a peptide within the verified scope into
one training record before any cap. Stable sample identity is bundle plus scoped
peptide identity, distinct from a positive source-observation ID and candidate
occurrence ID. If exactly one occurrence exists, retain its coordinates/parent and
flanks. If multiple occurrences exist, mask parent/flanks and expose their count;
the complete candidate sidecar remains recoverable by bundle + peptide. Never
select an arbitrary protein context or turn alternate occurrences into extra votes.
Keep leading-razor source assignment separate from candidate parent support.

Use a bounded-memory SQLite scratch index inside datacache's staging directory to
collapse all occurrences and validate consistent scope, search, labels and source
observation linkage. SQLite is scratch, not a second permanent data store. Publish
records.parquet, coverage.json, label_policy.json and README through datacache,
with code/version/options and all source hashes in the materialization identity.
Apply deterministic seeded reservoir caps only after the full projection; report
full and admitted support by evidence class and ambiguous occurrence counts.

## Host metadata and support

Add explicit scoped peptide/candidate/search/protocol/reference/evidence/count
fields to BulkMSRecord and PrestoSample. Carry them through existing host lineage,
collation and prediction exports; none are predictive input tensors. Positive
records retain their original observation ID; negative candidates do not invent
measured observation or assay IDs. Human receipts explain sidecar lookup and
retain acquisition/cell/citation policy without duplicating raw files per row.
Classify all three scoped labels as proxy support, separately from raw observed
product excision proxies, rule-derived controls and unknown provenance.

## Verification and subsequent integration

Use tiny real-schema bundles verified by Hitlist's actual verifier. Test bad pins,
missing/corrupt inventory, altered/incomparable scope, unknown replicate/depth,
MS/MS vs matching, candidate multiplicity, missing/conflicting source linkage,
ambiguous context masking, deterministic post-collapse caps and datacache reuse.
Exercise real dataset conversion, collation, output-support census and prediction
lineage. Compare legacy raw bulk behavior. Bump version, format, lint, full tests,
review, merge and verify the public GitHub release (PyPI namespace conflict #73).

The next PR must expose this only through the canonical trainer, explicitly scoped
bulk-only, with frozen peptide-disjoint within-search partitions. One experiment
cannot establish independent experiment/donor/specimen holdouts; those audits must
show shared/unknown axes and reject broader claims. Add scope-aware contrast
sampling only when labels and actual pairs support it. Register the real-data
preflight before running it, publish membership/hashes/support/audits in datacache,
and preserve prediction exports for subsequent bounded fitting and baselines.

## Resource plan

A full checkout encountered shared-workstation ENOSPC. The tree contains 418 MB of
tracked data and 904 MB of archived experiments, which does not explain the full
concurrent disk change. Use a sparse feature checkout; retain all original files.
The migration to pinned shared data artifacts is tracked separately in #80. Do not
bypass memory/disk safeguards or run a full corpus build on this pressured laptop.
