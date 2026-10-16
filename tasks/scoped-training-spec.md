# Canonical frozen scoped detectability training — Presto #70 / #48 / #53

## Source and claim

Add `data_source=scoped_bulk` to the existing canonical trainer and CLI. Consume
only the hash-verified public PXD013455 bundle (or an explicitly pinned local
copy) via `load_scoped_bulk_records`. Preserve its global one-vote-per-peptide
projection before deterministic capping. Do not fall back to TSV/live Hitlist/raw
bulk, add unpinned MHC/UniProt auxiliaries or fabricate canary modalities.

This is a peptide-disjoint evaluation within one documented HeLa search. It is
not an independent experiment/donor/specimen evaluation. Original and reanalysis
sources describe this same captured experiment. Shared proteins and ambiguous
candidate parents remain auditable limitations; matching peptides, cell names,
MHC genotypes and PMIDs never establish missing donor/specimen identities.
No new cleavage labels, depth targets or generated negatives are supplied.
MS/MS, matching and scoped absence remain separate evidence categories, all
proxy supervision for the explicitly named search-observation endpoint.

## Frozen membership and datacache

Before dataset conversion/training, freeze selected peptide identities and
train/validation/test membership. Use deterministic peptide groups with seed and
fractions recorded, or a validated hash-pinned assignment artifact. Require
exact scoped IDs, complete source scope and one row per peptide. Candidate IDs
and peptide lookup keys are distinct from measured observation IDs: an absent
candidate must never receive an invented assay ID. Preserve source partitions
through the existing records, samples, host batches and prediction exports.

Publish immutable datacache artifacts: assignment parquet, legible membership
TSV, label policy, source registry/citations, source/projection pins, selected
coverage and pairwise overlap audits. Audit peptides, candidates, measured IDs,
known source proteins and the shared search/experiment; donors/specimens are
explicitly unresolved. Include a zero-row generated-parent graph with an
explicit no-generated-controls policy, not fabricated parent measurements.
The canonical coverage preflight must pin this membership receipt so fitting
can require the exact preflight supervision fingerprint. A dataset conversion
that loses/adds/changes membership must fail rather than regenerate a split.

## Integration and verification

Use the existing canonical loader/coverage/loss/evaluation paths. Do not build a
second trainer. The scoped source bypasses unrelated merged-input requirements
and has no MHC records to resolve. Keep legacy and frozen MHC behavior intact.
Only peptide-disjoint within-search policy is permitted for this source; reject
unsupported stronger policies, row-random splits and mixed/live auxiliaries.
Both CLI entry points expose the source, input pins and cap. New model/checkpoint
contracts retain distinct Trypsin/P vocabulary and corrected scoring versions.

Test frozen membership reuse, wrong hashes, corrupt/duplicate/conflicting IDs,
external assignments, deliberate leakage, changed conversion, scope restrictions
and actual canonical preflight/CLI dispatch with small synthetic schema fixtures.
Then register a real-data experiment before running: bounded actual admission,
fixed splits, output-support audit, bounded fitting and held-out baselines.
Publish exact inputs, parameters, versions, label policies, per-example prediction
exports and recomputable metrics. A fitting smoke test alone is not readiness.
Every claimed output needs measured support evidence; other outputs remain
unsupported. The broader MHC cohort qualification proceeds separately in Hitlist
#680 and will use the existing frozen MHC trainer route.

Version 0.1.8; format/lint/full tests, reviewed PR, merge and verified distribution
before claiming the integration shipped. #70/#48/#53 and Hitlist #678 stay open
until their actual artifact/evaluation requirements are met.
