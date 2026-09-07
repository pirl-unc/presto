# Quantitative metadata preservation source audit (#49)

Agent/model: Codex / GPT-6. Date: 2026-09-07.

Run the actual merged adapter from before and after the metadata-only repair on
the identical local data/merged_deduped.tsv, preserving source hash and order.
Use head caps of one retained record per modality to bound memory. Instrument
quantitative constructors before cap selection to count all routed source
observations; do not describe these as post-curation split counts.

Freeze both loader function sources and the analyzer. Pin the production helper,
record, collator and vocabulary files by SHA-256. Preserve exact command/env,
git revision/dirty state and installed mhcseqs metadata for the real-row trace.
Record raw counts, descriptor missingness, per-axis selector counts and changes,
and ordered fingerprints of all non-descriptor quantitative record fields.
Require unchanged row counts, numeric/qualifier/provenance fingerprints, and
loader funnel statistics before interpreting newly supported columns.

Trace the real EVMPVSMAK / HLA-A*03:01 / 473 nM source example through
BindingRecord, PrestoDataset and the collator with the real MHC resolver. Save
resolved sequence/metadata and selected indices. Use the unit gradient test to
verify observed-column loss routing; no optimizer steps or model-quality claims.

No pretraining/training, new splits or synthetic augmentation. Validation/test
metrics and prediction dumps are intentionally absent because this is an
ingestion/selector census, not predictive evaluation. Register results with
runtime, hardware (local CPU, no GPU), snapshots, source-artifact paths,
before/after tables, numerical invariants and explicit remaining #48/#53 gaps.
