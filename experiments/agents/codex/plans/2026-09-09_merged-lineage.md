# Merged publication-lineage recovery and payload parity

Agent/model: Codex / GPT-6. Implementation and validation specification:
[`tasks/merged_lineage_spec.md`](../../../../tasks/merged_lineage_spec.md).
Base: merged PR #61, `8cebd17660b67c0e4908017e0de9fccf5b00c653`.

Before the first run, register a new timestamped experiment directory. Compare
the canonical merged loader before/after restoring publication metadata through
records, samples, collation and held-out export. Reuse the immutable merged file
from the completed [assay-routing audit](../../../2026-09-09_1133_codex_assay-routing/):
SHA-256 `46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c`,
3,423,737 source rows and 2,702,233 supported records before MHC filtering/caps
under the corrected routing policy. Do not rebuild or blacklist source data.

Use the pinned isolated inventory environment, local CPU and OMP/MKL threads
one; no GPU, pretraining, optimization, synthetic generation, split change or
predictive evaluation. Validation/test predictions and quality metrics are
intentionally absent because the experiment verifies ingestion metadata and
non-lineage payload parity. Separate deterministic fixtures verify actual model
input and label tensors plus row/MIL metadata export.

At the canonical reader/append boundaries, count source-field availability and
typed-record metadata preservation across every modality before a one-record
head cap bounds memory. Preserve every non-lineage record fingerprint in source
order. Both runs must conserve routing, skip counts and non-lineage payloads;
restored publication metadata must match its corresponding source row. Original
assay identifiers absent from this input remain absent, with fallback identity
explicitly distinguished from original observations.

Freeze exact invocations, commits/dirty state, source/dependency hashes and
launcher/production snapshots. Close per-field/per-modality CSV/JSON summaries,
runtime and all failures in the experiment README and canonical log before the
PR claims completion. This supplies #60 evidence; it does not close #48/#50/#53.
