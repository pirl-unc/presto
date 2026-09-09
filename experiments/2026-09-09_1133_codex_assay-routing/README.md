# Full-source assay routing correction

- Date/agent/model: 2026-09-09; Codex / GPT-6.
- Status: registered before/after audit; baseline not yet launched.
- Issue: [Presto #59](https://github.com/pirl-unc/presto/issues/59).
- Specification: [assay routing](../../tasks/assay_routing_spec.md).
- Prior evidence: [source inventory and flagged-study trace](../2026-09-09_1106_codex_output-coverage/).

## Contract

Compare the canonical merged loader before/after explicit assay routing using
the identical complete `data/merged_deduped.tsv`: 3,423,737 rows, SHA-256
`46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c`.
No upstream cache rebuild, study exclusions or source-file edits. Observe all
classifications and every typed record constructed before cap sampling; retain
one record per modality using the actual head sampler to bound memory. The
audit reports pre-cap counts and grouped ordered payload hashes, and reconciles
them with the loader statistics. No MHC filtering or effective training count
is inferred from this source audit.

Conditions are `before` (the unchanged main adapter) and `after` (the #59
correction). Descriptor groups retain source, record type, value type, assay type,
method, response and scalar-presence metadata. Group payload hashes include all
typed-record fields; changed descriptor routing is explained at closure and
unaffected groups must retain exactly identical payloads and order.

## Reproduction and environment

Run `bash reproduce/launch.sh before` or `bash reproduce/launch.sh after` from
this checkout. Each condition freezes its invocation, source state, production
module snapshots, launcher and dependency versions before reading data. Existing
output directories are refused; supply a fresh `--output-dir` for a repeat.

Use the pinned isolated environment at
`artifacts/2026-09-09_1106_codex_output-coverage/.venv`: Python 3.12.6, torch
2.7.0, Hitlist 1.59.1, mhcseqs 2.5.12, mhcgnomes 3.41.0. Its complete package
and biological-parser source hashes are recorded by the linked inventory;
each new invocation records its own installed versions and that receipt's hash.
Local CPU; OMP/MKL threads one; no requested or observed GPU. Runtime includes
the full source hashes before and after, classification and record construction.

Raw console logs: `artifacts/2026-09-09_1133_codex_assay-routing/`. JSON/CSV
summaries and fingerprints belong in this experiment's `results/<condition>/`.
The input remains in shared `data/`; no full duplicate is required.

## Scientific interpretation

No pretraining, optimization, generated data, model selection or predictive
evaluation occurs. No validation/test split is used, so prediction dumps and
held-out metrics are intentionally absent. Existing assay-label transformations,
loss terms and weights remain unchanged for supported measurements. The
correction must prevent unsupported structural/qualitative measurements from
being relabeled as nM affinity or MS outcomes, with explicit source-ingest counts.
This does not establish supervision adequacy (#48/#50) or model quality (#53).

## Handoff

- Status: baseline pending.
- Next step: full-source baseline, then finalize descriptor policy and implement.
- Remaining: after-condition comparison, semantic regressions, review and CI.
