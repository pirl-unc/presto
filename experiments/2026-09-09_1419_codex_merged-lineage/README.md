# Merged publication-lineage recovery

- Agent/model: Codex / GPT-6; date: 2026-09-09.
- Base: merged PR #61, `8cebd17660b67c0e4908017e0de9fccf5b00c653`.
- Status: registered; before phase pending.
- Plan: [lineage recovery](../agents/codex/plans/2026-09-09_merged-lineage.md).
- Specification: [Presto #60 repair](../../tasks/merged_lineage_spec.md).

## Source and measurement contract

Use the unchanged full `data/merged_deduped.tsv`, 3,423,737 rows, SHA-256
`46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c`.
The corrected #61 routing produces 2,702,233 typed records before caps and MHC
filtering. No study exclusions, cache rebuild or source-file edits. Observe
every input row and constructed typed record while the actual head sampler
retains one record per modality to bound memory. No post-MHC training population
is inferred from this audit.

At the reader/append boundaries compare the source's PMID, DOI, reference text,
evidence-row ID, assay IRI and reference IRI with each typed record. Record
availability, missingness, mismatches and invented values per field/modality,
including zero-count modalities. Preserve complete non-lineage record hashes
in source order. Both phases must have identical routing, skip counts, cap
statistics and non-lineage hashes. Unavailable original assay IDs stay absent;
publication identifiers cannot substitute for original observation identity.

## Execution and artifacts

Reproduce with `bash reproduce/launch.sh before` and `bash reproduce/launch.sh
after` at their recorded commits. The launcher freezes source/production hashes,
all dependency versions, invocation, git/dirty state and source snapshots before
running. Existing output directories are refused; repeat runs require a fresh
`--output-dir`. Summaries go to `results/<condition>/`; raw logs to
`artifacts/2026-09-09_1419_codex_merged-lineage/`.
Each phase freezes its actual bundle in `results/<condition>/reproduce/` and
its complete launch receipt in `results/<condition>/invocation.json`.

Use the isolated environment under
`artifacts/2026-09-09_1106_codex_output-coverage/.venv`: Python 3.12.6, torch
2.7.0, Hitlist 1.59.1, mhcseqs 2.5.12, mhcgnomes 3.41.0 and its frozen
transitive dependencies. Local CPU, OMP/MKL threads one; no GPU or Modal run.
The inventory's package-source hash receipt is linked from each invocation.

## Evaluation scope

No pretraining, optimization, synthetic generation, splitting or predictive
evaluation occurs. No validation/test split, prediction dumps or quality metrics
are appropriate for this source metadata audit. Separate deterministic tests
verify model-input/label tensor parity and row/MIL held-out lineage export.
Supported target transforms, loss terms/weights, source populations and sampler
behavior must remain unchanged. Diagnostic hashes including restored metadata
may change and are distinct from actual model-input parity.

## Handoff

- Status: before phase pending.
- Next step: capture full-source metadata loss before production changes.
- Remaining: metadata propagation, after-phase reconciliation, review and CI.
