# Merged publication-lineage recovery

- Agent/model: Codex / GPT-6; date: 2026-09-09.
- Base: merged PR #61, `8cebd17660b67c0e4908017e0de9fccf5b00c653`.
- Status: before/after reconciliation completed; PR review and full CI pending.
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

## Baseline result

The before phase ran at clean `15f28552a54ca2fb1c7a508ab7270db4a8ff4717` in
147.505 seconds. All 3,423,737 source rows and 2,702,233 supported records
reconciled. All available publication fields were lost at typed-record
construction. Exact availability by modality/field is preserved in
[`results/before/field_counts.csv`](results/before/field_counts.csv); DOI is
populated only on 1,316 emitted TCR-evidence rows. Original observation/assay/
reference-ID columns are absent from the input. No source or production hash
changed during the run. This establishes ingestion loss, not predictive quality.

## Handoff

The after phase ran at clean `87ce95b2515a9a3be2dcdc05017bff2e1b20e856` in
187.212 seconds. Comparison at clean `859be29e353d4937dfe61a584cda1953281aa0ec`
completed in 1.730 seconds. Both phases use identical dependency versions,
package-source receipt, Python/platform, source hash, caps, seed and OMP/MKL
settings. These instrumented scan runtimes are not a controlled throughput
benchmark. [Comparison JSON](results/compare/result.json) and
[per-field CSV](results/compare/field_counts.csv) reconcile all 42 field/modality
groups and verify every frozen launcher/production snapshot and CSV/JSON pair.

| Modality | Accepted records | PMIDs recovered | DOIs recovered | Reference texts recovered |
|---|---:|---:|---:|---:|
| Binding | 241,803 | 62,712 | 0 | 179,091 |
| Kinetics | 102 | 102 | 0 | 0 |
| Stability | 12,259 | 3,131 | 0 | 9,128 |
| Processing | 0 | 0 | 0 | 0 |
| Elution | 2,073,797 | 2,061,734 | 0 | 12,063 |
| T-cell | 207,987 | 171,899 | 0 | 36,088 |
| TCR evidence | 166,285 | 127,095 | 1,316 | 125,806 |
| Total | 2,702,233 | 2,426,673 | 1,316 | 362,176 |

Before values were zero in every publication field. After values match every
available source value, with **zero dropped, invented or changed values**.
Field counts overlap and are not additive observation counts. Original
observation/assay/reference IDs remain absent in this corpus. Processing has no
real rows here; its propagation is covered by the all-modality fixture.

All seven ordered non-lineage payload hashes, source metadata hashes, input
counts, routing/skip reasons and cap statistics match exactly. The unchanged
population excludes 156,765 invalid peptides and 564,739 unsupported or
missing-label classifications under #61's policy. No new training population,
supervision, validation/test performance or original assay identity is claimed.
The preferred condition is the repaired metadata path; output coverage and
fresh fitting/generalization evidence remain #48/#50/#53 work.

Validation passed 434 affected regression tests in 7.38 seconds and 48
focused-probe/audit checks in 22.08 seconds. The initial expanded test command
named a nonexistent focused-probe test file and collected no tests; that receipt
is retained as `focused-tests.log`, while the actual passing expanded suite is
`focused-tests-v2.log` under the raw-artifact root. Ruff 0.16.0 lint/format passed
(253 files); the experiment's canonical scripts were checked separately.
The tests prove actual input/target/selector tensor parity, missing-ID fallback
semantics, cap selection, distinct observations sharing a publication, device
transfer, and source-row alignment in real row/bag prediction export code.

The raw cache upstream defect remains Hitlist #444 (still open at review), with
independent evidence already posted there. Newly reproduced alternate-selector
descriptor loss is Presto #62; it is outside this metadata-only repair. Final
PR review/CI/merge remain pending. No deployment workflow exists in Presto.
