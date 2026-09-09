# Full canonical output coverage after the adapter repairs

Agent/model: Codex / GPT-6. This extends the question in
`2026-09-08_output-coverage-census.md` after the source/lineage/selector defects
were repaired. Because effective curation changed, use the new registered family
`experiments/2026-09-09_1541_codex_canonical-coverage/`; preserve the old inventory.

The complete prospective specification, including the six source/augmentation
conditions, canonical execution path, instrumentation constraints, hardware and
closure criteria, is `tasks/canonical_coverage_evidence_spec.md`.

First run only uncapped `merged_measured` with CPU 4/8, RAM 64/192 GiB and a
four-hour limit. Inspect real stage counts, peak memory and runtime before the
other conditions. No model fitting or predictive evaluation is part of that run.
Training candidates are selected solely from train; optimization settings and
supported claims must be frozen in distinct phase receipts before their use.
