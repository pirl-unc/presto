# Alternate binding-selector descriptor recovery

Date/agent/model: 2026-09-09; Codex / GPT-6. Status: registered before launch.
Plan: [selector repair](../agents/codex/plans/2026-09-09_binding-selectors.md).
Base is reviewed PR #63 head `2cce640`; the eventual PR follows its merge.

## Source and selection

Read `/Users/iskander/code/presto/data/merged_deduped.tsv` without mutation:
3,423,737 rows, SHA-256
`46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c`.
Run the actual panel and bootstrap merged selectors with HLA-A*02:01 and
HLA-A*03:01, seed 17. Panel retains one record using head sampling while the
audit observes every accepted record before the cap. Bootstrap selects at most
500 peptides / 2000 records / 4 rows per peptide with its existing algorithm.
Bootstrap counts describe that selected subset; neither condition claims full
corpus coverage or post-MHC support. No filtering or curation policy changes.

Compare normalized source assay type/method and effector/APC culture fields with
typed records. Use the real collator's descriptor factorization to count columns.
Preserve all non-descriptor payloads (including publication metadata), retained
record order, selector counts and source descriptor distributions.

## Execution

Use `bash reproduce/launch.sh before` and `after` at their recorded commits.
Each phase freezes its invocation/git state, exact package versions, source and
production hashes, and launcher snapshot in `results/<condition>/reproduce/`.
Production is pinned by git; pre-rebase commits will be preserved on an audit
branch. Source and production hashes are checked before and after each run.
Existing result directories are refused. Raw logs live under
`/Users/iskander/code/presto/artifacts/2026-09-09_1457_codex_binding-selectors/`.

Use the established isolated inventory Python 3.12.6 / torch 2.7.0 / Hitlist
1.59.1 release environment. Freeze every installed dependency at launch. The
temporary checkout is imported through the explicit isolated PYTHONPATH, with
an import-origin assertion; OMP/MKL threads one. Local CPU, no GPU/Modal run.

## Interpretation and closure

This is ingestion/selector verification, with no model inference, training,
pretraining, synthetic data, validation/test split or predictive metrics.
Prediction dumps are intentionally absent; fixtures test the downstream tensor
and loss-selector behavior. Existing target transforms, units, qualifiers and
loss equations/weights are unchanged; the repaired descriptors select the
observed output columns instead of fallback/unknown columns.

Before/after source counts, selected-column distributions, runtimes, exact
non-descriptor payload reconciliation and all failed receipts will be closed
here and in the canonical experiment log. No result or preferred condition is
claimed before execution. #48/#50/#53 remain separate acceptance work.
