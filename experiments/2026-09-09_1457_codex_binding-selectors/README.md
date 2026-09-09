# Alternate binding-selector descriptor recovery

Date/agent/model: 2026-09-09; Codex / GPT-6. Status: source comparison closed; PR #64 reviewed, CI passed and merged.
Plan: [selector repair](../agents/codex/plans/2026-09-09_binding-selectors.md).
PR base is merged #63, `48182ae1d2a36994470338b9422329590bc7c413`.
The earlier baseline used its identical reviewed tree, preserved on the audit branch.

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

Before/after counts, selected-column distributions, runtimes and exact payload
reconciliation are closed below and in the canonical experiment log. #48/#50/#53
remain separate acceptance work.

## Baseline

The before phase completed at clean `e6d6b607d95e801637a6f92be45ea3c2d866bd31`
in 111.439 seconds (panel 43.416, bootstrap 66.333). This original commit is
preserved on `codex/binding-selector-audit`; the working branch was then rebased
onto merged #63 (`48182ae`), whose tree exactly matches reviewed `2cce640`.

Panel: 23,210 accepted constructions, one retained. Bootstrap: 1,000 selected
records from 500 peptides. Both scopes lose assay type and method on every
record. Culture fields are absent in this subset. Fallback measurement labels
preserve the affinity-family column populations here, while every method/prep/
geometry/readout column becomes unknown. Complete column distributions and
field counts are in `results/before/`; these scopes overlap and are not additive.
No source/production hash changed during the scan. After repair/reconciliation
and final validation remain required.

The new 22-case regression suite reproduced **10 failures / 12 passes** before
the fix, including zero selected-panel gradients for both alternate loaders.
The eight-line correction copies the four normalized fields from the existing
UnifiedRecord into both BindingRecord constructors. The affected suite then
passed **247 tests in 21.46 seconds**, including source/target/selector parity
across all four loaders, missing and conflicting descriptors, preserved lineage
and selected gradients with invariant fixed predictions. Ruff 0.16.0 lint and
format passed (255 files). The before/after test logs are in the raw-artifact
root as `regression-before.log` and `regression-after.log`.

## Completed comparison

After ran at clean `37a06cdbde591bb08f883d5d84cbf6274fdba786` in 49.342
seconds. Comparison ran at clean `c6b0eeb` in 0.191 seconds. The production
repair is exactly eight assignments across two constructors. These scan timings
were not a controlled throughput benchmark and do not support a speed claim.

| Condition | Panel constructions / retained | Bootstrap selected | Total seconds |
|---|---:|---:|---:|
| Before: descriptors omitted | 23,210 / 1 | 1,000 | 111.439 |
| After: source descriptors retained | 23,210 / 1 | 1,000 | 49.342 |

| Field | Panel recovered values | Bootstrap recovered values |
|---|---:|---:|
| Assay type | 23,210 | 1,000 |
| Assay method | 23,210 | 1,000 |
| Effector culture | 0 | 0 |
| APC culture | 0 | 0 |

These scopes overlap and must not be added as distinct observations. Culture
fields have no real support in this subset; whitespace and missing/known culture
handling are tested in fixtures. Every post-repair field matches its normalized
source, with zero lost, invented or changed values. The actual post-repair
method/type/preparation/geometry/readout column distributions exactly match
those calculated from the input descriptors by the same collator. Affinity-family
columns already agreed through measurement-label fallback for these real rows;
conflicting type labels are covered by the regression fixtures.

Every selector statistic, observed/retained count, ordered non-descriptor payload
hash and source-descriptor hash is identical. Publication fields, numeric values,
units, qualifiers and selection order are conserved. This verifies the declared
allele/selection scope, not all-corpus prevalence, post-MHC support or prediction
quality. The preferred condition is the repaired descriptor path.

Reproduce comparison with `bash results/compare/reproduce/launch.sh` from this
experiment directory at the recorded commit. The script freezes the actual
root-relative invocation and environment. To repeat on a later checkout, run the
canonical analysis script from the repository root with `--output-dir` pointing
to a fresh directory and the same recorded environment.
[Comparison JSON](results/compare/result.json) and [field counts](results/compare/field_counts.csv)
verify CSV/JSON agreement, immutable launcher snapshots, production hashes at
the preserved before/after git commits, and identical source/dependency/selection
contracts. Complete before/after column distributions are in their result JSONs.

Four audit/comparison tests passed in 2.28 seconds, including rejection of changed
payloads and wrong selected columns; they overlap the previously reported affected
suite. Final full CI and author review passed. No deployment
workflow exists. Next: resume #48/#50 uncapped census/update evidence and #53
fitting/generalization acceptance with the repaired adapters.

PR #64 merged at `190360a7e2596eb5f62682088f1ef63270bf5eeb` on 2026-09-09
15:41:16 UTC. The merge tree is identical to reviewed `b1f4646`; #62 is closed.
Both final-head CI runs passed 2,061 tests / 3 skipped (663.26 and 941.30 seconds).
Their complete logs are preserved in the raw-artifact root. The posted review
was performed by the implementation author, not an independent reviewer.
