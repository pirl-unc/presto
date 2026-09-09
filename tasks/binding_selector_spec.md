# Preserve binding descriptors in alternate merged selectors — #62

Prepare in an isolated checkout of reviewed PR #63 head `2cce640`; publish the
PR only after #63 merges, rebasing the repair onto its verified identical merged
tree. Preserve any pre-rebase experiment commits on an audit branch so frozen
receipts remain reproducible. If the preceding production tree changes, reassess
and repeat the affected baseline. The canonical merged
loader and focused binding selector already preserve observed assay type,
method and culture descriptors. Panel and probe-bootstrap selectors currently
discard all four despite populating their intermediate UnifiedRecord. This
changes selected quantitative output columns and removes method-panel support.

The narrow repair is to propagate those already-normalized fields into both
BindingRecord constructors, preserving the fallback measurement label. Avoid a
broad loader rewrite unless one small shared helper eliminates actual semantic
drift across all four paths. Keep allele normalization, source/species defaults,
qualifiers, units, numeric values, ordering, reservoir sampling, bootstrap
peptide/pair selection and publication lineage unchanged. No model/loss change.

Before production edits, register a source verification family with immutable
TSV hash and the established isolated environment. Declare exact panel alleles
and bootstrap caps/seed before launch. Observe every accepted record at the
actual selector boundary, retaining bounded records, and verify against its
input row's observed fields. Capture before/after descriptor counts and
non-descriptor payload hashes; no changes to selection or source population are
allowed. Do not claim full-corpus impact from a configured allele panel.

Fixtures must exercise all four loaders on the same source rows, conflicting
assay_type/value_type, fallback labels, missing descriptors and culture fields.
Carry the resulting record through samples, selectors and loss resolution:
observed methods must choose the same columns as the canonical path. Actual
fixed model predictions remain independent of assay descriptors. Preserve
publication metadata and genuine observation identity from the preceding PR.

- [ ] Verify previous merge and create branch from main.
- [ ] Register exact source contract; capture baseline and expected selectors.
- [ ] Restore the omitted fields; test path parity and unchanged selection.
- [ ] Reconcile real-source counts and payloads; close experiment artifacts.
- [ ] Review, run full CI, merge, then resume #48/#50 census and #53 quality run.

## Exact source verification contract

Register `experiments/2026-09-09_1457_codex_binding-selectors/` before launch.
Use the same immutable merged input SHA-256
`46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c` and
pinned isolated environment as the lineage audit. Explicit panel alleles are
HLA-A*02:01 and HLA-A*03:01; panel retention cap 1 with head sampling, seed 17.
Observe every pre-cap BindingRecord construction. Bootstrap uses those same
alleles, seed 17, max_records 2000, max_peptides 500 and max_rows_per_peptide 4.
Its bounded selected records are the measured population, not a full-corpus
census. Both selectors scan the full file under their normal selection rules.

Capture each selector's normalized UnifiedRecord at classification, then compare
its four observed descriptor fields with the following typed record. Preserve
source descriptors, selected-column counts using the actual collator's
factorization, non-descriptor payload hashes, selector statistics and retained
record hashes. Never patch the global CSV module or the shared dataclass; hooks
remain local to the runner and restore in finally. Verify imported production
code comes from this checkout despite the shared isolated Python environment.

No MHC resolution, prediction, training, pretraining, synthetic generation or
validation/test split is performed in this audit. Fixture tests separately trace
actual selectors/loss routing. Record exact source/production hashes, environment
including isolated PYTHONPATH, git state and launcher snapshots. Reconciliation
must show unchanged population/non-descriptor payloads and zero descriptor loss.
