# Real-source output coverage and update audit

- Date: 2026-09-09; agent/model: Codex / GPT-6.
- Status: corrected inventory and source-routing trace completed. The full
  census and optimizer diagnostics have not run.
- Presto base: PR #58, `d9a666a07085289b168105089d29dd4e675add61`.
- Detailed plan: [coverage census](../agents/codex/plans/2026-09-08_output-coverage-census.md).
- Raw artifacts/environment: `artifacts/2026-09-09_1106_codex_output-coverage/`.

## Question and source contract

Measure actual supervision for every declared output/column, separating direct,
proxy, auxiliary and generated evidence, and verify label/gradient/update
incidence on a frozen subset of real training rows. This supplies evidence for
Presto #48/#50 and scopes the following #53 predictive baseline.

The uncapped canonical merged TSV, exclusive Hitlist indexes and explicit
bulk-MS supplement are separate source conditions. Freeze measured-only and
default-augmented populations with data seed 17 and peptide-disjoint train/val/test
80/10/10 splits (split/model seed 42). No modality load caps. Source resolution
uses `mask_unresolved`, strict complete MHC resolution and the actual adapter's
qualifier/assay policies. Persist the normalized funnel and all source hashes;
do not silently alter shared input files or caches.

The first phase inventories source files, Parquet schemas/build metadata and
study exclusions, including existing Hitlist #444. It reads only the fields
needed for metadata counts, not model predictions. Results must distinguish
current cache contents from current curation and from filtered training records.

## Environment and reproducibility

Use an isolated environment under the raw-artifact root. Initial pins are
Python 3.12, torch 2.7.0, numpy 2.2.6, pandas 2.3.3, pyarrow 23.0.1,
Hitlist 1.59.1, mhcseqs 2.5.12 and mhcgnomes 3.41.0. Resolve and freeze all
transitive packages before source analysis. Record package source hashes as
well as distribution versions. The shared environment's editable Hitlist source
reported 1.59.1 while its distribution metadata reported 1.55.8, so metadata
alone is not accepted as the curation contract. No shared package is upgraded.

`reproduce/launch.json` and `launch.sh` freeze each launched phase and relevant
environment overrides; `reproduce/source/` keeps launcher snapshots. Additional
phases receive their own invocation records without replacing prior receipts.
Runtime, git SHA/dirty state and source stability checks accompany every phase.

## Census, updates and scientific interpretation

The census uses PR #58's executable output/objective contract and exact distinct
counting. Preserve all columns and global zeros, measured versus generated
source families, original versus fallback observation identities, class/species/
context, response balance and censor qualifiers. Count #50's real T-cell pathway
bags and selected assay columns separately from row observations.

Before optimizer diagnostics, freeze selected training IDs, update budgets and
model/optimizer/loss configurations. Use fresh d128/l2/h4 models, both expanded
and collapsed topologies, and explicit initialized/frozen/unused controls. The
full corpus census remains independent of diagnostic subset coverage. No claim
of inactivity is made for a column omitted from those batches.

No predictive evaluation is planned in the census/update phase. Validation and
test splits are used for coverage counts; no prediction dumps, predictive losses,
regression metrics or binding classification metrics are claimed. #53 will
require those artifacts from an adequately trained selected checkpoint. No
pretraining or synthetic-only quality demonstration substitutes for that work.

## Upstream findings and closure

The initial inventory launch at `96b3ff6` stopped after 0.476 seconds because
the launcher treated the curation YAML as a mapping, while Hitlist stores a list
and exposes a canonical `load_pmid_overrides()` mapping adapter. No source scan
ran. Preserve `results/inventory/`, `reproduce/inventory_invocation.json` and
`reproduce/source/launch.py` as the failed receipt. The corrected launcher calls
the provider's loader; its three fixture/installed-curation checks precede the
second launch via `reproduce/inventory_v2.sh`. This is a local launcher error,
not a Hitlist bug. Later snapshots live under distinct phase-output directories.

Reproduction requires an environment created from `reproduce/environment.txt`
plus this checkout installed with `uv pip install --no-deps --editable .`.
The frozen scripts record the exact original paths; for a repeat inspection,
pass a fresh `--output-dir` with a unique basename. Existing evidence directories
and SQLite files are never overwritten.

The corrected inventory completed in **38.804 seconds** on local CPU at clean
Presto commit `aada2b19ac74346c5a71ea603498352045c4cc91` with the pinned release
environment. Reproduce with `reproduce/inventory_v2.sh`; its invocation, launcher
snapshot, status, source hashes and complete grouped counts are preserved under
`reproduce/` and `results/inventory_v2/`. All source stability checks passed.

| Frozen source | Total rows | Rows from studies curated `exclude_from_ms` |
|---|---:|---:|
| Hitlist observations | 4,439,643 | 40,355 |
| Hitlist binding | 891,885 | 472,497 |
| Default merged TSV | 3,423,737 | 514,190 |

The observation count independently reproduces Hitlist #444. Exclusion applies
to MS evidence, so the binding count is not a blanket removal recommendation.
The observations SHA-256 is
`f51440ab229fd187d2548b4dddcd1fc04580d97d45fb4d5b8e0222aa8080f928`;
the installed Hitlist 1.59.1 curation SHA-256 is
`e0270f2b417619a318ef03549e7cb7d46231bb3dbb6a088d8367a239cd6b3308`.
The cache's producer version is not inferred from the installed package version.

Loading the complete flagged merged subset through the actual Presto adapter,
with every modality cap disabled, drops 375 invalid peptides and emits 15,641
affinity, 11 melting-temperature, **496,976 elution** and 1,187 T-cell records.
Every emitted record lacks the source PMID despite that field being present in
the TSV. These counts precede MHC resolution and all downstream filtering;
they are not final training counts. The classifier's missing-numeric-value
fallback routes binding observations to elution without requiring an MS method.

Before the full census or training, run `reproduce/route_trace.sh` to attribute
those assay buckets to source methods/studies and reconcile exactly with the
actual loader totals. Preserve the original inventory and launcher snapshots.
File the confirmed adapter routing and lineage defects in Presto, and add the
independent cache evidence to existing Hitlist #444. Hitlist #361 remains an
existing bulk-candidate report, not a new finding from this inventory.

The route trace completed in **6.311 seconds** at clean
`4e949da09a9960031b09ee249eb3a57ad894c44e`. All 43 source-method/response groups
reconcile exactly with the actual loader buckets. Of the 496,976 elution rows,
**492,858 have non-MS methods**; 4,118 explicitly say cellular MHC/mass
spectrometry and require the upstream study-level curation decision. The
non-MS-method group includes 442,612 negative responses. PMID 32903714 alone
contributes 418,890 microarray rows (410,723 negative, 8,167 positive) mislabeled
as elution. Two numeric `3D structure`/`x-ray crystallography` rows are also
routed as affinity, with values 2.7 and 2.5 treated as nM by `BindingRecord`.
Both presence and absence of a scalar are insufficient assay definitions.

The read-only SQLite reconciliation confirms that the 40,355 excluded Hitlist
observation rows contain **33,101 distinct peptides** globally. Complete trace
tables, subset line references and source hashes are in `results/route_trace/`.
The raw flagged subset and SQLite database remain under the raw-artifact root
at the paths recorded in the result JSON. No peptide sequences are needed in
the public grouped trace to reproduce or describe these routing defects.

Close each informative phase with JSON/CSV summaries, raw artifact links,
runtime and source/config hashes, conclusions and canonical experiment-log
updates. Do not close #48/#50 on launcher code or fixture tests.

## Handoff

- Status: inventory and trace closed; full coverage/update phases remain pending.
- Next step: merge the independently registered
  [routing correction](../2026-09-09_1133_codex_assay-routing/), then preserve
  merged lineage in Presto #60 before resuming the full census.
- Open questions: source contamination, traceable evidence, rare assay support,
  canonical augmentation leakage and actual per-column update incidence.

Published upstream evidence is on [Hitlist #444](https://github.com/pirl-unc/hitlist/issues/444#issuecomment-5601129610).
Presto routing and lineage defects are filed separately as
[#59](https://github.com/pirl-unc/presto/issues/59) and
[#60](https://github.com/pirl-unc/presto/issues/60). The source inventory is
complete; the broader #48/#50/#53 work is not closed by these metadata results.
