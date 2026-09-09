# Real-source output coverage and update audit

- Date: 2026-09-09; agent/model: Codex / GPT-6.
- Status: registered preparation; no census or optimizer diagnostic has run.
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

File confirmed Hitlist data/curation problems with reproducible source versions,
study/record examples and impact counts. Add new evidence to existing issues
when applicable. Hitlist #444 and #361 are known reports, not findings measured
by this experiment yet. Presto adapter defects stay in Presto.

Close each informative phase with JSON/CSV summaries, raw artifact links,
runtime and source/config hashes, conclusions and canonical experiment-log
updates. Do not close #48/#50 on launcher code or fixture tests.

## Handoff

- Status: preparing isolated environment and first frozen inventory launcher.
- Next step: inventory and verify exclusions before the uncapped census.
- Open questions: source contamination, traceable evidence, rare assay support,
  canonical augmentation leakage and actual per-column update incidence.
