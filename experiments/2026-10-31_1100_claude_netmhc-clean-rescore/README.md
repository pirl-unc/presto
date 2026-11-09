# Tool-clean re-scoring of the external NetMHC benchmark

- Agent/model: claude (Claude Sonnet 5.5). CPU only, no Modal.
- Pre-registration (committed before any clean metric): `experiments/agents/claude/plans/2026-10-31_netmhc-clean-rescore.md`.
- Builds on PR #149 (`2026-10-31_0830_claude_netmhc-benchmark`). Harness reused unchanged
  (`evaluation/external_benchmark.py`: `unit_metrics`, `pooled_metrics`, n>=20 per unit,
  study-level cluster bootstrap 2000 draws, seed 20261031). The `full` subset reproduces the PR's pooled
  tables exactly (`results/reproduction_check.csv`, max abs diff 0.0 over 8 dumps x 140 cells).
- Git: plan commit 4f2bf4ad; analysis committed on top (branch `codex/netmhc-clean-rescore`).
- Reproduce: `reproduce/launch.sh` (env vars point at the extracted tool archives), `reproduce/launch.json`.

## Data and "seen" definition

Tool training data (not committed, not redistributed; sha256 in `reproduce/launch.json`):
NetMHCpan-4.1 and NetMHCIIpan-4.3 DTU training tarballs (BA c000-c004 plus EL label-1 lines via `allelelist`);
MHCflurry 2.2.0 `models_class1_pan/train_data.csv.bz2` UNION `curated_training_data.affinity.csv.bz2`.
Variants: `pair` (exact peptide + allele, primary), `peptide` (any allele), `nested` (shared substring, same allele).
Test-row overlap is released only as sha256("peptide|allele") plus boolean flags:
`results/presto_row_overlap_mask.csv.gz` (allele string: class I `HLA-A*02:01`, class II NetMHCIIpan style).

### Deviations from the pre-registration

1. MHCflurry: the curated affinity file is NOT a subset of the pan model's `train_data.csv` (53,597 affinity
   (allele, peptide) pairs differ, mostly allele-name spellings and non-HLA alleles). The registered text assumed
   a subset; seen = union of both files (more conservative, more rows marked seen).
2. The registered per-cell "common" subset for class II equals NetMHCIIpan's own clean subset (as registered).
3. Seen-vs-unseen within a study could be estimated in only 1 unit (see below), not pooled meaningfully.

## Seen fractions (test exact-BA rows, identical across families; pair variant)

| class | tool | rows | seen | fraction |
|---|---|---|---|---|
| I | NetMHCpan-4.1 | 1313 | 564 | 0.430 |
| I | MHCflurry | 1313 | 721 | 0.549 |
| II | NetMHCIIpan-4.3 | 982 | 633 | 0.645 |

Peptide-seen: class II 0.672; nested-seen: class II 0.680 (class I variants coincide with pair).
Seen-ness is almost all-or-nothing per study (whole IEDB studies are in the tools' training sets):
class I: pmid:17291638, iedb ref 1019353 and pmid:17942551 are 100% seen by both tools; pmid:34305923
(fluorescence) is 0% seen by NetMHCpan but 100% by MHCflurry; pmid:42048449 (radioactivity, n=592) is 0% seen
by both. Class II: pmid:34305923 (n=296) unseen; pmid:40313951 14% seen (n=21); the rest 81-100% seen.
Per-study tables: `results/seen_fraction_by_study.csv`, `results/headline_clean.md`.

## Result (headline: held-out test, pair variant)

Every clean cell is a single study, so no cell is headline-grade (>=3 studies); CIs are degenerate.
Tool-minus-Presto Spearman gap (full -> same-study matched -> clean; positive = tool ahead):

| split | class | detector | tool | full | matched | clean (n rows) |
|---|---|---|---|---|---|---|
| direct-s43 | I | fluorescence | NetMHCpan-4.1 | +0.192 | +0.068 | +0.068 (157) |
| direct-s43 | I | radioactivity | NetMHCpan-4.1 | +0.337 | +0.519 | +0.519 (592) |
| direct-s43 | I | radioactivity | MHCflurry | +0.279 | +0.488 | +0.488 (592) |
| direct-s43 | I | fluorescence | MHCflurry | +0.146 | n/a | n/a (all seen) |
| direct-s43 | II | fluorescence | NetMHCIIpan-4.3 | +0.329 | +0.274 | +0.145 (296) |
| direct-s43 | II | radioactivity | NetMHCIIpan-4.3 | +0.267 | n/a | n/a (all seen) |
| curated-s43 | I | radioactivity | NetMHCpan-4.1 | +0.137 | +0.122 | +0.122 (592) |
| curated-s43 | I | radioactivity | MHCflurry | +0.079 | +0.091 | +0.091 (592) |
| curated-s43 | I | fluorescence | either tool | +0.30 / +0.18 | n/a | n/a (all seen) |
| curated-s43 | II | fluorescence | NetMHCIIpan-4.3 | +0.213 | +0.242 | n/a (no unit n>=20) |
| curated-s43 | II | radioactivity | NetMHCIIpan-4.3 | +0.151 | n/a | n/a (all seen) |

direct-s42 is similar (e.g. class I radioactivity +0.381 / +0.350, class II fluorescence +0.153).
Pearson, RMSE and AUROC gaps, validation dumps and the peptide/nested variants: `results/gap_before_after.csv`
and `results/headline_clean.md`.

Reading: cleaning does not close the gap. In the one class I radioactivity study (pmid:42048449) the gap
WIDENS (tool .79-.82 vs Presto .30-.70). Class I fluorescence (pmid:34305923) and class II fluorescence
(pmid:34305923) gaps narrow (+0.07, +0.145 for direct-s43) but remain positive; in those two cells Presto's RMSE
is lower than NetMHCpan's/NetMHCIIpan's for class I (NetMHCpan RMSE gap +0.46) and class II (+0.29) on the
direct family, while ranking and AUROC stay worse. Under the registered verdict rule Presto beats no tool on clean
rows (no clean cell has a gap CI below 0, and none has >=3 studies).

## Seen vs unseen within a study

Only one unit (class II, pmid:11433380, fluorescence) has >=20 rows on both sides, and its unseen side has no
variance-defined Spearman; the question is therefore NOT answerable from this data. The tools' seen/unseen
contrast is confounded with study identity (see all-or-nothing seen fractions). `results/seen_vs_unseen_*.csv`.

## Limitations

Exact-string matching; the shipped tool weights may have used data not in the downloaded snapshots; clean
subsets are 1-2 studies (possible study-specific effects, e.g. a recent study with different assay conditions);
validation results are developmental. Not a fresh confirmation set.

## Files

`results/`: `gap_before_after.csv`, `headline_clean.md`, `pooled_all.csv`, `units_all.csv.gz`,
`seen_fraction_by_study.csv`, `seen_vs_unseen_{units,pooled}.csv`, `presto_row_overlap_mask.csv.gz`,
`reproduction_check.csv`, `dump_sha256.json`. Raw per-example dumps: `artifacts/2026-10-31_clean-rescore/`
(gitignored copies of the PR #149 dumps).
