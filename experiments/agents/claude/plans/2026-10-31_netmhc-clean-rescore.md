# Pre-registration: tool-clean re-scoring of the external benchmark

Registered 2026-10-31 BEFORE any clean-subset metric, seen-fraction or overlap
count was computed. Only file formats (column layout, allele-name styles,
archive listing) of the tools' training files and of the prediction dumps were
inspected. Follows `2026-10-31_external-netmhc-benchmark.md` (PR #149); the
scoring code (`evaluation/external_benchmark.py`: `unit_metrics`,
`pooled_metrics`, MIN_UNIT_ROWS=20, study-level cluster bootstrap, 2000 draws,
seed 20261031) is reused unchanged. Post-hoc changes go under "Deviations" in
the experiment README.

## Question

PR #149 scored Presto against NetMHCpan-4.1, NetMHCIIpan-4.3 and MHCflurry on
held-out Presto rows, but the tools may have trained on those rows (Presto's
test split is not their held-out set). How large is the tool-vs-Presto gap on
rows the tool has NOT seen?

## Tool training data (obtained from the tool vendors; NOT redistributed)

* NetMHCpan-4.1: DTU `NetMHCpan_train.tar.gz` (c000-c004 `_ba` and `_el`, `allelelist`).
* NetMHCIIpan-4.3: DTU `NetMHCIIpan_train.tar.gz` (same layout).
* MHCflurry 2.3.16 `mhcflurry-predict` (pan-allele affinity predictor): the file
  it was trained on, `models_class1_pan/train_data.csv.bz2` (affinity rows plus
  mass-spec rows reassigned to 100 nM by the vendor's own pipeline; per
  GENERATE.sh). The curated `curated_training_data.affinity.csv.bz2` (affinity
  only) is a strict subset and is reported as a sensitivity.
* Raw tool data stays in the scratchpad / gitignored `artifacts/`. Only
  sha256 hashes of (peptide|allele) keys are committed.

## Definition of "seen" (per test row, per tool; computed only for the tool's class)

Test row key: peptide (as dumped) + allele resolved in the TOOL's own naming
(`allele_netmhcpan41`, `allele_mhcflurry`, `allele_netmhciipan43` of the dump).

Allele normalisation (both sides): strip `HLA-`, `*`, `:`, uppercase; class II
names are compared as written in the tool's files (`DRB1_0401`,
`HLA-DQA10501-DQB10303`), which already match the dump's format.

Primary variant, **pair-seen**: the exact (peptide, allele) pair occurs in
* the tool's BA files (all five partitions), alleles resolved as above; or
* the tool's EL files with label 1 (eluted ligand; label-0 rows are random
  decoys, not binding observations, and are not counted). An EL line names a
  cell line; the cell line is mapped to its allele set through the tool's
  `allelelist`; the test pair counts as seen if the peptide is in a cell-line
  file whose allele set contains the test allele; or
* (MHCflurry) the pan train_data rows, allele normalised as above.

Stricter variant, **peptide-seen**: the peptide occurs in any BA row, EL
label-1 row, or MHCflurry train row for ANY allele (strictly a superset of
pair-seen).

Extra sensitivity, **nested-seen** (reported in fractions and a clean-subset
score, labelled secondary): the test peptide equals, is a substring of, or
contains (length >= 8 class I / 9 class II) a training peptide for the same
allele (same allele-resolution rules). Superset of pair-seen. Rationale: class
II training sets are nested length series of the same core.

Each variant is a boolean per test row per tool. Unmapped alleles never occur
among scored rows (they were excluded upstream).

## Subsets (identical rows for all methods within a subset)

* `full`: all rows scored in PR #149 (reproduction check against the PR's
  pooled tables; must match to numerical precision).
* `clean_<tool>`: rows the tool has not seen. Scored for Presto, the tool, and
  every other applicable method on the same rows. Computed per variant.
* `common`: rows no tool applicable to the row's class has seen (class I:
  neither NetMHCpan-4.1 nor MHCflurry; class II: NetMHCIIpan-4.3 only, so
  `common` equals its clean subset). Computed per variant.
* Scope: exact (`qualifier == "="`) BA rows, the same rows as the PR's
  regression tables, for each dump separately (direct s42/s43, curated s42/s43;
  test = held-out headline, validation = developmental). Headline conclusions
  use test only.

## Metrics and reporting rules

* Per class x endpoint x detector x study unit: n, Spearman, Pearson, RMSE
  (log10 nM), AUROC at <= 500 nM (plus the PR's other threshold metrics),
  exactly as `unit_metrics`; no per-unit row bootstrap (not needed here).
* Pooled per class x detector: mean-of-study with study-level cluster
  bootstrap CI and paired (tool minus Presto) difference, exactly as
  `pooled_metrics`.
* A unit with clean n < 20 is NOT REPORTABLE and drops out of pooling (the
  harness already does this; every table also lists the row counts per unit).
  A pooled cell is reportable if >= 1 unit has n >= 20, but its CI is called
  degenerate below 3 studies; only cells with >= 3 studies are headline-grade.
* Gap before/after: per pooled cell, tool-minus-Presto Spearman, Pearson,
  RMSE, AUROC on `full`, on `full` restricted to the same studies as the clean
  cell (to separate subset shrinkage from study loss), and on `clean`.
* Seen fraction per class x detector x study x tool x variant (all rows and
  exact-BA rows).
* Seen-vs-unseen: per unit, the tool's Spearman on its seen rows vs unseen
  rows (each side needs n >= 20), with Presto on the same two sides as a
  control; pooled over qualifying studies with a study-level bootstrap CI of
  (unseen minus seen). This tests whether memorisation inflates the tool score.
* Mask output: `results/presto_row_overlap_mask.csv.gz`: per unique test row,
  sha256("<peptide>|<allele>") with the allele in the tool-native naming used
  by the class (class I: MHCflurry-style `HLA-A*02:01`; class II: NetMHCIIpan
  style), and boolean seen flags per tool and variant. No raw tool data.

## Verdict rule (registered)

Presto "beats a tool on clean rows" only if the paired tool-minus-Presto
Spearman gap CI on `clean_<tool>` (headline-grade cell, >= 3 studies) is
entirely below 0. A gap CI that includes 0 is "not distinguishable", and a CI
entirely above 0 means the tool is still ahead. If gaps shrink but stay
positive that is reported as such, not as parity.

## Known limitations (stated up front)

* Exact string matching misses nested/near-duplicate peptides (covered only by
  the nested sensitivity) and allele-name synonyms outside the normalisation.
* Tool training data may include rows from sources not in the downloaded
  snapshots (the shipped binaries could be newer/older); unseen here does not
  prove unseen by the shipped model. MHCflurry's shipped weights are the 2.2.0
  download.
* Both tools also trained on data beyond binding assays for non-test alleles;
  that is not a leakage concern for these rows.
