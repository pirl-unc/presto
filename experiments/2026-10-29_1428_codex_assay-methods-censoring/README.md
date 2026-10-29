# Actual assays, hidden censoring and additional source coverage

Date: 2026-10-29. Agent/model: Codex / GPT-6. Status: completed descriptive audit;
protocol recuration and newly qualified data admission remain follow-up work.
[Prespecified plan](../agents/codex/plans/2026-10-29_assay-methods-censoring.md).

## Contract and reproduction

Frozen corrected physical TRAIN: 234,337 rows, including 184,637 concentration
observations. Source `/volume/2026-10-28_1540_codex_temporal-multi-output/revisions/kinetic-review-v2/prepared/hitlist-v4-iedb-train.jsonl`,
SHA256 `c76aeafbb47c66b7fa904c191502de95f8812d45275cdfe680821838a37200b1`.
Admission is the parent human, explicitly unmodified, resolved-groove cohort;
original IC50, KD, EC50, KD(~IC50), and KD(~EC50) labels remain distinct.
Explicit inequalities remain inequalities; missing raw inequalities currently
map to exact. This audit does not alter any qualifiers or frozen model input.

Method agreement uses only TRAIN rows currently represented as exact. Collapse
log10(nM) within pMHC/method/study by median, then across studies by median.
Match the same peptide plus both groove sequences and class using SHA256 keys.
Report both pooled concentration labels and literal-response-matched strata;
these are observational comparisons, not randomized protocol comparisons.
Source-census inputs are outcome-independent metadata from frozen binding
(821,470 rows) and other-assay (26,694 rows) indices. The separate ledger scans
canonical MS metadata (5,440,009 rows) and original TRAIN source attribution
(1,641,063 MS rows). Hashes, paths and counts are preserved in receipts.

No model fitting, pretraining, synthetic supervision, loss terms or output-head
changes. Validation/test outcomes and predictions are unused. Held-out regression,
500 nM classification and ligand AUC metrics/prediction dumps are consequently
inapplicable; this is a source audit, not a predictive evaluation.

- Starting commit: `c4b797b9effbb7e678ac40b62ff66d8c4c5545fb`, dirty state recorded
  verbatim in `reproduce/launch.json` and `source-ledger-launch.json`.
- Main command: `python -m modal run --detach experiments/2026-10-29_1428_codex_assay-methods-censoring/code/launch.py`.
  See `reproduce/launch.sh`, JSON and exact executed source snapshots.
- Ledger command: `python -m modal run --detach experiments/2026-10-29_1428_codex_assay-methods-censoring/code/source_ledger.py`.
  Exact executed source is `reproduce/source/source_ledger.py`; current code was
  subsequently formatted only. Preserve that snapshot when reproducing.
- Rerunning requires a new experiment ID/output directory and fresh call receipts;
  launchers deliberately refuse to overwrite existing immutable outputs.
- CPU only, no GPU requested or observed. Main: 4 CPUs/16 GiB; ledger: 4 CPUs/8 GiB.
  Python 3.12 image, NumPy 2.5.4, SciPy 1.18.1, PyArrow 26.0.0. No non-default
  environment overrides. Physical audit 30.080 s, excluding census/startup/archive;
  ledger full worker elapsed 263.975 s. Total main wall time was not instrumented.
- Initial ledger attempt failed importing PyArrow with base NumPy 1.26.3 before
  reading source data. Source, call and logs are retained under
  `attempts/source-ledger-missing-numpy/`; retry explicitly pinned NumPy 2.5.4.
- Main Modal app: https://modal.com/apps/iskandr/main/ap-eQlGbWHVdqXHZmZclqWWH9 .
  Both call IDs are preserved in `reproduce/`.

## Assays versus reported quantities

The following are original structured IEDB tags, **not fully source-reviewed
protocol classifications**. Every concentration row is tagged purified MHC.

| Literal quantity | Competitive radioactivity | Competitive fluorescence | Direct fluorescence | Direct radioactivity | Total |
|---|---:|---:|---:|---:|---:|
| IC50 | 44,442 | 10,294 | 0 | 0 | 54,736 |
| KD | 1,464 | 16,367 | 0 | 1 | 17,832 |
| EC50 | 0 | 0 | 212 | 0 | 212 |
| KD(~IC50) | 71,749 | 1,176 | 0 | 0 | 72,925 |
| KD(~EC50) | 0 | 0 | 38,932 | 0 | 38,932 |
| Total | 117,655 | 27,837 | 39,144 | 1 | 184,637 |

Mechanistically, purified tracer competition is related across detectors.
Cellular competition and cellular surface stabilization require additional
measurement context; fluorescence alone does not distinguish them. Direct
refolding/assembly readouts differ from tracer displacement. The literal quantity
also matters: an IC50 is an assay-dependent response concentration, not
unconditionally a thermodynamic KD. See the primary
[fluorescence competition study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4151172/).

A concrete ontology warning: 20 source components have comments citing
PMID12135423. Their 24,686 TRAIN rows comprise 16,319 tagged competitive
fluorescence/KD and 8,367 tagged direct fluorescence/KD(~EC50). The cited
[2002 primary paper](https://onlinelibrary.wiley.com/doi/10.1034/j.1399-0039.2002.590402.x)
describes a peptide-dependent refolding ELISA. Group-level citation is not proof
of an identical protocol for every row, but it prevents interpreting the tags
as verified fluorescent tracer competition. [Harndahl 2009](https://doi.org/10.1177/1087057108329453)
also compares an assembly-detection LOCI assay with ELISA. Preserve original
labels, source scope and unresolved disagreements in a reviewed method relation.

| Comparison of original method tags | MHC-I Spearman (shared pMHCs) | MHC-II Spearman (shared pMHCs) |
|---|---:|---:|
| Competitive fluorescence vs radioactivity, literal IC50 only | 0.62549 (125) | 0.30613 (42) |
| Same method pair, all concentration labels | 0.61403 (287) | 0.19190 (48) |
| Competitive fluorescence vs direct fluorescence, pooled | 0.69467 (179) | 0.80000 (9) |
| Competitive radioactivity vs direct fluorescence, pooled | 0.34374 (291) | unavailable (0) |

All pairs in the first two rows are cross-study; all class-I pairs in this table
are cross-study. Assay, lab, study selection, allele mixture and source reuse are
confounded. The nine-pair class-II estimate is too small for a general conclusion.
No cellular-versus-purified comparison is possible in this admitted affinity
cohort. These results do not establish a causal detector effect or demonstrate
that the same protocol always correlates better. They complement, rather than
replace, the prior endpoint audit whose IC50/KD(~IC50) overlap was dominated by
publication/submission source components.

Decision: represent preparation, physical mechanism, detector, reported quantity,
and reviewed protocol context separately within the common observation interface.
Test method-conditioned calibration of a shared affinity representation; retain
quantity-specific units and censored losses. Do not create unrelated models per
assay tag, or collapse all fluorescent measurements into one biological target.
This audit does not select or promote a new model architecture.

## Hidden censoring

Current TRAIN uses 136,233 exact labels and 48,404 explicit right bounds
(48,215 `>` and 189 `>=`). Of the exact labels, **100,544 have missing raw
inequality**, versus 35,689 explicit `=`. Missing inequality is not evidence that
all those rows are censored; it also is not evidence of confirmed exactness.

| Reported nM | Currently exact | Explicit `>` | Explicit `>=` |
|---|---:|---:|---:|
| 5,000 | 158 | 1,820 | 0 |
| 10,000 | 415 | 1,807 | 3 |
| 20,000 | 282 | 36,770 | 0 |
| 25,000 | 77 | 121 | 2 |
| 50,000 | 88 | 374 | 0 |
| 100,000 | 208 | 618 | 175 |

The original bound count did **not** count inferred limits hidden among exact
values. Prespecified screening groups by study, resolved MHC, method and literal
quantity; groups need at least 20 rows, and a heap needs at least 5 exact rows
and 10% of the group. Any repeated value can qualify, not only round thresholds.
There are 142 candidate group/value cells covering 2,556 exact rows. Of these,
840 rows have stronger right-limit review evidence: an explicit right bound at
the same value, or a maximum reported value with at least half qualitative
negatives. These are **review candidates, not 840 confirmed new bounds**.

Example: reference1005810, a 2007 large-scale HLA submission, contains multiple
negative upper heaps at 20,000 nM in direct-fluorescence/KD(~EC50) groups. Retained
comments do not establish that threshold as the detection limit. Source review
was inconclusive; no qualifier was overridden. Other flagged values include
nonround upper heaps and possible low-end floors. A blanket 20k/25k/50k rewrite
would erase legitimate measurements and cannot be justified by this screen.

Next curation should attach source-confirmed limits and original units/ranges to
specific assay scopes, retaining original reported values and review state.
Confirmed upper limits should use right-censored likelihood/inequality loss and
be excluded from exact-value ranking/regression metrics. Sensitivity analyses
can compare unchanged labels versus exclusion of unresolved candidate heaps.

## Sources actually contributing and available expansion

The parent build downloaded fresh full MHC-ligand exports from IEDB and CEDAR
on 2026-10-28, using Hitlist 2.0.40. In this snapshot, all 5,324,565 CEDAR MS,
119,608 binding and 25,311 other-assay index candidates duplicated IEDB assay
IRIs; original contributor provenance is retained. CEDAR added **zero unique
index assays**. See the parent `launch_logs/build-attempt4-reconnected.log`.

All physical index winners are IEDB, and all 1,641,063 original admitted TRAIN
MS observations have raw source `iedb`. Hitlist is the integration/curation layer,
not an independent measurement database. mhcseqs 2.6.13 supplies groove identity
and sequence support. Ensembl 112 supplies human protein context/background
candidates; generated ligand comparators are not measured nonbinders.
No NetMHC/MHCflurry predictions or TCRvdb labels enter these fits.

The source ledger finds 535,026 supplementary MS rows in the canonical export,
all with null `has_ptm`: 4,904,983 IEDB rows plus 535,026 supplement rows reconcile
to 5,440,009. The consumer requires explicit unmodified chemistry, so **none of
those supplementary rows reached TRAIN**. Of these, 332,866 class-II and 9,954
class-I rows have exact allele provenance and monoallelic restriction evidence;
this is promising metadata, not completed eligibility or independent-new-label
qualification. [Hitlist #854](https://github.com/pirl-unc/hitlist/issues/854) already
tracks loss of supplied peptidoform fields and source-backed curation of blanks.

Already downloaded, currently unused physical modalities include:

| Source subset | Candidate rows | What remains to qualify |
|---|---:|---|
| High-throughput multiplexed binding | 468,548 (468,203 II; 345 I) | assay-specific readout, literal qualitative/enrichment meaning, study holdouts and overlap |
| Human cellular MHC with a specified non-MS method | 23,384 | competition versus stabilization/functional interpretation; 7,087 rows have concentration units |
| Human cellular MHC with unspecified method | 714 | underlying method and target |
| Nonhuman MHC in the physical indices | 48,654 | species identity/groove resolution and modality; not an additional disjoint total |

The 7,087 concentration-labeled cellular records are not all verified numerical
pure-binding measurements. Some are T-cell inhibition; assigning all to IC50
binding would be wrong. Multiplexed records likewise must not be relabeled as
IC50 or mass-spectrometry ligands.

Another 43,136 physical admission candidates were deferred for unresolved catalog
names (26,354) or missing partners (16,682). Upstream indexing also excluded
3,273 class-II binding records shorter than 12 residues before Presto's own
minimum of 9; this is a source-review candidate, not proof all should be restored.
These categories are not independent/additive and counts precede new admission,
deduplication, temporal splitting and label review. A new IEDB download alone is
not the main near-term gain: the full current export is already present.

## Verification and artifacts

Six targeted tests passed. `analysis/collect.py` verified archive hashes,
184,637 records, 72 correlation cells, 100 threshold-count cells and 848,164
source-census rows. `analysis/verify_aggregates.py` independently rebuilt 989 pair
records (1,978 aggregate sides), all 142 heap cells, and source-ledger totals from
retained records. These scripts do not import the worker aggregation functions.
See `results/verification.json` and `results/aggregate-verification.json`.

Review tables, study metadata and paired aggregates are under `results/audit/`;
source attribution is `results/source-ledger.json`. Full TRAIN row dump contains
hashed pMHC/groove identity, values, methods, qualifiers and original assay IDs,
without peptide sequences. It remains outside Git at
`/Users/iskander/code/presto/artifacts/2026-10-29_1428_codex_assay-methods-censoring/audit/concentration-records.jsonl.gz`.
Durable archive: Modal volume `presto-checkpoints`,
`/2026-10-29_1428_codex_assay-methods-censoring/audit.tar.gz`, SHA256
`13b61c267b6e1f06afa8f06bfc800b9cf311ea4facaf4daf1999e8162f0d61ec`.
Full archive/file hashes and exact local paths are in the verification receipt.
No confirmed recensoring, source admission, retraining or predictive improvement
is claimed by this audit.
