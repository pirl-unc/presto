# Native PSMD2 reference and quantity qualification

Status: source qualification complete and independently verified. Canonical
Hitlist admission, release and biological fitting are still pending.

Agent: Codex (GPT-6), local CPU, no Modal/GPU or environment installation.
Git: 8454ef9c40913c89d56bca5cd02f923701af91f6 with documented uncommitted source/report work.

Dataset: original Sahu2021/PXD018722 Hi20S Dataset7, reusing the verified 0344
workbook, inventory, acquisition/search metadata and primary article by checksum.
The exact deposited reference and selected original processed text will establish
parent origins and exported quantity grain before canonical Hitlist admission.

No model fitting, synthetic training, assay loss, validation/test score or
prediction dump applies: this is source qualification, and no held-out modeling
partition is read. Full APC concentration and exact/discrete model comparison
remain unachieved.

See `reproduce/plan.md` and `reproduce/launch.json` for resource limits and
contracts. Original source assets live at `/Users/iskander/code/presto/.worktrees/presto-modular-model/data/native_psmd2_sources/2026-10-25_0424_codex_native-psmd2-source`; existing inputs are
referenced once. Stages refuse to overwrite previous attempts.

## Result and next decision

The original reference and processed tables support a prospective native-cell
processing comparison. They also rule out treating the publisher table as
unambiguous parent-resolved, unmodified-peptide concentration data. Preserve
these source distinctions in Hitlist #837 before fitting. No new predictive score,
RL advantage or calibrated APC concentration result is claimed.

| Evidence | Verified result |
|---|---:|
| Original peptide rows / fields | 10,647 / 80 |
| Original feature evidence / modification-specific rows | 40,679 / 10,935 |
| Publisher raw-intensity cells exactly reconciled | 74,529 |
| Six-run sequence-intensity cells | 63,882: 34,036 positive, 29,846 source zero |
| Raw versus LFQ cells with different values | 30,170 |
| Original gene annotations recovered from matching text rows | 13 |
| Deposited reference entries / distinct sequences | 73,113 / 72,855 |
| All candidate peptide spans / I-L-only alternatives | 38,641 / 403 |
| Peptides with one / multiple distinct parent sequences | 3,679 / 6,809 |
| Peptides with additional unreported accession origins | 582 |
| Peptides with a reporter origin / no reference origin | 53 / 159 |
| All-source dependency components | 1,689; largest contains 514 peptides |

The reference contains 24,735,594 residues and differs from the article's stated
reference count. Its engineered reporter is labeled human in the header; the
source construct assertion must override taxonomy for endogenous eligibility.
Twenty-four declared parent tokens absent from the deposited FASTA are
contaminant-prefixed. All declared alternatives and complete-reference matches
are preserved before any selection, including repeated spans and I/L ambiguity.

## Quantity and annotation interpretation

All 74,529 publisher `Intensity` cells (total plus six acquisitions) exactly
match the original peptide table. The original table adds distinct LFQ columns,
native IDs and feature/form links. All identifier memberships reconcile without
errors. Of the positive sequence/acquisition cells, 5,765 have positive modified
feature evidence, and 276 peptide sequences have multiple modification-specific
rows. These quantities cannot be relabeled unmodified-peptide amounts.

Feature sums exactly match 60,520 of 63,882 sequence/acquisition cells; form sums
match 63,243. Maximum relative discrepancies are 8.05122e-5 and 7.54988e-5. Those
small discrepancies are consistent with rounded exports, but the rounding
explanation is not proved. Every residual is retained; source values are never
silently replaced by sums. Independent regrouping by sequence/acquisition checks
every cell, without relying on the source-ID-list aggregation implementation.

Evidence types are 900 MSMS, 28,519 MULTI-MSMS, 10,283 MULTI-MATCH and 977
MULTI-SECPEP records. Preserve matching evidence and the original sample-level
identification types. The six run mappings agree across original summary,
evidence, experimental design and mqpar. Search-specific 8–25 bounds and the
general minimum-length field 7 remain separate source facts. No shorter-product
negative label, known censor limit or calibrated extraction recovery is inferred.

The publisher comparison covers 372,645 field positions: 270,301 literal
agreements, 2,465 decimal-equivalent representations, 80,231 paired blanks,
19,635 mass/PEP/score representation differences within the prespecified 2e-15
relative tolerance, and 13 gene-name differences. Openpyxl independently confirms
those 13 cells are date-formatted serials. The original text directly supplies
their gene annotations for matching sequence/native-ID rows. This recovers
source evidence; it does not infer names from dates or rewrite the workbook.
The initial source comparator retained them as generic literal differences;
the independent annotation pass establishes the date-specific explanation.

## Feasibility without outcome inspection

A descriptive strict-origin stratum excludes source contaminants/reverses,
PEP above 0.01, multiple complete parent sequences, reporter possibilities and
I/L-only alternatives. It contains 2,554 peptides before requiring observations
in both conditions. PEP and global FDR are different criteria. All-source
dependency groups are constructed before this stratum, including excluded rows.

| Positive acquisitions required in each condition | Parents with at least two peptides | Peptides in those parents | Dependency components | Possible single-reference contrasts |
|---|---:|---:|---:|---:|
| At least one | 138 | 603 | 131 | 465 |
| At least two | 113 | 462 | 109 | 349 |
| All three | 82 | 310 | 81 | 228 |

These are support counts, not a fitted cohort or numerical response calculation.
No reference peptide, split or model has been selected. Numbered acquisitions do
not prove independent biological replicates or paired control/KD specimens.
Same-parent contrasts can cancel shared precursor/run scale and fixed peptide
response only under the explicit observation assumptions in the next plan; they
cannot independently calibrate each condition's full peptide distribution.

## Acquisition, resources and reproducibility

Four bounded HTTP requests retrieved 40,969,817 bytes in 43.48 seconds. The exact
33,003,906-byte FASTA matches the deposited SHA1. Only 7,965,911 bytes of the
167,389,796-byte processed ZIP were retrieved: its tail/directory and two selected
ranges. Five selected members passed original local/central header, size and CRC
checks plus independent stdlib ZIP replay. The whole archive SHA1 is **not**
verified. No RAW/Andromeda data, cloud training or dependencies were downloaded.
All new source assets/receipts total 68,369,865 bytes, including extracted members.

| Local stage | Wall time | Peak process RSS |
|---|---:|---:|
| Source cell/ID/quantity reconciliation | 2.36 s | 306.4 MiB |
| Full-reference mapping plus independent all-query search | 106.05 s | 252.4 MiB |
| Independent quantity/date/run replay | 2.20 s | 274.6 MiB |
| Dependency graph and descriptive support | 0.30 s | 187.3 MiB |
| Independent source/reference/graph closure | 0.77 s | 241.3 MiB |

The prefix-index mapping took 4.31 seconds; most reference-stage time was the
independent whole-reference search over every query. Closure reparsed all 73,113
FASTA records with BioPython, checked every span, replayed all 10,647 peptide
dependency assignments using a different graph algorithm, and verified all
three descriptive support strata. The 1 GiB disk reserve, 64 MiB asset/member,
256 MiB family, 3 GiB RSS and 20-minute local-stage limits were unchanged.

The first independent annotation verifier failed on openpyxl `EmptyCell` objects.
Its code and log are preserved. The recorded recovery only skips cells without
coordinates; the numerical comparisons and source data were unchanged. The retry
and final closure pass. All eight current scripts pass isolated E9/F and syntax
checks; no production model code changed and no model unit-test claim is made.

`reproduce/executed-commands.json` records the exact stage/retrieval invocations;
`reproduce/source/` preserves current and earlier executed code. Run stages through
`sh reproduce/launch.sh STAGE [ARGS...]` in a fresh output directory. Existing
completed outputs intentionally refuse overwrites. The earliest three-member
extraction uses the archived initial extractor; subsequent evidence extraction
uses the current `--evidence` mode. `reproduce/environment.json` records packages.

Large reference records/support and the complete cell-exception list remain once
locally, excluded from Git. `results/large-local-artifacts.json` records exact paths,
sizes, hashes and reconstruction stages. Quantity residuals, compressed origins,
dependency groups, summaries and all source/code hashes are published. The
canonical experiment registry and #837 carry the qualified next-step decision.
