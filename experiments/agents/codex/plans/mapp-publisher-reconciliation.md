# MAPP publisher/deposit reconciliation and filtering

October 22, 2026; Codex / GPT-6. Previous goal turn was progress: ef4b4c1
published a source audit, an independently verified structural contrast candidate,
and a complete precursor-proteomics archive. This follow-up resolves the source
filters and sample mapping needed before a biological comparison.

## Sources and questions

Freeze the publisher's Supplementary Data2/Data3 and Extended MS Methods PDF from
links read directly on the public Nature article page. The PMC download attempts
returned HTML challenge pages and are not spreadsheet inputs. Reuse the immutable
PXD009092 and PXD009360 deposits and previous member receipts. Read only; do not
recalculate or change original workbooks. No patient spreadsheets are requested.

1. Does the publisher peptide table reproduce the reported4,983peptides and match
   original sequence IDs/quantities? Preserve worksheet/header/source-cell scope.
   Identify conditions from explicit column labels. Join by exact unmodified
   sequence plus any supplied peptidoform/parent semantics, not row order. Report
   missing, duplicate and conflicting mappings; do not resolve ambiguities by choice.
2. What bead-background rule accounts for the9,276positive-evidence quantities
   zeroed in the deposited peptide export? The source reports a twofold enrichment
   cutoff but does not yet specify its operational aggregation. Compare a bounded,
   prespecified set of diagnostic hypotheses at fixed ratio2: per-run intensity
   versus mean/median/maximum bead intensity; per-condition arithmetic mean or
   median over its three runs versus mean/median/maximum bead intensity, zeroing
   the entire condition if it fails. Include zeros in means/medians. Use >=2 as
   the primary boundary and report exact-boundary cases separately. Reconstruct
   evidence sums from original rows, preserving source rounding tolerance1e-4
   relative /1absolute. Do not tune thresholds or add variants after inspecting
   outcomes without a dated amendment. An exact arithmetic match supports a
   reconstruction; inferred rules remain distinct from directly reported methods.
3. Does publisher Data3 or the supplementary methods identify proteome runs448–456
   by condition or supply unambiguous specimen links? Preserve condition-only
   matching versus exact pairing. Do not assign roles from numeric order. If only
   group ratios are supplied, retain those as derived summaries without inventing
   replicate assignments or using selected outcome proteins to validate a model.

## Outputs and verification

Create immutable URL/size/hash receipts, workbook inventories, original-source
mapping and comparison dumps, per-rule discrepancy counts, and an explicit
admission decision. Independently re-read workbook cells and replay the selected
source comparison. Preserve all failed input retrievals as excluded provenance
without treating HTML as XLSX. Update shared Hitlist source-curation requirements
with decisive evidence. If mappings remain unknown, proceed with the next
available source analysis instead of guessing.

No model, training, pretraining, synthetic data, loss, GPU/Modal or predictive
train/validation/test split applies to this source reconciliation. Per-cell
comparisons are not predictions, and no concentration/APC generalization is
claimed. A separate fit-specific spec will freeze the actual admitted endpoint,
shared source export, protein-component holdouts and identical exact/RL objective.

## Comparison details frozen before computation, 2026-10-22 continuation

Prior goal turn was progress (published local training feasibility e80fd42).
Resume this source audit without changing the unqualified biological endpoint.
For Data2, preserve all cell values/coordinates and compare each of its four
quantitative columns to per-sequence deposited peptide medians for each of the
five non-bead conditions, both including zero entries and treating reported zero
as absent for aggregation. Exponentiate only the two explicitly log10 columns.
The cross-column comparisons diagnose source alignment; they do not authorize
relabeling or zero-as-absence training. Use the same1e-4relative/1absolute tolerance.
Counts and sequence joins precede interpretation. Preserve all nine fixed
background-filter hypothesis masks. Do not turn a best approximate match into a
source-verified filter. Data3 is a selected gene-level summary, not an independent
validation set or a specimen map. Inspect its complete headers/rows and the
supplementary methods; unresolved numeric-run mappings remain unknown.

## Source finding and bounded follow-up, after primary comparison

Run-level evidence intensity >=2 times median BEADS reproduces all160,095cells.
Publisher C/D reproduce4983/5311same-header condition medians. E/F instead match
CHX/UT medians in4868/4913rows, warranting a source conflict rather than relabeling.
Before follow-up computation: independently reconstruct all nine masks and source
cell comparisons with csv/statistics; reconstruct publisher C/D membership from
source reverse/contaminant exclusions, unique protein-group membership and at
least two of three nonzero filtered replicates. Count unmatched rows and exact
IDs; do not tune eligibility to force counts. Independently verify E/F mismatch
counts against both their headers and the unexpected condition matches. Read all
Data3 rows and supplementary-methods pages for explicit numeric-run identities;
no outcome-derived numeric-run assignment or posthoc group search is permitted.

Verification execution amendment: first independent verifier was deliberately
interrupted after identifying repeated compressed-NPZ reads in its innermost loop.
Exact process20987 terminated with130; preserve source/log/receipt under attempts.
Load immutable arrays once, retain all scalar comparisons, and additionally verify
all original input hashes. No scientific policy or tolerance changes.

Membership follow-up: the strict single-source-group filter retains4424/4707
rows; publisher has559/604additional rows with multiple listed groups. This is
not by itself a contradiction of a razor-assignment policy. Before computing,
compare the non-reverse/noncontaminant >=2/3positive population without the
single-group restriction to C/D, count distinct source Leading razor protein
values and preserve all mapping alternatives. A leading assignment is not proof
of unique biological origin. Validate all Data3 cells as a selected gene-level
summary, without assigning raw numerical runs or implying physical pairing.
