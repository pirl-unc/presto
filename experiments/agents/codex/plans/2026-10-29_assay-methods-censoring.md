# Actual assay methods, hidden censoring and source expansion

Date: 2026-10-29. Agent/model: Codex / GPT-6.
Family: 2026-10-29_1428_codex_assay-methods-censoring.

## Questions and constraints

The user asks whether experimental method is a better grouping than literal
IC50/KD(~IC50) labels, whether apparent exact values at common upper concentrations
are counted as bounds, and which source data are used or still available.
Keep method and measured quantity as separate axes; preserve original values and
labels. Current training uses explicit source inequalities only, including the
documented missing-qualifier-to-exact convention. Do not silently equate high or
repeated values with censoring. Current two temporal fits and test gate remain
unchanged. No new GPU fits in this audit; at most two H100! fits globally.

## Prospective descriptive audit

Use corrected frozen physical TRAIN source and exact SHA256 from the prior
assay-agreement audit. Read existing frozen Hitlist binding/other-assay indices
for method/response/source/identity metadata census; no held-out numerical values
or predictions are used. This is not predictive evaluation and needs no test
prediction dump. No fit, pretraining, loss or synthetic labels.

1. TRAIN method/response/class census. Resolve method into preparation (purified,
   cellular, lysate), competitive/direct mechanism, and detection readout; retain
   original string. Compare exact same peptide+both grooves+class after medians
   within study, then across studies, without Cartesian pairs. Report method-only
   exploratory pooling and literal-response-matched comparisons separately;
   mixing response definitions can confound a method-only comparison. Preserve
   same-study counts, original paired values and independent recomputation.
2. Censoring audit: count raw missing versus explicit exact qualifiers. Tabulate
   exact/bounded counts at 5k, 10k, 20k, 25k, 50k and 100k nM. Detect value heaps
   within study+modeled MHC+method+endpoint: at least 20 original observations,
   at least five exact observations at a value, and at least 10% of that group.
   Record position relative to other values, qualitative-negative fraction,
   explicit bounds at that same value, study metadata and curator comments.
   Heaps at arbitrary values are screened too, including lower plateaus. They
   are candidates for source review, not automatic qualifier corrections.
   Preserve candidate assay identities without peptide sequences. Inspect leading
   source-supported cases and freeze any subsequent curation decision separately.
3. Additional-data census: source/assay method/response/class/species/units and
   restriction/peptidoform support in frozen binding and other-assay indices.
   Reconcile original source counts and current admission exclusions. Counts are
   candidate rows, not additional independently usable training outcomes. Distinguish
   downloaded-but-excluded from sources not downloaded, and source wrappers from
   independent datasets. Consult current primary IEDB/CEDAR export documentation.

Source-ledger follow-up: the build log shows all CEDAR index rows deduplicated
against IEDB, plus 554,238 packaged MS supplement rows. The admission census has
no rows attributed to those supplement PMIDs. Read source and peptidoform/identity
metadata from the checksummed canonical MS export and count source categories in
the original TRAIN MS file. Do not report downloaded/indexed data as trained data.
Also record the upstream removal of 3,273 class-II binding rows shorter than 12
residues as a recoverability question, not 3,273 automatically eligible examples.

The first source-ledger worker failed at import before any source reads: PyArrow
26 with base-image NumPy 1.26.3. Preserve that attempt and explicitly pin NumPy
2.5.4 for the retry. This changes only the declared execution environment.

## Verification and closure

Register before launch, freeze exact source/invocation/git state. CPU scans must
verify input hashes and preserve summaries/pairs/candidates. Tests cover method
parsing, aggregation, qualifier missingness and heap screening without relabeling.
Independently recompute correlations and counts, document uncertainty/source
dependence, update README/log/tasks and the common-path experiment plan. Any
confirmed assay override belongs in common data curation with original/effective
values and evidence; active artifacts remain immutable.
