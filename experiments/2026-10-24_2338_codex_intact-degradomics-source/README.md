# Intact degradomics: original kinetic quantities available with explicit exceptions

Codex / GPT-6, October24,2026. Registered at Presto
`1de0a1208369ec3699dff457c24b93f174329a98` with preexisting task-file changes.
[Prospective plan](reproduce/plan.md), [launch/environment](reproduce/launch.json),
[closure](results/source-closure.json), [Hitlist831](https://github.com/pirl-unc/hitlist/issues/831).

**Original peptide/substrate time-course intensities are available.** All325,718
populated cells in eight workbooks independently reconcile to OOXML. Published
kinetic/organ tables contain3,427positive intensity rows;3,426agree exactly with
deposited quantities. No supervised cohort, model or predictive gain is claimed.

## Scientific contract

[Primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11874349/), DOI10.1073/pnas.2419607122,
PRIDE PXD058121. Substrate is human alpha-synuclein; enzyme preparations are
mouse-organ-derived20S. Expression host, substrate species and enzyme species are
separate provenance roles. These are purified mixtures without MHC capture or
intact-cell APC measurements. The methods place all organs in one animal; R1–R3
are reported labels with unresolved biological/technical independence.

Nominal zero already contains products. Source input concentrations are protocol
assertions; unknown peptide-specific MS response and run scaling prevent molar
interpretation. One substrate cannot establish new-protein generalization.
Mass-match uniqueness, fit settings, recurrence and selected MS/MS confirmations
are distinct confidence axes. Missing products are unknown-cause missingness,
never zero, negative or inferred censoring labels. No biological values are imputed.

## Assets and verification

The complete PRIDE inventory contains105unique accessions:102RAW and3SEARCH
workbooks. No RAW files were downloaded. Five supplementary workbooks, SI PDF,
publication XML and official metadata were also retrieved:4,718,213bytes total.
Active retrieval44.07seconds includes a preserved30second bundle timeout. An
initial incorrect inventory route returned an empty body; the valid two-page
inventory is retained. Page zero caps at100despite a larger requested page size.

| Workbook | Sheets | Populated cells | Content |
|---|---:|---:|---|
| F1/SF2/SF3/SF4/SF5 search |21|33,130|Input masses, assignments and controls|
| F2/SF6 search |21|59,978|Seven-time-point kinetic series|
| F3/SF7 search |44|97,562|Three organ preparations|
| Dataset S1 |28|51,999|Fit-factor, mass and envelope comparisons|
| Dataset S2 |6|11,578|Control/mixture assignment views|
| Dataset S3 |8|37,148|Charge-envelope quantities and fit information|
| Dataset S4 |3|12,917|Kinetic peptide quantities|
| Dataset S5 |2|21,406|Organ-specific peptide quantities|

Read-only openpyxl and a separate ZIP/ElementTree OOXML reader agree on every
stored value and2,991formulas/caches. Shared formulas are independently expanded
for comparison; no recalculation or date conversion. Original hashes, archive
CRCs, PRIDE processed-file sizes and PMC object-list sizes agree. Complete literal
cells remain in the local shared source directory.

| Table | Positive rows | Sequences | Sample labels | Lengths, including substrate | Exact quantities |
|---|---:|---:|---:|---|---:|
| S4 Kinetics |1,289|237|21|5–140|1,288 / 1,289|
| S5 Organs |2,138|313|44|4–140|2,138 / 2,138|

All3,427published rows resolve to one deposited identity through span/type and,
when necessary, exact observed mass. Intensity never selects a match. Multiple
masses can share one sequence; aggregation uncertainty remains.157/200sequences
appear in multiple sample labels, which can differ only by replicate or organ.

The search workbooks contain17,862input-mass rows and8,138assignment rows before
deduplicating figure reuse. All nine common F1/F2 sheets have identical input and
assignment values under explicit header/type equivalence. They are repeated views.

## Source exceptions

The [layout amendment](reproduce/layout-amendment.md) preserves original headers
and documents two body interpretations before admission. F2's40minuteR1has an
extra identifier in H, shifting assignment values into I:V. All68interpreted
spans validate and reconcile to S4. F3's brain-zeroR2input block lacks the
advertised Mass ID column; A:C contain two masses and intensity. No missing IDs
or assignment links are invented from row order. Source files remain unchanged.

- Spleen30minuteR3is absent from processed views and the RAW inventory.
- S4row20,100minuteR1,span128–140 disagrees with F2row246on mass, error and intensity.
- F1/F2,120minuteR2assignment row93names an absent input ID. Two F2,100minuteR1
  assignments and one F3,liver90minuteR3assignment disagree with input intensity.
- SI Figure S8contains machinery measurements, but original numeric densitometry
  and exact material links were not located. Organ is not numeric catalytic state.

The SI PDF was text-inspected and pages9,10,12visually reviewed. Figure S6uses
normalized/z-scored displays with recurrence/consecutive-time selection. Summed
deconvoluted intensity is not automatically TIC. No plot values were digitized
into labels. Publication assets report CC BY-NC-ND; PRIDE reports CC0. Original
assets remain local and are not vendored into this report.

## Execution and failures

Local CPU, one numerical thread; no installations/cloud/GPU. Caps:64MiBper asset,
256MiBnew raw bytes,3GiBRSS,20minutes per analysis,1GiBfree disk plus32MiBmargin.
Cell replay9.11seconds/84.94MiBpeak RSS; layout-qualified quantity audit1.17s/64.88MiB;
closure.013s/53.41MiB. All resource gates pass; these are single-run timings.

Initial inventory failed because shared formulas were not expanded. Initial
quantity parsing encountered an abbreviated header, then exposed body-layout
differences. Failed scripts/logs, partial dump and header-driven report remain.
The initial68position/count discrepancies and298nonnumeric-intensity flags are
superseded by the layout-qualified audit, not final biological findings. Lint
findings were corrected before successful runs. No source or scientific gate changed.

## Reproduction and artifacts

Raw inputs/receipts:
`/private/tmp/presto-modular-model/data/intact_degradomics_source/2026-10-24_2338_codex_intact-degradomics-source/`.
[Sanitized receipts](results/retrieval-receipts-sanitized.json) freeze URLs,UTC,hashes,
sizes/failures. [Inventory](results/source-file-inventory.json),
[cell audit](results/workbook-inventory.json),
[final reconciliation](results/quantity-audit-layout-qualified.json) and
[sheet summary](results/source-sheet-summary-layout-qualified.csv) retain evidence.
The initial quantity-audit.json/CSVare explicitly superseded. All executed source
versions and exact command lists remain in reproduce; reviewed logs are retained.

Use a fresh output location; scripts refuse to overwrite receipts. Fetch recorded
assets, then execute canonical inventory.py, quantities.py and close_source.py.
When using frozen snapshots, restore them to the original code layout so imports
and root paths resolve identically. launch.jsonrecords absolute raw paths.

No pretraining, optimizer, loss, synthetic biological labels, assay head,
validation/test partition or predictions: source qualification precedes shared
cohort admission. Regression and binding-threshold metrics are inapplicable.
Earlier held-outs remain closed; the full APCavailability goal remains unachieved.

**Status:** source qualification complete, with explicit exceptions.
**Next:** shared Hitlist ingestion/replay, then a separately registered exact-versus-
sampled kinetic comparison. **Open:** MS calibration, initial products, material/
replicate identity, missing acquisition and numeric machinery measurements.
