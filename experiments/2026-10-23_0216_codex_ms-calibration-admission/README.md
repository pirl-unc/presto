# Known-load MS-response data: source verified, training admission unresolved

**All four public PXD004886 supplements are retrieved and their 12,941 populated cells independently verified.** They supply a useful nominal standard panel and response evidence, but cannot yet be treated as one clean calibrated training table. This audit found amount-grid, molecular-form and normalization discrepancies that must remain explicit in shared Hitlist curation. No model was fitted and no MAPP failure cause was established.

Registered October 23, 2026 at 02:16 America/New_York; Codex / GPT-6. Execution base Presto `7020b62`, Hitlist `6e275fe`; dirty registration state and all stage commands/hashes are retained under [reproduce](reproduce/launch.json). The [prospective plan](reproduce/plan.md) predates retrieval and records source-layout/replay amendments. The [shared admission specification](reproduce/hitlist-admission-contract.md) targets [Hitlist #762](https://github.com/pirl-unc/hitlist/issues/762), not a separate Presto training path.

## Original sources and retrieval

[Collins et al., 2017](https://www.nature.com/articles/s41467-017-00249-5) and [the public archive](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5566333/supplementaryFiles). Original assets are under `/private/tmp/hitlist-pxd004886-assets`. The publisher returned actual XLSX files for Data2/8 and HTML challenge pages for Data1/4; those unsuccessful receipts remain recorded. No CAPTCHA was automated. The separate Europe PMC archive supplied all four originals, and both available publisher copies match byte-for-byte.

The full public archive is 59,882,038 bytes, SHA256 `d8c99067fee216285284623eadc25cfdc121b3e7e73b551c2da9b28e09a501f5`. The bounded GET took 174.37 s; a preceding HEAD request ended in a recorded 45 s timeout. Original source hashes/member names and exact URLs are in [retrieval](results/retrieval.json), [archive receipt](results/archive-retrieval.json) and [article/context](results/source-context.json). The source article specifies CC-BY4.0 with third-party exceptions; no conflicting restriction was found in these four workbook cells. Raw source files stay outside the repository; generated cell audits preserve exact coordinates and both XML/formula/cache representations.

| Supplement | Source role | Populated cells | Verified content |
|---|---|---:|---|
| Data1 | Nominal dilution design | 39 | 25 group/sample amount cells; exact threefold ratios |
| Data2 | Standard panel | 95 | 30 sequences; six in each of five groups |
| Data4 | S4 responses across sites | 6,249 | 31 reporters, 11 sites, nine original column slots per site, raw and normalized |
| Data8 | Site response curves | 6,558 | 31 reporters × 11 sites × two representations; its internal title calls it Data9 |

Independent ZIP/XML parsing checks all 12,941 cached values and formula expansions against the separate openpyxl records. The first verifier stopped on shared formula anchors; the second exposed inconsistent header placement/spelling. Both failed attempts remain frozen. The final verifier locates actual numeric headers and independently expands relative/absolute cell references, then passes in 0.181 s. These are source checks, not model metrics.

## Findings that affect admission

Data1 says “concentration” but gives **fmol on column**, an amount, not volumetric concentration. Its five top loads are 1, 10, 100, 1,000 and 10,000 fmol. Data8 uses a shared plotting grid. Under a five-point group-order correspondence, some differences exceed decimal rounding: group C ends at111 while Data1 specifies100; group A ends at1.1 instead of1.0. Across the682 reporter/representation rows,1,914 of3,410 planned grid positions differ by over5%; this includes missing-response positions and small amounts with large relative rounding error. The largest relative difference is19%. The positional correspondence remains an explicitly labeled hypothesis until sample identities are reconciled; no values are silently corrected or admitted as calibration targets.

Thirty panel sequences produce **31 reporter labels** because CNNVLYIR appears with both `C[CAM]` and `C[PCm]`. Other labels carry isotope tags or `sum3`/`sum4` aggregation suffixes, and some omit the heavy tag. These remain distinct source forms/channels. A nominal sequence spike does not establish known molar quantities of both modified forms.

Data4 covers **sample S4 only**, not all229 acquisitions mentioned in the study. Its source columns provide site and local slot, but no explicit file/day/replicate labels. There are3,069 planned reporter slots per representation. Raw has2,372 positive values,568 explicit empty strings and129 absent/unpopulated cells; normalized has2,398 positive values,572 explicit empty strings and99 absent/unpopulated cells. Neither representation contains a numeric zero. Missing cells are not zeros or confirmed nondetections.

**Twenty-six normalized responses have no positive raw counterpart.** All occur in source column X, the first slot under site3. Of97 columns with positive raw/normalized pairs,39 do not behave as a shared column multiplier at a1e-6 relative diagnostic tolerance. The largest deviation from its column median ratio is2.561. This is a source discrepancy requiring explanation; it does not prove which values are wrong or supply the missing normalization metadata. In contrast, all162 computable published Avg/SD/CV values match their own source matrices (maximum relative error1.96e-14). See [normalization audit](results/normalization-audit.json).

The observed panel was selected for LC–MS performance and assigned to dilution groups using response. Source selection, omitted low-end curve points and saturation must remain visible; the30 standards are not an unbiased sequence universe. The synthetic standard is added to an already prepared background digest, so background trypsin context is not antigen-processing supervision for the standard. Human background-cell identity is not an APC assignment. The assay supplies no MHC binding, machinery perturbation or intracellular concentration label.

## Decision, reproduction and next experiment

**Do not train from Data8's plotted amounts or treat normalized cells as automatic copies of raw observations.** Implement the shared source relations first, with explicit cell/form/amount/run-slot/conflict provenance. The initial bounded recipe may use supported raw S4 evidence and nominal panel amounts after resolving form eligibility, while keeping broad calibration and transfer claims out of scope. Full curve/sample and acquisition/exclusion reconciliation are still needed. The proposed source and recipe contract is detailed in [the Hitlist handoff](reproduce/hitlist-admission-contract.md).

There is no optimizer, loss, pretraining, synthetic training data, model checkpoint, validation/test split or prediction dump in this source audit. Predictive loss/correlation/nM/binder metrics do not apply. All work ran locally on CPU; no Modal or shared-environment changes. The first partial inventory took1.17 s and peaked at85,786,624 bytes. Public archive download and complete inventory are recorded separately. [Independent receipt](results/independent-verification.json), [complete inventory](results/workbook-inventory-complete.json), [all source join diagnostics](results/source-join-audit.json) and frozen stage-specific launch files retain enough information to replay without thread context. Run only against fresh output directories or inspected existing sources; the retrieval scripts intentionally refuse to overwrite original assets.

Compared with MAPP, this adds independently known nominal peptide loads, but it does not yet identify intracellular processing yield, free-pool mass, clearance or assay transfer. Calibration #762 remains separate from precursor/specimen #757 and controlled-processing #759. The full concentration/availability goal stays active.
