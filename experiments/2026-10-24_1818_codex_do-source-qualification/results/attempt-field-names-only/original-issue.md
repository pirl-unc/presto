Presto needs an independent quantitative APC dataset to test processing, downstream MHC selection and an independently fitted MS-response term. Nanaware2019 was already named in the architecture plan, but its original quantitative tables have not been recovered into canonical Hitlist evidence.

Primary: https://pmc.ncbi.nlm.nih.gov/articles/PMC6398211/ (PMID30573663; DOI10.1074/mcp.RA118.000956). Repository: https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000082570 (PXD010301).

Source qualification report: https://github.com/pirl-unc/presto/blob/codex/modular-presto-model-20261020/experiments/2026-10-24_1818_codex_do-source-qualification/README.md

## What is available versus missing

- Official dataset descriptor and full328-file inventory are retrieved and independently verified. Human LG-2 control/DO-deficient clones and mouse splenic B cells must remain separate systems.
- A source-named WT mzTab has3,385PSM rows/612protein rows; matching mzIdentML has4,800identification items. Complete fields contain no intensity/abundance/quantity values. Only these two files were decoded; this is not a statement that all repository results lack quantities.
- Three286–294MB MSF files are parental LG-2, not the WT/KO panels; not downloaded. No RAW acquisition/reprocessing occurred.
- Quantitative S10–S12, selected isotope-standard S13 and selected binding S14 were not retrieved through attempted official routes. EuropePMC supplementary API explicitly returns non-OA error XML, PMC index gives a browser challenge, and the documented PMC Cloud prefix has zero objects. The legacy OA API is retired; its404 must not be treated as source absence. Public publisher media guessing did not recover files. No author contact, private access or bypass was attempted.

Source-listed key filenames:

| Table | Original filename | Intended distinction |
|---|---|---|
| S10 |139110_1_supp_238824_pjk8j3.xlsx|Intensity analysis; exact grain pending source inspection|
| S11 |139110_1_supp_238825_p9k893.xlsx|Intensity analysis; exact grain pending source inspection|
| S12 |139110_1_supp_238826_pvk8v3.xlsx|Intensity analysis; exact grain pending source inspection|
| S13 |139110_1_supp_238827_pzk8z3.xls|Nine outcome-selected isotope-standard amounts|
| S14 |139110_1_supp_238828_plk8l3.xls|Selected peptide binding measurements|

## Acceptance

1. Recover authorized original supplements/quantitative exports with checksums, access/license provenance and original source cells. Preserve failure evidence; do not manufacture quantities from IDs or spectral counts.
2. Curate study/species/lineage/clone/control/DO-loss status, biological and technical replicate, acquisition, material and exact/surrogate pairing through existing typed Hitlist APIs. Legacy PMID30573663 explicitly has an IEDB axis mismatch; source curation must not imply its ambiguous observations became resolved.
3. Preserve per-peptide versus predicted-core grain, actual molecular forms/charges, raw versus sample-normalized intensity and coefficients, positive/zero/missing states, filtering/confidence and original candidate identities. A predicted core is not an experimentally observed binding register.
4. Keep MHC-bound amounts, loaded amount, measured response and free intracellular concentration separate. The nine standards were selected from observed outcomes and cannot serve as an unbiased global calibration cohort. Whole-cell protein measurements need their own material/replicate joins; RNA is not a protein concentration.
5. Keep DO/DM editing context separate from precursor cleavage. Expose source-backed quantitative recipes and exclusions with component/parent/clone/study leakage accounting before any model fit. No Presto-only workbook curator.

Search found no existing PXD010301 source owner; related broad sample-axis issue #362 and independent MS-response issue #762 remain distinct. Current biological baselines and the full APC-concentration goal are unchanged.
