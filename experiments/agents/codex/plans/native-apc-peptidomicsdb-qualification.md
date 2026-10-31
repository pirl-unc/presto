# Find direct native intracellular peptide evidence for APC availability

October 24, 2026, Codex / GPT-6. The completed Maben source audit cannot anchor
sequence-specific trimming. The main objective remains precursor-relative peptide
amounts in antigen-presenting cells, with matched differentiable and sampled fits.
This plan prioritizes measurements closer to that endpoint rather than more fitting
on the same HAP1 development data. No new biological cohort is assumed available.

## Source lead and read-only inventory

PeptidomicsDB, DOI10.1093/nar/gkag1032 (published October23,2026), describes public
reanalysis with sample/project metadata, peptide quantities and identification
provenance. The official landing page at http://www.inbirg.com/ppd/ was retrieved
successfully and exposes `/ppd/browse/` and `/ppd/download/`. It includes global,
neuro- and immunopeptidomics; database membership does not prove a native APC sample.
The initial landing page/receipt are retained as a post-run discovery asset under
`data/erap1_source_qualification/2026-10-24_0924_maben-v1/`.

1. Inspect public download metadata and original project accessions. Inventory
   fields and sample units without retrieving a massive peptide collection.
   Retain reanalysis version/date and distinguish original deposition from derived
   PEAKS outputs. Do not treat sample/file counts as independent specimens.
2. Filter metadata prospectively for human/mouse non-infected APC material or
   relevant non-pathogen cell systems, native intracellular peptide extraction,
   explicit treatment/control or time groups, and actual quantitative measurements.
   Distinguish MHC pull-down, native peptide extraction, extracellular secretion,
   and deliberately digested proteomics. Keep professional APC status explicit.
3. Confirm any candidate in its primary paper and deposition. Require extraction
   and proteolysis-arrest details, peptide identity/quantity units/qualifiers,
   biological versus technical replication, assay filtering, search provenance,
   machinery assertions, and whether matching precursor measurements exist.
   Database-search, de-novo, homolog and sequence-variant calls remain distinct;
   model-generated identifications are not unqualified exact-source matches.
4. Search existing Hitlist issues and use or create the appropriate upstream owner
   before production curation. Keep shared sample/context/source/recipe contracts;
   no Presto-only source parser. Register a dated audit family before quantitative
   source admission or fitting and freeze original manifests and selection rules.
5. If eligible material exists, freeze a biological experiment that predicts
   within-study native peptide quantities/responses with component grouping and
   appropriate precursor/detection controls. Preserve short/nonbinding fragments,
   exact reference and identical sampled objective. Define the measurement model
   and scale identifiability before claiming concentrations. Avoid cross-project
   intensity calibration and account for ex vivo degradation uncertainty.

## Stop conditions and scope

If the metadata contains no eligible APC/native samples, record the exact search
and exclusions and pursue the original dataset sources. Do not relabel MHC elution
or tryptic proteomics as the intracellular peptide pool. Unknown lineage, missing
quantities or access gaps remain upstream requirements, not generated labels.
HAP1 outer validation/test and prior closed tests remain untouched. No cloud
training or shared-environment changes are authorized by this plan.
