# Shared PXD024809 source contract

The source audit is complete; shared ingestion and quantitative recipe admission
remain pending. Reuse Hitlist's evidence/source-cell, molecular-form, sample,
assay, calibration-applicability, observation and recipe/disposition contracts.
Do not add a Presto-only workbook/TSV training path or a separate metadata registry.

## Retained input and observation layers

- Pin original project metadata, article XML/Table1, complete68-file inventory and
  the13,073,500byte tab-separated `Dilute_peptide_quantities.csv`. The inventory
  declares no checksums; retain our whole-fileSHA256 values without calling them
  deposit-supplied checksums. Record the original `.csv` name and real delimiter.
- Materialize all64,070original rows and14columns with source line/column/value
  provenance. They form6,407unique precursor×10run slots,6,406sequence/peptide IDs,
  two charge states. Preserve precursor, feature, peptide and group_label IDs.
  Do not collapse charge, isotope forms, aligned signals or confidence states.
- Keep58,063positive intensities,5,743literal zeros and264literalNA states distinct.
  Keep10,939missing m_score/peak_group_rank entries, all64,070alignment_rank values,
  RT/boundary missingness and raw string cells. No zero-to-negative conversion,
  imputation, q-value filter or guessed censor threshold in the source layer.
- Materialize the full10-standard intended panel separately from observed forms.
  Nine have90query rows,88positive values/twozeros; three lack m_score. The absent
  tenth is not ten measured zeros or established nondetections. Preserve its query
  eligibility as unresolved until original library/per-run evidence resolves it.
- Keep original protocol label, sequence UniMod695 and group_label isotope strings
  distinct with explicit same-form evidence and position. Standard isotope forms
  are not unmodified endogenous peptides. Stock/synthetic-solution concentration,
  dilution and on-column amount are separate typed quantities.

## Protocol, samples and shared conflicts

Represent JY HLA-selected matrix and post-isolation spike-in standards as different
material roles. HLA-A*02:01/B*07:02/C*07:02 annotate source MHC context; they are not
measured restrictions of the standards. No native intracellular or APC precursor
amount label follows from this experiment.

Store Table1's five nominal standard concentrations0.2/0.1/0.05/0.025/0.0125fmol/µL
with source coordinates. Project protocol independently supports the twofold grid;
the article Results claims1:10. Retain a conflict record, not a silent correction.
All ten quantification run labels match deposited raw files after the explicit
`#`→`_` filename normalization. This is a filename-derived link, not supplied
specimen/aliquot/biological pairing. Two injections per level share preparation.
Injection volume, purity/certification, lot, recovery and parameter provenance
must be source-backed or unknown. Supplementary protocol review remains pending.

Record Orbitrap Fusion Lumos/DIA and original LC/extraction details. Pinned
nf-core/diaproteomics1.2.2 code runs DIAlignR and renames DIAlignR.tsv to the
peptide_quantities.csv output; it is an aligned quantity export. Defaults alone
do not prove the actual study parameters. Libraries assembled across dilution
levels create an outcome-conditioned query universe, not a blind detection panel.

## Next qualification before a training recipe

1. Pin exact pipeline/container/package versions and parameter reports where
   available; preserve unknown overrides. Retrieve the existing per-run scored
   TSVs selectively to reconcile feature/run/form IDs and directly scored versus
   aligned/integrated observations. The necessary files are listed publicly;
   do not request their existence as if they were absent.
2. Inspect the existing master library and intended standard query membership
   before labeling the absent standard. Preserve declared/found/unresolved states.
3. Independently reconcile all source cells, dose arithmetic and observation
   joins in the shared output. Produce a complete disposition table before any
   choice of calibration eligibility, threshold or positive-only policy.
4. A bounded nominal-concentration response diagnostic may use the nine observed
   standards after these semantics are fixed. Preregister dose/standard holdouts,
   simple response controls, zero/missingness likelihood and selection handling.
   Do not claim nine sequences establish a broadly transferable MS encoder.
5. Any transfer to native non-MHC MS1 needs a separate protocol/form/matrix test;
   neither this source nor the crude synthetic observer identifies endogenous
   molar concentrations by itself. Preserve the full APC availability objective.

Related owner: Hitlist#762 (observation calibration). This source-specific request
supplies a concrete complementary dilution panel; it does not replace precursor
or controlled-processing metadata issues.
