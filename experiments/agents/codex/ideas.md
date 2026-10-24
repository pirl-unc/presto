# Codex Ideas

Use this file for rough experimental hypotheses and brainstorms.

- Keep items short.
- Promote actionable items into `todo.md`.
- Promote non-trivial items into `plans/`.

## 2026-10-22 — quantitative availability source candidates

Goal-directed source reconnaissance, not source admission or a biological fit:

- [Wolf-Levy et al., Nature Biotechnology 2018](https://www.nature.com/articles/nbt.4279)
  describes MAPP capture of proteasome-associated endogenous fragments alongside
  standard proteomics. Supplementary Data 2 and 8 are peptide tables; PRIDE
  accessions include PXD009092, PXD009360, PXD007263, PXD009181, PXD009438,
  PXD009112 and PXD009508. Candidate for relative product-pool supervision;
  captured local fragments and measured intensities are not necessarily free
  cellular concentration. Audit the exact tables, paired sample identities,
  original values, precursor quantities and observation scope before fitting.
- [Stopfer et al., Nature Communications 2020](https://www.nature.com/articles/s41467-020-16588-9)
  / PXD017407 reports calibrated copies-per-cell for selected pMHCs and broader
  relative immunopeptidomic data. Useful possible downstream quantity check;
  the measured pool is MHC-bound, not all intracellular products. A small number
  of calibrated targets cannot by itself anchor an entire concentration model.

Neither source has been downloaded or schema-qualified in this program yet.
The earlier generic optimizer study does not establish any biological endpoint.


## October24: transfer a digestion-trained cleavage prior to cellular MAPP

The nested sequence-context comparison has a stronger digestion development gain
from wider cleavage context than from fragment-length dependence. Before adding
more mechanistic state, test whether frozen digestion-trained rates transfer to
the existing cellular captured-composition endpoint that defeated direct fitting.
This is a candidate next experiment, not a launched or qualified result.

- Reuse canonical MAPP min2 manifestdf78d02f8200809c659e43adfae29866c48b712122b740379106994622f4ac80; parent and example files are JSONL. No new consumer curator.
- Audit cross-source parent/construct and peptide overlap before claiming transfer; exact separation is not homology/specimen separation.
- Compare frozen adjacent versus wider-context digestion rates, plus uniform-cut and uniform-composition controls. Fit at most one global effective exposure scalar initially; do not silently call it known intracellular digestion time. Enzyme-context reference and MS capture/response remain explicit assumptions.
- Preserve all-fragment physical support; normalize only for the observed-set MAPP evaluation. This endpoint still cannot identify total yield, clearance or concentration. Do not transfer instrument-specific MS coefficients without support.
- Use an observation-aligned centered-log10 loss, retain earlier KL-based results as different-objective references, and qualify a matched sampled-gradient counterpart for the same new loss. Freeze the full specification and verification scope before any new fit.
- Treat prior MAPP held-outs as previously exposed; original digestion test remains closed. Independent cellular concentration validation remains an unmet objective.


## October24: native intracellular peptidome perturbations

[Hitlist788](https://github.com/pirl-unc/hitlist/issues/788) now owns admission of
Dasgupta2014 (10.1371/journal.pone.0103604) and related Gelman2013
(10.1371/journal.pone.0053263) TMAB-labeled cellular peptide extracts. These are
a distinct candidate measured population beyond proteasome capture/MHC elution.
Primary articles describe public supplementary ratios, censoring and reused
epoxomicin/bortezomib measurements. No workbook has been schema-qualified or fitted
in this program; do not treat literature row counts as admitted examples.
Curation must preserve drug/vehicle/exposure, cell/species, extraction/labeling,
channel/control covariance, PTMs/charge/identification, original mappings and
cross-paper measurement lineage. No drug-to-KO inference or absolute-concentration
claim. Shared canonical admission must precede any consumer fitting.


## October24: APC-derived endolysosomal component measurements

Primary source lead: Öztemiz2026, PMID42846314, DOI10.3389/falgy.2026.1936335 ([paper](https://www.frontiersin.org/journals/allergy/articles/10.3389/falgy.2026.1936335/full)). Extracted dendritic-cell endolysosomal preparation and peptide time-course signals could help separate processing from MHC capture, but are not native intact-cell concentrations. Original quantitative files and dependencies remain unqualified. Public PeptideStudio sample_data at18dd99da4a619abd8a28c7622e0809eb2c03e0ff is explicitly synthetic; do not use it as biology. [Hitlist812](https://github.com/pirl-unc/hitlist/issues/812) filed and body read back, including source identity, activity/sample context, quantity/modification/detection semantics and demo exclusion. No numeric assets admitted, no experiment or training claim. Register an original-source audit before numeric inspection/admission.
