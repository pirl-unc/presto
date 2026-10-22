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
