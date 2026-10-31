## Modeling need

Related to #762: Presto needs independent observation-response constraints before
interpreting native peptide intensities as precursor-relative availability. The
native response-only feature/objective comparisons failed their fixed gates; we
are now qualifying baseline amounts and their observation contract.

Please add a shared PXD024809 source and, after semantic qualification, a bounded
nominal-concentration response recipe using the existing evidence/catalog/query
and calibration-applicability system. Do not create a parallel metadata system.

## Original sources and verified support

- [PRIDE PXD024809](https://www.ebi.ac.uk/pride/archive/projects/PXD024809) and
  [primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12406853/).
- The complete inventory has 68 files. The selected
  `Dilute_peptide_quantities.csv` is tab-separated, 13,073,500 bytes, SHA256
  `646f8054ca9f60d9ebad46722a994c507c0b30c0c897a75df356586f0bfcccb8`.
- Independent csv/ElementTree and pandas/DOM audits reconcile all 64,070 rows and
  896,980 original string cells. There are 6,407 precursor × 10 run slots, 6,406
  sequence/peptide IDs, two charges and no duplicate precursor/run slots.
- The nominal ten-standard panel contains nine forms present in the table: 90
  standard/run entries, 88 positive intensities, two literal zeros and three
  missing m_score values. The tenth standard has no row; it is not ten negatives.
  Preserve the full isotope/position annotation and separate intended panel from
  observed query universe.
- Across all slots, preserve 58,063 positive intensities, 5,743 literal zeros and
  264 literal NA values, plus 10,939 missing m_score/peak_group_rank entries.

## Required semantic and metadata work

1. Retain original rows/cells, raw strings, peptide/precursor/feature/group IDs,
   charge/form, run, RT, integration boundaries, intensity, confidence and both
   rank fields. Source ingestion must not convert zero/NA/absent into one state.
2. Pin original sample and protocol assertions with provenance. Table1 and PRIDE
   protocol support five twofold levels, whereas the article Results says 1:10.
   Keep the conflict visible. Nominal solution concentration is not certified
   on-column amount; qualify injection volume, purity/lot, aliquot and preparation
   relationships or leave them unknown. Run links are presently filename-derived.
3. Distinguish the HLA-selected JY background from standards spiked after isolation.
   The source HLA context does not label standard restriction or calibrate native
   intracellular amounts, extraction recovery or precursor abundance.
4. Pinned nf-core/diaproteomics 1.2.2 runs DIAlignR and renames its TSV to this CSV.
   Materialize aligned quantities as such. Reconcile original parameters and
   directly scored versus aligned/integrated records using the existing ten
   per-run scored TSVs; those assets are already publicly listed. Preserve
   missing scores without assigning an unsupported confidence/censor meaning.
5. Inspect the existing master library to resolve the absent standard's query
   membership and cross-dilution library selection. A library conditioned on all
   levels does not establish unconditional detection or a blind query panel.

## Acceptance and scope

Conserve all source rows and states through the shared APIs; independently replay
form/sample/dose joins and publish every recipe disposition before fitting.
Prespecify any later nominal-dose diagnostic's likelihood, dose/standard holdouts
and controls. Nine observed standards cannot validate a general sequence encoder
or transfer to native non-MHC MS1. Source ingestion is supported; calibration
training is not yet admitted, and no APC concentration model is claimed.

Registered audit, frozen source, failed retrieval/parser receipts and full shared
contract: [Presto family0813](https://github.com/pirl-unc/presto/blob/codex/modular-presto-model-20261020/experiments/2026-10-25_0813_codex_dilution-observer-admission/README.md).
