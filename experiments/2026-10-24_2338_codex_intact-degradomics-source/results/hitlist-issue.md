# Admit PXD058121 intact-degradomics trajectories with mass-assignment and preparation provenance

Presto needs an independently curated kinetic constraint for peptide generation
and subsequent destruction, including short products and remaining substrate.
This source provides original quantitative tables, but those values must retain
their measurement and identification limitations before they enter training.

Primary study: [Tracking proteasome degradation: A cross-organ analysis via intact
degradomics mass spectrometry](https://pmc.ncbi.nlm.nih.gov/articles/PMC11874349/),
DOI 10.1073/pnas.2419607122, PRIDE PXD058121.

## Original assets and verified coverage

- PRIDE project plus both inventory pages: **105 files, 102 RAW and 3 processed
  XLSX**. Only processed assets were downloaded. Project license is CC0; publication
  assets carry their separately reported CC BY-NC-ND license.
- Deposited `F1_SF2_SF3_SF4_SF5_SearchResults.xlsx`: 21 sheets, 3,235 input-mass rows,
  1,323 assignment rows, including substrate-only and proteasome-only controls.
- `F2_SF6_SearchResults.xlsx`: 21 sheets, 5,064 input-mass rows, 2,617 assignment rows.
  Seven nominal times (0 through 120 minutes) and three replicate labels.
- `F3_SF7_SearchResults.xlsx`: 44 sheets, 9,563 input-mass rows, 4,198 assignment rows.
  Brain/liver/spleen preparations, five nominal times, three declared replicates.
- Five original supplementary XLSX plus SI PDF were retrieved through the public
  PMC Open Data bucket. Eight workbooks total: **133 sheets / 325,718 populated
  cells**, independently compared with OOXML, including **2,991 formulas and their
  stored cached values**. No workbook was edited or recalculated.
- Dataset S4: **1,289 positive intensity rows, 237 sequences, 21 sample labels**.
  Dataset S5: **2,138 rows, 313 sequences, 44 sample labels**. Both include intact
  substrate; their length ranges are 5–140 and 4–140 respectively.
- All 3,427 published rows reconcile to one deposited identity when source span,
  fragment type and, when needed, exact observed mass are retained. **3,426 intensity
  values agree exactly**; one conflict remains below. These are source reconciliation
  results, not a newly admitted training cohort.

Public processed files reside under
`https://ftp.pride.ebi.ac.uk/pride/data/archive/2025/08/PXD058121/`.
Supplement objects are under
`https://pmc-oa-opendata.s3.amazonaws.com/PMC11874349.1/`.

## Required homogeneous source representation

- Separate original input masses and their sum intensities, candidate assignments,
  charge/envelope measurements, supplement views, and source-context assertions.
  Keep unassigned masses, every assignment alternative, literal coordinates,
  original/cached formulas, missing/error tokens, fit settings and modifications.
- Link every quantity to its source file/sheet/row and acquisition/preparation.
  Source Mass ID is local to its sheet. Repeated rows across figures are not new
  observations: all nine common F1/F2 sheets have identical input and assignment
  values after literal header/type equivalence is documented.
- Use the common material/context graph: substrate is recombinant human
  alpha-synuclein, enzyme preparations are mouse-organ-derived 20S, and expression
  host is a distinct provenance role. These are purified degradation mixtures,
  not MHC elutions or intact-cell APC pools. MHC capture/class/allele are inapplicable.
- The methods explicitly say all organs came from the same animal. Preserve R1–R3
  as reported labels with unknown biological/technical independence; never create
  three animal IDs or independent cell types from them.
- Preserve actual substrate sequence and span checks, source concentrations as
  protocol assertions, nominal time and pre-zero processing. A nonempty product
  pool exists at nominal zero. Protocol input amount does not calibrate per-peptide
  MS response or turn intensity into molarity.
- Fit factor, mass tolerance, assignment multiplicity, source recurrence selection,
  MS/MS-confirmed examples and confidence are separate axes. A unique mass match
  is not broad sequence validation. Missing products are not zero or negatives.
- Keep raw summed/deconvoluted intensity distinct from total-spectrum-normalized
  plots and z scores. The sum of listed deconvoluted input intensities must not be
  silently called TIC or a calibrated normalization standard.

## Concrete source exceptions to preserve and resolve

1. **Missing replicate:** `Spleen_30min_3` is absent from F3, Dataset S5 and the PRIDE
   RAW inventory. There are only R1 and R2 RAW files for this condition. Record
   absence and request the processed export/acquisition or its exclusion reason.
2. **F2 shifted body:** `20S_AlphaSyn_40min_R1` has an extra unlabeled composite ID
   in column H. Headers H:U describe values in I:V. Preserve the literal sheet and
   a documented interpretation. All 68 assignment spans validate under that
   interpretation and their S4 counterparts reconcile; do not silently shift data.
3. **F3 missing input IDs:** `Brain_20S_aSyn_0min_2` has input values in A:C while
   A:D headers imply a Mass ID column. Actual input IDs are absent. Do not invent
   IDs from row order or assert assignment-to-input links from that inference.
4. **Publisher/deposit disagreement:** S4 `Kinetics` row20, `100min_Rep1`, span
   128–140, disagrees with F2 `20S_AlphaSyn_100min_R1` row246 on observed mass,
   error and intensity. Preserve both values and mark conflict until resolved.
5. **Other linkage/quantity discrepancies:** F1/F2 `20S_AlphaSyn_120min_R2` row93
   references an input Mass ID not listed in its input block. F2
   `20S_AlphaSyn_100min_R1` assignment rows219–220 and F3
   `Liver_20S_aSyn_90min_3` row13 disagree with their named input intensity.
   Do not substitute one value for another.
6. **Machinery measurements:** SI Figure S8 reports catalytic-subunit blots and
   relative quantification, but original numeric densitometry/sample links were
   not located in the retrieved workbooks. Request those original tables and
   normalization details; organ identity alone must not become numeric PSMB state.
7. **Replicate/material precision:** request preparation-to-reaction-to-aliquot-to-
   acquisition links and replicate semantics, plus any available spike/calibration
   information. Preserve unknowns and protocol-claim conflicts without guessing.

## Acceptance

- [ ] Shared source adapter, schema/catalog registration and public EvidenceStore query.
- [ ] Pinned original assets, complete inventories, all measurement/assignment/source
  decisions and reusable provenance, including the listed exceptions.
- [ ] Independent replay of original cells, assignment-to-input and supplement links;
  tests for shifted/missing columns, string/numeric IDs, duplicate masses, ambiguity,
  missing replicates, repeated figures and genuine missing-versus-zero values.
- [ ] Explicit recipe support/eligibility axes before any supervised cohort. Retain
  conflicting and ambiguous records in the source even if excluded downstream.
- [ ] Documentation, complete CI, merge and verified package release.

No predictive result or training admission is claimed by this issue. A subsequent
registered experiment should compare direct time controls with exact and sampled
generation/destruction models on the same observable endpoint. One substrate and
one animal cannot establish new-protein or APC-concentration generalization.
