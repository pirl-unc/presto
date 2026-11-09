# Independent ERAP1 source qualification

Registered before workbook-cell inspection on October 24, 2026 by Codex / GPT-6.
The completed HAP1 experiment found no improvement from the current trimming model.
This family asks whether the Maben2021 deposited source data can independently
constrain substrate-dependent processing. It does not fit a model or establish
cellular availability. Plan: `reproduce/plan.md`.

Source: DOI 10.1038/s41467-021-25564-w, public Source Data ZIP, SHA256
`a9d7bcbae52b25e55aec5f725578238fd0e6719720153dc829b999dd4bde5d2c`.
Original archive: `/private/tmp/hitlist-erap1-809-assets/maben2021-source-data.zip`.
The 2023 correction concerns an acknowledgment. Source assets remain unchanged.

No pretraining, optimizer, loss, synthetic targets, assay predictions, or held-out
split is used: this is source qualification before training admission. All prior
biological comparisons and the reserved HAP1 outer splits remain unchanged.
Execution is local, with no installations or environment changes.

## Outcome

The archive does **not** qualify as an independent, sequence-diverse training
cohort for the current peptide-processing model. It contains real biochemical
evidence, but the directly relevant panels use one unmodified peptide and one
BPA-modified probe, both with the same N-terminal residue. They do not supply a
broad product-resolved trimming dataset. No new model was fit and no prior
biological result or baseline changes.

The original 479,402-byte ZIP and both workbook byte streams pass hash/CRC checks.
There are 34 sheets, 3,726 populated cells, 3,390 numeric cells and no formulas.
Numeric cells include axes, structural calculations, uncertainties and fitted
parameters: they are not independent biological observations. A separate openpyxl
3.1.5 reader reproduces every populated value and numeric float exactly. Inventory
took 0.116 seconds; independent verification 0.044 seconds with 55.9 MiB peak RSS.

| Panel category | Sheets | Processing-model interpretation |
|---|---:|---|
| Structural measurements or structural calculations | 21 | No peptide-turnover labels |
| Peptide assays | 2 | Narrow evidence with unresolved assay-scale and experiment-group metadata |
| Reporter activity, sometimes alongside fitted summaries | 9 | Reporter readouts cannot be relabeled peptide-specific rates |
| Derived reporter fit parameters | 1 | Preserve derivation and dependence on the fitted curves |
| Gel image panel | 1 | No tabular quantitative peptide assay |

The complete per-sheet disposition table is in `results/sheet-dispositions.csv`.

## Relevant cells and unresolved semantics

- Main workbook `Fig 2h`, C5:I16: one peptide across 12 substrate concentrations
  and three enzyme conditions. There are 70 numeric response cells in 72 duplicate
  slots. H14 and I15 are missing and remain missing. C4 declares molar substrate
  concentration. The workbook does not establish the response's rate unit. The
  article describes a coupled absorbance assay and multiple independent experiments,
  while this sheet shows two columns per condition without independent-experiment
  identifiers. Do not invent all claimed experiments or interpret these as cells.
- Supplementary workbook `Fig S8d`, C4:E13 and H4:J13: two peptide probes at a
  reported 20 uM with nine time points. There are 35 numeric responses in 36 slots;
  D5 is missing, whereas I5 and J5 are reported zero. Four values exceed 100
  (E10, E11, J9, J10). Preserve them. The precise response normalization, product
  identity, and independent-experiment grouping remain unqualified. A percent-like
  scale is not evidence of an absolute concentration or a bounded probability.
- Main workbook `Fig 2i`: K11:K16 and L11:L16 are identical concentration values,
  although L9 labels the enzyme condition. L must not be misread as a fourth rate
  replicate. The kinetic summaries elsewhere on this sheet and in `Fig S12b` are
  derived results, even though the workbook contains no formulas.
- The model ovalbumin peptide appears in `Fig S2a` as a structural condition.
  That does not provide a measured peptide-processing trajectory.
- Different ERAP1 constructs/variants and independent-assay declarations in the
  article must remain explicit. None identifies the HAP1 enzyme state or supplies
  a cellular precursor abundance, transport, MHC selection or MS response label.

These observations are source qualification, not a claim that the biochemical
experiments are invalid. A narrow assay could later be admitted with its actual
units/context, but it cannot support the planned sequence-generalizing fit alone.

## Provenance and reproduction

Primary article: [Maben et al., 2021](https://www.nature.com/articles/s41467-021-25564-w).
The public article XML is preserved with SHA256
`3776aa5739d7ba0609539e1af601a313b0b831ebeba59c6e6955f2ca8ea5d65b`.
The [2023 correction](https://www.nature.com/articles/s41467-023-37300-7) changes
an acknowledgment. Article license is CC BY 4.0, subject to its third-party
exceptions. The source attribution remains intact.

`reproduce/launch.sh` and `launch.json` freeze the commands and Git state. Actual
pre-format scripts are under `reproduce/executed/`; final formatted scripts are
under `code/` and `reproduce/source/`. Read-only extraction uses the bundled
Python 3.12 runtime and standard OOXML parsing; the independent reader is
openpyxl 3.1.5. No workbook is recalculated, edited or exported. Original workbook
copies, all cell-addressed fields and HTTP receipts are retained under
`data/erap1_source_qualification/2026-10-24_0924_maben-v1`. Inventory and
verification summaries are copied into `results/`. No data adapter or production
training recipe was added to Presto. Canonical admission belongs in Hitlist809.

Independent reader closure covers every populated cell, original bytes, worksheet
identity, numeric counts and absence of formulas. Semantic classification is a
manual source interpretation, explicitly retaining unresolved metadata. It is not
an independently proven assay reconstruction. There is no fitting/test split or
prediction dump because no modeling cohort was admitted. No synthetic data,
pretraining, assay loss, parameter updates or concentration predictions occur.

Status: workbook inventory and the proposed-training admission decision are closed.
Next: preserve narrow biochemical evidence upstream and seek a source with broader
substrate/product measurements or native intracellular peptides from APC material.
Open questions: response calibration, independent experiment IDs, source replicate
availability and a sufficiently varied training cohort. The full goal stays active.
