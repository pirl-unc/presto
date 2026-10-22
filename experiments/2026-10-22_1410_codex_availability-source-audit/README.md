# Endogenous availability source audit

October 22, 2026; Codex / GPT-6. **Initial PXD057588 inventory completed;
biological training admission and shared Hitlist implementation remain pending.**
This supplies acceptance evidence for the [curation contract](../../docs/study_curation_contract.md),
[Hitlist materials/protocols #736](https://github.com/pirl-unc/hitlist/issues/736)
and [cross-modality #737](https://github.com/pirl-unc/hitlist/issues/737).
It is not a second production metadata registry.

## Data and measurement contract

The [Cui et al. study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12616400/)
(DOI 10.1158/0008-5472.CAN-24-4674) contains macrophage/tumor experiments,
relative MHC-I immunopeptidomics and global proteomics. The six inspected workbooks
concern mouse experiments; the paper's human and targeted measurements are outside
this inventory. Do not broadcast a study-level species or MHC-loss annotation.

Inputs: PXD057588 project/file API snapshots, community SDRF, publisher Tables 7–12
and their Figshare metadata; PXD009092 project/file discovery snapshots. Seventeen
files, **11,532,567 bytes**, installed with datacache under
`data/availability_sources/aa7ef2db2e19dacd55bd0fce1c332469b895942b58f3d18f4873a7e52d7bdf27/`.
URLs, sizes and SHA-256 are frozen in [input_assets.json](reproduce/input_assets.json).
All six workbook publisher sizes/MD5s pass. Licenses/versions and retrieval limits
are in [retrieval_receipts.json](results/retrieval_receipts.json): supplements
declare CC BY 4.0; PRIDE PXD057588 declares CC0. API drift requires a new reviewed
snapshot, not silent refresh. Original bytes remain unchanged.

No model, pretraining/training, loss, synthetic data, assay-head mapping or
train/validation/test split applies. Prediction dumps, held-out losses,
correlations and binding metrics are inapplicable to this source audit.
No Modal/GPU. No biological rows have been admitted for fitting.

## Quantitative coverage

Counts refer to the leftmost primary sequence block, not independent subjects.
All **309,126 quantity cells** are positive numbers. Every table describes
complete-case filtering; this does not establish coverage of discarded/unobserved
peptides. All 35,772 source rows have distinct sequences within each table;
sequences can recur across tables/conditions.

| Table | Published comparison / measured pool | Peptide rows | Channels | Distinct parent labels |
|---|---|---:|---:|---:|
| [7](https://doi.org/10.1158/0008-5472.30883082) | Pro-inflammatory versus untreated macrophage MHC-I | 2,824 | 6 | 2,139 |
| [8](https://doi.org/10.1158/0008-5472.30883079) | Anti-inflammatory versus untreated macrophage MHC-I | 3,554 | 6 | 2,562 |
| [9](https://doi.org/10.1158/0008-5472.30883076) | Macrophage MHC-I after GL261/CT2A coculture versus untreated | 1,309 | 9 | 1,096 |
| [10](https://doi.org/10.1158/0008-5472.30883115) | Macrophage MHC-I with MHC-I-loss tumor partners versus untreated | 2,613 | 11 | 1,997 |
| [11](https://doi.org/10.1158/0008-5472.30883112) | Macrophage global-proteomics digest peptides across coculture states | 25,110 | 9 | 4,495 |
| [12](https://doi.org/10.1158/0008-5472.30883109) | Tumor-cell MHC-I before/after macrophage coculture | 362 | 12 | 348 |

There are 10,662 MHC-I rows and 25,110 digest-peptide rows. Parent counts are literal
source labels, not reference-validated proteins. Table 11 A5 is `Sequence` and L5
is `Protein`: its title does not make each row a protein measurement. Protein
aggregation/ambiguity remain unresolved. Tables 9 and 11 share 663 parent labels;
433 Table 9 labels have no exact Table 11 match. This is not a validated precursor join.

The first panels are labeled raw TMT intensity; other panels contain log2 changes.
Retain both scales and normalization evidence. Unequal channel totals alone do not
reconstruct processing history. These quantities do not label free intracellular
concentration, isolated production rates or calibrated probabilities over all
peptides. The reported q≤0.05 search policy and missing-channel exclusion need
explicit eligibility treatment and earlier-stage source data.

## Concrete shared-curation acceptance cases

1. **Scope channel identity to assay/plex.** Eight of nine Table 9/11 channel
   mappings differ: `128N` is `G3+` in Table 9 and `C2+` in Table 11 (B4:B5).
   `133N` is the sole same-label mapping. Channel joins would mix conditions;
   matching sample names still do not prove shared specimens. The paper describes
   post-IP lysate proteomics, which is workflow evidence, not resolved aliquot IDs.
2. **Separate measured population and perturbed partner.** Table 10 A2:A3 describes
   macrophage measurements with MHC-I-loss tumor partners. The paper identifies
   tumor-cell H2-Kb/H2-Db loss. This must not train a macrophage processing-gene
   knockout effect. Ordinary coculture separation and loss-arm attribution differ.
3. **Preserve conflicts and replication evidence.** Table 8 A2 calls M2
   pro-inflammatory, conflicting with its title and other descriptions. Table 10
   A3 says n=3 while B5:L5 contains G1–G4, C1–C4, U1–U3. Preserve both assertions;
   do not discard a fourth channel or infer independent donors from suffixes.
4. **Preserve incomplete SDRF scope.** Its 72 rows and 45 columns annotate twelve
   channel-associated names across six E38 CD45-negative tumor raw files. The
   deposit has 50 RAW, seven PEAK, seven RESULT and one design file: 44 raw files
   are unannotated. Treatment is `not available`; replicate labels 1–12 do not
   establish twelve independent donors. All twelve Table 12 channels match SDRF
   labels after adding `TMT`, but this does not prove a specimen join or coculture
   treatment for Table 12's untreated controls.
5. **Keep repeated fields and independent panels.** SDRF repeats template fields
   three times and modification fields four times; preserve their positions.
   Workbook significance panels have independent sequence keys/order: all 157
   populated Table 8 R-column sequences differ from their same-row A-column
   sequences. Spreadsheet row numbers are not joins across horizontal panels.
6. **Separate measured pools, missingness and calibration scope.** MHC ligands
   and digest peptides are different evidence. Filtered observations are not
   measured zeros. Targeted copies/cell results elsewhere in the paper have their
   own calibration/recovery scope; they do not calibrate all relative tables.

Machine-readable evidence: [workbook cells/coverage](results/workbook_inventory.json),
[SDRF fields/coverage](results/sdrf_inventory.json),
[cross-assay checks](results/cross_assay_checks.json),
[compact counts](results/table_summary.csv). These retain native labels and
coordinates without inventing normalized biological subjects.

## Decision and remaining qualification

**Ingestion candidate; training deferred.** Curate shared material/condition/assay
identities, scope species/MHC/partner loss, resolve or retain conflicts, qualify
specimen links, reconcile transformations/reference mappings/search selection,
then freeze per-row eligibility and lineage-aware holdouts through Hitlist.
The measured endpoints support observed presentation/digest signals, not the
complete intracellular pool. Six tables are not six independent studies.
This source audit neither changes the synthetic optimizer conclusions nor
qualifies a concentration model.

Wolf-Levy remains a distinct candidate. PXD009092 lists nineteen files including
a 51,736,986-byte SEARCH.7z. Two attempts timed out; the partial archive is excluded
and no measurement-table claims are made. PMC returned an HTML supplement download
page, not XLSX; official AACR Figshare supplied PXD057588's workbooks. No access
controls were bypassed.

## Reproduction and verification

Base commit `932ccefceda4aa82e98a43fbe01ebb8896e9dfec`, dirty audit additions;
source/invocation frozen in [reproduce](reproduce/launch.json). Python 3.12.6,
datacache 1.29.2, openpyxl 3.1.5, macOS arm64 CPU. Initial acquisition used public
curl requests; datacache verifies the imported originals. Final local inspection
[runtime](results/runtime.json) excludes network acquisition/discovery and is not
a runtime comparison. Large originals are ignored by git; the receipts reproduce
their exact public downloads or fail explicitly on upstream drift.
Small original API/SDRF snapshots are tracked under
`data/availability_sources/snapshots/2026-10-22/`; the launcher imports these
unchanged bytes so mutable API counters do not prevent replay.

```bash
AUDIT_PYTHON=/path/to/python bash experiments/2026-10-22_1410_codex_availability-source-audit/reproduce/launch.sh
/path/to/python experiments/2026-10-22_1410_codex_availability-source-audit/code/verify.py
```

Independent OOXML reconstruction without openpyxl verifies all seventeen hashes,
six table/header inventories, **309,126 quantity cells** and **3,240 SDRF cells**;
[verification passed](results/independent_verification.json). Explicit Ruff
E4/E7/E9/F checks pass (repo defaults exclude experiments). No workbook was edited
or recalculated. Full SDRF ontology validation, normalized upstream metadata
round-trip and biological admission have not been performed.
