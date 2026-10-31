The current Abelin2019 MAPTAC class-II import merges the HLA-DM conditions before observation admission. That loses the contrast needed to evaluate DM-dependent ligand selection separately from binding and generic MS detection.

Verified against Hitlist **2.0.42**, commit `13b91424295e01286db46e8e9818fcc2fe816834`:

- [`supplementary.yaml`](https://github.com/pirl-unc/hitlist/blob/13b91424295e01286db46e8e9818fcc2fe816834/hitlist/data/supplementary.yaml#L54) explicitly says the two conditions are merged per allele.
- [`pmid_overrides.yaml`](https://github.com/pirl-unc/hitlist/blob/13b91424295e01286db46e8e9818fcc2fe816834/hitlist/data/pmid_overrides.yaml#L4636) keeps separate DM arms and states that the imported observations cannot be assigned to them.
- [`abelin_2019_maptac_class2.csv`](https://github.com/pirl-unc/hitlist/blob/13b91424295e01286db46e8e9818fcc2fe816834/hitlist/data/supplementary/abelin_2019_maptac_class2.csv) has only `peptide,mhc_class,mhc_restriction`, with no source sample/DM discriminator.

Two independent readers (Python CSV and Arrow CSV/group-by) agree on 43,503 imported rows. **11,184 rows** use the four literal restrictions with both DM arms in current curation:

| Imported restriction | Merged import rows |
|---|---:|
| HLA-DRB1*12:01 | 2,276 |
| HLA-DRB3*01:01 | 2,357 |
| HLA-DPB1*06:01 | 4,587 |
| HLA-DQB1*06:04 | 1,964 |

These are counts of the merged import, **not** recovered per-condition counts or independent acquisitions. The original DataS1B workbook was not recovered in this audit: the publisher download returned HTTP403. The source-level contrast is also described in the [primary study abstract, PMID31495665](https://pubmed.ncbi.nlm.nih.gov/31495665/). Do not infer original workbook cells or per-arm membership from this audit.

## Requested work and acceptance

1. Recover and pin the original Table S1/DataS1B source (or an authorized deposited equivalent), then retain the actual source block/row, sample and condition identifiers before merging peptide/allele records. Independently reconcile the admitted cells to that source.
2. Preserve observations from both conditions when a peptide occurs in both. Absence from one identification list is not a measured biological negative. Keep merged historical rows distinguishable and do not count them again alongside their recovered contributors.
3. Link observations to their source-backed material/condition assertions in the shared EvidenceStore and exports. Preserve complete DP/DQ pair context without inventing a peptide-specific restriction or arbitrary alpha-chain default.
4. Distinguish **no added DM co-transfection** from measured loss/absence of endogenous DM, and distinguish the introduced HLA construct from the additional DM intervention. Retain unknown endogenous expression, unresolved sample pooling and assay/acquisition scope.
5. Add real-source regressions showing that the four paired contrasts remain distinct after ingestion, export and query, including shared peptides and unresolved cases. Report admission/attribution coverage and reasons for any remaining gaps.

Scope is the upstream loss of observation-to-DM-arm identity. #593 owns material grouping/engineering normalization; this issue requires recovering the missing import discriminator, which normalization of the merged rows cannot supply. #366 is closed but the current source explicitly retains this unresolved case. #151 already addressed beta-chain-to-heterodimer matching and does not restore DM membership. Keep #817's separate Nanaware source-admission work separate.

This enables a useful MHC-II condition comparison; it does not establish a causal processing model or turn MS ligands into affinity measurements.
