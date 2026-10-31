The HLA Ligand Atlas separates observed MS peptides from computational peptide–HLA associations. Its Methods state that class-II associations use NetMHCIIpan/MixMHC2pred and can include multiple donor alleles; analyses without specified allele associations include detected peptides classified as nonbinders. These named associations therefore need explicit computational provenance before they can be used as independent restriction evidence.

Primary source: https://pmc.ncbi.nlm.nih.gov/articles/PMC8054196/ (PMID 33858848; “HLA binding prediction” Methods), DOI https://doi.org/10.1136/jitc-2020-002071.

### Reproduction and scope

- Current read-only Hitlist 2.0.42, commit `13b91424295e01286db46e8e9818fcc2fe816834`.
- Older native human export producer: Hitlist 1.78.0, SHA256 `f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd`.
- In Presto's admitted, unmodified class-II MS subset, 24,399 Atlas observations have explicit paired names but native `restriction_evidence=unknown`.
- Replaying all of those source rows through current `restriction_evidence_for_row` still returns `unknown` for all 24,399. This is a current classifier replay, not a newly regenerated export.
- Examples: IEDB assays 11198595, 11198624 and 11198625, whose source names are HLA-DPA1*01:03/DPB1*04:01.
- The count defines the scope needing a source-association audit. It does **not** prove every IEDB row was derived from the paper's prediction table; exact row joins remain necessary.

### Requested behavior

1. Join native named assignments to the original association evidence and retain the assignment method/version/source. Mark confirmed prediction-derived associations accordingly.
2. Keep the observed-positive MS label and donor typing separate from peptide-specific restriction evidence. Preserve candidate sets and incomplete/unphased typing.
3. Do not blanket-relabel every PMID row, including any separately measured binding assays. Do not turn unassigned peptides into negatives or supply missing alpha partners.
4. Add source-backed regression fixtures for confirmed predicted associations and observed MS detections without allele assignments.

The donor-aware primary Atlas loader already exists (#702, closed); this is not a request to build it again. #31 is the older completed computational-assignment request. A missing assignment-provenance distinction still prevents using these native named pairs as independent restriction validation.

Consumer evidence: `experiments/2026-10-26_1954_codex_mhc2-ms-transfer-cohort/results/current-source-qualification.json` on Presto branch `codex/mhc2-improvement-20261026` (publication pending). Verified original XML SHA256 `1d901f85a12c3772d785688b84473044d34c699d8d339af005b43d97c6dd7edb`.
