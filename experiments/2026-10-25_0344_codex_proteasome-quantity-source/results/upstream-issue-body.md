The availability experiments need native intracellular peptide quantities under a documented processing-machinery perturbation. Sahu2021 provides an unadmitted candidate that is distinct from MHC elution and proteasome-trapped peptidomics.

Primary source: [Sahu et al., Nature Communications2021](https://doi.org/10.1038/s41467-021-26427-0), [PMC8548400](https://pmc.ncbi.nlm.nih.gov/articles/PMC8548400/), [PXD018722](https://www.ebi.ac.uk/pride/archive/projects/PXD018722). Related infrastructure: #788, #795, #811. This HEK293T population does not close the professional-APC source gap in #811.

## Recovered evidence

Dataset7, `41467_2021_26427_MOESM10_ESM.xlsx`, sheet `peptides`, contains **10,647 distinct peptide sequences**, 35 columns and six sequence-level intensity columns: Cont1–3 and KD1–3. There are **34,036 positive intensity cells and 29,846 source zeros**, before any training eligibility. SHA256: `f29811e09144f6d526ce1f2955681bb5f07a7ed0a4957e1a9f596740fc492ea5`.

The study associates this panel with native intracellular peptide extraction from human HEK293T control versus inducible PSMD2/Rpn1 knockdown cells. It separately measures purified digests, proteasome-trapped peptides, hypoxia and heart tissue; those populations must not be pooled under one capture label.

The official project inventory has48unique files, with an empty next page. `Hi20S_experimentalDesignTemplate.txt` (156bytes) and `Hi20S_mqpar.xml` (22,385bytes) match deposited sizes/SHA1. Both independently map Cont1–3/KD1–3 to `Seq61406_HFX.raw` through `Seq61411_HFX.raw`, which exist in the inventory. No RAW spectra were downloaded.

Independent openpyxl/OOXML readers verified293,009populated cells across Dataset7 and the separately recovered Dataset6. Dataset6 is34branched peptide-spectrum rows across six sheets, not a quantitative availability endpoint. Original recovery descriptors, hashes, failed responses and independent CRC replay are preserved in Presto family `2026-10-25_0344_codex_proteasome-quantity-source`.

## Metadata and defects the shared schema must preserve

- The actual Hi20S search length range is **8–25**, although the article broadly states minimum7. Shorter products are unsearched, not biological negatives.
- Matching between runs is enabled; its FDR option is false. Preserve each sample's `Identification type`, plus peptide-level PEP and separate global peptide/protein FDR settings. Matching-only values are not independent MS/MS identifications.
- Source columns are `Intensity`, not `LFQ intensity`. Enabled LFQ settings do not establish that these exported sequence-level values are LFQ-normalized or unmodified-peptidoform quantities. Original `Hi20S_txt.zip` (167,389,796bytes) is available to resolve aggregation; prefer selective member retrieval to a full archive download.
- **13cells in `Gene names` are date-formatted Excel serials.** Preserve original serial/style/rendered date and flag the annotation; do not infer a gene symbol from a date. Protein-accession alternatives remain available.
- There are6,202literal parent-accession tokens before reference resolution. Preserve all alternatives and reference-version/construct provenance rather than selecting the leading razor protein. The deposited `HUMAN_NS3_Cyclin.fasta` is33,003,906bytes; human endogenous products must be distinguishable from reporter-derived records.
- PSMD2 knockdown, measured proteasome composition and numeric activity are different assertions. Do not encode complete KO or assign the paper's aggregate composition estimate to every sample.
- Numbered labels do not establish biological replicate independence, exact control/KD pairing or clone independence. Reporter, doxycycline, other exposure and handling context need evidence at the appropriate material scope.
- Source zeros, decoy/contaminant flags, missing parent annotations and sequence/form aggregation stay explicit. No imputed negatives or inferred censor limits.
- No independently matched precursor abundance, extraction recovery or molar calibration has been established. Protein groups calculated from the same peptidome are not independent precursor measurements.

## Acceptance work

- [ ] Add a pinned native-intracellular source to the common evidence/processing catalog, preserving all original rows, columns, cells and exception dispositions.
- [ ] Resolve the original reference and all human candidate origins; retain reporter/ambiguous records as explicit separate dispositions and build dependence groups before eligibility.
- [ ] Reconcile workbook quantities/annotations to the relevant original processed text where available; preserve disagreements rather than silently repair.
- [ ] Encode source-supported species, cell type, perturbation, specimen/preparation uncertainty, capture/extraction, search and observation metadata in the homogeneous system.
- [ ] Expose source quantities and six acquisition mappings through the standard query/materialization interfaces, with explicit units/normalization/form scope and matching evidence.
- [ ] Provide a registered recipe only after freezing its endpoint and source support. Candidate first endpoint: within-parent control/KD relative-product contrasts, cancelling shared precursor/run scale and fixed peptide response under stated assumptions. It must not be labeled absolute concentration.
- [ ] Add tests for all exceptions above and independent source/reference/quantity replay; follow normal versioned PR, full checks, merge and tested release workflow.

The source workbook is promising evidence, not a trained model or biological validation. Existing inspected Presto test cohorts remain closed. Broader APC transfer and calibrated complete-pool quantities remain separate requirements.
