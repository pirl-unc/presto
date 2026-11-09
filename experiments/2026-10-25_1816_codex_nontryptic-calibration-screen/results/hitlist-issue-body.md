## Problem and scope

Extend #24 with reusable **known-dose standard / raw-response calibration**, separately from inferred endogenous copies per cell. Presto needs to distinguish MS response from peptide availability and processing. Current Hitlist source/curation search at 5a5a416 finds the study's sample metadata, but no Iso18 calibration-cell ingestion. Preserve the corrected SKMEL5/binimetinib identities from #436.

Primary source: [Stopfer et al., PNAS 2021](https://doi.org/10.1073/pnas.2111173118), PMID 34497125, [PMC8449407](https://pmc.ncbi.nlm.nih.gov/articles/PMC8449407/), PRIDE PXD024917. The panel comprises 18 selected tumor-antigen sequences, selected as predicted HLA-A*02:01 binders. This does not establish measured allele restriction for every source sample.

## Public processed assets verified

Official source base: https://pmc-oa-opendata.s3.amazonaws.com/PMC8449407.1/

| Asset | Role | SHA256 |
|---|---|---|
| pnas.2111173118.sd01.xlsx | selected panel; predicted affinity metadata | 71f8bd65d178d78014faf8d4e01a227fbb20ee90eee9ee8e9fbe6c5d311d0d2c |
| pnas.2111173118.sd03.xlsx | separate added fmol and summed fragment-ion signals; transition counts | a51dce88e3fa91b6f38d6f5f86cf3347ee285bcf7e6ca0384a16de1a8d9d0803 |
| pnas.2111173118.sd06.xlsx | isotope and acquisition definitions | ed26e8b29dda9d48e5784e392b30271cfc1e5a7b7ada3fadfcbe344fe68131b5 |
| pnas.2111173118.sd07.xlsx | deposited raw/processed filename map | ab93052b47887c19df62a2c5580a7b2245cf98e5aa91eee88d4d61c31607cfa9 |
| pnas.2111173118.sapp.pdf | standard preparation, spike stages and analytical methods | 5f8b2868251304250a0a1e18d0c27ffa49a4df886bfcaf57b44329084d991265 |

Official metadata MD5s and all 18,911 stored worksheet cells across four workbooks were independently checked with OOXML and openpyxl. This is a literal-cell count including formatted blanks, not a measurement count. Presto source-qualification family: `experiments/2026-10-25_1816_codex_nontryptic-calibration-screen/`. No training admitted.

## Preserve the different observation processes

- S3A: free 1H peptide added to a previously purified HLA-peptide background. **18 sequences, 71 reported dose rows, 213 positive signal cells** across unlabeled columns C/D/E. Seventeen sequences have 0.1/1/10/100 fmol; GLFDQHFRL[+7] at A16 has only 1/10/100 fmol. There is no 0.1-fmol source row: do not create a zero or pretend three missing cells have coordinates.
- S3B: embedded 2H hipMHC spike before immunoprecipitation. **9 sequences, 45 dose rows, 135 worksheet slots: 126 positive and 9 blank**. Doses 0.1/1/10/100/1000 fmol. Preserve original missing cells without inferring their cause.
- S3C: selected transition counts at dose levels, **not peptide signal intensities or molecule amounts**.
- S1 affinities are predictions, not experimental binding labels.
- Free-peptide content is reported as amino-acid-analysis assigned, HPLC purity >97%, with isotopologue balancing. Embedded-complex amounts use reported stable-complex ELISA after exchange. Retain reported assignment method and uncertainty; do not label either independently certified.
- Targeted SureQuant/IS-PRM summed selected product-ion response is distinct from MS1 detectability. The 4H trigger is an analytical reagent; preserve its dose/protocol, not as biological machinery.
- Free spikes constrain downstream response/matrix. Embedded spikes additionally involve recovery and capture. Different isotope/spike designs do not by themselves isolate causal purification loss.
- Neither route directly measures native intracellular free-peptide concentrations, APC-wide availability, or precursor-normalized production. Do not map these calibration labels to APC machinery or availability supervision.

## Exact metadata and source conflicts to resolve

1. **C/D/E are not individually named by acquisition.** S7 lists three numbered raw filenames per dose, but positional correspondence is not proven. Retrieve the deposited exports named literally `Iso18_DirectSpike_TransitionResults.xlsx` and `Iso18_hipMHCspiikein_TransitionResults.xlsx` (the double-i spelling is in S7). Preserve exact source spelling, checksums, URLs and acquisition/aliquot lineage. Verify column-to-run links, transition identities, summed-area reconstruction, interference exclusions and missingness; do not infer C = run 1.
2. **Valine isotope conflict:** S6A A10:C10 reports V / 13C3,15N1 / 4.007 Da, while S3 uses V[+6] and S6B contains nominal shifts consistent with +6. An arithmetic audit finds 54/72 S6B isotopologue records inconsistent with summing S6A masses, plus one uncheckable by S6A alone. This is a source-consistency diagnostic, not 54 independent chemical failures. Preserve both assertions until stronger source evidence supports a resolution.
3. **Methionine table omission:** S6B B99 includes labeled M and E99 reports M:13C5,15N1, but S6A has no M entry. Preserve that distinction; do not claim isotope identity absent from the whole source.
4. S6C supplies trigger/method transitions; establish whether these are the exact quantification transitions before joining them to S3 summed areas. Retain unresolved charge/offset discrepancies and source fields.
5. Keep matrix, spike timing, sample/aliquot/injection, input amount/volume, antibody/purification, instrument/acquisition, target selection and transition aggregation separate in homogeneous protocol/assertion tables. Unknowns must remain typed unknowns.

## Acceptance criteria

- [ ] Source resolver and reusable package through existing Hitlist data-family/schema conventions; no Presto-only adapter.
- [ ] Stable study/material/analyte-form/protocol/measurement identities, raw source cells, literal blanks and lineage; raw quantity and standardized units separately.
- [ ] Explicit roles for nominal standard amount, reported signal, transition count, predicted affinity and inferred endogenous abundance.
- [ ] Row-level support/exclusion decisions for unresolved isotope identities and acquisition joins, without silently rewriting original values.
- [ ] Replay all 348 S3A/B worksheet slots, the absent dose row, panel joins and every curation decision independently.
- [ ] Tests for missing versus zero, ordinal columns versus acquisition IDs, corrupted source hashes, isotope contradictions, and separation of calibration from biological observations.
- [ ] A recipe may admit a clearly scoped empirical sequence/dose response subset only with documented uncertainty and peptide-grouped splits. Do not claim MS1 or APC concentration transfer from this selected panel.
