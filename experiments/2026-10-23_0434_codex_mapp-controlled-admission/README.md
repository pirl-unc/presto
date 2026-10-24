# Controlled MAPP panels: source inventory and admission

Registered before new downloads or source analysis. Full availability objective remains active. **Completed source audit:** the public controls archive contains an identifiable epoxomicin comparison, and all 146,832 peptide/run quantity comparisons across the controls and time-course archives independently replay. Shared Hitlist admission and biological fitting remain next steps.

## Contract and reproduction

- Date: October 23, 2026. Agent/model: Codex / GPT-6.
- Presto launch commit: `9f8e4e0761a6acf76c3db94a55bea640b2f6bd65`; dirty status is preserved in `reproduce/launch.json`. Existing unrelated work is excluded from this family.
- Hitlist launch commit: `e186dd7831976f505b06571fe7b5419ee16e003b`, clean. Canonical source generation: `/private/tmp/hitlist-v2-calibration-203-canonical`, manifest SHA256 `bce500f34f111aaf3b41573e96fa630252204a3403d5a2e93e9c50caa97b8552`.
- Local CPU only; no Modal/GPU. Python: `/Users/iskander/code/shared-virtual-env/bin/python`. No package/environment changes. Frozen source scripts and prospective stage configurations are in `reproduce/`.
- Raw assets: `/private/tmp/presto-mapp-controlled-assets`; existing MAPP source tables remain under the canonical generation's `external/pxd009092/assets`.
- No model, pretraining, synthetic data, optimization, losses/weights, or assay-to-model-output mapping is used in this source audit. No held-out split or validation/test predictions are produced because source discovery/qualification precedes training. This is not a new predictive performance estimate.
- Missing, filtered and absent records remain distinct. No affinity filtering, missing-to-zero conversion, or inference of live-cell treatment from a project title/filename is allowed.

## Original study and deposited projects

Primary source: [Wolf-Levy et al., 2018](https://doi.org/10.1038/nbt.4279), accepted manuscript available through the Weizmann institutional collection. The manuscript's Figure 1 and treatment/capture methods were inspected, including complete rendered PDF pages 14 and 19. The original paper links all seven projects below. Project titles and protocols alone do not establish sample-level condition mappings.

| Project | Raw acquisitions in public inventory | Processed archive bytes | Source description / qualification scope |
|---|---:|---:|---|
| PXD009092 | 18 | 51,736,986 | Previously admitted UT/TNF+IFN MAPP; reused canonical metadata/tables |
| PXD009360 | 9 | 733,580,954 | Tryptic whole-lysate precursor candidate; prior audit found unresolved numeric sample labels (#757) |
| PXD007263 | 12 | 205,717,009 | Processed archive verified: source labels identify 0/1/2/6 time points and three numbered replicates |
| PXD009181 | 24 | 1,971,273,522 | Time-course part 2; deposited protocol is tryptic lysate analysis |
| PXD009438 | 14 | 199,785,170 | SLE nonspecific MAPP search; not a defined pharmacological intervention |
| PXD009112 | 14 | 1,202,687,972 | SLE tryptic lysate analysis |
| PXD009508 | 18 | 386,079,581 | Processed archive verified: alpha6/isotype/GAPDH capture labels crossed with untreated/epoxomicin, three numbered replicates |

The paper's live-cell treatments include epoxomicin (1 micromolar, 4 h), Velcade (50 nM, 4 h), and cycloheximide (100 micrograms/ml, 2 h). These are separate from inhibitors added to lysate during preservation/capture. A pharmacological treatment does not establish a measured residual activity fraction or a genetic knockout. The ZsGreen reporter is fused to a degron; its four displayed peptide sequences are lookup aids, not a recovered quantitative table or verified full construct sequence.

The inhibitor comparison motivates measuring both total detected signal and composition. A unit-sum peptide distribution cannot represent a global reduction in signal. Neither uncalibrated total intensity nor its change establishes total molar production or intracellular concentration; capture, loading, observation response, and background remain relevant.

## Retrieval outcomes and amendments

- PRIDE metadata are available for all seven projects. All inventories contain fewer than 100 entries, so no further result page is needed. Only two processed archives were selected for new download; no raw acquisitions were downloaded.
- EuropePMC's MAPP article XML request failed. Its supplementary-files endpoint returned HTTP 200 with a 296-byte XML error stating the article is not open access, not a ZIP. The original transport receipt is preserved; transport success must not be interpreted as asset qualification.
- A previously cached institutional manuscript was found after the downloader launched. `results/cached-manuscript.json` identifies the inspected bytes. The fresh institutional download also succeeded; its generated cover/date changes its byte hash. Both provenance records are retained; the second fetch was redundant.
- Initial full HTTP 200 archive transfers ended after roughly 146 seconds with only 16,510,976 and 22,343,728 bytes. Expected-size checks correctly rejected both. Failed partials and logs remain intact.
- An HTTP 206 probe returned the exact requested range. `reproduce/range-recovery.json` prospectively switches to bounded 4 MiB ranges with eight workers and at most three attempts. Each range must match its exact Content-Range/length; the assembled archive must match the entire PRIDE size and SHA1 before source inspection/admission. A separate EBI mirror prefix matched but did not replace the registered download host.
- All 143 ranges ultimately passed. Recovery took 1,290.53 seconds. Both complete archives match their public byte counts and SHA1 checksums. SHA256: controls `87d3711ad4a0e59d2618d079c2d6e95e14b8c544739701349d2b36bb4d5a14b8`; time course `05e46c2be2bc6046a4dee1f264d2c4ffd5fd3cba821ae778ece8c34906e8dde4`.

## Recovered panel coverage and quantitative replay

Each archive has 18 members. Five original tables per project were extracted: peptide, evidence, protein group, summary and parameters. All 30 deposited raw-file names match actual summary/evidence runs; no raw acquisition is inferred from an unrelated filename. `results/archive-inspection.json` retains the full inventories, table schemas, source metadata rows and literal run assignments.

| Source | Peptide rows | Evidence rows | Protein groups | Run columns | Positive agreeing quantities | Both-zero comparisons |
|---|---:|---:|---:|---:|---:|---:|
| PXD009508 controls | 6,060 | 19,581 | 1,058 | 18 | 15,484 | 93,596 |
| PXD007263 time course | 3,146 | 27,196 | 733 | 12 | 20,940 | 16,812 |

These are unfiltered source counts, not admitted training examples. Controls include 104 reverse and 29 contaminant peptide records; 710 other rows are not unique to one protein group. Time-course rows include 91 reverse records and 519 other non-unique-group rows. No protein-group ID is blank in these two archives. Observed lengths are 6–40 and 7–40 residues, respectively. Match-between-runs is enabled; evidence includes 7,136 and 12,131 `MULTI-MATCH` rows, respectively. There are 400/500 evidence rows with unreported intensity; their identities and detection type remain separate from numerical sums.

The first arithmetic audit stopped because the controls' peptide columns use labels such as `1511_7181-300a_alpha6-ut_1`, whereas evidence and summary use `1511_7181-300a_alpha6_ut1`. The failed script/config/log/partial output remain in `attempts/01-literal-experiment-label-mismatch/`. A prospective amendment allowed only a unique alias after removing underscores/hyphens from the complete identifier. Both original labels remain preserved; this derived relation is not a supplied specimen crosswalk.

Under those explicit candidate aliases, every reported peptide quantity agrees with the sum of its linked evidence or is zero alongside a zero sum. The fixed tolerance is relative 1e-4 / absolute 1, unchanged from the earlier MAPP source audit. Maximum positive relative discrepancies are 7.203e-5 and 7.217e-5. There are no reported-zero/positive-evidence discrepancies in these two panels, unlike the earlier bead-filtered PXD009092 export. A source zero still does not establish biological absence. Compressed per-comparison dumps preserve all source values, contributors and states.

An independent verifier re-extracted all ten tables using BSD tar, checked complete archive checksums/member bytes, and recomputed quantities with pandas grouping by evidence `Peptide ID` rather than the producer's peptide `Evidence IDs` lookup. It verified all 146,832 comparisons, source memberships and run crosswalks, with zero difference in summed intensities. Source inspection took 3.28 seconds; arithmetic audit 1.35 seconds; independent replay 7.63 seconds and peak process RSS 416,595,968 bytes.

The source flags and metadata constrain the next recipe. `parameters.txt` reports PSM FDR 0.01 and protein FDR 1; do not restate the latter as a 1% protein confidence guarantee. It references `Human_UP_2016_02.fasta`, whose original bytes are not present in either archive. The controls provide numbered replicate labels but no independent specimen/aliquot crosswalk. The time-course literal names contain `TNF`; the paper describes TNF-alpha plus IFN-gamma. Preserve both assertions and their scope instead of erasing that distinction. Capture-target aliases such as `alpha6` also need explicit source-backed interpretation in shared curation.

No Velcade- or ZsGreen-labeled run and none of the four displayed reporter peptide sequences were found in the two selected processed panels. This does **not** prove their absence from all deposited raw data, and a human-only search reference cannot establish a negative for a nonhuman reporter. Their exact PRM/construct/quantity assets remain unresolved under #759.

## Quantitative fallback inventory

While archive recovery ran, primary-source methods and public inventories identified a more direct kinetic candidate. [Hitlist #770](https://github.com/pirl-unc/hitlist/issues/770) now owns shared curation of [Soh et al., 2024](https://doi.org/10.1038/s41467-024-45339-3), [MSV000092631 / PXD044463](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000092631). The parsed public inventory has 505 entries, including a 963,763,954-byte aSPIRE output ZIP, a 28,339-byte identifier workbook, methods and 15 substrate FASTAs. The ZIP/schema and exact sample joins are not yet qualified; only metadata and methods were inspected in this family.

This study reports MS1 product time courses from 15 purified full-protein substrates. Its background subtraction, technical-replicate averaging, and final selection for a positive 4 h measurement (3 h for alpha-synuclein) must be replayed from original contributors. The selected output is not a complete negative set. Contiguous and spliced products need separate endpoint policies; neither purified proteasome origin nor this assay establishes APC context.

InvitroSPI/ProteasomeDB (2023; Figshare 17143865.v1) was also screened. Its documented 26-column schema contains identification scores/scan/sequence/condition metadata, with no quantitative abundance column; PSM support is not silently relabeled as concentration. The earlier 2015 quantitative proteasome-dynamics article includes fluorogenic-substrate kinetics and mechanistic models; this is a distinct assay, not interchangeable with a full peptide-product distribution. These are source-selection findings, not fitted model comparisons.

## Verification and pending work

Public metadata, receipts, full inventories and issue body/readback are retained in `results/`. The GitHub connector could read but returned 403 on issue creation; the configured CLI successfully created #770, and exact body (including final newline) and open state were read back. No author outreach occurred. Hosted CI on both launch commits now passes, including Hitlist's full hosted test workflow; the earlier normal local full-suite memory refusal remains a refusal, not a local pass.

The preferred next biological source is now the recovered controls panel under #759, with #770 as a complementary kinetic source. Implement the source/alias/metadata relations and a declared recipe in shared Hitlist, independently verify that export, then assess repeated observations and freeze the next exact-versus-sampled comparison. The [minimal matched kinetic design](../agents/codex/plans/controlled-fragment-kinetics.md) defines an additional bounded direction once time-course labels qualify. No source-only audit changes the failed biological MAPP benchmark or establishes intracellular concentration prediction.
