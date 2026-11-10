# Paired HLA-DP/DM source recovery

Date: 2026-11-10. Agent/model: Codex / GPT-6. Presto base `04a9d2d9`;
working tree dirty only with this source-recovery family and task records.
Status: original source recovery and independent extraction verification complete;
shared Hitlist integration and prospective training qualification remain open.

This family recovers a real class-II DM-present/absent source following the
previous audit's confounded CIITA candidate. It supplies source data and exact
molecular identities, not a new model-performance result. No training, prediction,
checkpoint selection or designated held-out outcome access occurred.

## Source and molecular identity

PXD017154 / PMID33025005 is the engineered HeLa HLA-DP study by Meurer et al.
[Primary article](https://doi.org/10.1182/blood.2020008464),
[primary supplement](https://ars.els-cdn.com/content/image/1-s2.0-S000649712100313X-mmc1.pdf),
[PRIDE archive](https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD017154).

Supplemental Methods p2 explicitly supplies the alpha/beta constructs:

| Construct | Alpha | Beta |
|---|---|---|
| DP401 | DPA1*01:03 | DPB1*04:01 |
| DP402 | DPA1*01:03 | DPB1*04:02 |
| DP10 | DPA1*02:01 | DPB1*10:01 |

All three resolve through the existing Presto `HumanMHCCatalog` with no partner
imputation. The original catalog hash and groove hashes are preserved in
`results/common-mhc-resolution.json`; no alternative model/resolver was added.

The primary Methods also describes the shared CD74/CD80/DP-engineered HeLa
background and additional DM transduction. Preserve that scope when encoding
presence/absence; a DM-negative arm does not lack every historical transduction.
The original mzIdentML `Sample` names independently agree with all 32 acquisition
labels in the community SDRF. These identify 11 biological-source labels: ten DP
samples and one no-DP control, each with technical acquisitions. DP401_DM_A has
two deposited acquisitions; the other source labels have three. Do not manufacture
its absent third acquisition. DM/DO cells occur in the paper's cellular assays;
these 32 MS files do not establish a DO immunopeptidome arm. See
`source-review/constructs.json` and `results/source-samples.json`.

## Confidence correction and extraction

The complete paginated API inventory contains 193 files, including 32 original
MSF processed databases and 32 original mzIdentML files. The generated mzTab links
returned errors; original mzIdentML and native MSF files were retrieved instead.
Every original checksum was verified. The API's mzIdentML size is the uncompressed
XML size, while its SHA1 is for the compressed download; both are checked.

All 121,783 converted spectrum matches say `passThreshold=true`, including 2,017
records with decoy protein mappings. The conversion's literal peptide threshold
is `10`; it must not be interpreted as the paper's high-confidence filter.
In the pilot, 1,909 converted matches comprise 1,662 native high-confidence targets,
243 low-confidence targets and 4 decoys. Ten additional native high-confidence
records lack protein references and do not appear in the converted file.
`results/confidence-pilot-comparison.json` preserves this concrete discrepancy.

The original databases retain `Peptides.ConfidenceLevel`, validation parameters,
rank, protein references, and amino-acid/terminal modification tables. All32
report `TargetFPRHigh=0.01` and the target-decoy PSM validator. Level 3 is the
high-confidence category, consistent with the Methods and the
[PRIDE parser documentation](https://github.com/PRIDE-Archive/pride-converter-2/wiki/ProteomeDiscoverer).
No new confidence score or target-decoy recalculation is performed.

| Native source stage | Count / unit |
|---|---:|
| Target-table PSMs | 445,319 PSMs |
| Native high-confidence targets retained in raw extraction | 69,327 PSMs |
| Reported modified high-confidence records | 9,365 PSMs |
| Missing complete original protein references | 152 PSMs |
| Noncanonical or outside 10–512 residues | 2,133 PSMs |
| Not a unique visible rank-1 match | 1,235 PSMs |
| Conservative sequence/form/rank/protein candidates | 57,259 PSMs |
| Collapsed peptide-by-biological-sample records | 21,634 rows |
| No-DP control sequences | 17 peptides |
| Ordinary DP sample rows overlapping those controls | 70 rows |
| Remaining source-qualified DP candidate records | **21,547 rows** |
| Distinct sequences among those candidates | **10,143 peptides** |

Exclusion categories overlap and must not be summed. The candidate subset uses
native high confidence, a unique visible rank-1 match, canonical unmodified
10–512-aa sequence, and complete native protein references. Control detections
remain in a separate role; overlap is flagged and excluded from this conservative
candidate tier. All 69,327 original high-confidence records remain recoverable,
including modified, alternative-rank and incomplete-protein records. This is not
an assertion that excluded records are biologically false, or an exact recreation
of the paper's Progenesis quantity analysis. No abundance is inferred from Mascot
scores or PSM counts. The public 193-file archive has no original Progenesis
peptide quantity table.

| Arm | Biological samples | Candidate peptide–sample rows |
|---|---:|---:|
| DP10, DM absent | 2 | 6,604 |
| DP10, DM present | 2 | 3,415 |
| DP402, DM absent | 2 | 5,771 |
| DP402, DM present | 2 | 2,558 |
| DP401, DM present | 2 | 3,199 |

These counts are identifications, not measured treatment-effect sizes or
independent peptide counts. The source contributes two paired allotypes in one
engineered cell line; it cannot alone establish general processing transfer.

## Verification and runtime

- Independent lxml replay verifies all 121,783 converted PSMs against original XML:
  identities, forms, scores, protein/decoy mappings, sample names and grouping.
- Native extraction retains 69,327 high-confidence records. Independent replay of
  original SQLite query outputs and separate SQL histograms verifies all 69,327
  records and 21,634 peptide–sample groups, including controls and qualification.
- Original file and derived artifact checksums, failed retrievals, the first
  extraction stop on missing protein references, and corrected v2 are retained.
- mzIdentML extraction: 16.69s; independent verification: 11.85s. Native extraction:
  3.71s; independent verification:2.54s. These are local CPU source-processing times.
- Native transport used at most two Modal workers, each 1 CPU / 2 GiB, requested region
  `eu`, timeout 900s. No GPU. The 32 original MSF files total 6,397,444,096 bytes;
  summed per-file download time 836.31s, range 7.09–54.34s. This sum is not wall time
  or total billed runtime. Actual host CPU model/location was not separately logged.
- The first Modal deployment rejected unsupported region `eu-west-1` before any
  function ran; its receipt is retained. The corrected pilot and all 32 full calls
  completed. The slow local native download was explicitly stopped only after
  the same source was checksum-verified remotely; its partial is retained.

No predictive validation/test metrics or prediction dumps apply: this is an
unpartitioned source-recovery experiment. No losses, pretraining, synthetic labels,
training exposures or model comparisons apply. Every recovered record remains
explicitly not train-eligible pending shared-source integration and independence
qualification. Software/source verification is not biological qualification.

## Reproduction and artifacts

`reproduce/launch.sh` runs both extractors and independent verifiers on a fresh
output root. `reproduce/launch.json` records exact environment/invocations;
`reproduce/source/` preserves scripts, including failed versions. The pinned public
retrieval requests and complete inventory are committed. Raw data are outside Git. The fresh-output `launch.sh` replay passed both
independent verifiers and reproduced all 121,783 converted PSMs, 69,327 native
high-confidence PSMs and 21,634 peptide–sample records byte-for-byte after
decompression; see `results/reproduction-verification.json`.

Durable Modal volume `presto-checkpoints`:

- `2026-11-10_1454_codex_dp-dm-source-recovery/native-v1/`: all 32 original MSF files
  with verified source checksums and download receipts.
- `2026-11-10_1454_codex_dp-dm-source-recovery/source-recovery-v1.tar.gz`:
  58,883,809 bytes, SHA256
  `b8ecc4413dde2f76a271d81f6f8359580d11ea794bfd61f0ae13813888a1abcc`.
  Contains original mzIdentML, native query results, derived records and metadata;
  `results/archive-manifest.json` pins all 77 members. Remote bytes were read back
  and verified in `results/publication.json`. Article PDFs/text are excluded.

Local raw root: `/private/tmp/presto-dp-dm-source-20261110`. To restore, extract
archive members to a fresh root; use its `raw-mzid` subdirectory as the source for
`extract_mzid.py`, its `native-results` for `extract_native.py`, and
`metadata/pride-sdrf` for the sample cross-check. The convenience launch script
uses the original local layout with mzIdentML files directly under the source root.
For independent native retrieval, deploy the frozen `modal_native.py`, queue
`native-full-manifest.json`/`native-full-queries.json` with `launch_native.py`, and
collect the returned call IDs with `collect_native.py`. The reusable
`launch_native.py` was added after the original equivalent inline queuing; the
actual executed calls and timestamps are frozen in `native-full-calls.jsonl`.
Reuse verified native
cache bytes; never restart a job merely because a poll times out.

## Handoff

**Next step:** integrate these source records through Hitlist's existing
supplementary/provenance path, then apply Presto's source/sequence embargo and
freeze the comparison in [Presto155](https://github.com/pirl-unc/presto/issues/155).
[Hitlist895](https://github.com/pirl-unc/hitlist/issues/895) owns the source.
An isolated Hitlist checkout is prepared at
`/private/tmp/hitlist-dp-dm-source-20261110`, branch `codex/dp-dm-source-20261110`,
with `tasks/dp_dm_source_integration_spec.md`. No Hitlist implementation change
or release is claimed yet. The broader binding, ligand and processing objective
remains active; no existing checkpoint or model baseline is promoted here.

Published in [PR156](https://github.com/pirl-unc/presto/pull/156), source record
commit `e21d6dbf135400b5d8a598fca4016a8cbc217f53`. Hitlist895 and Presto155 were
updated with these counts, confidence findings and remaining acceptance steps;
both were read back successfully. Local source checks pass, and hosted CI for
the publication commit was still running at handoff. The parent scientific
PR131 remains draft. Earlier stability, assay calibration, coverage and sampling
observations remain in the committed post-fit handoff and issues #138–#142.
