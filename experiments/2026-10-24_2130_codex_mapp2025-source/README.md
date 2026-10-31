# Newer MAPP quantities recovered; source selection and identity require explicit curation

Codex / GPT-6, October24,2026. Registered at Presto
`35a2b97267bafe8d5da9d63aa17234267f60ab36` with unrelated task dirt and new
experiment-local code. [Prospective specification](reproduce/plan.md).
Local CPU only; no environment changes, cloud or training.

**A substantial baseline quantity source exists.** Publisher Table5 has79,025
positive values across six control MAPP runs. Every baseline intensity cell,
including zeros, agrees exactly with the deposited original results. However,
the publisher table is not a qualified human-only training cohort. Its selected
rows include nonhuman source annotations, and thousands of unflagged human-annotated
rows with baseline signal are omitted. Shared source ingestion and an explicit
eligibility policy are required before fitting. [Hitlist825](https://github.com/pirl-unc/hitlist/issues/825)
owns this work. No predictive improvement is claimed.

## Sources and fixed scope

[Goldberg2025](https://www.nature.com/articles/s41586-025-08615-w),
DOI10.1038/s41586-025-08615-w, public PXD052314. The audit targets baseline human
captured-peptide quantities, addressing the inability of KO/WT ratios to identify
baseline amounts. Activity scoring and pathogen-response modeling are outside
this experiment. No activity-based peptide selection or new biological target.
Other source arms are retained as provenance, never silently relabeled basal.

- Publisher `41586_2025_8615_MOESM5_ESM.zip`:48,110,489bytes,10members.
  Inspect Table3, Table5 and original captions; retain original member names/hashes.
- PRIDE project/files metadata:41files, including36RAW acquisitions, three processed
  archives, original deposition workbook and separate community SDRF. Original
  deposition workbook and SDRF are downloaded; no RAW spectra are downloaded.
- Original MAPP processed archive:356,132,312bytes,17members,
  SHA256`a7314be5a1fd268736bd40b6a96e92ff74303bde53822bdac7d6941f367f6fe4`.
  PRIDE supplies no checksum for this asset. Size, local SHA256 and selected-member
  ZIP CRCs are checked; do not describe the digest as a supplied upstream checksum.
  Proteomics694,747,970bytes and proteasome-composition204,242,482bytes remain
  inventoried but undownloaded. All actual transfers fit the registered resource caps.

Raw assets live at `data/mapp2025_source/2026-10-24_2130/`, with original bytes
and receipts. The expanded40MBsource-value summary and caption extraction stay
there; the compact published [summary](results/summary.json) points to their
hashes/paths. There is no Presto-only training curator or admitted recipe.

## Quantitative and material coverage

| Source | Actual coverage | Interpretation |
|---|---|---|
| Table5 MAPP |21,082unique peptide rows;12intensity columns;6baseline columns with79,025positive/47,467zero cells|Captured MS signal, not free intracellular concentration|
| Baseline MAPP support |17,282peptides with any positive control signal;8,025positive in all6;positive lengths7–40|Source support counts before organism/confidence/reference eligibility|
| Identification |48,885positive baseline cells By MS/MS;30,140 By matching|Matching provenance must survive; zeros are not negative labels|
| Whole-cell shotgun |5,387protein-group rows;6baseline LFQ columns with29,037positive cells|Independent assay candidate; exact physical pairing unresolved|
| Proteasome-associated proteins |2,068rows|Separate captured-protein assay, not independent precursor amount|
| Basal secretome |963rows;16intensity columns;2,087positive cells|Extracellular material, distinct from intracellular availability|

The original deposition table matches all36deposited raw-file names,12per assay.
Publisher ReadMe supplies control1-hour/control4-hour and biological-repeat1–3
labels. Parallel assay statements and matching repeat labels do not prove shared
specimen/aliquot identity. Retain link precision explicitly.

Community SDRF conflicts with original evidence: all36assays are called tryptic
shotgun; biological-repeat labels are1/4; nonexistent raw file5582 replaces
deposited5566. Preserve these assertions separately, with original names and
provenance. No alias or biological replicate is guessed.

## Original archive reconciliation and selection limits

Original MaxQuant `peptides.txt` contains30,364rows. All21,082publisher IDs and
sequences match. Ten quantity column names match exactly: **210,820cells agree
with zero numerical difference**, including **all126,492baseline cells**.
The two remaining publisher columns end5450/Infect4_2 and5454/Infect4_3, while
original names end5450/Infect4_3 and5454/Infect4_4. Preserve both; no speculative
alias is used and no twelve-column agreement is claimed.

The publisher omits9,282original rows:2,465have reverse/contaminant flags;
**6,817have neither flag**. Source protein-group FASTA-header assertions produce:

| Source disposition | Rows | Rows with positive baseline | Positive baseline cells |
|---|---:|---:|---:|
| Published, unflagged, human-only annotation |20,052|16,427|74,976|
| Omitted, unflagged, human-only annotation |6,633|4,862|19,058|
| Published, unflagged, nonhuman-only annotation |130|113|473|
| Published, unflagged, mixed/unresolved annotation |900|742|3,576|

Full dispositions, including omitted nonhuman/unresolved and flagged rows, are
in [source-dispositions.json](results/source-dispositions.json). These are
**source assertions**, not independently validated organism assignments or a
training admission policy. The omission rule is unrecovered. Neither trusting
the supplement wholesale nor automatically adding every unflagged row is justified.

Original parameters name `Human_SP_27042020_with_E-coli.fasta`, whereas project
metadata describes a human reference. The archive contains no FASTA. Two supplied
protein headers end in truncated taxonomy tokens OX=9/OX=960 despite asserting
Homo sapiens. The final audit retains those raw tokens and treats incomplete
assertions as unresolved. Recover complete sequence/reference provenance and
all possible origins before qualifying a human precursor/processing cohort.

## Pooled Table3: numbers exist, identities remain unresolved

Table3 has199,281rows and six columns. Its `Treatment` strings contain
**3,095,695numeric tokens:1,773,019positive and1,322,676zero**. Split-based and
independent regex decoding agree on every token. There are141distinct labels,
33ordered label patterns, and197,785rows repeat labels within a cell. These
patterns do not establish33independent experiments or unique acquisitions.
Preserve order and repeated slots; a dictionary would lose measurements.
Run/sample/cell/intervention/quantity-unit/aggregation and sequence-reference
crosswalks remain unresolved, including overlap with earlier MAPP sources.
Table3 also has202date-styled `Gene names` cells; no gene symbol is repaired.

An early commentary statement incorrectly called this table identification-only
after header inspection. Full value inspection corrected that statement before
source admission or fitting. The issue/report preserve the numeric evidence and
do not repeat the absence claim.

## Verification, execution and preserved failures

- Independent OOXML decoding matches **3,822,731populated cells** against the
  read-only workbook parser, including original serial values for date-styled
  cells. Cached source values are audited; no workbook is recalculated or edited.
- Workbook pass24.52s/0.165GiB; independent XML22.38s/0.682GiB;
  original archive reconciliation6.68s/0.389GiB. Initial metadata/supplement
  retrieval97.26s; processed archive539.59s. These are local audit timings,
  not training-throughput benchmarks.
- Sandbox DNS failure was preserved, then public retrieval succeeded with network
  permission. No access bypass or author outreach. Original failed parser/logs
  remain: datetime serialization, secondary SDRF filename equality, unmatched
  archive quantity headers, and CSV128KiBfield limit. Frozen recovery notes
  explain each change; no source quantity or biological gate was altered.
- The CSV parser ceiling became16MiB. Taxonomy parsing was qualified to exclude
  truncated numeric tokens from organism classification; initial and final
  summaries are retained. No model outcome motivated any recovery.
- Exact commands, versions, source snapshots, hashes and selected artifact
  inventory are under `reproduce/` and `results/`. Logs are reviewed and archived
  losslessly. Original large assets remain under `data/`.

## Training contract, decision and handoff

No training, pretraining, optimizer, synthetic biological labels, loss terms,
assay-to-model output, validation/test split or predictions. This is source
qualification before modeling, so regression/binder metrics do not apply.
No prior held-out outcomes are reopened and no baseline is promoted.

**Status:** source audit complete; shared ingestion and supervised cohort admission
remain pending in Hitlist825. Source availability advances; predictive quality does
not change. The full APC availability/concentration goal remains unachieved.

**Next step:** canonically admit original MAPP quantities and all source dispositions,
using publisher values as an independently reconciled view. Qualify human/reference
identity, selection/confidence and material links; define baseline composition and
precursor-adjusted endpoints separately. Then preregister a new grouped biological
comparison with sequence/length controls and matched exact/sampled objectives.

**Open questions:** complete search reference, publisher selection rule, exact
specimen/aliquot links, capture/detector calibration, Table3 identities, and transfer
from A549 to professional APCs. None can be supplied by treating observed MS signal
as intracellular molarity or by changing the optimizer alone.
