# MHC-II source and context audit

Status: closed **literal source inventory**; predictive improvement remains open.
Codex / GPT-6, October 26, 2026. Presto `ddfd7a1` plus the registered audit files;
exact dirty state and input/executed-source hashes are in
`reproduce/execution-inputs.json`. [Prospective plan](../agents/codex/plans/2026-10-26_mhc2-source-context-audit.md).

The public export contains substantial class-II binding and MS evidence, but
counts are not training eligibility. A confirmed MAPTAC import limitation loses
DM-condition membership for restrictions covering 11,184 merged rows. Filed and
read back [Hitlist #853](https://github.com/pirl-unc/hitlist/issues/853).

## Sources and curation contract

Read the unchanged Oct16 public `ci-corpus-v4` binding, observation and other-assay
Parquets at the explicit absolute paths in `reproduce/execution-inputs.json`.
These three files contain 5,327,796 rows, use 164,041,987 bytes, and are separate
export partitions. Filter literally on `mhc_class == "II"`, without a species,
peptide-length, PTM, outcome, assay, positivity or provenance admission filter.
No deduplication across studies or exports is asserted. Distinct peptides and
known PMIDs are counted within each partition, not treated as independent samples.

Current Hitlist source is separately pinned at commit
`13b91424295e01286db46e8e9818fcc2fe816834`, version2.0.42. The 49,774,247-byte
archive and extraction live under `/private/tmp/presto-mhc2-20261026/`; its hash
and acquisition receipt are in `reproduce/source-archive.json`. It did not rebuild
the older data export. The current release inventory is retained in
`reproduce/releases.json`; v4 is the latest general corpus in that response.
No shared environment/package or dirty user checkout was changed.

## Literal coverage

| Partition | Class-II rows | Distinct peptides | Known PMIDs |
|---|---:|---:|---:|
| Binding |590,479|109,636|863|
| MS observations |1,444,135|576,894|178|
| Other assays |832|689|156|

Binding labels include468,203 `MHC binding` rows without units,46,551 explicit
IC50/nM,30,186 qualitative binding,23,928 KD(~IC50)/nM,18,650 KD(~EC50)/nM,
2,061 directKD/nM,423half-life/min,422EC50/nM,27dissociation-temperature,
20off-rate and8on-rate rows. These literal labels do not establish exact positive
affinity measurements. Qualifiers, numeric states and literal methods remain
separate in `results/inventory-corrected.json`.

All1,444,135 class-II MS rows have the observation flag. Candidate allele sets
are nonempty in1,296,591;401,686 have the monoallelic flag;788,373 have a cell-line
field and936,252 a cell-type field. These flags do not independently prove complete
alpha/beta pairing or correct source attribution. The allele-set provenance is
727,616exact,428,889sample-allele-match,57,232sample-locus-match,
76,855PMID-locus-pool,5,999PMID-class-pool and147,544unmatched.

The export's Stražar2023 subset has290,461class-II rows/170,180distinct peptides;
Abelin2019 has48,404/41,410. These are export counts, not the original supplementary
CSV counts. Stražar's current source CSV is described as308,418rows in curation.
Do not subtract them and invent one filtering cause without a lineage audit.

Nonempty `peptide_extended` is counted as a literal field only. It does not prove
real parent flanks, resolved origin or full-parent support; those need the mapping
and contributor relations. Similarly, study comments do not assign per-row routes
or conditions. This inventory has not completed that downstream qualification.

## Confirmed condition gap and existing owners

The current MAPTAC class-II CSV has43,503rows and only three columns: peptide,
class and restriction. Current curation explicitly says DM-positive/negative
peptides are merged while the arm definitions remain separate. Two independent
CSV readers agree on11,184rows for the four restrictions with both arms:
DRB1*12:01(2,276),DRB3*01:01(2,357),DPB1*06:01(4,587),DQB1*06:04(1,964).
These are merged import counts, not original observations per condition.

Issue853 requests original source recovery, lossless observation-to-condition
membership, retained ambiguity/shared observations, full pair context and correct
semantics for additional DM expression. No added DM is not measured endogenous DM
loss, and absent identification is not a measured biological negative. Its body,
readback and the source review are retained under `results/`.

Existing owners remain distinct: #593 covers engineering/material grouping;
#817 is open for Nanaware2024 admission. #366(arm-discriminator umbrella),
#151(beta-chain/pair matching) and #736(general context normalization) are closed.
The missing DM discriminator is still explicit in current source, so issue853
does not duplicate a completed representation-only fix. Github discovery receipts
preserve the states observed during this audit.

Seven of eight prospective primary-literature operations were used; requests and
failures are retained in `reproduce/literature-ledger.json`. The Abelin abstract
supports DM-sensitive ligand selection. Original workbook access returned403;
Stražar full-text routes returned a challenge/tool error/HTTP500. No original
workbook-cell counts, recovered scientific tables, new route labels, or complete
Methods interpretation are claimed. No author/private-data outreach occurred.

## Verification, runtime and reproduction

Execute the three commands in `reproduce/launch.json`: inventory, source review,
then coverage verification/correction. Scripts and executed snapshots are retained;
existing outputs are refused. Raw inputs remain at the hash-pinned receipt paths.
Local CPU, one Arrow/BLAS thread; no training, pretraining, GPU, synthetic data,
loss weights or assay-to-prediction mapping. Validation/test splits and per-example
prediction dumps are inapplicable to this metadata inventory.

- Independent Arrow aggregation verifies all5,327,796rows/177joint class-assay
  groups against the Python row census, with a different batch size.
- A subsequent Arrow-kernel pass verifies29nonzero coverage counts and all
  distinct peptide/PMID counts, and reconstructs four categorical distributions.
- Initial scalar-category serialization truncated labels to their first character.
  The original artifact/script remain unchanged and those category tables are
  invalid. **Use `inventory-corrected.json`**; only category tables changed. The
  original joint tuples and coverage counts passed independent replay.
- MAPTAC CSV and Arrow counts agree; issue853 readback matches its local body.
- Inventory88.42s/504,856,576peakRSSbytes; coverage verification3.85s/
  445,202,432bytes. Source-review runtime is in its result. All within300s/3GiB.
  These are descriptive audit timings, not a hardware comparison.
- Explicit Ruff checks and formatting pass for the three executed scripts.
  Initial line-length findings were fixed before each script ran.

Preferred next step: qualify source-backed class-II pair/parent/context joins and
the actual retained IC50 cohort, then freeze binding/MS comparisons. Stražar provides
a sizable single-context ligand source, not a DM intervention contrast. MAPTAC
condition learning requires the upstream source repair. No predictor improved in
this audit; all three full-goal scientific gates remain open.
