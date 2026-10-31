# MHC-II MS and context admission

Status: positive-evidence admission and independent replay passed; predictive
partitioning and training remain open.
Codex / GPT-6, 2026-10-26.

[Prospective plan](../agents/codex/plans/2026-10-26_mhc2-ms-admission.md).
Downloaded the canonical corrected human training table, native manifest,
qualification report and checksums. All three content files match the published
SHA256SUMS;188,213,823content bytes,4,071,463rows/161columns. Reproduction uses
`reproduce/launch.sh` followed by the existing Python environment executing
`code/verify_source.py`; receipts and the verification source snapshot are retained.
Raw bytes remain at `/private/tmp/presto-mhc2-20261026/ms-source/`.

The upstream qualification states that the Oct16 export is human source/MHC and
lengths8–25. It has no resolved donor/specimen/acquisition identities and claims
no real flanks or unique parent proteins. This source cannot alone satisfy the
full class-II length range, parent-context or independent-sample requirements.
The older fullv4 export retains longer peptides; extending compatible canonical
context admission beyond this published cohort remains necessary.

Two source-row inspections confirm usable distinctions: Stražar's exact allele
attribution retains a condition ID and HLA-DM transfection; MAPTAC's unresolved
DM arms retain empty conditions and `pmid_ambiguous` attribution. Neither should
be overwritten with guessed conditions. `matched_sample_count` is the number of
curated study samples, not the number of uniquely matched arms for the row:
Stražar shows42despite exact attribution. Current Hitlist `export.py` confirms
the PMID-level count; do not require that field to equal1for unique attribution.

## Admitted evidence and complete dispositions

The frozen source scan retains **98,220 positive observations**, **56,853 distinct
peptide sequences**, **20 displayed studies**, and **84 molecular candidate sets**.
Every retained observation has an explicit complete candidate set whose grooves
resolve consistently against the pinned human IMGT catalog. No best-binder
assignment, DRA default or missing partner is supplied. Molecularly equivalent
candidates retain all their reported names. Reported candidate provenance is
preserved; this is not an experimental claim that every candidate presents the
peptide. Source-specific prediction/assignment policies still need review when
defining a predictive cohort.

The adapter requires an explicitly unmodified canonical peptide, positive MS
source status, indexed source IDs and human MHC. It retains the entire native
row, source ordinal/hash and condition evidence. Observations remain
`split=predict`, `group_id=unpartitioned`: these are evidence records, not a
training or held-out partition. No negative observation or intensity is invented.

| First applicable disposition | Rows |
|---|---:|
| Admitted positive with complete candidates |98,220|
| Deferred: unresolved candidate scope |618,756|
| Deferred: unspecified peptidoform |293,282|
| Deferred: missing explicit MHC partner |209,466|
| Deferred: predicted restriction |78,744|
| Excluded: modified peptidoform |40,937|
| Excluded: unsupported peptide length/sequence |268|
| Total source class-II MS rows |1,339,673|

All retained candidates belong to one locus family per observation:
DP63,730;DQ28,508;DR5,982. These are observation counts, not independent samples.
The catalog is the unchanged61,437,179-byte local reference with SHA256
`ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1`.
Reference producer version is not established by the cache filename.

The cached parent-v4 contributor file matches the native manifest's declared
SHA256 `41ff93e08bc6ea4af692fee16044726af26c4e20ec650ec6c7cd851974b71db1`.
All196,304links for98,220retained provenance IDs are preserved and independently
matched against its9,381,101original links. Retained contributor study IDs are
known and none has an unresolved-overlap flag. That does not establish donor,
specimen, parent-protein or homology independence; original contributor study
sets, rather than displayed PMID alone, must govern subsequent partitioning.

## Real context support

The new adapter maps source-scoped, independently named interventions and
cytokines only for resolved annotated conditions. Literal `none` supplies an
absent state within that study's declared factor vocabulary; blank fields and a
control label alone do not. No pathway activation, residual enzyme activity or
numeric dose is inferred. Cell identity, intervention, cytokine and source
preparation/acquisition metadata remain separate.

One intervention has an actual present/absent contrast in the admitted source:
CIITA transduction in PMID33592498. Raw observed ligand counts are:

| Curated material | Explicit absent transduction | Present transduction |
|---|---:|---:|
| HROG02 |35|2,923|
| HROG17 |167|1,071|
| RA |6|618|
| Total |208|4,612|

These counts are **not** a measured treatment effect: missing peptides have no
negative/intensity label and acquisition/specimen independence is unresolved.
TAX1BP1 and cytokine/drug contrasts appear in the broader profile but do not pass
this complete-pair admission. No processing-pathway factor is activated here.

## Verification and upstream repair

The one-pass source profile independently replays every group and categorical
value. A separate checker imports neither the row adapter nor HumanMHCCatalog:
it reconstructs eligibility, every retained groove/candidate set, source row,
context claim, all1,339,673dispositions and all196,304contributor links. All pass.
The checker initially failed on empty Arrow selection indices, then on numeric
native PMID versus text profile keys. Both failed sources and diagnoses are
preserved under `reproduce/verification-attempt01/` and `...attempt02/`.
Typed indices and normalized comparison keys repair the reader; admitted bytes
and scientific criteria stay unchanged. The final replay passes.

Current Hitlist2.0.42 separately drops supplied `has_ptm` and
`peptide_modifications` columns in supplementary scanning. An artificial two-row
software fixture reproduces both omissions. [Hitlist #854](https://github.com/pirl-unc/hitlist/issues/854)
requests preservation and source-backed curation; its published body was read
back exactly. Unknown chemistry stays unknown pending source evidence.
#853 remains the separate MAPTAC DM-arm issue. Neither issue is a fitted result.

| Operation | Seconds | Peak RSS bytes |
|---|---:|---:|
| Literal profile |3.902|209,338,368|
| Admission and contributor joins |132.528|981,123,072|
| Final independent replay |132.118|668,254,208|

Local CPU, one numerical thread, existing Python3.12.6/PyArrow26, no install,
cloud or GPU. The final85focused adapter/context/training tests, repository lint
and strict documentation build pass. A preliminary extended-test command named
a nonexistent test file and ran no tests; that diagnostic is preserved alongside
the corrected passing log in `tasks/verification/`. No model,
pretraining, optimizer, loss, selected condition or synthetic training data is
used. Assay mapping is positive MS source evidence to `elution_observation=1`;
it supplies neither binding nM nor unconditional detection probability.
Validation/test predictions and metrics are intentionally absent because no
predictive split or fit occurred in this admission family.

## Reproduction and next step

Execution base is Presto `7560af3` with the newly added adapter and experiment
scripts recorded in `reproduce/admission-launch-attempt01.json`; their exact
executed sources are copied under `reproduce/source/`. Source files stayed fixed
during each scan. Use `reproduce/qualify.sh` after acquisition in a fresh output
directory; scripts refuse to overwrite their registered outputs. Launch receipts
record exact argv, package/runtime, path and nondefault environment information.

Large evidence artifacts stay under
`/private/tmp/presto-mhc2-20261026/ms-admitted/attempt01/`: observations JSONL,
every source disposition, retained contributor links and contributor summaries.
`results/admission-attempt01/summary.json` pins every artifact's size and hash.
Summary tables, profiles, issue receipts and independent verification live here.

Next: register and execute the actual binding/MS/context comparison, with
source-aware partitioning, explicit generated-contrast semantics if used, frozen
controls and source-policy checks. The CIITA contrast permits a real intervention
ablation but does not by itself identify distinct intracellular processing routes.
No predictive improvement is claimed; the full three-track goal remains active.
