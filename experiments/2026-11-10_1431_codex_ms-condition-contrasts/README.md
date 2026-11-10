# Native MS condition-contrast admission audit

Date: 2026-11-10. Agent/model: Codex / GPT-6. Code base: `038ffbcc` (PR156).
Status: source admission and independent replay complete. No biological fit.

The shared condition interface is usable, but the current source data do not yet
supply a qualified class-II treatment/control benchmark. This audit locates the
specific gaps and preserves a source-linked observation inventory for the next
cohort. It does not count an adapter-accepted row as a scientifically qualified
ligand or a training example.

## Inputs and exact scope

The frozen Hitlist2.1.3 canonical MS export has 5,440,019 rows, 112,357,418 bytes,
SHA256 `4a739794ab7a21ceb633df6e4cde4c01054f3e53213a5a8dcdc4d78604dde58f`.
It is on `presto-checkpoints` at
`2026-11-07_1645_codex_unified-data-audit/admission/canonical-ms-export.parquet`.
The original admission catalog is pinned to SHA256
`ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1`.
No newer YAML annotations were broadcast onto this older export.

The first pass inventories literal source-arm metadata, yielding 1,347 signatures
and 109 condition IDs with admitted factor claims. The second applies the common
MS adapter to condition-bearing observations, retaining its source chemistry,
MHC-candidate, identity and attribution gates and its existing declared human
DRA*01:01 reference assumption. Whole candidate sets remain intact. Both existing
DM/DO-related and newly preserved factor families are considered.

This is an **unsplit, native positive-source inventory**, not a TRAIN-only cohort.
It reads native source observations; no designated TEST, confirmation, comparator
prediction or binding-outcome file was opened. Every output remains `predict` /
`unpartitioned` with `audit_only_not_train_eligible=true`. Contributor lineage,
source qualification, reuse/sequence embargo and partitions remain required.

## Results

| Admission stage | MHC-I rows | MHC-II rows |
|---|---:|---:|
| Source-qualified condition factors | 291,426 | 357,750 |
| Shared adapter accepted | 102,939 | 36,495 |
| Unknown peptidoform status | 117,655 | 305,607 |
| Unresolved candidate scope | 46,371 | 7,522 |
| Missing explicit MHC partner | 0 | 7,297 |
| Unresolved catalog name | 7,929 | 0 |
| Reported modified form | 16,489 | 829 |
| Unsupported peptide | 43 | 0 |

These are source evidence rows, not independent specimens. The 139,434 accepted
rows form 148 distinct arm/molecule/protocol/factor contexts. Separate peptide,
source-ID and observation counts are retained in `results/admitted-arms.json`.
Candidate molecule multiplicity does not inflate the observation count.

The class-II accepted rows come from four studies:

| PMID | Positive source rows | Condition context |
|---|---:|---|
| 33592498 | 35,609 | CIITA transduction and parental arms |
| 35561310 | 763 | Source-supplied intervention context |
| 31020640 | 92 | Source-supplied infection context |
| 35051231 | 31 | Source-supplied condition context |

There are three within-study factors with both explicit present and absent states
across both classes. Only one is class II: CIITA, with 35,071 transduced and 538
parental observations. **Do not fit this as a clean positive-pMHC contrast.**
Current Hitlist curation, following the authors' interpretation, explicitly
qualifies the parental class-II identifications as probable background. Existing
[Hitlist567](https://github.com/pirl-unc/hitlist/issues/567) fixed arm attribution;
it did not establish those peptides as HLA-II presentation. They are also not
measured affinity negatives. The review disposition is in
`source-review/findings.json`. Other molecular/capture/independence qualifications
remain necessary even for transduced arms.

The 305,607 class-II peptidoform deferrals comprise 290,461 Stražar2023 rows and
15,146 Abelin2019 cross-presentation rows. A separate vectorized source projection
verified this breakdown. [Hitlist896](https://github.com/pirl-unc/hitlist/issues/896)
tracks original-form recovery and preparation provenance. The generic importer
preservation fix in #854 is already complete; source curation is still missing.
The original Stražar Methods describe preparation chemistry and several search
modifications, so blanket unmodified imputation is inappropriate. Preparation-
induced derivatization must also remain distinct from native binding chemistry.
[Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10519123/).

## Next source and scientific decision

[Hitlist895](https://github.com/pirl-unc/hitlist/issues/895) tracks
PXD017154 / PMID33025005, a source explicitly comparing HLA-DP with and without
HLA-DM. PRIDE metadata and its community sample sheet were recovered and hashed.
The sheet contains 32 acquisitions / 11 source labels, including DM-present and
DM-not-coexpressed DPB1*04:02 and *10:01 groups, a DM-present *04:01 group, and a
no-DP control. These community annotations still need primary-source verification;
DPA identity is unresolved in the reviewed metadata. Processed mzTab/mzIdentML
files are present, so source qualification can proceed without a raw-spectra
reanalysis first. The API inventory examined only page zero; it is not a full
file count. [Primary dataset](https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD017154).

Preserve the current model and experiments. Next actions are original-form
recovery for the large supplemental cohort and exact molecule/run/condition
qualification of the paired DM source, followed by the prospectively frozen
condition-aware comparison in [Presto155](https://github.com/pirl-unc/presto/issues/155).
This audit supplies no evidence of improved binding, MS AUROC or processing
prediction. It keeps the broader three-part objective open.

## Reproduction and verification

`reproduce/launch.json` freezes source hashes, the exact invocation and environment.
`reproduce/launch.sh` refuses overwrite; use fresh `PRESTO_CONDITION_OUTPUT` and
`PRESTO_CONDITION_OBSERVATIONS` paths for replay. The package import root points
to this worktree through `/private/tmp/presto-conditions-import/presto`; an
alternative checkout must supply its own `PRESTO_IMPORT_ROOT` with the same code.
Inputs are verified before reading. The source snapshot is under
`reproduce/source/`; `code/verify.py` independently replays the admitted rows.

Local CPU, one Arrow thread, Python3.12.6 / PyArrow26.0.0. The audit took 54.58s.
Independent verification took 14.43s and matched all **139,434 original rows × 56
projected fields**, explicit condition claims and candidate-contrast counts.
All source observations remain unpartitioned. Results and verification receipts
are committed. Full raw observations (12,528,670 compressed bytes; SHA256
`6bc2ff0dad1497cbbb12e6402a92a49f2b190c60443bc379c2904d241f44ead9`)
are outside Git at the path in `results/summary.json` and durably published on
`presto-checkpoints` at
`2026-11-10_1431_codex_ms-condition-contrasts/audit-v1/condition-observations.jsonl.gz`.
`results/publication.json` verifies the remote bytes and records the uncompressed
content hash for replay despite gzip timestamp changes. They are reproducible
from the pinned export. Public retrieval receipts preserve successful and failed requests.
Full copyrighted article text remains in the local retrieval cache; only factual
summaries and its receipt are committed.

No GPU was requested/used. No training/pretraining, loss terms, synthetic labels,
model selection, predictive validation/test metrics or prediction dumps apply.
Artificial semantic regression tests are in the preceding interface family;
PR156's code head `038ffbcc` passed full GitHub lint/test/package and strict docs
checks in runs42885525626 and42885514023. The scientific data audit is verified
by original-row replay, not inferred from those software tests.
