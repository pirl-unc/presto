# Full eligible purified IC50 source admission

Registered before execution, 2026-10-21 UTC; Codex / GPT-6. **Closed: source
admission, mechanical readiness and independent source/partition replay pass.**
Follow the [prospective contract](../agents/codex/plans/full-ic50-admission.md).

This is data-only admission for matched assay-only comparisons. Independent
same-input measurements remain; original-outcome copies are deduplicated and
contradictory exports deferred. Exact-only and exact-plus-censored training use
the same complete held-out observations. Source hashes, catalog aliases and
historical partition exposure remain explicit. No pretraining, optimizer,
predictive metrics, synthetic labels or model qualification is implied.

## Frozen source and execution

Clean executed commit `2214086783b83b741a6c9f0d0dda7137cd718c09`;
[actual invocation/environment](reproduce/executed-launch.json),
[rerun helper](reproduce/launch.sh) and [executed source](reproduce/source/).
Run the helper from this family with fresh output and cohort paths, under the
recorded Python 3.12.6 environment and import shim pointing at the frozen checkout.
The exact source paths and full SHA256s are in the receipt:

- Hitlist v4 binding parquet: `2b3f6ea2dd647e7fc54d214eef420feaebc4e81fab97ffc0565a2de1f8695ba0`.
- Existing MHC catalog: `ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1`.
- Historical membership: `8959e60c0d9b8049aec25095b9c354d456b48f97ba8fb31d7b786e2861113e99`.
- Prior 32-row memorization cohort: `145957e201c596ba82db4c900eb3715c34c81605ffadf887d0811f6f1d282ad5`.

The catalog's producer version is unknown; installed runtime versions do not
establish it. No new download, latest-catalog claim or generated labels were used.
Execution was local CPU, one Torch/BLAS thread, no GPU/Modal requested or observed.
Preparation took **42.6482 seconds**, peak process RSS **1,202,814,976 bytes**.
A separate four-worker repository regression ran concurrently; this is an
admission timing, not an isolated performance benchmark. Independent replay is
excluded from that timing.

## Actual population and dispositions

The source contract accepts direct IEDB/CEDAR IC50, explicitly purified MHC,
canonical unmodified peptides and explicit human HLA molecular assignments.
Other endpoints/species/cellular measurements are outside this comparison.
Supported class-I lengths are 1–512, class-II lengths 9–512, with one whole-peptide
class-I input and explicit complete class-II registers. Actual retained rows are
all IEDB and nM; other supported concentration units were not present among
admitted records. Raw values and censor bounds are unchanged, without clipping.

| Final raw-row disposition | Rows |
| --- | ---: |
| Admitted | 29,283 |
| Deferred: missing explicit MHC partner | 33,165 |
| Deferred: unresolved catalog name | 30 |
| Excluded | 751,790 |
| Total | 814,268 |

Exclusions are 740,001 outside direct IEDB/CEDAR IC50; 7,349 without an explicit
purified-MHC method; 1,259 modified/unspecified peptidoforms; 14 non-peptide
ligands; 176 unsupported peptide/class inputs; 157 nonpositive/nonfinite
measurements; and 2,834 without an explicit human HLA restriction.
Every original row has a [final disposition](results/source-dispositions.jsonl.gz).
Missing class-II partners are deferred, never filled with an assumed DRA chain.

All 29,283 eligible original assay IRIs are unique in this actual source, so no
original-outcome copies/conflicting exports needed removal. Software checks cover
both cases for other exports. **1,002 sets of identical model inputs have
different independent reported outcomes and remain in the cohort.** Differences
include values or qualifiers; they do not necessarily imply mutually inconsistent
censor intervals. They are not averaged or filtered using the capacity
diagnostic's consistency rule.

All 84 used catalog names resolve unambiguously; their actual source aliases,
sequence membership and current strict groove extraction agree. Catalog
annotations supply molecular species/class only. No class/species prediction,
annotation loss, APC default or inferred cellular supervision is introduced.

## Paired training conditions and support

| Split / class | All measurements | Exact measurements | Exact peptide groups | Exact binders / nonbinders at 500 nM |
| --- | ---: | ---: | ---: | ---: |
| Train I | 15,266 | 14,375 | 6,532 | 7,815 / 6,560 |
| Train II | 8,289 | 7,261 | 1,117 | 1,789 / 5,472 |
| Validation I | 2,051 | 1,968 | 863 | 1,045 / 923 |
| Validation II | 874 | 758 | 141 | 212 / 546 |
| Test I | 1,845 | 1,758 | 777 | 991 / 767 |
| Test II | 958 | 842 | 133 | 204 / 638 |

The **exact-only** condition has 21,636 training measurements. The
**exact-plus-censored** condition adds 1,919 training bounds, for 23,555 total.
Both have the identical 2,925 validation and 2,803 test observations, including
their censored measurements. Actual total exact/censored counts are 26,962/2,321.
There are 9,908 distinct peptide groups across the corpus. The mechanical gate
(50 exact measurements/20 groups per class/split, ten exact binders/nonbinders
per class/held-out split, and multi-register II training support) passes unchanged.

Raw measurements above 50,000 nM remain: 1,577 train, 214 validation and 188 test.
These counts include exact and censored measurements. Censored rows retain their
original direction; ambiguous bounds are not forced into a binary binding label.
The [admission receipt](results/admission.json) preserves full qualifier/source/
class/binding-state and allele counts, target ranges and component sizes.

Components connect exact peptides and original assays. Existing historical
assignments and all 32 memorization peptides' training exposure are preserved.
Only entirely new components use the frozen `presto-direct-ic50-v1` hash80/10/10
policy; no target values or reseeding enter assignment. No incompatible historical
component was admitted or encountered in the eligible set. Historical assignment
rows are 2,117 train, 230 validation and 219 test; new-component rows are
21,438/2,695/2,584 respectively. Preserving old held-outs does not make them fresh:
their prior inspection remains a limitation.

Train/validation, train/test and validation/test share **114, 117 and 90 studies**.
Donor/specimen/acquisition identity is absent in this binding export; this is a
peptide-disjoint diagnostic cohort, not a study/donor/homology-independent or
unseen-allele qualification set. Both classes and threshold support must still
receive separately reported predictive evaluation.

## Independent verification and artifacts

[Independent replay](analysis/verify.py) imports no Presto preparation functions.
It verifies all source/manifest/normalized-file hashes, all 814,268 dispositions
against raw values and exclusion/deferral predicates, all 29,283 retained raw
records/values/units/bounds, all used catalog aliases and resolved grooves,
original-assay uniqueness/identities, independent-input disagreement counts and
all historical/hash assignments. It proves that the two manifests differ only
by censored training exposure. [Replay passed](results/independent-replay.json).
The first analysis setup failed from an unscoped lambda replacement before
replaying binding rows; its source/log are preserved under
[verification-attempt0](reproduce/verification-attempt0/). The corrected analyzer
passes explicit Ruff and full replay. No preparation or fitting was rerun.

Shared cohort: `data/modular_cohorts/2026-10-21-full-ic50/`. Its generated JSONL
files (about 137 MiB) remain local and hash-pinned, not duplicated in this family.
Tracked manifests:

- Exact-only: `914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be`.
- Exact-plus-censored: `7fd143e22082dc18c1b7332954f66608bb7b0840813e5e12d74b3c85806dfc7c`.

Raw preparation artifacts (about 407 MiB), including SQLite, full per-row
preparation dispositions and independent-measurement disagreement records, are
under `artifacts/2026-10-21_0647_codex_full-ic50-admission/`. The receipt links every
canonical file and hash. This family retains source dispositions, all component
memberships, used catalog aliases, both preflight reports, actual source/launch
snapshots and [admission](launch_logs/admission.log)/[replay](launch_logs/independent-replay.log)
logs.

No model condition has been fitted or selected. There are no executed loss
weights, optimizer/pretraining parameters, synthetic labels, predictions,
validation/test loss or IC50 discrimination/regression metrics in this data-only
family. The manifest allocates only direct IC50 with supplied class/species and
class-II registers; a later registered fit must declare exact budgets, baselines,
selection, permutation controls and scientific quality gates. This is readiness
for that comparison, not a qualified checkpoint or completion of the wider model.
