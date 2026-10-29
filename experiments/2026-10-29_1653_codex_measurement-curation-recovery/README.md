# Shared measurement curation and source recovery

Status: common curation implemented and verified; source recovery partially complete.
See [plan](../agents/codex/plans/2026-10-29_measurement-curation-recovery.md).

Codex/GPT-6; CPU curation validation, no model training or held-out prediction
evaluation in this family. Source decisions, acquisition hashes and qualified
counts are preserved in `results/` and `reproduce/`. Existing temporal runs remain
immutable. Follow-on fitting is registered separately under
[curated-data-retrain](../2026-10-29_1956_codex_curated-data-retrain/README.md).

## Recovery admission and the short-peptide exclusion

The source-reviewed Stražar recovery contains 14,187 unmodified peptide–MHC
observations. The common MS adapter admits 13,898 before partitioning; a further
348 observations are withheld because their peptide occurs in an existing held-out
study. This leaves 13,550 candidate TRAIN observations. These are prepared data,
not observations already used to retrain a model. See the
[admission receipt](results/recovery-admission.json).

The earlier phrase “289 unsupported peptides” means **289 MHC-II observations
shorter than the model's required nine-residue core**, representing **184 distinct
peptide sequences**. It does not mean failed experimental identifications, invalid
amino-acid characters, or unknown modification status. All 289 already passed the
source chemistry review. Their source records remain available; only their
admission into the current model is rejected.

| Peptide length | Excluded observations |
|---|---:|
| 7 residues | 132 |
| 8 residues | 157 |
| Total | 289 |

The adapter's generic `unsupported_peptide` reason also covers other problems in
other datasets, but the [length audit](results/recovery-length-exclusions.json)
establishes that every exclusion in this recovery is caused by the minimum length.
The present class-II representation enumerates complete nine-residue registers;
these sequences have none. Supporting them requires an explicitly defined and
validated representation of incomplete cores, rather than just removing the
admission check. This curation family has not changed the architecture.

## Source-scoped inferred limits

The [threshold table](results/thresholds.md) lists all 71 local panels and their
source references, assay tags, quantities, thresholds and affected observation
counts. The frozen TRAIN pass inferred 506 bounds (482 ceilings and 24 floors)
and withheld 618 unresolved observations. These are statistical inferences, not
verified assay limits. No shared threshold is approved or transferred across
sources; cross-source evidence is summarized separately under stricter criteria.

The physical TRAIN cohort changes from 234,337 to 233,719 observations. The shared
`data/measurement_curation.py` parser/screen is used by
`data/modular_hitlist.py::curated_purified_assays`. Explicit structured/text
qualifiers, conflicts, missing qualifiers, units and source-fingerprinted decisions
remain distinguishable. Missing original inequality is not evidence of exactness.
The screen examines extrema within reference × class × MHC × reported method ×
literal quantity. Repeated values alone are insufficient; inferred rules need
same-limit explicit-bound or stronger directionally consistent panel evidence.
The full policy thresholds and all decisions are frozen in the curation receipt.

Shared-limit review requires at least 3 independent study components, explicit
same-limit bounds in at least 2, at least 100 affected observations, 95% directional
agreement and verified protocol-family equivalence. This summary never creates
cross-source row decisions automatically. Nearby numerical thresholds are not
collapsed. The original source method remains an unverified tag where appropriate.

## Source fixes and acquired data

[Hitlist#871](https://github.com/pirl-unc/hitlist/issues/871) records the source-
scoped competitive-fluorescence/KD versus cited refolding-ELISA discrepancy.
Presto preserves the reported method and quantity separately and flags the cited
protocol conflict; literal KD is not described as proof of a direct assay.
[Hitlist PR872](https://github.com/pirl-unc/hitlist/pull/872) repairs supplementary
peptidoform transport and chemistry-aware deduplication. The actual recovery run
pins the tested 2.1.1 development wheel and its hash, independently of later release
version changes. No unknown chemistry was filled with False globally.

The [source ledger](results/source-ledger.md) distinguishes downloaded archives
from admissible independent additions. Six MHCflurry archives, NetMHCpan 4.2 and
NetMHCIIpan 4.3 training archives, NetMHCstabpan stability, original Abelin supplement,
HLA LigandAtlas 2020.12 and Human HLA PeptideAtlas 2026.09 were acquired and hashed.
Their duplicate/source/assay/modification/holdout review remains explicit.
The initial batch retrieved 11 of 132 selected Stražar PSM files before persistent
public rate limits; anonymous FTP timed out. A later bounded retry retrieved nine
more and stopped at the first new HTTP 429. There are now 20 verified files and
112 remaining. Source recovery is incomplete.

## Verification and reproducibility

Common source/curation checks: 146 passed. Independent collected-artifact replay
verifies effective qualifiers, unchanged numerical values/raw records, decision
fingerprints, removed rows and exact counts. A separate follow-on audit verifies
all 13,550 candidate recovered observations against the original CSV and every
linked native-terminal/modification declaration. The candidate's compatibility
with the parent negative blacklist is checked in the retraining family; source
admission alone does not certify that combination.

Base source commit `858f5383094ad1a8e840ddb1b4df8233b189b2df` was dirty; exact statuses
and executed source hashes are in each `reproduce/*-launch.json`. Curation ran
on Modal CPU 4/16 GiB in 141.75 s; recovery admission CPU 4/16 GiB in 49.97 s. No GPU,
pretraining, optimization, losses or synthetic labels were used in this family.
Validation/test prediction dumps and predictive metrics do not apply to this
source-validation work. No model winner is selected.

Raw artifacts reside at
`/Users/iskander/code/presto/artifacts/2026-10-29_1653_codex_measurement-curation-recovery/`
and the same family prefix on Modal volume `presto-checkpoints`. The collected
`curated.tar.gz` and `recovered-admission.tar.gz` hashes and constituent hashes are
in `results/curation-receipt.json` and `results/recovery-admission.json`.
Exact Modal invocation/source snapshots are in `reproduce/curation-*` and
`reproduce/recovery-admission-*`; acquisition manifests and invocation receipts
are in `reproduce/acquisition*`. Local table/recovery scripts remain under `code/`.

Follow-on combination audit found 35 generated TRAIN comparisons matching reviewed
positive sequences and three suspect generated TEST labels. One further recovered
observation receives a held-out embargo: the retrain uses 13,549 positives. This
is a downstream cohort amendment, preserving the original source-admission receipt.

## Later immutable source batch

`results/recovery-late.json` records 33,683 source-verified unmodified pMHC
observations: 19,496 additional identities beyond the initial 14,187. This source
count precedes model admission and is not an addition to the running retrain. The
source batch preserves all original positives, with 77,523 unmodified support rows
across 20 files. Model length/restriction admission, whole-study/peptide embargo
and old-background compatibility still apply before future training.

The pure recovery function was refactored to accept explicit inputs/output root;
Modal replay reproduced the original CSV byte-for-byte and all original counts.
The new run used CPU 4/16 GiB, no GPU, 10.94 seconds. Collected archive and both
constituent hashes were independently checked, along with every CSV identity,
evidence fingerprint, source-file fingerprint and native-terminal/modification
declaration. `code/verify_late_recovery.py` and its saved source/receipt reproduce
the independent check. Raw files remain at the family artifact path under
`recovery-batch-late/`; the archive SHA is
`f74aad46b9d132d00cfd56ed34c4eb5585ebc11fae8891d323ba0d5d3282c174`.
The original recovery and active training inputs remain unchanged.

The same common adapter, using the verified published Hitlist 2.1.3 wheel,
admits 33,235 rows, excludes 448 shorter than its MHC-II minimum, and embargoes
607 peptide overlaps with any fixed validation/test source (including generated
comparisons). **32,628 candidate TRAIN observations** remain: 32,534 pMHCs not
seen in the parent's admitted positives and 94 also observed in other studies.
This stage took 61.60 s on CPU 4/16 GiB. The candidate archive SHA is
`6d855faeefc8ad45577c2f7eefedd730e639156f3810edae4072daccaf298a2b`;
`results/recovery-admission-late-v2.json` pins all constituents, heldout sources,
catalog and actual dependency versions. Background-conflict review is separate.
The first late admission failed while querying a noninstalled package's version;
its partial output and executed source are preserved in
`attempts/late-admission-environment/` and the `late` reproduction bundle.
The corrected `late-v2` recipe records the pinned mhcseqs catalog as an artifact
provider, separately from installed reader dependencies. No failed output is used.

Independent collected-candidate and frozen-background verification passed:
`results/late-candidate-audit.json`. All 13,549 positives already used by the
active fit remain present; there are exactly **19,079 additional candidate pMHCs**.
No candidate peptide intersects a held-out source. Source rejection by length is
186 seven-residue and 262 eight-residue observations, reflecting the current
architecture rather than invalid experimental measurements.

| Later-batch check | Count | Disposition |
|---|---:|---|
| Source-verified unmodified pMHCs | 33,683 | Retain source evidence |
| Common-adapter eligible before embargo | 33,235 | Current model geometry |
| Held-out peptide exclusions | 607 | Keep outside TRAIN |
| Candidate TRAIN observations | 32,628 | Prepare a subsequent frozen cohort |
| Candidate additions beyond active-fit recovery | 19,079 | Not inserted into active runs |
| Current generated TRAIN rows matching later source positives | 33 | Remove in next cohort; 28 distinct sequences |
| Fixed generated TEST rows matching later source positives | 4 | Suspect labels; retain fixed comparison |

The last count includes the three originally declared suspect TEST rows plus one
new row. Original prespecified three-row sensitivity remains unchanged; this later
source audit is an additional limitation, not permission to change the denominator
or to select a model on test results. Source presentation does not establish binding
to every comparison MHC. All reviewed sequences, including short excluded forms,
were checked: 474 are absent from the original blacklist. The audit used CPU
4/8 GiB, no GPU, 206.52 seconds, and verified every scanned frozen source hash.
Candidate archive/constituent checks, source identities, peptide chemistry, study
provenance and immutable current manifest were independently validated.

## Upstream release closure

[Hitlist PR 872](https://github.com/pirl-unc/hitlist/pull/872) merged at
`ff5247f787ce79fbb640a00bcbe702fb7b7910cb`; version **2.1.3 is published on PyPI**.
Clean-main workflow 40948383762 passed the full release gate. Both public wheel
and source-archive bytes match the tested artifacts, as recorded in
`results/hitlist-pypi-2.1.3-verified.json`. The immediate post-upload inventory check
failed; subsequent inventory matched, and strict reverification succeeded without
additional uploads. Both attempts are preserved. The earlier recovery retains its
original development-wheel pin; publication does not mutate existing inputs.
The immediate-inventory behavior is already tracked by
[Hitlist issue 867](https://github.com/pirl-unc/hitlist/issues/867).

Presto's corrected curation commit `865eac08` passed hosted run 40951565780:
2,879 tests / three skips, 543.86 s for the test phase, plus full lint and docs.
An initial formatting-only failure was fixed with verified identical Python AST.
