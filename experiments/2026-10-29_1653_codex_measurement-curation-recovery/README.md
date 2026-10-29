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
Only 11 of 132 selected Stražar PSM files were obtainable before persistent public
HTTP rate limits; anonymous FTP also timed out. Source recovery is incomplete.

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
