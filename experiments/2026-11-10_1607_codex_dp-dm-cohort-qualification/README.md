# DP/DM cohort qualification and condition-interaction capacity

2026-11-10. Codex / GPT-6. **Identity audit and structural probe complete;
source release, model change and scientific fitting remain open.**
Plan: [frozen specification](../agents/codex/plans/2026-11-10_dp-dm-cohort-qualification.md).

## Data that remain usable for the next experiment

The preceding source integration had 21,547 PXD017154 / PMID33025005
peptide–sample candidates and 13,209 rows clear only of presentation/Atlas
exact/core8 identity checks. The full known embargo now leaves **10,034 rows /
4,871 distinct peptides** and excludes 11,513 rows. This removes another
3,175 rows from that partial-clear set. No row is yet marked TRAIN-eligible: the
source release and prospective comparison/background/split contract remain open.

| Exact DP pair | Clear source rows | DM-absent peptides | DM-present peptides | Detected in both conditions |
|---|---:|---:|---:|---:|
| DPA1*01:03 / DPB1*04:01 | 1,275 | 0 | 792 | 0 |
| DPA1*01:03 / DPB1*04:02 | 3,790 | 1,573 | 701 | 358 |
| DPA1*02:01 / DPB1*10:01 | 4,969 | 1,995 | 924 | 465 |

DM-absent rows total 6,058 and DM-present rows 3,976. Both paired allotypes
retain A/B source labels in both conditions. DP04:01 has no absent-DM arm and
cannot supply a paired contrast. Within-condition detection is unioned over
source labels for the peptide columns above; these are not independent donors
or confirmed physical specimens. Counts do not establish a causal DM effect or
quantitative abundance. Nondetection is not measured nonbinding.

## Complete known identity contract

Seventeen namespaces are checked separately, preserving existing TRAIN overlap
versus protected evaluation identities:

- Presentation TRAIN and consumed development, plus the **full original reserved
  sequence embargo** of 180,116 peptides (reserved study positives and final held
  comparisons), with its reserved study-alias closure. Locked confirmation
  observation files/outcomes are never opened.
- Atlas TRAIN/validation/TEST identities, including excluded unsupported bags.
- Seven frozen affinity namespaces: Stage-B TRAIN/development, historical
  source-review-v2 TRAIN/validation/previously consumed TEST identities, and both
  NetMHC BA TRAIN archives. Reusable identity-only projections cover 546,876
  original rows and retain original byte hashes and row-level source fields.
- T-cell TRAIN/development/calibration/TEST membership plus embargo-drop
  identities. Component metadata supplies PMID/DOI for study-collision checks.

Every namespace checks exact peptide, shared contiguous 8-mer and same-length
Hamming<=1. The primary uses same-position deletion masks; independent replay
enumerates one-residue substitutions and separately builds the reference core
index. Noncanonical reference strings remain in the embargo; no sequence is
repaired or dropped to make overlap vanish. All 17 primary/replay membership
sets match. The final per-row union, uniqueness and ineligibility flags are
recomputed locally after checksum-verified download. No source-study token
match was found in the checked lineage fields/closures; this is scoped to these
known namespaces, not a universal novelty claim. TRAIN overlaps remain explicit
and are not automatically interpreted as duplicate measurements.

The largest additional individual protected overlap is the full presentation
reservation: 9,318 candidate rows (9,317 core8, plus a Hamming-only match).
Atlas validation gains eight additional matches beyond core8. Per-namespace
counts overlap and must not be summed; complete tables are in `results/`.

## A model limitation found before fitting

The current staged `affinity_downstream` path computes processing from full
peptide + source/APC context, then adds MHC-dependent but context-independent
affinity/stability. With the remaining inputs fixed, its DM-dependent logit
change for a peptide cannot depend on MHC identity. Its binding-register weights
also cannot respond to DM. A sigmoid probability difference can still vary with
baseline affinity; that is not an MHC-by-condition logit interaction.

The frozen artificial probe tests two synthetic MHCs per class, three peptides
per class, seeds42/43 and both existing modes (24 condition groups, no optimizer).
Staged processing deltas are exactly equal across MHCs and register-weight changes
are exactly zero; logit interaction is at most 7.45e-9 roundoff. Existing
`legacy_joint` fusion can express interaction (maximum absolute 0.0012063).
Purified-affinity predictions are exactly unchanged by DM in both modes.
These are capacity checks, **not trained accuracy measurements or evidence that
the legacy model predicts better**. Full fixture values/config/source hashes are
in `results/condition-interaction-probe.json` and `reproduce/`.

[Presto157](https://github.com/pirl-unc/presto/issues/157) owns a shared
MHC-by-condition selection function and matched comparison. It must retain a
common encoder/register path for both classes, distinguish precursor supply
from downstream selection, preserve purified affinity/MS-detector dependence,
and prove feature/update coverage. The biological motivation is the primary
[DP/DM study](https://pubmed.ncbi.nlm.nih.gov/33025005/) and
[DO repertoire study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6398211/); neither
proves a particular proposed architecture improves predictions.

## Reproduction, runtime and verification

- Base `4da30f74`; projection source frozen in `e0759165`; full audit source in
  `b8913a69`. Runtime source and every input byte count/hash are frozen in the
  respective `reproduce/` launch bundles. The later artificial fixture has its
  own exact runtime hashes and invocation, explicitly marked dirty at execution.
- `bash reproduce/launch.sh`: seven source identity projections, 176.84 seconds,
  Modal app `ap-MN3I9rgLpwuDE0QP8lItSe`.
- `bash reproduce/audit-launch.sh`: full comparison and independent replay,
  122.47 seconds, Modal app `ap-notjaWQVECNpRJIURzw90u`.
- Each job requests one 8-CPU / 16-GiB container, blocked network, no GPU. Actual
  CPU model and peak RSS were not recorded. Projection uses Python3.12 stdlib;
  comparison uses Python3.12, numpy2.5.4 and pyarrow26.0.1. Shared environment
  packages were not changed.
- Five focused fixtures pass: outcome exclusion, forbidden allowlists, exact/core
  overlap, correct Hamming position/length and retained noncanonical references.
  Isolated Ruff passes. No production model/trainer code changed in this family.
- Large per-row artifacts remain on `presto-checkpoints` under this family's
  `affinity-identities-v1/` and `cohort-audit-v1/`. Receipts pin every file. The
  latter receipt SHA256 is
  `a54fea6b1800c704260e84d16eeaf125126a92a0985ff18d0646e529d9e2b4a5`.
  Downloaded summary/dispositions match their remote byte counts and hashes.
- The first directory download placed a single parquet at the destination file
  path. Explicit receipt/summary downloads and checksum verification recovered
  all outputs; no audit was relaunched. Client final-log timeouts occurred after
  both jobs returned passed results; the apps completed normally.

## Decision and scientific limits

Paired support remains sufficient to develop a tightly scoped within-study
sequence-generalization experiment. It is one engineered HeLa system and two
paired allotypes, not external pathway qualification. Compare a condition-masked
control, the current staged response and a shared MHC-conditioned selection
model with matched source rows/backgrounds/exposures. Report within-condition
and within-pMHC contrasts, so a generic DM-dependent detection offset cannot
masquerade as learning editing specificity. Freeze peptide-component splits and
weak nondetection semantics before fitting; do not convert them to affinity
labels or select an evaluation cohort using model scores.

No pretraining, biological fit, loss optimization, model selection, GPU, or
held-out biological prediction occurs here. Predictive validation/test metrics
and dumps are inapplicable. Source-release checks remain in Hitlist898/897;
Presto155/157 own the next qualified model/fit steps. Scientific PR131 remains
draft and the broader binding/MS/processing objective is not complete.

The [next implementation/fit gates](../agents/codex/plans/2026-11-10_shared-selection-comparison.md) preserve the full shared architecture and matched-comparison requirements. That plan is not an implemented model or a launched fit.
