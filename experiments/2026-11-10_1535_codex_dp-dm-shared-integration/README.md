# Shared DP/DM source integration

Date: 2026-11-10. Agent/model: Codex / GPT-6. Source and identity audits passed;
Hitlist release qualification and scientific training remain open.
Plan: [detailed specification](../agents/codex/plans/2026-11-10_dp-dm-shared-integration.md).

The recovered PXD017154 / PMID33025005 observations now traverse the existing
Hitlist supplementary scanner, contributor capture, sample/condition/lineage
joins and public `generate_training_table` API into Presto's common MS adapter.
There is no additional model or trainer path. [Hitlist draft PR898](https://github.com/pirl-unc/hitlist/pull/898)
contains the packaged source, curation and tests at `4e700744b196c7cb088be9772f1b7cd37a61406e`.

## Dataset and semantics

Start from the independently replayed native extraction in
[the preceding source recovery](../2026-11-10_1454_codex_dp-dm-source-recovery/README.md).
Its hash, all source-code/data hashes and the MHC catalog hash are frozen in
`reproduce/launch.json`. Original native high-confidence PSMs, explicit
unmodified forms, canonical 10–512-aa sequence, unique visible rank-1 identity,
complete protein references and no-DP-control exclusion define the candidate
subset. Converted mzIdentML pass flags are unsuitable for admission.

There are **21,547 peptide–sample rows / 10,143 distinct peptides**, with all
21,547 contributor links checked against original native PSM, protein, acquisition
and sample evidence. Ten registered sample assets preserve exact alpha/beta
chains; a separate 87-row control companion contains 17 control sequences and
70 overlapping ordinary-sample rows and is excluded from positive ligands.
A/B sample labels do not establish independent physical specimens or donors;
the common lineage retains these as unknown and preserves `origin:PXD017154`.

| Exact DP pair | Rows |
|---|---:|
| DPA1*01:03 / DPB1*04:01 | 3,199 |
| DPA1*01:03 / DPB1*04:02 | 8,329 |
| DPA1*02:01 / DPB1*10:01 | 10,019 |

All three pairs resolve through the common catalog without partner imputation.
There are 12,375 DM-absent and 9,172 DM-present rows. Absence means no additional
DM transduction relative to the declared CD74/CD80/DP-engineered background.
No DO-MS arm, measured component activity or biological cleavage target is
inferred. The assay maps to positive MS detection (`value=1`), not affinity or
abundance. Controls, nondetections and spectral counts never become affinity
negatives. There are no concentration qualifiers, synthetic labels or losses.

## Identity-only overlap

The frozen presentation identity index contains 1,349,378 TRAIN and 48,530
consumed-development rows. The frozen Atlas identity index contains 993,120
rows across TRAIN, validation and sealed TEST. Only allowlisted identity fields
were projected; no held-out outcomes or predictions were accessed. Exact-peptide
and shared-contiguous-8-mer membership was checked against all fold identities,
including excluded Atlas candidates. Counts below overlap and must not be summed.

| Identity cohort | Exact-overlap source rows | Shared-8-mer source rows |
|---|---:|---:|
| Presentation TRAIN | 52 | 7,904 |
| Presentation consumed development | 7 | 83 |
| Atlas TRAIN | 2,611 | 11,330 |
| Atlas validation | 1,083 | 6,458 |
| Atlas sealed TEST (identity only) | 1,169 | 6,467 |

The protected union embargoes **8,338 rows**. The other **13,209 rows / 6,434
peptides** clear only these presentation/Atlas identity checks. Quantitative
affinity and other reserved cohorts remain unchecked. Consequently **zero rows
are currently TRAIN-eligible**. TRAIN overlap itself does not establish an
independent measurement or a label conflict. No PXD017154/PMID33025005 token
was found in the checked presentation identity fields; this is not a universal
source-absence claim. Independent replay uses a reverse candidate-core lookup
and verifies all 21,547 rows × 12 membership/eligibility fields.

## Verification and reproducibility

- Presto base: `9a99bbf0207af70b4e168385a1daa0cf98cc5ece`; Hitlist base:
  `43e226621434a483141764ed54faba725bbb3869`. Executions used dirty, hash-pinned
  experiment/source files; the exact invocations and snapshots are in `reproduce/`.
- `reproduce/launch.sh` is the source/export/common-adapter audit;
  `reproduce/identity-overlap-launch.sh` is the identity audit;
  `reproduce/verify-identity-overlap.sh` is its independent replay. Restore the
  pinned files to the recorded paths or supply equivalent paths with the same
  hashes. Use a fresh output directory when repeating a launch.
- Shared Python 3.12.6 and explicit `PYTHONPATH` are recorded in the launch.
  The source-only index uses an isolated `HITLIST_DATA_DIR`; existing indexes
  and cohorts are untouched. Source integration took 12.104s local CPU; the
  identity audit took 4.885s. Peak RSS was not measured for these two audits.
- Attempt01 failed in the verifier's identity comparison against a pandas
  nullable boolean; attempt02 checks values and preserves Python-bool conversion
  before the common adapter. Both launch/source snapshots remain in `reproduce/`.
- Hitlist format, lint and isolated `pip check` pass. The wheel includes all
  12 source assets byte-for-byte. See `results/wheel-check.json`.
- Full pre-marker-fix default Hitlist tests: **3,729 passed / 91 failed**.
  Nine existing failures reproduce on clean base; [Hitlist897](https://github.com/pirl-unc/hitlist/issues/897)
  owns compatibility with current mhcgnomes. One new real-source test now has
  the required integration marker. Isolated post-fix test launches were refused
  before pytest by the unchanged host memory guard. No full post-fix/integration
  pass is claimed. Compressed logs and software status are under `results/`.
- Presto's prior exact head `9a99bbf0` passed hosted lint-test/docs checks.
  Publication-head checks are reported in PR156, not assumed from that result.

Raw source-only exports, contributor artifacts, per-row overlap membership and
the identity-only input indexes are archived on Modal volume `presto-checkpoints`
under `2026-11-10_1535_codex_dp-dm-shared-integration/shared-integration-v1.tar.gz`.
`results/publication.json` records the archive/member hashes and readback status.
Original native source files remain at the preceding recovery family's documented
locations. Packaged positive/control assets are in Hitlist PR898; raw exports,
source inputs and wheels stay outside this Git repository.

## Decision and remaining work

Prefer the shared source/export path with native confidence and explicit context.
This audit establishes usable source records and a partial embargo; it does not
establish performance gains or independent training additions. No model fit,
optimizer, pretraining, selection or GPU occurred. Validation/test predictive
metrics and prediction dumps are inapplicable because this is source integration
and identity reconciliation, not an evaluation experiment.

[Hitlist895](https://github.com/pirl-unc/hitlist/issues/895) remains open through
source release qualification. [Presto155](https://github.com/pirl-unc/presto/issues/155)
owns the remaining overlap checks and a prospective, matched condition-aware
versus masked fit through the common model. The conversation's broader
[future-work map](../agents/codex/plans/2026-11-10_conversation-followups.md)
links stability, affinity, censoring, data coverage and transfer requirements.
Scientific parent PR131 stays draft until its qualification milestones pass.
