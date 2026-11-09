# Post-fit scientific handoff — 2026-10-30

Codex / GPT-6. Publication follow-up for PR #131. No new model, labels or training
run is introduced here. Current source is shared-v7; the recent throughput change
is already committed as `a5db249b8b1d5a8a0ec140b00286369b837d40f2`.

## Evidence to carry forward

- [Direct factorized fits](../../../2026-10-30_1154_codex_direct-factorized-affinity/README.md): both seeds completed
  4,000 updates over the same prepared rows and sampling contract as the pooled
  controls. These are development comparisons, not complete training epochs or
  fresh blind tests. At batch 128, 4,000 updates supply 512,000 draws, including
  repeat sampling, from 3,518,242 available TRAIN observations.
- [Curated complete-epoch fits](../../../2026-10-29_1956_codex_curated-data-retrain/README.md):
  both runs and saved-prediction analyses are closed. Validation selects s43;
  that is not automatic scientific qualification or model promotion. Do not
  compare their 51,471-update results to short direct-decoder runs as matched compute.
- [Preparation benchmark](../../../2026-10-30_1725_codex_preparation-throughput/README.md):
  29.813 to 7.986 seconds, 3.73x, on a 15,000-row / 116.3 MB actual-cohort prefix.
  Scales, sampling and tiny-run updates match. Full 30 GB speedup is unmeasured.

## Interpretations and failure modes

The MHC-I aggregate validation IC50 correlation (.374 pooled to .626 direct)
is dominated in part by assay/study target distributions: radioactivity has median
9 nM and fluorescence 8,327.85 nM. A post hoc assay-only grouping score gives
Spearman .636 without sequence. It is a diagnostic, not a prospectively qualified
baseline. Per-assay metrics and final test metrics remain authoritative; class-II
radioactivity worsens in validation and final test results are weak.

Only 61/258 and 62/258 validation literal-IC50 bounds are satisfied. The loss sign
is correct. All these rows are lower bounds on numeric IC50; failed predictions
are too low and imply binding that is too strong. Report violation magnitude,
threshold, assay, class and original versus inferred evidence as well as pass rate.

Stability shares the encoder but is outside the new categorical concentration
decoder. Test half-life Spearman is .0648/.3160 for class I (TRAIN 8,588; TEST 62)
and -.0726/-.1806 for class II (TRAIN 135; TEST 148; no II validation). Test class-I
td50 is .3163/.3824 (TRAIN 489; TEST 140). Koff has only seven TRAIN observations;
Tm has none. Half-life, koff, td50 and Tm have distinct units/interpretations.
An allocated or updated output is not a qualified predictor.

Class-II TRAIN has 306,303 observed ligands, 306,302 generated comparisons and
61,143 concentration observations, including 35,512 literal IC50. NetMHCIIpan-4.3
reports 675,364 positive ligands, 6,886,973 generated negatives and 129,110 BA
points in a different full-development-corpus contract. Do not compare our
both-class total or TRAIN-only counts as identical populations. Measured coverage
and negative-sampling ratios both contribute to the gap.

Generated ligand backgrounds are unobserved comparison peptides, not measured
nonbinders. No random >10k/50k/100k/200k nM thresholds have been assigned. The
user's proposal is preserved as an explicitly synthetic, controlled experiment.
It must not overwrite measured bounds, fabricate a protocol or inflate measured
sample counts.

## Existing source evidence, rather than duplicate recovery work

The NetMHCstabpan archive was already audited in
[measurement-curation recovery](../../../2026-10-29_1653_codex_measurement-curation-recovery/results/stability-review.json):
28,166 rows, 22,487 positive half-lives, 5,679 literal zeros, 5,012 positive rows
matching current TRAIN pMHC half-lives, 4,991 matching exact converted values,
and 1,135 unresolved mutant-construct rows. No rows were admitted by that audit.
The preceding conversational suggestion to audit overlap should be read as
finishing this existing qualification, not starting a new inventory.

Reuse [Hitlist #875](https://github.com/pirl-unc/hitlist/issues/875) for those
stability semantics, [#871](https://github.com/pirl-unc/hitlist/issues/871) for
ELISA-referencing KD protocol tags, [#721](https://github.com/pirl-unc/hitlist/issues/721)
for assay conditions/replicates and [#785](https://github.com/pirl-unc/hitlist/issues/785)
for binding coverage. Reviewed recovery candidates and later positive/background
conflicts must enter a new frozen preparation, not mutate prior experiments.

## Actionable next work

| Priority | Issue | Outcome required |
|---|---|---|
| 1 | [#140: assay-level qualification](https://github.com/pirl-unc/presto/issues/140) | Within-assay/study calibration, censor violation metrics, and identical-population external benchmarks; no headline aggregate shortcut. |
| 2 | [#139: MHC-II coverage](https://github.com/pirl-unc/presto/issues/139) | Source/assay/curation waterfall against the published corpus and qualified independent additions. |
| 3 | [#141: training experiments](https://github.com/pirl-unc/presto/issues/141) | Separate exposure, schedule and ranking effects with matched controls before attributing gradient conflict. |
| 4 | [#138: stability decoder](https://github.com/pirl-unc/presto/issues/138) | Extend the common measurement interface, preserve physical quantities and obtain useful held-out stability results. |
| 5 | [#142: synthetic affinity priors](https://github.com/pirl-unc/presto/issues/142) | Test the proposed thresholds as declared assumptions with real-measurement evaluation and false-negative sensitivity. |

Source qualification and evaluation tooling can proceed independently. Every fit
must use the canonical registry/trainer, explicit H100! hardware and the approved
maximum of two concurrent GPU workers. These issues do not claim an experiment
has run or a gain has been established. Prior identity-pretraining ideas remain
in #133–#137; they are not replacements for measured affinity qualification.

## Artifact and publication state

Large raw predictions/checkpoints remain at the recorded local and Modal volume
paths; collection receipts contain hashes and archive locators. The final plots,
summary tables, source snapshots and machine-readable results are committed here.
Live JSON and watch-process records are historical publication snapshots; the
finished/superseded monitors were replaced by static local report servers.
The old residual/scalar-calibration queue is superseded, not a completed six-arm
comparison; its collection error is preserved as historical evidence.

The curated s43 startup receipt reports 1,723.614 s, while its subsequently
collected `preflight.json` reports 2,065.021 s. Both receipts are preserved; these
are separate recorded timings and must not be silently substituted as a matched
throughput comparison.

Local verification for this publication: JSON parsing, CSV shape, Python snapshot
syntax, closeout/reporting hashes, visual plot inspection, lint and strict docs.
No runtime source changes are included; the full suite at the preceding source
commit passed 2,929 tests with four skips. Scientific milestone issues #130/#132
remain open and PR #131 stays draft.
