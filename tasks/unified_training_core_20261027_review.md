# Unified training core — software review

Implemented in the isolated `codex/unified-training-core-20261027` checkout.
Base: `9dd71348124faa49a4c6a60872385c86895d50b0`. The user approved simplifying
new training to one observation-manifest interface.

## Result

- `presto.models.Presto` is the single current model. The former modular names are
  aliases to the same class/config; they do not select another implementation.
- All 38 train/pretrain/probe/sweep/benchmark launcher aliases call the common
  manifest entry point. Public `Trainer` uses the same `fit_index` and optimizer
  step. Identity pretraining, MHC-I/II binding, MS, paired TCR and joint subsets
  are exercised with synthetic fixtures through that same model.
- Class II retains all `L - 8` nine-residue registers and a variable-length softmax.
  Full ordered peptide token states supply separate N/C-flank summaries for every
  register, replacing the three-residue composition cutoff. Class I uses the
  whole peptide. There is one shared sequence encoder; no new flank encoder.
- Historical default/modular/TCR implementations are explicit compatibility code.
  Old modular-v2 checkpoints replay with their original feature contract. New fits
  use `presto.shared.v3`; old checkpoints cannot silently warm-start/resume it.
- Retired flags and separate TCR constructors reject use. Historical scripts are
  archived, with exact reproduction delegated to their frozen source commits.
  Old core-supervision/configurable-core-length machinery and architecture bakeoffs
  are not active training options. Their old outputs are not claimed equivalent.
- Resume fingerprints include the promoted model/entry points and ordinal context.

## Verification

The broad suite passed **2,724 tests, with four skips**. Three skips require an
explicit pinned real-corpus bundle; one is a platform-specific sharing-strategy
check. After adding the final ordinal-context source pin and four rejection cases,
all **38 resume/shared-core tests passed**. This final edit does not alter model
computation. Repository Ruff checks, formatting and whitespace checks pass.

MkDocs, wheel and sdist builds pass. A wheel extracted outside the checkout passes
an actual synthetic training/checkpoint/prediction smoke with one/seven registers
for class I/II. All 22 pinned source files match the wheel bytes. No dependency
install, shared-environment maintenance, cloud job or release was performed.

A read-only compatibility regression exactly replays 24 already-recorded training
predictions from three scientific DO checkpoints. All 168 files in that frozen
experiment retain their hashes. Earlier shared-core/context checkouts remain clean.
The original user worktree was kept separate from implementation.

Initial test failures and the interrupted catalog-heavy repair run are preserved.
Repairs updated retired-interface assertions, historical imports and explicit test
paths; the DRA existence checks retain every DRA record in the checked-in catalog.
A wheel fixture initially supplied identical regression labels and correctly
failed the target-scale guard; its corrected distinct-label fixture passes.

## Scientific and migration limits

Three flank residues are a published baseline approximation, not demonstrated
sufficiency for Presto. Full context is now represented, but no accuracy gain or
MHC-I transfer is established. The native DO result remains negative. TCRvdb
access and scientific PR #131 milestones remain open. Software passing does not
make that PR ready to merge.

The new model needs fresh training and qualification. Old direct cleavage/transport
and repertoire outputs are not automatically reproduced by current observation
endpoints. No elution label is reinterpreted as purified binding or independently
measured antigen processing. Historical experiment records are unchanged.

## Evidence and use

- [Migration and capability map](../docs/unified_training.md)
- [Implementation spec](unified_training_core_20261027_spec.md)
- [Verification metadata, exact commands and source hashes](unified_training_core_20261027_verification.json)
- [Full suite](unified_training_core_20261027_suite.xml)
- [Final resume/shared-core suite](unified_training_core_20261027_final_resume.xml)
- [Historical replay](unified_training_core_20261027_replay.json)
- [Packaged smoke](unified_training_core_20261027_wheel.json)
- [Archive inventory](unified_training_core_20261027_archive.json)

```bash
python -m presto train --manifest /absolute/path/manifest.json --preflight
```

Remove `--preflight` and supply a registered experiment directory for fitting;
follow `experiments/EXPERIMENT_WORKFLOW.md` before a new scientific run.
