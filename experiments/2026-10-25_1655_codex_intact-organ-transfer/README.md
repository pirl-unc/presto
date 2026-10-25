# Frozen organ-transfer evaluation: stopped at quantitative admission

2026-10-25 · Codex/GPT-6 · **Closed: support failed; no model scoring.**

The original PXD058121 F3 workbook leaves `Unlocalized Mod` blank for every one
of its 4,198 assignments. All 44 source headers name the expected column, and
independent OOXML inspection confirms that the corresponding column-K cells are
absent. Hitlist's canonical null values are faithful to the original file. They
cannot satisfy the prespecified explicitly-unmodified gate.

| Stage / condition | Result |
|---|---|
| Metadata coverage | 44 F3 acquisitions; Brain/Liver/Spleen nominal 0/30/60 supported |
| F2/F3 recorded acquisition reuse | None found; physical preparation independence remains unresolved |
| Qualified primary examples | Brain 0; Liver 0; Spleen 0 |
| Qualified extrapolation examples | Brain 0; Liver 0; Spleen 0 |
| Independent admission replay | All 8,138 assignment dispositions agree |
| Original modification-cell replay | 4,198/4,198 absent K cells; no parser mismatch |
| Model fitting, selection or scoring | None |

No winner, validation/test loss, Pearson/Spearman, RMSE, bootstrap interval or
prediction dump exists: the fixed minimum of 20 sequences and 40 primary cells
per organ failed before scoring. IC50 and 500 nM binding metrics do not apply.
This is an admission result, not evidence that a model predicts biology poorly.

## Frozen scientific contract

Same study, one source animal and human alpha-synuclein substrate, purified
mouse-organ 20S preparations; not an APC concentration or independent-protein test.
The planned target was change in log10(product/intact) from a supplied organ/peptide
nominal-zero mean. Shared run-gain and fixed response cancellation were assumptions;
time-dependent response and absolute concentration remained unresolved. Primary
30/60-minute measurements and secondary 90/120-minute measurements could not select
models. Missing values were not converted to zeros or inferred censoring.

The 21 frozen arms were zero change; five-fold shared/length/sequence slope
ensembles; three five-fold physical ensembles; and the 14 original uniform/residue
exact-versus-conditional-score fits (16/64 draws, seeds 42/43/44). The latter are
exact-forward/REINFORCE-gradient hybrids, not simulator-only RL or GRPO. All 31
checkpoint/selection files were hashed before admission and remain unchanged.
No new training, synthetic labels, loss updates, pretraining, organ adaptation,
or old F2 held-out outcome access occurred. The planned comparison loss was
weighted squared error of measured log-ratio changes, balanced across organs,
peptide curves, times and replicate cells. No validation selection split was
planned because all predictors were already fixed.

## Implementation, reproducibility and verification

- Presto base `712cce2523665981496c9d8d66d9472c34dddb53`; pre-existing task-note
  dirt was preserved. The frozen plan is `reproduce/plan.md`.
- Metadata used clean released Hitlist `d76c225a0aa741a1a1b0aee707bff328ca467c7c`.
  Admission used isolated tested commit `7c427e390daafdfb457581c511de88c82af3efbd`,
  version 2.0.41 **unreleased**, in draft [Hitlist PR848](https://github.com/pirl-unc/hitlist/pull/848).
  No shared Python installation was changed.
- Shared recipe software checks: format/lint passed, 17 focused checks including
  unchanged-F2 output parity passed, and 3,582 default-suite tests passed. These
  synthetic fixtures are software checks, not biological training evidence.
- Local CPU; one numerical thread; 3 GiB startup free+speculative and process-tree
  RSS limits, 20 minutes/stage, 1 GiB disk reserve plus startup margin. No Modal.
  Five successful guarded stages took 19.561 s total; maximum child RSS 745.73 MiB.
- Attempt 1 failed before guard initialization because `inspect.py` shadowed the
  standard library. The separate `PYTHONSAFEPATH=1` wrapper preserves the original
  source. Attempt 2 was refused below the memory floor, without a child. Attempt 3
  ran after the unrelated processes ended. Neither failure queried the source.
- `reproduce/launch-safe.sh` retains the existing guard. `inspect` performs metadata
  queries, `verify` materializes the shared endpoint, and `fetch` runs explicitly
  pinned **local** replay scripts; no network retrieval occurs. Requests and source
  snapshots retain exact invocations/inputs. Admission's request explicitly pins
  its feature checkout; actual import origin is recorded in `results/admission.json`.

Executed commands, from the Presto worktree:

```sh
reproduce/launch-safe.sh inspect --attempt 3 --requests reproduce/requests/inspect-2.json
reproduce/launch-safe.sh verify --attempt 1 --requests reproduce/requests/verify-1.json
reproduce/launch-safe.sh fetch --attempt 1 --requests reproduce/requests/fetch-1.json
reproduce/launch-safe.sh fetch --attempt 2 --requests reproduce/requests/fetch-2.json
reproduce/launch-safe.sh fetch --attempt 3 --requests reproduce/requests/fetch-3.json
```

Paths above are relative to this experiment directory; use its full prefix when
launching from the repository root. Existing attempts refuse overwriting: consult
receipts before rerunning. Shared raw source and recipe tables remain once under
`data/intact_degradomics_canonical/2026-10-25_0021_codex_intact-canonical` and
`data/intact_organ_recipes/2026-10-25_1655_codex_intact-organ-transfer`.
`results/recipe-{manifest,contract,summary}.json` pin their contents. Coordinates,
source metadata, all stage receipts, source snapshots and independent replays are
included here. No scientific data or unique checkpoint was deleted.

## Decision and handoff

This family is closed at the fixed support gate; do not relax it or score models
in this family. The source needs documented F3 search/export semantics or original
explicit modification annotations. A documented interpretation that blanks mean
none would need separate provenance and a newly registered cohort/comparison.

The precise source request is https://github.com/pirl-unc/hitlist/issues/831#issuecomment-6400806453.
The full precursor-relative APC availability goal remains active and unachieved.
