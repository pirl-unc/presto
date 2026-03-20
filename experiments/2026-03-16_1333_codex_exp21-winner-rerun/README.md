# EXP-21 Winner Rerun via Canonical Launcher Layout

- Agent: `codex`
- Source script: `code/launch.py`
- Source baseline: `experiments/2026-03-15_1226_codex_exp21-seed-epoch-confirmation`
- Status: `completed`
- Result: `exact structure-check reproduction of the best EXP-21 seed`

## Goal

Verify that the current best known baseline can be rerun end-to-end through the canonical experiment-local layout:

- `code/launch.py`
- `manifest.json`
- `launch_logs/`
- `results/runs/`
- `reproduce/`

This is not a new sweep. It is a one-run replay of the strongest prior EXP-21 condition so the structure itself is validated.

## Rerun Contract

- Model family: `groove`
- Condition: `cond_id=2`
- Content conditioning: `off`
- Epochs: `50`
- Batch size: `256`
- Seed: `43`
- Requested Modal GPU: `H100!`

## Dataset Contract

- Source: `data/merged_deduped.tsv`
- Alleles: `HLA-A*02:01`, `HLA-A*24:02`
- Measurement profile: `numeric_no_qualitative`
- Included assay families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- Qualifier filter: `all`
- Split: `peptide_group_80_10_10_seed42`
- Split sizes: train `15465`, val `1974`, test `1925`

## Result

The rerun reproduced the original EXP-21 best-seed result exactly on held-out metrics.

| run id | test Spearman | test Pearson | test AUROC | test AUPRC | test RMSE log10 | test loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dist-ba-v6-rerun-groove_c02-c02-cc0-e050-s43` | `0.85413903` | `0.85440373` | `0.94411862` | `0.91761374` | `0.81867343` | `0.02374651` |

Source comparison:

- Original run: `dist-ba-v6-confirm-groove_c02-c02-cc0-e050-s43`
- Original test Spearman: `0.85413903`
- Original test AUROC: `0.94411862`
- Original test AUPRC: `0.91761374`
- Original test RMSE log10: `0.81867343`

Best validation checkpoints also matched:

- best val loss: `0.02550036` at epoch `25`
- best val Spearman: `0.84519303` at epoch `39`
- final val Spearman: `0.84358341`

This means the canonical `code/launch.py` experiment layout is not merely compatible with the winner contract; it reproduces the known best-seed result exactly.

## Interpretation

- The baseline itself did not change.
- `model_to_beat.md` remains anchored on the full EXP-21 robustness sweep, not this one-run replay.
- What changed is confidence in the workflow: the experiment-local launcher structure is now validated on the current best contract.

## Artifact Notes

- Raw run artifacts were fetched locally under `results/runs/`.
- Derived summaries/plots were regenerated with the shared aggregation tool.
- This shared v6 runner does not emit per-example val/test prediction CSVs, so this rerun preserves the same closure contract as the original family:
  - `summary.json`
  - `probes.jsonl`
  - `metrics.jsonl`
  - `step_log.jsonl`

## Artifacts

- launch manifest: `manifest.json`
- launch log: `launch_logs/dist-ba-v6-rerun-groove_c02-c02-cc0-e050-s43.log`
- raw run dir: `results/runs/dist-ba-v6-rerun-groove_c02-c02-cc0-e050-s43/`
- summary table: `results/condition_summary.csv`
- summary bundle: `results/summary_bundle.json`
- plots:
  - `results/test_spearman_ranking.png`
  - `results/test_metric_grid.png`
  - `results/training_curves.png`
  - `results/final_probe_heatmap.png`
- reproduce bundle: `reproduce/`

## Decision

Keep the canonical experiment-local structure:

- shared inputs under repo-level `data/`
- shared multi-experiment machinery in `scripts/`
- experiment-local launcher at `code/launch.py`
- fetched raw outputs in `results/runs/`
- frozen launch metadata in `reproduce/`

This structure is now validated against the current best known binding baseline.

## Handoff

- Status: closed
- Next Step: use this same structure for future Claude/Codex experiment families unless a launcher is truly shared enough to remain in `scripts/`
- Open Questions:
  - should the shared v6 runner start emitting per-example val/test predictions so structure-check reruns also satisfy the richer eval-artifact contract?
  - should the next confirmation run use a fresh seed (`46+`) instead of replaying the historical best seed?
