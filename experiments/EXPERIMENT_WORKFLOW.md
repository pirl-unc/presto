# Experiment Workflow

This document defines how Codex, Claude, or any future agent should use `experiments/`.

For Codex/Claude coordination details beyond the basic lifecycle, also read [AGENT_COORDINATION.md](./AGENT_COORDINATION.md).
For the current stable baselines by dataset/metric, use [model_to_beat.md](./model_to_beat.md).

Read this before starting or extending experimental work. It explains:
- where rough ideas belong
- where coarse experiment TODOs belong
- where detailed plans belong
- where completed experiment families belong
- how to write the canonical markdown record so later agents can reuse the results without thread context

## Goals

- keep experiment history readable and reusable
- let multiple agents work in parallel without stepping on each other
- make dataset/training contracts explicit
- separate rough ideas from planned work from completed results

## Workflow

1. Put rough ideas in `agents/<agent>/ideas.md`
2. Promote promising items into `agents/<agent>/todo.md`
3. For non-trivial experiments, create a detailed plan in `agents/<agent>/plans/`
4. When launching/running an experiment family, create `experiments/YYYY-MM-DD_HHMM_agent_slug/`
5. Put the local experiment README, reproducibility bundle, summary tables, plots, and artifact links there
6. Add a final summary entry to `experiment_log.md`
7. Update `model_to_beat.md` only if the new result changes the practical baseline for a contract

Every new experiment family must contain a reproducibility bundle under:
- `reproduce/launch.sh`
- `reproduce/launch.json`
- `reproduce/source/` (launcher snapshot when available)

Launcher placement rule:
- shared launch/collection infrastructure stays in `scripts/`
- if a launcher or posthoc analyzer is specific to one experiment family, put it in that experiment directory instead, usually under `code/launch.py` or `analysis/`
- do not keep one-off experiment wrappers in `scripts/` after the experiment is closed unless they are being generalized for reuse

Canonical experiment-local layout:
- `README.md` for the narrative writeup and handoff state
- `code/launch.py` for the canonical launcher for that experiment family
- `analysis/` only for one-off analysis code tied to that family
- `manifest.json` for launched condition/run metadata
- `launch_logs/` for local launch stdout/stderr
- `results/runs/` for locally fetched raw run directories
- `results/*.csv`, `results/*.json`, `results/*.png` for derived summaries and plots
- `reproduce/` for the frozen launch bundle

Canonical data split:
- shared reusable input data belongs under repo-level `data/`
- experiment directories record the data contract and store generated outputs, not duplicate shared raw inputs
- if an experiment needs a frozen code snapshot, keep that under the experiment dir; if it needs a frozen large dataset slice, record the exact source path and curation contract unless duplication is intentionally required

These files freeze the exact launch command and relevant environment overrides so later agents can rerun the same family even if launcher defaults change.

An experiment is only considered complete when:
- the runs have finished
- metrics have been extracted from available logs/artifacts
- the experiment directory contains the summary tables and any useful plots
- the experiment `README.md` has been updated with dataset/training/assay/loss details and conclusions
- `experiment_log.md` has been updated with a contextualized summary
- validation/test prediction dumps and held-out metrics have been saved for eval experiments, or the absence of a test split has been explicitly justified

## Parallel-Agent Rules

- Each agent owns its own:
  - `ideas.md`
  - `todo.md`
  - `plans/`
- Completed runs are shared and live at top-level timestamped experiment directories.
- If one agent is extending another agent's plan or run family, say so explicitly in the experiment README and log entry.
- Use stable slugs so related runs are easy to recognize.
- Do not split the canonical completed-experiment history by agent; use [experiment_log.md](./experiment_log.md) as the shared result registry and keep per-agent separation in `agents/<agent>/`.

## Naming

Experiment directory:
- `YYYY-MM-DD_HHMM_agent_slug`

Examples:
- `2026-03-11_1300_codex_directness-round3`
- `2026-03-11_1500_claude_assay-ablation`
- `2026-03-11_1700_codex_claude_joint-runtime-bakeoff`

## What To Record

Minimum information for any completed experiment family:
- reproducibility bundle path and whether it matches the original launch
- exact dataset contract
- exact training/pretraining contract
- exact assay families and qualifier/censor policy
- synthetic-data contract, if any
- loss terms and weights
- assay-label -> output mapping
- requested Modal GPU and actual observed hardware / GPU memory when applicable
- tested variants
- runtime metrics
- evaluation metrics
- takeaway

For predictive evaluation experiments, do not stop at aggregate loss plus probe values. Save enough held-out information to recompute:
- validation/test loss
- exact-value regression metrics on the primary assay family where applicable:
  - Spearman
  - Pearson
  - RMSE in target space and/or `log10(nM)`
- thresholded binding metrics such as `<=500 nM`:
  - accuracy
  - balanced accuracy
  - precision
  - recall
  - F1
  - AUROC
  - AUPRC

The preferred implementation is:
- save per-example validation predictions
- save per-example test predictions
- record the split policy and random seed

If a test split is intentionally omitted, explain that decision explicitly in both the experiment README and the canonical log.

Always include:
- whether comparisons are apples-to-apples
- whether probes are fit targets or true generalization probes
- where raw artifacts live
- a comparison table whose condition descriptions are explicit enough that another agent can rerun the same matrix
- enough context to compare the experiment to earlier runs without relying on chat history

## Reproducibility

Every `initialize_experiment_dir()` call auto-creates a `reproduce/` bundle containing:

- **`launch.json`** — argv, environment overrides, git state (commit, branch, dirty), timestamps
- **`launch.sh`** — executable relaunch script with the same git state recorded as a comment header
- **`source/<launcher>.py`** — frozen snapshot of the launcher script at launch time

The git commit hash pins the exact model and training code used. The launcher snapshot pins the sweep config and any mutable defaults (batch size, alleles, etc.). Together they make the experiment fully reproducible even after launcher defaults change.

If launching from a dirty working tree (uncommitted changes), note that in the experiment README so reviewers know the `reproduce/` bundle alone may not be sufficient — the uncommitted diff matters too.

Never rely on mutable launcher defaults as the only reproduction path. The `reproduce/` bundle is the source of truth for what was actually run.

## Modal Hardware Policy

- Treat hardware as part of the experiment contract when it affects runtime, memory, or optimization behavior.
- Set the GPU explicitly rather than relying on the repo default.
- Default choice:
  - `H100!`
- Exceptions:
  - `A100` only for historical-comparison experiments
  - `H200` only when you intentionally want a different hardware tier
- Record both the requested GPU string and any observed hardware evidence from logs.
- Do not mix `A100`, `H100!`, and `H200` inside one comparison table without calling out the hardware difference in the conclusion.

## What Not To Do

- Do not maintain a separate hand-edited machine-readable registry in parallel.
- Do not leave important results only in `modal_runs/`.
- Do not rely on mutable launcher defaults as the only reproduction path; preserve a frozen launch bundle in the experiment directory.
- Do not compare runs across different contracts without saying so explicitly.
- Do not rely on chat history as the only explanation for an experiment.
