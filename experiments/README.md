# Experiments

This directory is the canonical coordination layer for experimental work in this repo.

Use it for:
- completed experiment families
- cross-agent handoff
- per-agent brainstorms
- per-agent coarse TODOs
- detailed experimental plans that may later become runs

Do not rely on thread context, detached Modal app names, or scattered `modal_runs/` directories as the primary record.

Start here:
- [EXPERIMENT_WORKFLOW.md](./EXPERIMENT_WORKFLOW.md) for the experiment lifecycle
- [AGENT_COORDINATION.md](./AGENT_COORDINATION.md) for Codex/Claude planning, handoff, and canonical-record rules
- [model_to_beat.md](./model_to_beat.md) for the current stable baselines by dataset/metric

## Directory Layout

```text
experiments/
  README.md
  EXPERIMENT_WORKFLOW.md
  AGENT_COORDINATION.md
  model_to_beat.md
  experiment_log.md
  agents/
    codex/
      ideas.md
      todo.md
      plans/
    claude/
      ideas.md
      todo.md
      plans/
  YYYY-MM-DD_HHMM_agent_slug/
    README.md
    code/
      launch.py
    analysis/
    reproduce/
      launch.sh
      launch.json
      source/
    manifest.json
    launch_logs/
    results/
      runs/
      *.csv
      *.json
      *.png
    ...
```

## Canonical Experiment/Data Split

Shared repo-level data stays in `data/`.

Use `data/` for:
- curated input corpora shared across many experiments
- canonical merged tables
- reference sequence indexes and similar reusable inputs

Do not copy heavyweight shared input data into each experiment directory by default. Instead, each experiment should record the exact data contract in its `README.md` and launcher metadata.

Each experiment directory should own:
- `code/launch.py` for the canonical experiment-local launcher
- `analysis/` only for posthoc code specific to that experiment family
- `manifest.json` for launched run metadata
- `launch_logs/` for local launcher stdout/stderr captures
- `results/runs/` for locally fetched raw per-run outputs from Modal
- `results/*.csv`, `results/*.json`, `results/*.png` for derived summaries and plots
- `reproduce/` for the frozen launch bundle

Keep only reusable multi-experiment machinery in `scripts/`, such as:
- experiment registry helpers
- shared Modal fetch/collection tools
- shared aggregation/backfill tools
- shared training/evaluation backends

If a launcher or analyzer exists only to support one experiment family, move it into that experiment directory rather than leaving it in `scripts/`.

## What Goes Where

### 1. `experiment_log.md`

This is the single canonical registry.

Every completed or materially informative experiment family should have an entry here using the richer narrative/table format:
- title / experiment id
- date
- agent/model
- experiment directory link
- exact dataset and curation contract
- training/pretraining contract
- tested conditions table
- winner / preferred condition
- takeaway

### 1a. `model_to_beat.md`

This is the stable baseline summary, not the historical registry.

Use it for:
- the current architecture to beat on a given dataset contract
- the primary metric and caveats for that baseline
- links back to the supporting experiment directory

Do not use it as a replacement for `experiment_log.md`, and do not paste full experimental result tables into it.

### 2. `agents/<agent>/ideas.md`

Rough brainstorms.

Use this for:
- speculative hypotheses
- possible architectural directions
- “things worth trying later”

This is intentionally low-friction and does not need rigorous metrics.

### 3. `agents/<agent>/todo.md`

Coarse-grained experimental backlog for that agent.

Use this for:
- ordered next experiments
- blocked experiments
- handoff notes

Keep items concise and actionable.

### 4. `agents/<agent>/plans/`

Detailed experiment specs that are not yet completed.

Use this when an experiment needs:
- a non-trivial matrix of conditions
- exact dataset contract
- explicit success criteria
- implementation notes

If a plan is executed, link it from the final experiment directory and summarize the outcome in `experiment_log.md`.

### 5. `YYYY-MM-DD_HHMM_agent_slug/`

One directory per experiment family or sweep.

Minimum required contents:
- `README.md`
- `code/launch.py`
- `reproduce/launch.sh`
- `reproduce/launch.json`
- copied or generated summary tables/JSON when available
- per-example validation/test prediction dumps for evaluation experiments, unless the README explicitly documents why a test split was not used
- links or paths to raw artifacts in `modal_runs/`, `artifacts/`, or elsewhere

Recommended:
- `analysis/` for experiment-local analyzers only when needed
- `manifest.json`
- `launch_logs/`
- `results/runs/` with locally fetched raw run directories
- `reproduce/source/` with a snapshot copy of the launcher script used to start the experiment

Suggested additional contents:
- plots
- probe trajectories
- manifest / launcher metadata
- notes on failed conditions

Large raw outputs can remain outside `experiments/`, but the experiment `README.md` must point to them explicitly.

## Required Technical Content For Each Experiment README

Every experiment directory `README.md` should make rerun/extension possible without chat context.

Include:
- purpose / question
- agent/model who ran it
- reproducibility bundle location and any non-default environment overrides
- exact source data and curation filters
- exact assay families included and excluded
- qualifier/censor policy
- train/val split policy
- effective sampled examples per epoch if batching changes exposure
- pretraining contract
- synthetic-data contract, if any
- training hyperparameters
- loss terms and weights
- which assay labels map to which model outputs / losses
- conditions tested
- explicit comparison table with condition descriptions
- runtime environment
- requested Modal GPU and actual observed hardware / GPU memory if run on Modal
- held-out validation metrics and held-out test metrics when a test split exists
- enough saved per-example predictions/labels to recompute downstream metrics later
- evaluation metrics
- result summary
- takeaway / decision
- artifact paths

## Required Post-Run Closure

A finished experiment is not complete until the experiment directory and canonical log are updated.

For every completed eval/sweep/benchmark, do all of the following:
- extract all available metrics from logs, summaries, and artifacts
- write or copy comparison tables/JSON into the experiment directory
- add plots when they materially improve interpretation
- update the experiment directory `README.md`
- update [experiment_log.md](./experiment_log.md)

The closure writeup must preserve:
- the reproducibility bundle and launch command used
- exact dataset and curation contract
- training and pretraining details
- synthetic / augmentation contract
- loss terms and weights
- assay-label -> output mapping
- per-example validation/test prediction dumps or an explicit reason they were not produced
- held-out validation/test metrics, not just probe summaries or aggregate loss
- tested conditions with explicit descriptions
- within-experiment comparisons
- contextual conclusions relative to earlier experiments when relevant

For predictive evaluation experiments, the preferred held-out metrics are:
- overall validation/test loss
- exact-value regression metrics for the primary assay family where applicable:
  - Spearman
  - Pearson
  - RMSE in target space and/or `log10(nM)`
- thresholded binding metrics for a biologically meaningful cutoff such as `<=500 nM`:
  - accuracy
  - balanced accuracy
  - precision
  - recall
  - F1
  - AUROC
  - AUPRC

If a test split is intentionally omitted, say so explicitly in the experiment README and in `experiment_log.md`, and explain why.

## Recommended README Template

```md
# <Experiment Title>

- Date: `YYYY-MM-DD`
- Agent: `codex` / `claude` / `codex_claude`
- Purpose: `<what question this experiment answers>`

## Reproducibility
- Launch bundle: `reproduce/launch.sh`
- Launch metadata: `reproduce/launch.json`
- Launcher snapshot: `reproduce/source/`
- Non-default environment overrides:

## Dataset Contract
- Source:
- Curation:
- Included assay families:
- Excluded assay families:
- Qualifier / censor policy:
- Split:
- Effective train examples per epoch:

## Training Contract
- Pretraining:
- Model:
- Hyperparameters:
- Losses:
- Assay label -> output mapping:
- Synthetic / augmentation policy:
- Runtime:

## Conditions Tested
| Variant | Description | Data / loss / model deltas | Notes |
| --- | --- | --- |

## Results
| Variant | Runtime | Main metric | Probe metrics | Notes |
| --- | --- | --- | --- | --- |

## Takeaway
- Winner:
- What to keep:
- What to discard:
- Next step:

## Artifacts
- `modal_runs/...`
- `plots/...`
- `summary.json`
```

## Rules

- Keep one canonical markdown registry.
- Prefer readable, explicit dataset/training descriptions over compact but ambiguous shorthand.
- When comparing runs, state clearly if contracts differ:
  - exact vs censored
  - narrow vs broad assay family
  - allele panel differences
  - synthetic vs real-only
  - Modal GPU / hardware differences (`A100` vs `H100!` vs `H200`)
- If multiple agents contribute to a family, say so explicitly.
- If a result is provisional or not apples-to-apples, label it that way.

## Modal GPU Guidance

- Default to `PRESTO_MODAL_GPU=H100!` for Modal experiments unless the experiment explicitly requires another GPU family.
- `A100` should be used only when preserving comparability with historical baselines is the point of the experiment.
- Use `H200` only intentionally and document it as a distinct hardware condition.
- Every Modal-backed experiment should record:
  - requested GPU string
  - any hardware evidence from logs (for example total VRAM)
  - whether the run should be compared only within the same hardware family or across hardware

For Codex, Claude, or any future agent: read [EXPERIMENT_WORKFLOW.md](./EXPERIMENT_WORKFLOW.md) before starting or extending experimental work. It explains how to use `ideas.md`, `todo.md`, `plans/`, timestamped experiment directories, and the canonical markdown log together.
