# Agent Coordination

This document defines how Codex and Claude should share experimental work in this repo without fragmenting the scientific record.

Use this together with [EXPERIMENT_WORKFLOW.md](./EXPERIMENT_WORKFLOW.md). The workflow doc explains the lifecycle of an experiment. This doc explains how two agents should divide planning, handoff, and canonical documentation.

## Core Decision

Keep one shared completed-experiment history.

Do not create:
- a separate Codex experiment log
- a separate Claude experiment log
- parallel "canonical" result summaries that can drift

The canonical completed-experiment record stays:
- [experiment_log.md](./experiment_log.md)
- one top-level directory per completed experiment family: `experiments/YYYY-MM-DD_HHMM_agent_slug/`

## Canonical Log Concurrency

`experiment_log.md` is shared, so concurrent edits are expected.

Treat it as append-only by default:
- add new experiment entries at the end
- only edit older entries for closure backfill, factual corrections, or explicit supersession notes
- do not reorder earlier entries unless a merge conflict forces it

If Codex and Claude both touch `experiment_log.md` concurrently:
- keep both entries
- resolve the git conflict by preserving both blocks in chronological order
- do not drop or rewrite the other agent's result just to make the merge cleaner
- if both edits modify the same historical entry, preserve the factual union of those edits and note the correction in the entry when helpful

The point is one shared history, not one shared branch tip. Merge conflicts are operational friction, not a reason to split the canonical log.

## Separate The Right Things

Separate agent-local planning, not completed results.

Use:
- `experiments/agents/codex/ideas.md`
- `experiments/agents/codex/todo.md`
- `experiments/agents/codex/plans/`
- `experiments/agents/claude/ideas.md`
- `experiments/agents/claude/todo.md`
- `experiments/agents/claude/plans/`

These are the right places for:
- rough hypotheses
- incomplete experimental designs
- proposed next steps
- agent-specific work queues
- handoff notes before an experiment is launched

Completed or materially informative runs do not stay only in per-agent areas. They must move into a top-level experiment directory and into [experiment_log.md](./experiment_log.md).

When either Codex or Claude creates a new experiment family, mirror the same on-disk layout:
- `README.md`
- `code/launch.py`
- `analysis/` only if the analysis is experiment-specific
- `manifest.json`
- `launch_logs/`
- `results/runs/`
- `reproduce/`

Use repo-level `data/` as the shared source of reusable input datasets. Do not invent agent-specific data copies unless the experiment explicitly requires a frozen private snapshot, and if it does, record that decision in the README.

## Stable Baseline Summaries

Do not turn every experiment `README.md` into a rolling scoreboard for the "current best model."

Use three layers instead:
- experiment `README.md`: what this experiment asked, ran, and concluded
- [experiment_log.md](./experiment_log.md): canonical historical registry
- [model_to_beat.md](./model_to_beat.md): current stable architecture(s) to beat by dataset / contract / metric

Rules for [model_to_beat.md](./model_to_beat.md):
- update it only when a result changes the practical baseline or clarifies baseline scope
- organize it by dataset contract, not by chronology
- include the exact experiment directory link for each baseline claim
- include the metric, split policy, and caveats
- do not duplicate full experiment tables there

## Ownership And Naming

Every launched experiment family must have a clear agent label in the directory name:
- `YYYY-MM-DD_HHMM_codex_slug`
- `YYYY-MM-DD_HHMM_claude_slug`
- `YYYY-MM-DD_HHMM_codex_claude_slug` for genuinely joint work

Use `codex_claude` only when both agents materially contributed to the executed family. If one agent is merely extending or analyzing a prior run from the other, keep the directory tied to the launching agent and record the extension explicitly in the README/log entry.

For Modal run ids, manifests, and launcher metadata:
- include the agent slug in run names where practical
- avoid generic names that can collide across agents

## Planning Before Launch

For any non-trivial experiment, create a plan under the launching agent's `plans/` directory before launching.

A usable plan should contain:
- question being answered
- exact dataset contract
- exact training / pretraining contract
- factors being varied
- success criteria
- artifacts expected at closure
- any known risks or dependency on another agent's prior work

If one agent is picking up the other agent's unfinished idea:
- create a new plan in the current agent's plan directory
- link the prior plan or experiment directory
- state explicitly what is being reused vs changed

## Handoff Rules

When handing work from one agent to another, leave enough context that the receiver does not need thread history.

Minimum handoff artifacts:
- a plan file in the agent's `plans/` directory for not-yet-launched work, or
- a top-level experiment directory for launched/completed work

Every experiment README should end with a short handoff block:
- `Status`
- `Next Step`
- `Open Questions`

Use that block to make it obvious whether the next agent should:
- rerun something
- extend the sweep
- treat the result as closed
- challenge an assumption before proceeding

If an experiment is incomplete but already launched:
- keep the top-level experiment directory
- update the README with current state and missing closure steps
- do not hide the state only in a per-agent todo

## Extending Another Agent's Experiment

When extending a prior experiment family, preserve the original record and create a new experiment directory if the new work changes any of the following:
- dataset contract
- split policy
- training schedule
- model family
- evaluation contract
- scientific question

Only append to the original experiment directory when the new work is strictly closure work:
- fetching missing artifacts
- regenerating summaries from existing raw outputs
- fixing documentation drift
- adding missing plots or metric tables from already finished runs
- retrospective Modal artifact collection / summary backfill for already finished runs

If the new work changes the actual experimental contract, create a new top-level directory and reference the earlier one.

## Isolation Rules

Do not try to isolate Codex and Claude by splitting the canonical history. Isolate them by working surfaces.

Recommended isolation:
- separate `ideas.md`, `todo.md`, and `plans/`
- separate git branches or worktrees when both agents may touch code concurrently
- explicit ownership in experiment directory names and run ids
- explicit README/log attribution when one agent extends the other's work

If both agents are likely to touch the same code path at the same time, prefer separate worktrees over informal coordination.

## Conflict Resolution

If Codex and Claude disagree:
- do not fork the canonical log
- preserve both proposals in agent-local plans
- run the comparison as an explicit experiment if the disagreement is empirical
- update `model_to_beat.md` only after the disagreement is resolved by data or an explicit user decision

If both agents produce analyses of the same completed runs:
- keep one canonical experiment README
- summarize competing interpretations in the README or log entry if they matter
- do not create duplicate canonical result entries for the same run family

## Minimum Cross-Agent Compatibility Contract

Any agent should be able to pick up another agent's experiment if the following are present:
- `code/launch.py` as the canonical experiment-local launcher
- `reproduce/launch.sh`
- `reproduce/launch.json` with git commit hash and dirty status
- launcher snapshot under `reproduce/source/` when available
- `manifest.json` if anything was launched
- `launch_logs/` if launch diagnostics matter
- `results/runs/` with fetched raw artifacts for completed Modal jobs
- local summary tables / JSON copied into the experiment directory
- raw artifact paths recorded explicitly
- held-out metrics recorded explicitly
- a concise README with dataset/training/eval contracts and a handoff block

If those are missing, the experiment is not properly closed.

## Recommended Review Questions

Before launching:
- is this plan in the right agent-local area?
- is it genuinely a new experiment family or just closure work on an old one?
- will another agent understand the contract without thread context?

Before calling an experiment complete:
- are all raw artifacts harvested locally or explicitly linked?
- is the README focused on this experiment rather than the global leaderboard?
- is [experiment_log.md](./experiment_log.md) updated?
- does the stable baseline belong in `model_to_beat.md` instead of in this README?

## Short Version

- One canonical experiment log.
- One top-level directory per completed experiment family.
- Separate Codex and Claude planning spaces.
- Use the same experiment-local layout, especially `code/launch.py` and `results/runs/`.
- Keep shared input data in repo-level `data/`; keep generated experiment outputs in the experiment dir.
- Separate stable-baseline documentation from experiment READMEs.
- Use explicit handoff blocks and reproducibility bundles so either agent can continue the work.
