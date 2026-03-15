## Workflow Orchestration

### 1. Upfront Planning
- For ANY non-trivial task (3+ steps or architectural decisions): write a detailed spec before touching code
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use planning/verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 3. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between the latest code and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 4. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 5. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Fix failing unit tests without being told how

---

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Experiment Registry

When running any experiment, benchmark, sweep, ablation, training comparison, or runtime study:

1. **Use the canonical registry**
   - Every experiment must be registered under `experiments/`
   - Do not rely on thread context, `modal_runs/`, or ad hoc notes as the primary record
   - `experiments/experiment_log.md` is the single canonical registry for completed or materially informative experiment families
   - Before running or extending experiments, read `experiments/EXPERIMENT_WORKFLOW.md`

2. **Create one experiment directory**
   - Directory format:
     - `experiments/YYYY-MM-DD_HHMM_agent_slug`
   - `agent` must identify who ran it in a tool-agnostic way, e.g.:
     - `codex`
     - `claude`
     - `codex_claude`
   - `slug` should describe the experiment family briefly

3. **Required contents**
   - Each experiment directory must contain at least:
     - `README.md`
     - `reproduce/launch.sh`
     - `reproduce/launch.json`
     - copied or generated summary tables/JSON for the experiment
     - per-example validation/test prediction dumps for eval experiments, unless the README explicitly justifies why a test split was not used
     - links or paths to raw artifacts in `modal_runs/`, `artifacts/`, or elsewhere
   - Large raw outputs may remain outside `experiments/`, but the experiment directory must point to them explicitly
   - Prefer also keeping `reproduce/source/` with a snapshot of the launcher script used for the experiment

4. **Required metadata**
   - Every experiment entry must record:
     - agent/model responsible
     - reproducibility bundle location and non-default environment overrides
     - git commit hash and dirty status
     - exact dataset and curation contract
     - pretraining and training parameters
     - tested conditions / variants
     - runtime metrics
     - evaluation metrics
     - takeaway / decision
     - artifact paths
   - For predictive eval experiments, evaluation metrics must include held-out validation and test metrics when a test split exists. At minimum, preserve enough information to recompute:
     - overall validation/test loss
     - exact-value regression metrics for the relevant primary assay family, including:
       - Spearman
       - Pearson
       - RMSE in the target space and/or `log10(nM)` where relevant
     - thresholded binding metrics for a biologically meaningful cutoff such as `<=500 nM`, including:
       - accuracy
       - balanced accuracy
       - precision
       - recall
       - F1
       - AUROC
       - AUPRC
   - If a test split is intentionally not used, the experiment README and canonical log entry must say so explicitly and explain why.

5. **Required post-run closure**
   - A launched experiment is not complete just because the Modal apps stopped
   - Every finished eval/benchmark/sweep must be closed out by:
     - extracting all available metrics from logs, summaries, and artifacts
     - copying or generating all available summary tables/JSON into the experiment directory
     - adding plots when they materially help interpret the results
     - updating the experiment directory `README.md`
     - updating `experiments/experiment_log.md`
   - The writeup must preserve:
     - reproducibility bundle and the exact launch command/env used
     - dataset/curation details
     - pretraining/training details
     - synthetic-data contract
     - loss terms and weights
     - assay-label -> output mapping
     - per-example validation/test prediction dumps or an explicit reason they were not produced
     - held-out validation/test metrics, not just probe summaries or aggregate loss
     - tested conditions with explicit condition descriptions
     - within-experiment comparisons
     - contextual conclusions relative to earlier experiments when relevant

6. **Update the unified logs**
   - Add or update entries in:
     - `experiments/experiment_log.md`
   - The markdown log is the canonical registry
   - If machine-readable summaries are needed later, generate them from experiment directories; do not hand-maintain a parallel registry

7. **Future-first behavior**
   - New launchers, sweep scripts, and experiment runners should write their summaries into the corresponding `experiments/` directory by default
   - New launchers should also freeze their invocation in the experiment directory by default so that later agents can rerun the exact same family without relying on current defaults
   - Backfilling existing experiments is acceptable, but new experiments should not require backfill to appear in the canonical registry

8. **Cross-agent compatibility**
   - These rules apply equally to Codex, Claude, or any other agent/model working in this repo
   - If multiple agents contribute to one experiment family, record that explicitly in the experiment metadata

9. **Required markdown format**
   - The experiment log should follow the richer narrative/table style already used in `experiments/experiment_log.md`
   - Each experiment entry must include:
     - title / experiment id
     - date
     - agent/model
     - experiment directory link
     - exact dataset and curation contract
      - training/pretraining contract
      - assay families used/excluded and qualifier/censor policy
      - synthetic-data contract, if any
      - loss terms and weights
      - assay-label -> output mapping
      - requested Modal GPU and actual observed GPU/memory when run on Modal
      - held-out validation metrics and held-out test metrics
      - a table or bullet list of tested conditions with metrics
      - winner / preferred condition
      - takeaway
   - Each experiment directory `README.md` should contain enough technical detail that another agent can rerun or extend the experiment without thread context

10. **Modal GPU guidance**
   - Do not leave Modal GPU selection implicit when an experiment is performance- or memory-sensitive
   - Default to `PRESTO_MODAL_GPU=H100!` for Modal experiments unless the experiment explicitly requires a different hardware baseline
   - Treat `A100` as a historical-comparison exception, not the default
   - Use `H200` only intentionally and treat it as a different hardware contract from `H100!`
   - Every Modal experiment entry must record:
     - requested GPU string
     - actual observed GPU memory / hardware evidence from logs if available
   - Do not compare A100, H100!, and H200 runs as if they were the same contract without saying so explicitly

11. **Per-agent working area**
   - Use:
     - `experiments/agents/codex/ideas.md`
     - `experiments/agents/codex/todo.md`
     - `experiments/agents/codex/plans/`
     - `experiments/agents/claude/ideas.md`
     - `experiments/agents/claude/todo.md`
     - `experiments/agents/claude/plans/`
   - `ideas.md` is for rough brainstorms
   - `todo.md` is for coarse-grained experimental backlog
   - `plans/` is for detailed experiment specs before launch
   - Completed results still belong in timestamped top-level experiment directories and `experiment_log.md`

---

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Scientific Domain Knowledge
- **Read the literature**: if some code involves scientific or biological concepts, feel free to search for review papers and read those before changing code that expresses scientific concepts. 
- **Flag inconsistencies**: if code expresses a scientific model that's at odds with your understanding, note that inconsistency and ask for clarification. 
