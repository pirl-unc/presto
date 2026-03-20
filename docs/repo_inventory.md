# Repo Inventory

This document explains what is intentionally committed in this repo, what is intentionally left local-only, and what should be reviewed later if repo size or clarity becomes a problem.

## Current Contract

Treat the repo as having three storage tiers:

1. `Committed and canonical`
   - source code, tests, docs, compact reference data, and experiment history needed to understand how Presto evolved
2. `Local and ignored, but reproducible or regenerable`
   - raw Modal pulls, ad hoc analysis artifacts, large extracted datasets, caches, and launcher logs
3. `Review later`
   - committed material that is useful now, but may eventually deserve consolidation or relocation if the repo becomes too heavy

## Committed and Canonical

### Source and Tests

Keep committed:
- `presto/`, `models/`, `data/*.py`, `scripts/`, `inference/`
- `tests/`
- root/project metadata such as `pyproject.toml`

This is the executable codebase and must remain reviewable and portable across machines.

### Core Documentation

Keep committed:
- `README.md`
- `docs/`
- `TODO.md`
- `tasks/`
- `experiments/README.md`
- `experiments/EXPERIMENT_WORKFLOW.md`
- `experiments/AGENT_COORDINATION.md`
- `experiments/experiment_log.md`
- `experiments/model_to_beat.md`
- `experiments/agents/**`

These files define the current architecture contract, workflow rules, and the reasoning trail behind experimental choices.

### Compact Reference Data

Keep committed when the artifact is both useful and reasonably small.

Examples currently committed:
- IMGT/IPD reference FASTA and metadata under `data/imgt/` and `data/ipd_mhc/`
- compact IEDB zip bundles under `data/iedb/`
- VDJdb compact artifacts under `data/vdjdb/`
- McPAS, STCRDab, 10x, and small overlay/reference files
- `data/manifest.json`
- compact derived summary artifacts such as `data/merged_deduped_funnel.tsv` and `data/merged_deduped_funnel.png`

Rule of thumb:
- small, reusable, cross-machine reference artifacts belong in git
- huge extracted tables do not

### Experiment History

Keep committed:
- timestamped experiment directories under `experiments/YYYY-MM-DD_HHMM_agent_slug/`
- experiment READMEs
- reproduce bundles
- experiment-local launchers and analysis scripts
- summary CSV/JSON tables
- final plots that materially help interpretation
- per-run summary JSON and prediction dumps when they are part of the durable evaluation record

This is the scientific history of how Presto is being shaped. It is intentionally in-repo.

## Local and Ignored

These are intentionally not part of normal git history.

### Raw and Disposable Artifact Roots

Ignored roots:
- `artifacts/`
- `modal_runs/`
- root `modal_train_result*.json`

These are useful for local debugging and recovery, but they are not the canonical record.

### Large Derived Datasets

Ignored examples:
- `data/merged_deduped.tsv`
- large backup copies like `data/merged_deduped.pre_*.tsv`
- large extracted raw datasets such as:
  - `data/iedb/*.csv`
  - `data/vdjdb/vdjdb.txt`
  - `data/vdjdb/vdjdb.scored.txt`
  - `data/vdjdb/vdjdb_full*.txt`

These are too large or too regenerable to be a good default fit for git.

### Transient Per-Experiment Local Noise

Ignored examples:
- `experiments/*/launch_logs/`
- `experiments/*/__pycache__/`
- `experiments/*/analysis/__pycache__/`

The canonical experiment record is the README plus the saved summaries/results, not the raw launcher stderr/stdout.

### Standard Local Build / Cache State

Ignored examples:
- `.venv/`
- `.pytest_cache/`
- `.ruff_cache/`
- `__pycache__/`
- `build/`

## Review-Later Candidates

These are committed now because they are useful, but they are the first things to revisit if the repo becomes unwieldy.

### 1. Experiment Plot Density

Some experiment directories contain many PNGs and per-run outputs. These are useful now, but if repo weight becomes a real problem, the likely first reduction is:
- keep experiment README + summary tables + final key plots
- move very dense figure sets to release artifacts or another artifact store

### 2. Reproduce Snapshots

Many experiment directories keep a frozen `reproduce/source/launch.py`. This is good for reproducibility, but there is duplication.

Keep it for now.
Review later only if:
- repo size becomes painful, and
- the launcher snapshots are demonstrably identical to a canonical shared launcher

### 3. Compact Upstream Bundles

Some zipped upstream bundles are committed because they are compact enough and useful across machines. If the data layer grows a lot more, we may want a stricter policy such as:
- keep only provenance manifests and retrieval scripts in git
- fetch compact bundles on demand

That is not the current policy.

## Practical Keep / Regenerate / Ignore Rules

When deciding whether to keep a new file:

- Keep it in git if it is:
  - source code
  - tests
  - core docs
  - compact reusable reference data
  - experiment history needed to understand or reproduce results

- Keep it local-only if it is:
  - a raw Modal pull
  - a launcher log
  - a cache
  - a one-off scratch artifact
  - a huge derived dataset that can be rebuilt

- Ask for a retention decision if it is:
  - large but not obviously regenerable
  - duplicated across many experiments
  - expensive to recompute yet not clearly canonical

## Current Reality Check

As of the current cleanup pass:
- the tracked worktree is clean
- the durable source/data/history slice is committed
- ignored local bulk still exists on disk for safety

So the repo is now understandable enough to continue work on another machine, and future cleanup can be done against this document instead of rediscovering the boundaries from `git status`.

## Operational Checklist

Use this section when you are deciding what can be deleted locally versus what should stay around until you have recreated it elsewhere.

### Safe To Delete Locally Now

Delete these if you need disk space and do not need the raw local scratch state:
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `build/`, local virtualenvs
- `experiments/*/launch_logs/`
- root `modal_train_result*.json`
- `artifacts/` when the corresponding experiment summaries and plots are already committed under `experiments/`
- `modal_runs/` when the corresponding experiment has already been harvested into `experiments/` and no further raw recovery is needed
- large extracted raw tables that are ignored and reproducible, such as:
  - `data/iedb/*.csv`
  - `data/vdjdb/vdjdb.txt`
  - `data/vdjdb/vdjdb.scored.txt`
  - `data/vdjdb/vdjdb_full*.txt`

### Keep Locally Until Regenerated Elsewhere

Keep these on disk until you have confirmed another machine or workflow can recreate them:
- `data/merged_deduped.tsv`
- any large merged/intermediate dataset that is currently the active default input for training
- `modal_runs/` for experiments that are still open, partially collected, or likely to need raw-log debugging
- `artifacts/` for analyses that have not yet been promoted into a committed experiment directory
- any compact upstream bundle you rely on if you are unsure the download path or provenance manifest is complete

### Keep Committed; Do Not Treat As Disposable

Do not delete these from git history as part of normal local cleanup:
- source code under `data/*.py`, `models/`, `scripts/`, `inference/`
- tests
- `docs/`
- `experiments/` history and coordination docs
- compact reference data already committed under `data/`

### Quick Decision Rule

If a file answers “what did we run?” or “how does the code work?”, it probably belongs in git.

If a file answers only “what raw temporary byproducts happened on this machine?”, it is probably safe to keep local-only or delete after closure.

### Useful Commands

Check clean tracked state:

```bash
git status --short
```

Check ignored local bulk that is still present on disk:

```bash
git status --ignored --short
```

Check disk usage before deleting local-only bulk:

```bash
du -sh artifacts modal_runs data/merged_deduped.tsv
```
