# PF07 Local Resume + Apple Silicon Support

## Goal

Make the active PF07 all-class-I cleanup-validation rerun resumable on another machine without Modal, and add a clean local Apple Silicon launch path that preserves the existing experiment contract and artifact layout.

## Why This Is Needed

- The active family `2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun` is blocked by exhausted Modal credits.
- The experiment directory already contains the canonical launcher, manifest, and repro bundle, but the launcher still assumes:
  - `modal run --detach`
  - remote warm-start checkpoint paths under `/checkpoints/...`
- The focused affinity runner still auto-selects only `cuda` vs `cpu`, so local Apple Silicon (`mps`) is not explicitly supported.

## Fixed Contract

- Same experiment family:
  - `pf07_control_constant`
  - `pf07_dag_method_leaf_constant`
  - `pf07_dag_prep_readout_leaf_constant`
  - each at `10 / 25 / 50` epochs
- Same dataset contract:
  - `data/merged_deduped.tsv`
  - source filter `iedb`
  - all class-I numeric rows
  - peptide-group `80/10/10`
  - split seed `42`
- Same train seed `43`
- Same probe panel:
  - alleles: `HLA-A*02:01`, `HLA-A*24:02`
  - peptides: `SLLQHLIGL`, `FLRYLLFGI`, `NFLIKFLLI`, `IMLEGETKL`
- Same 1-epoch MHC pretrain warm start, but local resume must use the checked-in local checkpoint path rather than the Modal volume path.

## Required Changes

### 1. Focused Trainer Device Selection

Add explicit runtime device selection to `scripts/focused_binding_probe.py`:

- New CLI arg: `--device {auto,cpu,cuda,mps}`
- `auto` resolution:
  - `cuda` if available
  - else `mps` if available
  - else `cpu`
- Explicit unavailable-device requests should fail fast with a clear error.

Also make runtime defaults safe for non-CUDA execution:

- disable `pin_memory` automatically for non-CUDA devices
- keep CUDA-only telemetry / TF32 logic gated to actual CUDA runs
- preserve the current artifact contract and summary fields

### 2. Local Backend In Experiment Launcher

Extend `experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun/code/launch.py`:

- Add `--backend {modal,local}`
- Add `--device {auto,cpu,cuda,mps}` for local execution
- Add explicit local warm-start checkpoint resolution pointing at:
  - `experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt`
- For local backend:
  - run `python -m presto.scripts.focused_binding_probe`
  - write outputs to `results/runs/<run_id>`
  - skip runs whose required local artifacts already exist
  - aggregate summaries locally after successful runs when possible

Do not change the Modal backend semantics beyond recording better resume metadata.

### 3. Resume Documentation

Update the active experiment README and repro notes so another machine can:

- understand the blocked Modal state
- see where the local warm-start checkpoint lives
- resume locally with:
  - CPU
  - Apple Silicon `mps`
- preserve the same experiment-local artifact structure

## Verification

- targeted tests for device resolution / non-CUDA runtime behavior
- `py_compile` for modified launcher / trainer
- launcher dry-run for:
  - Modal backend
  - local backend
- local command emission must show:
  - same condition matrix
  - local `results/runs/<run_id>` output path
  - local warm-start checkpoint path

## Non-Goals

- No new experiment family
- No metric claims about same/better yet
- No architecture changes
- No broad refactor of every training script to support `mps`; this pass is scoped to the active focused PF07 path
