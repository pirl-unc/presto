# MPS-Safe Smoke Rerun

## Goal

Validate the new MPS-safe training workaround in a fresh registered experiment family.

## Contract

- Same tiny focused PF07 smoke contract as `2026-03-19_1007_codex_cpu-vs-mps-focused-pf07-smoke`
- Same warm-start checkpoint
- Same data cap, seeds, and probe panel
- Same two devices:
  - `cpu`
  - `mps`
- New runtime difference:
  - `focused_binding_probe.py` now applies MPS-safe zero-dropout stabilization automatically on `device == "mps"`

## Success Criterion

- `mps` no longer diverges on epoch `1`
- experiment produces normal local summaries under `results/runs/`
- CPU and MPS both complete so we can compare held-out metrics and decide whether MPS is usable for local continuation
