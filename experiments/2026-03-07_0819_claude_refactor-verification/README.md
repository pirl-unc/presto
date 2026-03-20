# Refactoring Behavior Verification

**EXP ID**: EXP-28
**Date**: 2026-03-07
**Agent**: Claude Code (claude-opus-4-6)

## Overview

Verification runs after code refactoring to ensure training behavior was preserved across commits.

## Dataset & Training

1 epoch each on full profile, batch 128. d_model=128, n_layers=2, n_heads=4. Each run tagged with a git commit hash.

## Source Modal Runs

- `modal_runs/refactor-e1-57725c3a/`
- `modal_runs/refactor-e1-cc80425a/`
- `modal_runs/refactor-e1-d428f09a/`
- `modal_runs/refactor-e1-dabcd3da/`

## Conditions

| label |
| --- |
| refactor-e1-57725c3a |
| refactor-e1-cc80425a |
| refactor-e1-d428f09a |
| refactor-e1-dabcd3da |

## Plots

## Artifacts

- Condition summary: `results/condition_summary.csv`
- Epoch summary: `results/epoch_summary.csv`
- Probe predictions: `results/final_probe_predictions.csv`
- Reproduce: `reproduce/launch.json`
