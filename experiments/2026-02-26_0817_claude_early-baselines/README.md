# Early Foundation Training Runs

**EXP ID**: EXP-32
**Date**: 2026-02-26
**Agent**: Claude Code (claude-opus-4-6)

## Overview

Early full-scale and probe-tracking training runs that established baseline behavior before the diagnostic/groove refactoring.

## Dataset & Training

Full profile, various batch sizes. d_model=128, n_layers=2, n_heads=4. Limited metrics data (empty metrics files in some runs).

## Source Modal Runs

- `modal_runs/full-bs128-fastprobe-perflive-20260228/`
- `modal_runs/unified-probe10-20260226b/`

## Conditions

| label |
| --- |
| full-bs128-fastprobe-perflive-20260228 |
| unified-probe10-20260226b |

## Plots

## Artifacts

- Condition summary: `results/condition_summary.csv`
- Epoch summary: `results/epoch_summary.csv`
- Probe predictions: `results/final_probe_predictions.csv`
- Reproduce: `reproduce/launch.json`
