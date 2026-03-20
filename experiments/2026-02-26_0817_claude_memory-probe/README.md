# GPU Memory OOM Boundary Exploration

**EXP ID**: EXP-33
**Date**: 2026-02-26
**Agent**: Claude Code (claude-opus-4-6)

## Overview

Batch-size memory profiling to find the OOM boundary. Tested batch sizes from 64 to 192.

## Dataset & Training

Memory profiling only (no training metrics). Tested batch sizes: 64, 96, 112, 128, 160, 176, 180, 192.

## Source Modal Runs

- `modal_runs/memory_probe_20260226/`

## Plots

## Artifacts

- Condition summary: `results/condition_summary.csv`
- Epoch summary: `results/epoch_summary.csv`
- Probe predictions: `results/final_probe_predictions.csv`
- Reproduce: `reproduce/launch.json`
