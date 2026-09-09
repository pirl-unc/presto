#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: 87ce95b2515a9a3be2dcdc05017bff2e1b20e856
#   branch: codex/merged-source-lineage
#   dirty: no
cd '/Users/iskander/code/presto'
export OMP_NUM_THREADS='1'
export MKL_NUM_THREADS='1'
'/Users/iskander/code/presto/artifacts/2026-09-09_1106_codex_output-coverage/.venv/bin/python' 'experiments/2026-09-09_1419_codex_merged-lineage/code/launch.py' 'after'
