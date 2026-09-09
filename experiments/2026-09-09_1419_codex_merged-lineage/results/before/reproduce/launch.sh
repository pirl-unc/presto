#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: 15f28552a54ca2fb1c7a508ab7270db4a8ff4717
#   branch: codex/merged-source-lineage
#   dirty: no
cd '/Users/iskander/code/presto'
export OMP_NUM_THREADS='1'
export MKL_NUM_THREADS='1'
'/Users/iskander/code/presto/artifacts/2026-09-09_1106_codex_output-coverage/.venv/bin/python' 'experiments/2026-09-09_1419_codex_merged-lineage/code/launch.py' 'before'
