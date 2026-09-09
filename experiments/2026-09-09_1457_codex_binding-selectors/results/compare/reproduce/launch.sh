#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: c6b0eebfc552a7e7cd22dbfcc1b3dfbc5f533db7
#   branch: codex/merged-binding-descriptors
#   dirty: no
cd '/private/tmp/presto-selector-repair/presto'
export PYTHONPATH='/private/tmp/presto-selector-repair'
export OMP_NUM_THREADS='1'
export MKL_NUM_THREADS='1'
'/Users/iskander/code/presto/artifacts/2026-09-09_1106_codex_output-coverage/.venv/bin/python' 'experiments/2026-09-09_1457_codex_binding-selectors/analysis/compare.py'
