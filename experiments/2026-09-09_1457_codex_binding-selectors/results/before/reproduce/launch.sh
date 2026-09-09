#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: e6d6b607d95e801637a6f92be45ea3c2d866bd31
#   branch: codex/merged-binding-descriptors
#   dirty: no
cd '/private/tmp/presto-selector-repair/presto'
export PYTHONPATH='/private/tmp/presto-selector-repair'
export OMP_NUM_THREADS='1'
export MKL_NUM_THREADS='1'
'/Users/iskander/code/presto/artifacts/2026-09-09_1106_codex_output-coverage/.venv/bin/python' 'experiments/2026-09-09_1457_codex_binding-selectors/code/launch.py' 'before'
