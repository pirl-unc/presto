#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: 27cde9717ae7e59997581e4ae633e0cd644a7faa
#   branch: feat/scoped-proteomics-consumer
#   dirty: yes
cd '/private/tmp/presto-scoped-proteomics'
export PYTHONPATH='/private/tmp/presto-scoped-import:/private/tmp/hitlist175-wheel-import'
export OMP_NUM_THREADS='1'
export MKL_NUM_THREADS='1'
'/Users/iskander/code/shared-virtual-env/bin/python' 'experiments/2026-10-15_2100_codex_scoped-proteomics-admission/code/launch.py'
