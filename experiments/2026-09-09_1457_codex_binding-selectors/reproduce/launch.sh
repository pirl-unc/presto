#!/usr/bin/env bash
set -euo pipefail
cd /private/tmp/presto-selector-repair/presto
export PYTHONPATH=/private/tmp/presto-selector-repair
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec /Users/iskander/code/presto/artifacts/2026-09-09_1106_codex_output-coverage/.venv/bin/python \
  experiments/2026-09-09_1457_codex_binding-selectors/code/launch.py "$@"
