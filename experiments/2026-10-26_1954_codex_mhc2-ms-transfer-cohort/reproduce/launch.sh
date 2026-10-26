#!/bin/sh
set -eu
export PYTHONPATH=/private/tmp/presto-mhc2-20261026:/private/tmp/presto-pr-completion-20261026/dependencies
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_PYTHON=/Users/iskander/code/shared-virtual-env/bin/python
TASK_FAMILY=experiments/2026-10-26_1954_codex_mhc2-ms-transfer-cohort
"$TASK_PYTHON" "$TASK_FAMILY/code/inventory.py"
