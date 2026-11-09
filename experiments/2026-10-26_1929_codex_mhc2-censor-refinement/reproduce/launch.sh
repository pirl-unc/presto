#!/bin/sh
set -eu
export PYTHONPATH=/private/tmp/presto-mhc2-20261026:/private/tmp/presto-pr-completion-20261026/dependencies
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_ENABLE_MPS_FALLBACK=0
TASK_PYTHON=/Users/iskander/code/shared-virtual-env/bin/python
TASK_FAMILY=experiments/2026-10-26_1929_codex_mhc2-censor-refinement
"$TASK_PYTHON" "$TASK_FAMILY/code/launch.py" prepare
"$TASK_PYTHON" "$TASK_FAMILY/code/continue.py"
