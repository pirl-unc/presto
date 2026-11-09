#!/bin/sh
set -eu
TASK_FAMILY=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$TASK_FAMILY/code"
exec /Users/iskander/code/shared-virtual-env/bin/python -m pytest -q "$TASK_FAMILY/code/test_actual.py"
