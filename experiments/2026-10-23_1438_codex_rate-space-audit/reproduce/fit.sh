#!/bin/sh
set -eu
TASK_FAMILY=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/private/tmp/presto-modular-imports
exec /Users/iskander/code/shared-virtual-env/bin/python "$TASK_FAMILY/code/fit_direct.py"
