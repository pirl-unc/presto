#!/bin/sh
set -eu
TASK_FAMILY=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/private/tmp/presto-modular-imports
export MPLCONFIGDIR=/private/tmp/presto-boundary-matplotlib
exec /Users/iskander/code/shared-virtual-env/bin/python "$TASK_FAMILY/code/report_fit.py"
