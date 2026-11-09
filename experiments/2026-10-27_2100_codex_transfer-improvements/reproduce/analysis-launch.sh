#!/bin/sh
set -eu
cd "$(dirname "$0")/../analysis"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/private/tmp/presto-benchmark-matplotlib
PYTHON=/Users/iskander/code/shared-virtual-env/bin/python
if [ ! -f ../results/validation-decision.json ]; then "$PYTHON" selection.py; fi
"$PYTHON" verify.py
"$PYTHON" exposure.py
"$PYTHON" compare.py
"$PYTHON" historical_compare.py
