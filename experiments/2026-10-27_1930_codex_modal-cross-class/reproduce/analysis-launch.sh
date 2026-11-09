#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/private/tmp/presto-benchmark-matplotlib
PYTHON=${PRESTO_ANALYSIS_PYTHON:-/Users/iskander/code/shared-virtual-env/bin/python}
"$PYTHON" experiments/2026-10-27_1930_codex_modal-cross-class/analysis/verify.py
"$PYTHON" experiments/2026-10-27_1930_codex_modal-cross-class/analysis/compare.py
