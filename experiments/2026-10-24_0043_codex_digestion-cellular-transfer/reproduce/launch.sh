#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/private/tmp/presto-boundary-matplotlib
exec /Users/iskander/code/shared-virtual-env/bin/python code/launch.py
