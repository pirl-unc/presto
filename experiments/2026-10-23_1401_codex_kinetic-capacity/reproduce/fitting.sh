#!/bin/sh
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/private/tmp/presto-modular-imports
/Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-23_1401_codex_kinetic-capacity/code/fit_capacity.py
