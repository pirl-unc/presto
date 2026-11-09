#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export PYTHONPATH=/private/tmp/hitlist-goldberg-composition-20261024
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_2244_codex_mapp2025-baseline-prediction/code/launch.py "$@"
