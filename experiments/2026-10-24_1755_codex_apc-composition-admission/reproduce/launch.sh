#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export PYTHONPATH=/private/tmp/hitlist-release-2026
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_1755_codex_apc-composition-admission/code/inventory.py
