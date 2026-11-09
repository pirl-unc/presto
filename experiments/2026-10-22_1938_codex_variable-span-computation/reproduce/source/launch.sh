#!/bin/bash
set -euo pipefail
cd /private/tmp/presto-modular-model
export PYTHONPATH=/private/tmp/presto-modular-imports
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
/Users/iskander/code/shared-virtual-env/bin/python -u experiments/2026-10-22_1938_codex_variable-span-computation/code/launch.py
