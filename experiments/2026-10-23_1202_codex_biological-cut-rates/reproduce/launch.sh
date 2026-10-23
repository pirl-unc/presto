#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_1202_codex_biological-cut-rates/code/check_math.py
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_1202_codex_biological-cut-rates/code/fit.py
