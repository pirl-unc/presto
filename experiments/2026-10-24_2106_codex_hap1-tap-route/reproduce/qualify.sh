#!/bin/sh
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_2106_codex_hap1-tap-route/code/qualify.py
