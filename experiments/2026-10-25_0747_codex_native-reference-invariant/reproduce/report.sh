#!/bin/sh
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
exec /Users/iskander/code/shared-virtual-env/bin/python -B "/Users/iskander/code/presto/.worktrees/presto-modular-model/experiments/2026-10-25_0747_codex_native-reference-invariant/analysis/report.py"
