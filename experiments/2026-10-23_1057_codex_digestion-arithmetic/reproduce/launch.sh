#!/bin/sh
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /private/tmp/presto-modular-model/experiments/2026-10-23_1057_codex_digestion-arithmetic
[ ! -e results/summary.json ] || { echo "Refusing existing results" >&2; exit 1; }
/Users/iskander/code/shared-virtual-env/bin/python -u code/replay.py
