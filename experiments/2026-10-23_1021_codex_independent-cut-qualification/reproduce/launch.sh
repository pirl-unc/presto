#!/bin/sh
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /private/tmp/presto-modular-model/experiments/2026-10-23_1021_codex_independent-cut-qualification
/Users/iskander/code/shared-virtual-env/bin/python -u code/run.py
