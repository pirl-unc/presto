#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
/Users/iskander/code/shared-virtual-env/bin/python -u experiments/2026-10-24_2018_codex_hap1-context-ablation/code/launch.py
