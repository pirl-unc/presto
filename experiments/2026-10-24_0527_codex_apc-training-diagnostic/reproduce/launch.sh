#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/private/tmp/presto-native-import
exec /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_0527_codex_apc-training-diagnostic/code/launch.py "$@"
