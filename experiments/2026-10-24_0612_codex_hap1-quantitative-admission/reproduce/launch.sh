#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/private/tmp/hitlist-dc798:/private/tmp/presto-native-import
exec /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_0612_codex_hap1-quantitative-admission/code/launch.py "$@"
