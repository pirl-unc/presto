#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export PYTHONPATH=/private/tmp/hitlist-native-fix:/private/tmp/presto-native-import
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec /Users/iskander/code/shared-virtual-env/bin/python \
  experiments/2026-10-24_0449_codex_apc-precursor-response/code/"$1".py
