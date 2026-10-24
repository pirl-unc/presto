#!/bin/sh
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-23_1257_codex_digestion-observation-effects/code/layers.py
/Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-23_1257_codex_digestion-observation-effects/code/analyze.py
