#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
PYTHONPATH=/private/tmp/hitlist-lymph-20261024 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_1903_codex_lymph-canonical-admission/code/launch.py
