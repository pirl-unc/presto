#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/private/tmp/hitlist-digestion-recipe
exec /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_1123_codex_digestion-timecourse-admission/code/admit.py
