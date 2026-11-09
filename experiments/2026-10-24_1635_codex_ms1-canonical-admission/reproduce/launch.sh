#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export PYTHONPATH=/private/tmp/hitlist-proteometools-ms1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_1635_codex_ms1-canonical-admission/code/launch.py
