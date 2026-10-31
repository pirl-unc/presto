#!/bin/sh
set -eu
export PYTHONPATH=/private/tmp/hitlist-digestion-source
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/opt/homebrew/bin:/Users/iskander/code/shared-virtual-env/bin:/usr/bin:/bin
cd /private/tmp/presto-modular-model/experiments/2026-10-23_1018_codex_shared-digestion-source
/Users/iskander/code/shared-virtual-env/bin/python -u code/admit.py
