#!/bin/sh
set -eu
cd /private/tmp/presto-mhc2-evaluation-20261027/presto
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-27_0107_codex_mhc2-human-cell-contrast/code/fetch_sdrf.py
