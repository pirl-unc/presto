#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/Users/iskander/code/shared-virtual-env/bin/python -u experiments/2026-10-24_2041_codex_hap1-affinity-control/analysis/verify.py
MPLCONFIGDIR=/private/tmp/presto-mpl-2041 /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_2041_codex_hap1-affinity-control/analysis/report.py
