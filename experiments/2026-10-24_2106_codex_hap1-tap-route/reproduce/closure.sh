#!/bin/sh
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/private/tmp/presto-mpl-2106
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_2106_codex_hap1-tap-route/analysis/verify.py
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_2106_codex_hap1-tap-route/analysis/boundary_check.py
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_2106_codex_hap1-tap-route/analysis/report.py
