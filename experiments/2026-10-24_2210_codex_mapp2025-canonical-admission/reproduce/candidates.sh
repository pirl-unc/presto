#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export PYTHONPATH=/private/tmp/hitlist-mapp2025-20261024
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec /Users/iskander/code/shared-virtual-env/bin/python -c 'import subprocess; subprocess.run(["/Users/iskander/code/shared-virtual-env/bin/python", "experiments/2026-10-24_2210_codex_mapp2025-canonical-admission/code/candidates.py"], check=True, timeout=1200)'
