#!/bin/sh
set -eu
cd /private/tmp/presto-mhc2-evaluation-20261027/presto
export PYTHONPATH=/private/tmp/presto-mhc2-evaluation-20261027:/private/tmp/presto-pr-completion-20261026/dependencies
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0
exec /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-26_2036_codex_mhc2-ms-length-background/code/verify_controls.py
