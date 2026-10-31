#!/bin/sh
set -eu
cd /private/tmp/presto-mhc2-20261026/presto
export PYTHONPATH=/private/tmp/presto-mhc2-20261026:/private/tmp/presto-pr-completion-20261026/dependencies
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_ENABLE_MPS_FALLBACK=0
exec /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-26_2036_codex_mhc2-ms-length-background/code/continue_native.py
