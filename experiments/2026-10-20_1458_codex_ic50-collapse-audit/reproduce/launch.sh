#!/bin/sh
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/2026-10-20_1458_codex_ic50-collapse-audit/analysis/audit.py
