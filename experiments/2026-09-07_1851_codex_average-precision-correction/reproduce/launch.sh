#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python experiments/2026-09-07_1851_codex_average-precision-correction/analysis/recompute.py
