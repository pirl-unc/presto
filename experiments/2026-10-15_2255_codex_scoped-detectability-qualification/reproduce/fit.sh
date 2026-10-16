#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/2026-10-15_2255_codex_scoped-detectability-qualification/code/fit.py
