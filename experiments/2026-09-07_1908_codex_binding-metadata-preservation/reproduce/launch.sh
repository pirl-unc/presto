#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python experiments/2026-09-07_1908_codex_binding-metadata-preservation/analysis/audit.py
