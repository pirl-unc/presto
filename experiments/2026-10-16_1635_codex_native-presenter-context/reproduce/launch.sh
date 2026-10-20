#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec python code/audit_v2.py
