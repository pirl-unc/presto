#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$(dirname "$PWD")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-27_2235_codex_shared-support-audit/code/audit.py
