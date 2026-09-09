#!/usr/bin/env bash
set -euo pipefail
cd /Users/iskander/code/presto
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HITLIST_DATA_DIR=/Users/iskander/.hitlist
exec artifacts/2026-09-09_1106_codex_output-coverage/.venv/bin/python \
  experiments/2026-09-09_1106_codex_output-coverage/code/launch.py route_trace "$@"
