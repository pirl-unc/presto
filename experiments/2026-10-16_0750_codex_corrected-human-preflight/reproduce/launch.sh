#!/usr/bin/env bash
set -euo pipefail
family=experiments/2026-10-16_0750_codex_corrected-human-preflight
test -z "$(git status --porcelain)"
export PRESTO_QUALIFICATION_WHEEL="$(python "$family/code/fetch_wheel.py")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
modal run "$family/code/launch.py"
