#!/usr/bin/env bash
set -euo pipefail
experiment_output=${1:?fresh experiment output directory required}
cohort_output=${2:?fresh cohort directory required}
repo_root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$repo_root"
export PYTHONPATH=/private/tmp/presto-modular-imports
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
/Users/iskander/code/shared-virtual-env/bin/python \
  experiments/2026-10-21_0647_codex_full-ic50-admission/code/launch.py \
  --output "$experiment_output" --cohort "$cohort_output"
