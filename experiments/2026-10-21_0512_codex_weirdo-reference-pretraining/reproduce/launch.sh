#!/usr/bin/env bash
set -euo pipefail
# Run from the frozen fit-source commit; use fresh output and cohort directories.
experiment_output=${1:?provide a fresh experiment output directory}
cohort_output=${2:?provide a fresh shared cohort directory}
repo_root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$repo_root"
export PYTHONPATH=/private/tmp/presto-modular-imports
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python_bin=/Users/iskander/code/shared-virtual-env/bin/python
"$python_bin" experiments/2026-10-21_0512_codex_weirdo-reference-pretraining/code/prepare.py \
  --output "$experiment_output" --cohort "$cohort_output" --reconstruct-missing-groups
"$python_bin" experiments/2026-10-21_0512_codex_weirdo-reference-pretraining/code/launch.py \
  --output "$experiment_output" --manifest "$cohort_output/manifest.json" \
  --preparation "$experiment_output"
