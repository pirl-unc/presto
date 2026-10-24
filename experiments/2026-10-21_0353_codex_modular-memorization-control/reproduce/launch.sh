#!/bin/bash
set -euo pipefail
# A fresh destination preserves the original run. Use the git commit recorded in
# launch.json and the exact input pins from source-pins.json / code/launch.py.
experiment_output=${1:?provide a fresh experiment output directory}
cohort_output=${2:?provide a fresh shared cohort directory}
repo_root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$repo_root"
export PYTHONPATH=/private/tmp/presto-modular-imports
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python_bin=/Users/iskander/code/shared-virtual-env/bin/python
launcher=experiments/2026-10-21_0353_codex_modular-memorization-control/code/launch.py
"$python_bin" "$launcher" prepare --cohort "$cohort_output" --output "$experiment_output"
"$python_bin" "$launcher" fit --cohort "$cohort_output" --output "$experiment_output"
