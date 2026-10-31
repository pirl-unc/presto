#!/usr/bin/env bash
set -euo pipefail
# Run in the original worktree or a separately restored copy with pinned inputs.
# Qualification/training refuse existing output directories; never overwrite an attempt.
# Source commit 5699b9f4e73a31c65eddc414f5d6a41f2fae24d2 plus frozen new files.
cd /private/tmp/presto-modular-model
export PYTHONPATH=/private/tmp/presto-native-import
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/private/tmp/presto-erap1-matplotlib
python_bin=/Users/iskander/code/shared-virtual-env/bin/python
experiment=experiments/2026-10-24_0810_codex_erap1-trimming
"$python_bin" "$experiment/code/launch.py" qualify
"$python_bin" "$experiment/code/launch.py" train
"$python_bin" "$experiment/code/verify.py"
"$python_bin" "$experiment/code/report.py"
