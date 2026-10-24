#!/usr/bin/env bash
set -euo pipefail
cd /private/tmp/presto-modular-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/private/tmp/presto-native-import
python_bin=/Users/iskander/code/shared-virtual-env/bin/python
experiment=experiments/2026-10-24_1007_codex_erap1-length-response
"$python_bin" "$experiment/code/launch.py" qualify
"$python_bin" "$experiment/code/launch.py" train
"$python_bin" "$experiment/code/verify.py"
