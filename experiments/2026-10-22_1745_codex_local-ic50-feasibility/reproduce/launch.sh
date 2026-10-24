#!/bin/bash
set -euo pipefail
# Run from the matching Presto checkout with a PYTHONPATH parent containing
# a `presto` symlink to that checkout. Original launch used the values below.
cd /private/tmp/presto-modular-model
export PYTHONPATH=/private/tmp/presto-modular-imports
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_ENABLE_MPS_FALLBACK=0
python_bin=/Users/iskander/code/shared-virtual-env/bin/python
launcher=experiments/2026-10-22_1745_codex_local-ic50-feasibility/code/launch.py
# Executed separately, sequentially, preserving each stdout/stderr log.
"$python_bin" -u "$launcher" benchmark --device cpu
"$python_bin" -u "$launcher" benchmark --device mps
# CPU selected by the frozen measured-time rule in results/device-selection.json.
"$python_bin" -u "$launcher" fit --device cpu
"$python_bin" -u experiments/2026-10-22_1745_codex_local-ic50-feasibility/analysis/verify.py
