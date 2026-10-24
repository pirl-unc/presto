#!/usr/bin/env bash
set -euo pipefail
family="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${family}/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
python_bin="${PRESTO_EXPERIMENT_PYTHON:-/Users/iskander/code/shared-virtual-env/bin/python}"
output="${PRESTO_FACTORIAL_OUTPUT:-${family}/results/run}"
"${python_bin}" "${family}/reproduce/source/launch.py" run --output "${output}"
