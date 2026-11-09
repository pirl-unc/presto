#!/bin/bash
set -euo pipefail
family=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$family/../.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/Users/iskander/code/shared-virtual-env/bin/python
cd "$root"
"$python" "$family/code/launch.py"
"$python" "$family/code/verify.py"
