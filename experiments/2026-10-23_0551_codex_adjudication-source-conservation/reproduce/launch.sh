#!/bin/bash
set -euo pipefail
FAMILY=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=/private/tmp/hitlist-successor-compat
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
/Users/iskander/code/shared-virtual-env/bin/python "$FAMILY/code/launch.py"
/Users/iskander/code/shared-virtual-env/bin/python "$FAMILY/code/verify.py"
