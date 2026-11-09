#!/bin/sh
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
FAMILY_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec /Users/iskander/code/shared-virtual-env/bin/python "$FAMILY_DIR/code/launch.py"
