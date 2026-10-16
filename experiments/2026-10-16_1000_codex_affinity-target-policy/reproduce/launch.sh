#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$(dirname "$0")/../code/audit.py" "$@"
