#!/usr/bin/env bash
set -euo pipefail
family=$(cd "$(dirname "$0")/.." && pwd)
repository=$(cd "$family/../.." && pwd)
import_root=$(mktemp -d "${TMPDIR:-/tmp}/presto-allocation-import.XXXXXX")
trap 'rm -rf "$import_root"' EXIT
ln -s "$repository" "$import_root/presto"
export PYTHONPATH="$import_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$family/code/audit.py" "$@"
