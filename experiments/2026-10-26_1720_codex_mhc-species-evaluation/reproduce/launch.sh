#!/bin/bash
set -euo pipefail
family=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$family/../.." && pwd)
export PYTHONPATH=/private/tmp/presto-pr-completion-20261026/pr131:/private/tmp/presto-pr-completion-20261026/dependencies
export PRESTO_MHC_SPECIES_ARTIFACT_ROOT=/Users/iskander/code/presto/artifacts/2026-10-26_1720_codex_mhc-species-evaluation
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/Users/iskander/code/shared-virtual-env/bin/python
cd "$root"
case "${1:?supply prepare, baselines, seed42, seed43, seed44, report, or verify}" in
  seed42|seed43|seed44)
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    exec "$python" "$family/code/launch.py" "$@" ;;
  prepare|baselines) exec "$python" "$family/code/launch.py" "$@" ;;
  report) shift; exec "$python" "$family/code/report.py" "$@" ;;
  verify) shift; exec "$python" "$family/code/verify.py" "$@" ;;
  *) exit 2 ;;
esac
