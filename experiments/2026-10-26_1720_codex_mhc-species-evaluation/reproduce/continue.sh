#!/bin/bash
set -euo pipefail
family=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=/private/tmp/presto-pr-completion-20261026/pr131:/private/tmp/presto-pr-completion-20261026/dependencies
export PRESTO_MHC_SPECIES_ARTIFACT_ROOT=/Users/iskander/code/presto/artifacts/2026-10-26_1720_codex_mhc-species-evaluation
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /Users/iskander/code/shared-virtual-env/bin/python "$family/code/continue.py"
