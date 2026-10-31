#!/usr/bin/env bash
set -euo pipefail

# Continue this existing attempt; refuses retries or overwritten attempt logs.
PRESTO_FAMILY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PRESTO_REPO_DIR=$(cd -- "$PRESTO_FAMILY_DIR/../.." && pwd)
export PYTHONPATH="$(dirname -- "$PRESTO_REPO_DIR"):/private/tmp/presto-pr-completion-20261026/dependencies"
export PRESTO_IC50_ARTIFACT_ROOT=/Users/iskander/code/presto/artifacts/2026-10-26_1610_codex_ic50-convergence
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_ENABLE_MPS_FALLBACK=0
cd -- "$PRESTO_REPO_DIR"
exec /Users/iskander/code/shared-virtual-env/bin/python "$PRESTO_FAMILY_DIR/code/continue.py" \
  --wait-receipt "$PRESTO_FAMILY_DIR/reproduce/seed42-1793031157825971000.json" \
  --pid 12942
