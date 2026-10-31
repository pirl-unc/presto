#!/usr/bin/env bash
set -euo pipefail

# Exact local invocation. Override paths explicitly for a different checkout.
PRESTO_FAMILY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PRESTO_REPO_DIR=$(cd -- "$PRESTO_FAMILY_DIR/../.." && pwd)
PRESTO_CHECKOUT_PARENT=$(dirname -- "$PRESTO_REPO_DIR")
PRESTO_EXPERIMENT_PYTHON=${PRESTO_EXPERIMENT_PYTHON:-/Users/iskander/code/shared-virtual-env/bin/python}
PRESTO_HITLIST_OVERLAY=${PRESTO_HITLIST_OVERLAY:-/private/tmp/presto-pr-completion-20261026/dependencies}
PRESTO_IC50_SOURCE_MANIFEST=${PRESTO_IC50_SOURCE_MANIFEST:-/Users/iskander/code/presto/.worktrees/presto-modular-model/data/modular_cohorts/2026-10-21-full-ic50/manifest-exact.json}
export PRESTO_IC50_ARTIFACT_ROOT=${PRESTO_IC50_ARTIFACT_ROOT:-/Users/iskander/code/presto/artifacts/2026-10-26_1610_codex_ic50-convergence}
export PYTHONPATH="$PRESTO_CHECKOUT_PARENT:$PRESTO_HITLIST_OVERLAY"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_ENABLE_MPS_FALLBACK=0
cd -- "$PRESTO_REPO_DIR"

"$PRESTO_EXPERIMENT_PYTHON" "$PRESTO_FAMILY_DIR/code/launch.py" prepare \
  --source-manifest "$PRESTO_IC50_SOURCE_MANIFEST"
for condition in baselines seed42 seed43 seed44 permuted42; do
  "$PRESTO_EXPERIMENT_PYTHON" "$PRESTO_FAMILY_DIR/code/launch.py" "$condition" \
    > "$PRESTO_FAMILY_DIR/launch_logs/$condition.log" 2>&1
done
