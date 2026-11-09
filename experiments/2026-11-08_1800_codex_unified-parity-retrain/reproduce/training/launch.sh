#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PRESTO_EXPERIMENT_DIR="$EXPERIMENT_DIR"
export PRESTO_MODAL_GPU=H100!
exec /Users/iskander/code/shared-virtual-env/bin/modal run --detach \
  "$EXPERIMENT_DIR/code/fit_pair.py"
