#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PRESTO_EXPERIMENT_DIR="$EXPERIMENT_DIR"
exec /Users/iskander/code/shared-virtual-env/bin/modal run --detach \
  "$EXPERIMENT_DIR/code/runtime_tree_import_probe_v2.py"
