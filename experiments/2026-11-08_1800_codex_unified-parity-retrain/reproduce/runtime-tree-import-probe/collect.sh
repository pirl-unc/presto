#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PRESTO_EXPERIMENT_DIR="$EXPERIMENT_DIR"
exec /Users/iskander/code/shared-virtual-env/bin/python \
  "$EXPERIMENT_DIR/reproduce/runtime-tree-import-probe/source/collect.py"
