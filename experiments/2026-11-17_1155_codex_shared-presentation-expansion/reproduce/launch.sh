#!/usr/bin/env bash
set -euo pipefail
: "${PRESTO_MODAL_GPU:=H100!}"
: "${MODAL_BIN:=/Users/iskander/code/shared-virtual-env/bin/modal}"
export PRESTO_MODAL_GPU
EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "$MODAL_BIN" run --detach "$EXP_DIR/code/launch.py" "$@"
