#!/bin/sh
set -eu
EXPERIMENT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec /Users/iskander/code/shared-virtual-env/bin/python "$EXPERIMENT_DIR/code/launch.py" "$@"
