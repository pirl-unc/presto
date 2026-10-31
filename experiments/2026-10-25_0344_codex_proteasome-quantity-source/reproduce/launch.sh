#!/bin/sh
set -eu
EXPERIMENT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
STAGE=$1
shift
exec /Users/iskander/code/shared-virtual-env/bin/python -B "$EXPERIMENT_DIR/code/$STAGE.py" "$@"
