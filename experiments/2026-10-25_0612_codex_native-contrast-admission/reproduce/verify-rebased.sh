#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec /Users/iskander/code/shared-virtual-env/bin/python -B "$ROOT/code/launch_rebased.py" verify_rebased
