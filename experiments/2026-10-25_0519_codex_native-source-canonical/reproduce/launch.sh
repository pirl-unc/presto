#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
exec /Users/iskander/code/shared-virtual-env/bin/python -B code/launch.py "$@"
