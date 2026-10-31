#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/private/tmp/presto-native-screen-mpl
exec /Users/iskander/code/shared-virtual-env/bin/python -B "$ROOT/analysis/report.py"
