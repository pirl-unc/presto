#!/bin/sh
# Qualification entry point; original launch.json records this stage.
# Biological training was frozen separately in fitting.json and fit.sh.
set -eu
TASK_REPRO=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$TASK_REPRO/qualify.sh"
