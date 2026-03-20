#!/usr/bin/env bash
set -euo pipefail
cd /Users/iskander/code/presto
exec python scripts/benchmark_target_space_bakeoff.py "$@"
