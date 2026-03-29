#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: 50cd7bdc4fbb882699500b6fbc99a1856c96d69d
#   branch: main
#   dirty: no
cd '/Users/iskander/code/presto'
export PRESTO_EXPERIMENT_AGENT='claude'
'/Users/iskander/code/shared-virtual-env/bin/python3' 'experiments/2026-03-28_claude_class1-best-hits/launch.py' '--dry-run'
