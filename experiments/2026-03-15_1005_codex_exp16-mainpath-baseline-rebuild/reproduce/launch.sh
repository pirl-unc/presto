#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: e17aa284c89767d1b9827753dd7dd26c5750171e
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
'/Users/iskander/code/shared-virtual-env/bin/python' 'scripts/benchmark_distributional_ba_v6_backbone_compare.py' '--dry-run'
