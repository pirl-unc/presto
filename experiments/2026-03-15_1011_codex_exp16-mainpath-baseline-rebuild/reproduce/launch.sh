#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: e1cf8cc60022bdd073eff21dae2924cb23cae819
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
'/Users/iskander/code/shared-virtual-env/bin/python' 'experiments/2026-03-15_1011_codex_exp16-mainpath-baseline-rebuild/code/launch.py' '--dry-run'
