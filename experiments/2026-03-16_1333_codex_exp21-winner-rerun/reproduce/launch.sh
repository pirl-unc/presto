#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: e1cf8cc60022bdd073eff21dae2924cb23cae819
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
export PRESTO_MODAL_GPU='H100!'
'/Users/iskander/code/shared-virtual-env/bin/python' 'experiments/2026-03-16_1333_codex_exp21-winner-rerun/code/launch.py'
