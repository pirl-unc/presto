#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: a1cdcc0d27d63a7440188f98f0dc503c596c8dfd
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
'/Users/iskander/code/shared-virtual-env/bin/python3' '/Users/iskander/code/shared-virtual-env/bin/modal' 'run' '--detach' 'experiments/2026-09-02_1720_codex_source-junction-masking/code/launch.py'
