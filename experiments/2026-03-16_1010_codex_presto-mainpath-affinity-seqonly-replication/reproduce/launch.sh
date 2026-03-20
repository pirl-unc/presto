#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: e17aa284c89767d1b9827753dd7dd26c5750171e
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
'/Users/iskander/code/shared-virtual-env/bin/python' 'experiments/2026-03-16_1010_codex_presto-mainpath-affinity-seqonly-replication/code/launch.py'
