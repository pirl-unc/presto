#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: bf7fcbfcc2cfdcbaa32f6424b7f7ad0e09ceb857
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
'/Users/iskander/code/shared-virtual-env/bin/python' 'experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/code/launch.py' '--phase' 'all' '--wait-for-pretrains' '--wait-timeout-sec' '7200' '--poll-interval-sec' '30'
