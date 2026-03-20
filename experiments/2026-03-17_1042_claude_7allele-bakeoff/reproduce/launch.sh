#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: bf7fcbfcc2cfdcbaa32f6424b7f7ad0e09ceb857
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
export PRESTO_EXPERIMENT_AGENT='claude'
'/Users/iskander/code/shared-virtual-env/bin/python3' 'experiments/2026-03-17_1042_claude_7allele-bakeoff/launch.py' '--out-dir' 'experiments/2026-03-17_1042_claude_7allele-bakeoff' '--skip-launched'
