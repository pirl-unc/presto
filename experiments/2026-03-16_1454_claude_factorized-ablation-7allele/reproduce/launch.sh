#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: bf7fcbfcc2cfdcbaa32f6424b7f7ad0e09ceb857
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
'/Users/iskander/code/shared-virtual-env/bin/python3' 'scripts/benchmark_factorized_ablation.py' '--agent-label' 'claude'
