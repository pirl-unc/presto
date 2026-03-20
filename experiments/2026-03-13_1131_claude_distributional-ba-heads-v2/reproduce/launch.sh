#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: e01eea1e91eb3b07ac5a8d75e65956ef688cccfb
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
'/Users/iskander/code/shared-virtual-env/bin/python3' 'scripts/benchmark_distributional_ba_heads_v2.py' '--agent-label' 'claude' '--prefix' 'dist-ba-v2' '--cond-ids' '5,6,7,8,9,10,11,12'
