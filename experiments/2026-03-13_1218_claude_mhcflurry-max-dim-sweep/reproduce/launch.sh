#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: e01eea1e91eb3b07ac5a8d75e65956ef688cccfb
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
'/Users/iskander/code/shared-virtual-env/bin/python3' 'scripts/benchmark_distributional_ba_heads_v3.py' '--agent-label' 'claude' '--prefix' 'dist-ba-v3'
