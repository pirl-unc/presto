#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/presto-mpl-config}"
exec /Users/iskander/code/shared-virtual-env/bin/python \
  "${experiment_dir}/reproduce/source/smoke_prime.py" \
  --prime /Users/iskander/code/PRIME/PRIME \
  --mixmhcpred /Users/iskander/code/weirdo/tmp/MixMHCpred-3.0/MixMHCpred \
  --backend-python /Users/iskander/code/weirdo/tmp/prime-env/bin/python \
  --output "${experiment_dir}/results/smoke.json"
