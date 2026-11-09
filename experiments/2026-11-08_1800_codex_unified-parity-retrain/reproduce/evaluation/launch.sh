#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
input_root="${PRESTO_EVALUATION_INPUT_ROOT:?set PRESTO_EVALUATION_INPUT_ROOT}"
output_root="${PRESTO_EVALUATION_OUTPUT_ROOT:?set PRESTO_EVALUATION_OUTPUT_ROOT}"
python="${PRESTO_PYTHON:-/Users/iskander/code/shared-virtual-env/bin/python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/presto-mpl-config}"
export PYTHONPATH="${PRESTO_PYTHONPATH:-$(cd "${experiment_dir}/../../.." && pwd)}"

"${python}" "${experiment_dir}/code/evaluate_pair.py" \
  --contract "${experiment_dir}/reproduce/evaluation/contract.json" \
  --input-root "${input_root}" \
  --output "${output_root}"

"${python}" "${experiment_dir}/code/verify_evaluation.py" \
  --evaluation-root "${output_root}" \
  --output "${output_root}/verification.json"
