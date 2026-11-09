#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
report_output_root="${PRESTO_REPORT_OUTPUT_ROOT:-${experiment_dir}/reproduced-reports}"
python="${PRESTO_PYTHON:-/Users/iskander/code/shared-virtual-env/bin/python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/presto-mpl-config}"
export PYTHONPATH="${PRESTO_PYTHONPATH:-$(cd "${experiment_dir}/../.." && pwd)}"

"${python}" "${experiment_dir}/code/build_manifest.py"
"${python}" -m presto.evaluation.performance_report \
  --manifest "${experiment_dir}/report_manifest.json" \
  --output-root "${report_output_root}"
