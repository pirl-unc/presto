#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
report_output_root="${PRESTO_REPORT_OUTPUT_ROOT:-${experiment_dir}/reproduced-reports}"

presto-report \
  --manifest "${experiment_dir}/report_manifest.json" \
  --output-root "${report_output_root}"
