#!/usr/bin/env bash
set -euo pipefail
family="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$family/../.." && pwd)"
cd "$repo"
export PYTHONPATH="${PRESTO_IMPORT_ROOT:-/private/tmp/presto-conditions-import}"
"${PRESTO_AUDIT_PYTHON:-/Users/iskander/code/shared-virtual-env/bin/python}" \
  "$family/reproduce/source/audit.py" \
  --source "${PRESTO_MS_EXPORT:-/private/tmp/presto-ms-condition-contrasts-20261110/canonical-ms-export.parquet}" \
  --catalog "${PRESTO_MHC_CATALOG:-/Users/iskander/code/presto/artifacts/2026-11-07_1645_codex_unified-data-audit/upload/mhc-full-seqs.csv}" \
  --output "${PRESTO_CONDITION_OUTPUT:-$family/results}" \
  --observations "${PRESTO_CONDITION_OBSERVATIONS:-/private/tmp/presto-ms-condition-contrasts-20261110/condition-observations.jsonl.gz}"
