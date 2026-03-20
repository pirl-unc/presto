#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

LOCAL_DEVICE="${PRESTO_LOCAL_DEVICE:-cpu}"
LOCAL_MPS_SAFE_MODE="${PRESTO_LOCAL_MPS_SAFE_MODE:-auto}"
LOCAL_DATA_DIR="${PRESTO_LOCAL_DATA_DIR:-data}"
LOCAL_INIT_CHECKPOINT="${PRESTO_LOCAL_INIT_CHECKPOINT:-experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt}"

python experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun/code/launch.py \
  --out-dir experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun \
  --backend local \
  --device "${LOCAL_DEVICE}" \
  --mps-safe-mode "${LOCAL_MPS_SAFE_MODE}" \
  --local-data-dir "${LOCAL_DATA_DIR}" \
  --local-init-checkpoint "${LOCAL_INIT_CHECKPOINT}" \
  "$@"
