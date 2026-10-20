#!/usr/bin/env bash
set -euo pipefail
# Run in the repository root with Presto importable from this checkout.
# The source archive is acquired/verified by reproduce/launch.sh first.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/2026-10-20_1458_codex_ic50-collapse-audit/analysis/training_paths.py \
  --run "${PRESTO_AUDIT_RUN:-/private/tmp/presto-ic50-run}" \
  --output experiments/2026-10-20_1458_codex_ic50-collapse-audit/results/training-paths
