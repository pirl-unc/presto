#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_ROOT="$(pwd)"
TASK_IMPORT_ROOT="$(mktemp -d /private/tmp/presto-apc-replay.XXXXXX)"
ln -s "$TASK_ROOT/reproduce/runtime" "$TASK_IMPORT_ROOT/presto"
export PYTHONPATH="$TASK_IMPORT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PRESTO_MODAL_GPU=H100!
if [[ ! -f /private/tmp/presto-hapc-reference.pt ]]; then
  cp results/parent-reference.pt /private/tmp/presto-hapc-reference.pt
fi
case "${1:-verify}" in
  verify) python code/verify_cohort.py ;;
  fit-v1) modal run --detach code/fit.py ;;
  fit-v2) modal run --detach code/fit_v2.py ;;
  fit-global) modal run --detach code/fit_global.py --arm shuffled_global_fixed ;;
  collect)
    python code/collect.py
    python code/collect.py --prefix 2026-11-16_1925_codex_harmonized-apc-v2 --out results/runs-v2
    python code/collect.py --prefix 2026-11-16_1925_codex_harmonized-apc-v2 --out results/runs-v2 --arms shuffled_global_fixed
    ;;
  analyze)
    python code/analyze.py
    python code/analyze.py --runs results/runs-v2 --data results/cohort-v2/data/validation.jsonl --out results/analysis-v2-final
    ;;
  *) echo 'Use verify, fit-v1, fit-v2, fit-global, collect, or analyze' >&2; exit 2 ;;
esac
