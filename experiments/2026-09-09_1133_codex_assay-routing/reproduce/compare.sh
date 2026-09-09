#!/usr/bin/env bash
set -euo pipefail
cd /Users/iskander/code/presto
exec artifacts/2026-09-09_1106_codex_output-coverage/.venv/bin/python \
  experiments/2026-09-09_1133_codex_assay-routing/code/compare.py "$@"
