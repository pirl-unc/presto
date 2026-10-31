#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python -m modal run --detach experiments/2026-10-29_1428_codex_assay-methods-censoring/code/launch.py
