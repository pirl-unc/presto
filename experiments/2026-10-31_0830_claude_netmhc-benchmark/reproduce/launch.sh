#!/usr/bin/env bash
# CPU-only. Needs: netmhc-bundle, mhcflurry-predict, and this repo importable as `presto`
# (PYTHONPATH must contain a directory whose child `presto` is this checkout).
# Non-default overrides: none beyond the env vars below.
set -euo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
ART="${ARTIFACTS_DIR:-$EXP/../../artifacts/2026-10-31_0830_claude_netmhc-benchmark}"
export PATH="/Users/iskander/code/shared-virtual-env/bin:$PATH"
mkdir -p "$ART"
python -m presto.scripts.external_benchmark \
  --manifest "$EXP/reproduce/launch.json" \
  --out-dir "$EXP" --artifacts-dir "$ART" \
  --bundle-home "${NETMHC_BUNDLE_HOME:-/Users/iskander/code/netmhc-bundle}" \
  --mhcflurry "${MHCFLURRY_PREDICT:-/Users/iskander/code/shared-virtual-env/bin/mhcflurry-predict}"
