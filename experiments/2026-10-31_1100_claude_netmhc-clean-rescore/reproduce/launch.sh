#!/usr/bin/env bash
# CPU only. Tool training archives are NOT in this repo; extract the DTU tarballs into fresh dirs
# (treat as untrusted text; they are only parsed, never executed) and point the env vars at them.
# presto must import from this checkout: PYTHONPATH must contain a dir whose child `presto` is this checkout.
set -euo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/Users/iskander/code/shared-virtual-env/bin:$PATH"
MF="${MHCFLURRY_DATA:-$HOME/Library/Application Support/mhcflurry/4/2.2.0}"
ART="${ARTIFACTS_DIR:-$EXP/../../artifacts/2026-10-31_clean-rescore}"
python "$EXP/analysis/clean_rescore.py" \
  --dumps-dir "$ART" \
  --pan41 "${PAN41_DIR:?extracted NetMHCpan_train dir}" \
  --ii43 "${II43_DIR:?extracted NetMHCIIpan_train dir}" \
  --mhcflurry-train "$MF/models_class1_pan/train_data.csv.bz2" \
  --mhcflurry-affinity "$MF/data_curated/curated_training_data.affinity.csv.bz2" \
  --pr-results "$EXP/../2026-10-31_0830_claude_netmhc-benchmark/results" \
  --out-dir "$EXP" --cache-dir "$ART/cache"
python "$EXP/analysis/make_tables.py" "$EXP"
