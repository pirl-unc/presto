#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e ".[dev]"

scripts_dir="$(python - <<'PY'
import sysconfig
print(sysconfig.get_path("scripts"))
PY
)"

if command -v presto >/dev/null 2>&1; then
  presto --help >/dev/null
  echo "presto CLI available: $(command -v presto)"
else
  echo "presto CLI not on PATH."
  if [ -x "${scripts_dir}/presto" ]; then
    echo "Add ${scripts_dir} to your PATH or activate your virtual environment."
  fi
  python -m presto --help >/dev/null
  echo "presto CLI available via: python -m presto"
fi
