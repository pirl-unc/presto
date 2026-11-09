#!/usr/bin/env bash
set -euo pipefail
audit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${AUDIT_PYTHON:-python3}" "$audit_dir/reproduce/source/audit.py" "$@"
