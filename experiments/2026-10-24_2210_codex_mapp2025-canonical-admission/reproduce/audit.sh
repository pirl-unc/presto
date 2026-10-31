#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
exec /Users/iskander/code/shared-virtual-env/bin/python -c 'import subprocess; subprocess.run(["/Users/iskander/code/shared-virtual-env/bin/python", "experiments/2026-10-24_2210_codex_mapp2025-canonical-admission/code/audit.py"], check=True, timeout=1200)'
