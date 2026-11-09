#!/bin/bash
set -euo pipefail
cd /private/tmp/presto-modular-model
python_bin=/Users/iskander/code/shared-virtual-env/bin/python
family=experiments/2026-10-22_1914_codex_pxd057588-shared-sequence-admission
"$python_bin" -u "$family/code/verify_package.py"
"$python_bin" -u "$family/code/audit.py"
"$python_bin" -u "$family/code/verify.py"
