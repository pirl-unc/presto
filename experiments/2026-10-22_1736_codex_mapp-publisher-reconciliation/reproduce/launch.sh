#!/bin/bash
set -euo pipefail
cd /private/tmp/presto-modular-model
python_bin=/Users/iskander/code/shared-virtual-env/bin/python
family=experiments/2026-10-22_1736_codex_mapp-publisher-reconciliation
"$python_bin" -u "$family/code/audit.py"
"$python_bin" -u "$family/code/verify.py"
"$python_bin" -u "$family/code/membership.py"
"$python_bin" -u "$family/code/verify_membership.py"
