#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-22_2251_codex_shared-mapp-references/code/verify.py > experiments/2026-10-22_2251_codex_shared-mapp-references/launch_logs/independent-verification.log 2>&1
