#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
PYTHONPATH=/private/tmp/hitlist-mapp-source /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-22_2251_codex_shared-mapp-references/code/launch.py
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-22_2251_codex_shared-mapp-references/code/verify.py
