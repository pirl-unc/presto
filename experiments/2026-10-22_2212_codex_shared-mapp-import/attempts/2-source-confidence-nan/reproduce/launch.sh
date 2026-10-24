#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
PYTHONPATH=/private/tmp/hitlist-mapp-source /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-22_2212_codex_shared-mapp-import/code/launch.py
