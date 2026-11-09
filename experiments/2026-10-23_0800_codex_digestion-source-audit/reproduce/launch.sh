#!/bin/sh
set -eu
# Frozen local source audit. No biological model training or environment changes.
cd /private/tmp/presto-modular-model
family=experiments/2026-10-23_0800_codex_digestion-source-audit
python=/Users/iskander/code/shared-virtual-env/bin/python
"$python" "$family/code/fetch_sources.py"
"$python" "$family/code/audit_archive.py"
/opt/homebrew/bin/Rscript --vanilla "$family/code/export_rdata.R" /private/tmp/presto-digestion-assets/original-layers
"$python" "$family/code/audit_metadata.py"
"$python" "$family/code/replay_h4.py"
