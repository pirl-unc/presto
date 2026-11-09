#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
PYTHON=/Users/iskander/code/shared-virtual-env/bin/python
EXP=experiments/2026-10-24_2130_codex_mapp2025-source
# Use a fresh raw-output destination when reproducing retrieval; scripts refuse overwrite.
"$PYTHON" "$EXP/code/fetch.py"
"$PYTHON" "$EXP/code/fetch_metadata.py"
"$PYTHON" "$EXP/code/inventory.py"
"$PYTHON" "$EXP/code/audit_workbooks.py"
"$PYTHON" "$EXP/analysis/verify_xml.py"
"$PYTHON" "$EXP/analysis/summarize.py"
"$PYTHON" "$EXP/code/fetch_archive.py"
"$PYTHON" "$EXP/analysis/verify_archive.py"
"$PYTHON" "$EXP/analysis/source_dispositions.py"
