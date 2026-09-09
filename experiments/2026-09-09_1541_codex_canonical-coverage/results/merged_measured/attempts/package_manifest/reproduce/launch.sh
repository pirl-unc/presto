#!/bin/sh
set -eu
cd /Users/iskander/code/presto
exec /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-09-09_1541_codex_canonical-coverage/code/launch.py execute --condition merged_measured --attempt package_manifest --snapshot /Users/iskander/code/presto/artifacts/2026-09-09_1541_codex_canonical-coverage/prepared/d8f06ee40a69d213f85175be05c4e294a7914930/presto
