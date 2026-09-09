#!/bin/sh
set -eu
cd /Users/iskander/code/presto
exec /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-09-09_1541_codex_canonical-coverage/code/launch.py execute --condition merged_measured --snapshot /Users/iskander/code/presto/artifacts/2026-09-09_1541_codex_canonical-coverage/prepared/37cbb1742591e7c29f07496c8d49404b15306bbe/presto
