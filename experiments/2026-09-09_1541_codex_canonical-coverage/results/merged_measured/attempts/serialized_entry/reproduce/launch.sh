#!/bin/sh
set -eu
cd /Users/iskander/code/presto
exec /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-09-09_1541_codex_canonical-coverage/code/launch.py execute --condition merged_measured --attempt serialized_entry --snapshot /Users/iskander/code/presto/artifacts/2026-09-09_1541_codex_canonical-coverage/prepared/d6a690a510750ee66b9224ba4faa76c647e4f0b6/presto
