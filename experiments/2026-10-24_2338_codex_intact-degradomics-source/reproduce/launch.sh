#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
exec /Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-24_2338_codex_intact-degradomics-source/code/fetch.py "$@"
