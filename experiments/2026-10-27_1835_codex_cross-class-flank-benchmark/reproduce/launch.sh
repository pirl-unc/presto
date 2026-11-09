#!/bin/sh
set -eu
cd "/private/tmp/presto-flank-benchmark-20261027/presto"
exec "/Users/iskander/code/shared-virtual-env/bin/python" "/private/tmp/presto-flank-benchmark-20261027/presto/experiments/2026-10-27_1835_codex_cross-class-flank-benchmark/code/launch.py" all
