#!/bin/sh
set -eu
cd "/private/tmp/presto-flank-benchmark-20261027/presto"
PRESTO_MODAL_GPU=H100! /Users/iskander/code/shared-virtual-env/bin/modal run --detach "/private/tmp/presto-flank-benchmark-20261027/presto/experiments/2026-10-27_1930_codex_modal-cross-class/code/modal_run.py"
