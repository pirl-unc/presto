#!/bin/sh
set -eu
cd "/private/tmp/presto-transfer-improvements-20261027/presto"
PRESTO_MODAL_GPU=H100! /Users/iskander/code/shared-virtual-env/bin/modal run --detach "/private/tmp/presto-transfer-improvements-20261027/presto/experiments/2026-10-27_2100_codex_transfer-improvements/code/launch.py"
