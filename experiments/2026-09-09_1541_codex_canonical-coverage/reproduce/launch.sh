#!/bin/sh
set -eu
# Registered entry point; exact launched invocations receive phase-local bundles.
exec python experiments/2026-09-09_1541_codex_canonical-coverage/code/launch.py "$@"
