#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_1829_codex_lymph-source-qualification/code/retrieve.py experiments/2026-10-24_1829_codex_lymph-source-qualification/reproduce/design-and-identifications.json experiments/2026-10-24_1829_codex_lymph-source-qualification/results/design-and-identifications.json
