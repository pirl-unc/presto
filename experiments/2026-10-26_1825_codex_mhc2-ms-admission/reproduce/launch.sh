#!/bin/sh
set -eu
TASK_RAW=/private/tmp/presto-mhc2-20261026/ms-source
mkdir -p "$TASK_RAW"
gh release download human-cohort-corrected-v1 --repo pirl-unc/hitlist \
  --pattern training.parquet --pattern manifest.json --pattern SHA256SUMS \
  --pattern QUALIFICATION.md --dir "$TASK_RAW"
