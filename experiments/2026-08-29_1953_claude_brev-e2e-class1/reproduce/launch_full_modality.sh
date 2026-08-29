#!/usr/bin/env bash
# Variant: adds the merged TSV so T-cell / TCR / IEDB-processing modalities
# train too. hitlist carries only binding and MS evidence, so without this the
# immunogenicity, recognition and TCR heads get no gradient at all.
#
# Extra prerequisite on the box: /root/presto_data/merged_deduped.tsv (~1.0 GB,
# 31 columns, no flank columns -- which is why hitlist still supplies binding
# and elution).
set -euo pipefail

INSTANCE="${INSTANCE:-rc14-gcp-provision-full-training-3}"
OUT="${OUT:-./brev_runs/presto-e2e-07}"

PRESTO_EPOCHS=2 \
PRESTO_BATCH_SIZE=128 \
PRESTO_D_MODEL=128 \
PRESTO_N_LAYERS=2 \
PRESTO_N_HEADS=4 \
PRESTO_HITLIST_MHC_CLASS=I \
PRESTO_MAX_ELUTION=60000 \
PRESTO_MAX_BINDING=40000 \
PRESTO_MAX_BULK_MS=40000 \
PRESTO_HITLIST_BUILD=0 \
PRESTO_LATENT_TOPOLOGY=expanded \
PRESTO_MERGED_TSV=/root/presto_data/merged_deduped.tsv \
PRESTO_EXTRA_ARGS="--max-mil-instances 16 --max-tcell 4000 --max-vdjdb 1000 --max-processing 1000 --num-workers 8" \
RUNPLZ_MIN_GPUS=1 \
  runplz brev --instance "$INSTANCE" --outputs-dir "$OUT" scripts/train_remote.py
