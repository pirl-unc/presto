#!/usr/bin/env bash
# Presto end-to-end training on Brev, single GPU. Frozen invocation.
#
# Runs on ONE GPU of an already-running shared 4xA100 box rather than
# provisioning a new instance: presto training is single-GPU work, and taking
# one device of a box that is already billing adds no marginal cost while
# leaving the other three free.
#
# Prerequisites on the box (not shipped by runplz, both discovered the hard
# way -- see README "What it took to get here"):
#   /root/.hitlist/            244 MB of BUILT hitlist parquets. Do NOT rebuild
#                              there; the raw IEDB/CEDAR exports are 14.7 GB
#                              and the proteome index cache is 16 GB, and none
#                              of it is needed to read the built tables.
#   /root/.cache/mhcseqs/mhc-full-seqs.csv
#                              60 MB, 56,276 sequences. Without it every allele
#                              silently fails to resolve and the model trains
#                              peptide-only. There is now a guard that refuses
#                              to start in that state.
set -euo pipefail

INSTANCE="${INSTANCE:-rc14-gcp-provision-full-training-3}"
OUT="${OUT:-./brev_runs/presto-e2e-06}"

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
PRESTO_EXTRA_ARGS="--max-mil-instances 16 --max-tcell 4000 --max-vdjdb 1000 --max-processing 1000 --num-workers 8" \
RUNPLZ_MIN_GPUS=1 \
  runplz brev --instance "$INSTANCE" --outputs-dir "$OUT" scripts/train_remote.py
