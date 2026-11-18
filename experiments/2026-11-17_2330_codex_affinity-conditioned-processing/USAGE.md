# Use the retained baseline or processing candidate

The general-purpose default remains the previously qualified broad checkpoint.
The new boundary-aware model is a provisional processing candidate: pooled
matched ranking improves, but class-II paired ordering is not established.
See [the decision](DECISION.md) before interpreting its scores.

Run from `/Users/iskander/code/presto` using the existing shared Python environment.
These commands use the frozen canonical runtime, without reinstalling Presto:

```bash
# Retained general-purpose default, both classes and measured IC50:
/Users/iskander/code/shared-virtual-env/bin/python \
  experiments/2026-11-17_2330_codex_affinity-conditioned-processing/code/predict_example.py

# New selected boundary-aware processing candidate:
/Users/iskander/code/shared-virtual-env/bin/python \
  experiments/2026-11-17_2330_codex_affinity-conditioned-processing/code/predict_example.py \
  --checkpoint experiments/2026-11-17_2330_codex_affinity-conditioned-processing/results/fits/paired-junction-s42-n2000/raw/selected.pt \
  --output /private/tmp/presto-processing-predictions.json
```

Both commands were verified on CPU. Replace `--inputs` with canonical label-free
JSONL observations (`split: "predict"`, no value). The four complete examples in
`results/example-queries.jsonl` include typed MHC groove sequences, both classes,
presentation and IC50 endpoints. Use biological MHC candidates; do not put allele
names in residue fields. Public inference is `ModularPredictor.predict`.

Source flanks and APC context are optional, with explicit knownness. Unsupported
APC annotations raise by default; `--unsupported-apc-policy unknown` explicitly
masks them if that is the intended policy. Do not treat unknown as a WT/vehicle
control. The model uses a 64-dimensional shared APC state. Its new boundary head
is active in the candidate and inactive in the retained default.

The returned presentation score ranks the sampled-background objective. It is
not a calibrated absolute surface-abundance probability. Intrinsic IC50 does not
receive APC or source-flank inputs. No external NetMHC executable is needed for
inference; those programs were used only to construct affinity-matched data.
Processing diagnostics are already included in the prediction; do not add their
contributions a second time. See [architecture](ARCHITECTURE.md).

## Exact checkpoints and recovery

The default SHA256 is
`1b6011396a563f52ea1d6da4bb1d4e69b8dbf0103148ba829225581c4d312f68`, from
`../2026-11-17_1155_codex_shared-presentation-expansion/results/fits/expanded-state-sqrt-s42-n6000/raw/selected.pt`.
The candidate SHA256 is
`e4898af3a7a442a999177273687ea4e4ce5d4216e908f2e435b2d73226e56963`.
The helper verifies the default against `results/preferred-checkpoint.json`.

Raw fits and validation predictions remain on Modal volume `presto-checkpoints`:
`/volume/2026-11-17_2330_codex_affinity-conditioned-processing/`.
Large tensors and raw datasets are intentionally excluded from the source commit;
local copies and the remote locations/hashes are recorded in the artifact manifest.
`code/collect.py --fits paired-junction-s42-n2000` retrieves the candidate and its
training receipts using the existing Modal account. It does not rerun training.
The prior experiment's collection instructions retrieve the retained default.

## Reproduction

`reproduce/launch.sh` records the six-fit H100! comparison; `reproduce/launch.json`
and the model-v2 source snapshot preserve exact arguments, source and data hashes.
Completed remote directories are immutable: rerunning requires a new registered
experiment namespace rather than overwriting this family. The Python model runtime
is core `0c21bc4d`, frozen under `reproduce/runtime/`; the package alias under
`reproduce/inference-runtime/` points to it.

The user-requested Hitlist 2.1.12 refresh is a separate next-generation source
contract. Its exports are in `data/shared-presentation-20261118-hitlist2112` and
verification is in `tasks/hitlist-2.1.12-refresh`. It does not change these trained
weights or frozen evaluation rows. Rebuild protected model inputs through the
central adapter before training on the refreshed annotations. TEST remains sealed.
