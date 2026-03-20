# 2026-03-14 MHCflurry / LogMSE Cold vs Warm Start

## Goal

Run a minimal canonical follow-up to the clean self-contained BA head benchmark that isolates one remaining question: does warming the fixed backbone from `mhc-pretrain-20260308b` materially improve the two viable regression heads on the broad 7-allele numeric contract?

## Fixed contract

- Source: `data/merged_deduped.tsv`
- Alleles:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `HLA-A*03:01`
  - `HLA-A*11:01`
  - `HLA-A*01:01`
  - `HLA-B*07:02`
  - `HLA-B*44:02`
- Assay families:
  - `IC50`
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  - `EC50`
- Qualifier policy: `all`
- Split: deterministic peptide-group `80/10/10`, seed `42`
- Backbone: self-contained `FixedBackbone(embed=128, layers=2, heads=4, ff=128)`
- Batch size: `256`
- Epochs: `20`
- GPU: `H100!`
- Optimizer: `AdamW(weight_decay=0.01)`
- LR / schedule: `1e-4`, `warmup_cosine`, warmup fraction `0.1`, min lr scale `0.1`
- No synthetics
- No ranking losses

## Conditions (4)

- `mhcflurry_additive_max200k` cold-start
- `mhcflurry_additive_max200k` warm-start
- `log_mse_additive_max200k` cold-start
- `log_mse_additive_max200k` warm-start

## Warm-start contract

- Warm-start source: `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
- This is a `Presto` checkpoint, not a direct `FixedBackbone` checkpoint.
- Load only shape-compatible encoder weights into the fixed backbone.
- Expected compatible subset:
  - `aa_embedding.weight`
  - `stream_encoder.layers.*.self_attn.*` -> `encoder.layers.*.self_attn.*`
  - `stream_encoder.layers.*.norm{1,2}.*` -> `encoder.layers.*.norm{1,2}.*`
- Expected incompatible subset:
  - pretrain feed-forward blocks (`linear1/linear2`) because the pretrain encoder uses a larger FF dimension
  - all non-backbone heads / task-specific blocks
  - positional embeddings that do not map cleanly to the local segment-wise `pos_embedding`
- The trainer must record:
  - whether warm-start was requested
  - checkpoint path
  - loaded key count
  - skipped missing keys
  - skipped shape-mismatch keys

## Runtime boundary

- Use a new self-contained experiment directory, not the older clean 12-condition directory.
- Keep runtime ownership local to the new experiment dir:
  - condition config
  - trainer
  - evaluation
  - launcher
- Modal may use only a thin wrapper to mount `/data` and `/checkpoints` and prepend the experiment `code/` directory to `PYTHONPATH`.

## Verification plan

1. Local smoke run on one warm-start condition using the downloaded checkpoint copy.
2. Modal smoke run on one warm-start condition using `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`.
3. Launch the 4-run detached sweep on `H100!`.
4. Do not stop at "apps finished"; pull artifacts, regenerate tables/plots locally, update the experiment README, and update `experiments/experiment_log.md`.

## Questions to answer

1. Does warm-start improve held-out Spearman / RMSE / AUROC for `mhcflurry`?
2. Does warm-start improve `log_mse`, and if so by how much relative to `mhcflurry`?
3. Does warm-start preserve or improve the probe ordering for `SLLQHLIGL`, `FLRYLLFGI`, and `NFLIKFLLI`?
4. Is the canonical next baseline still cold-start `mhcflurry_additive_max200k`, or does the warm-started variant replace it?
