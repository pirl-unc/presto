# H100! Batch-Size Bakeoff

- Date: `2026-03-12`
- Agent: `codex` / `gpt-5`
- Purpose: determine the best batch-size regime on `H100!` for the current broad-contract canonical Presto assay-head neighborhood, while holding data, losses, and hardware fixed.
- Source script: `scripts/benchmark_h100_batchsize.py`

## Dataset Contract

- Source: merged IEDB-derived broad numeric class-I binding corpus
- Alleles:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `HLA-A*03:01`
  - `HLA-A*11:01`
  - `HLA-A*01:01`
  - `HLA-B*07:02`
  - `HLA-B*44:02`
- Included assay families:
  - `IC50`
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  - `EC50`
- Excluded assay families:
  - qualitative binding
  - presentation / elution
  - stability
  - immunogenicity / T-cell
- Qualifier / censor policy:
  - `qualifier_filter=all`
  - exact and `>` rows included
  - censor-aware loss on the corresponding assay outputs
- Train / val split:
  - `32,855 / 8,194`
- Synthetic data:
  - none
- Ranking / contrastive:
  - none

## Training Contract

- Warm start:
  - `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
- Epochs:
  - `5`
- GPU:
  - requested `H100!`
- Runtime defaults:
  - `allow_tf32=true`
  - `matmul_precision=high`
  - `num_workers=0`
  - `pin_memory=true`
- Positional base:
  - peptide `concat(start,end,nterm_frac,cterm_frac)`
  - groove `concat(start,end,nterm_frac,cterm_frac)`
- Assay label -> output mapping:
  - `IC50` -> `assays.IC50_nM`
  - direct `KD` -> `assays.KD_nM`
  - `KD (~IC50)` -> merged or split depending on `kd_grouping_mode`
  - `KD (~EC50)` -> merged or split depending on `kd_grouping_mode`
  - `EC50` -> `assays.EC50_nM`

## Conditions Tested

Assay-head variants:
- `A03`
  - `shared_base_segment_residual`
  - `split_kd_proxy`
- `A05`
  - `shared_base_factorized_context_residual`
  - `split_kd_proxy`
- `A06`
  - `shared_base_factorized_context_plus_segment_residual`
  - `merged_kd`
- `A07`
  - `shared_base_factorized_context_plus_segment_residual`
  - `split_kd_proxy`

Batch sizes:
- `64`
- `128`
- `192`
- `256`

## Results

| design | batch | val loss | epoch s | peak alloc GiB | SLL A02 / A24 | SLL ratio | FLR A02 / A24 | FLR ratio | NFL A02 / A24 | NFL ratio |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |
| `A03` | `64` | `0.7769` | `76.2` | `8.3` | `47.9 / 12472.1` | `260.5` | `14.9 / 5576.9` | `373.9` | `3472.3 / 3856.9` | `0.9` |
| `A03` | `128` | `0.9093` | `55.2` | `16.4` | `25.7 / 12794.4` | `498.7` | `10.4 / 5863.4` | `564.7` | `2563.7 / 786.5` | `3.3` |
| `A03` | `192` | `0.8137` | `59.1` | `24.6` | `50.7 / 9495.2` | `187.1` | `134.4 / 2878.3` | `21.4` | `890.0 / 1025.5` | `0.9` |
| `A03` | `256` | `0.9850` | `58.8` | `32.8` | `15.7 / 25116.8` | `1601.5` | `10.5 / 2025.7` | `192.6` | `2497.9 / 128.5` | `19.4` |
| `A05` | `64` | `2.0222` | `93.3` | `8.3` | `1347.6 / 1347.6` | `1.0` | `1347.6 / 1347.6` | `1.0` | `1347.6 / 1347.6` | `1.0` |
| `A05` | `128` | `0.8518` | `67.9` | `16.4` | `94.4 / 5868.7` | `62.1` | `23.9 / 1249.0` | `52.3` | `292.2 / 309.2` | `0.9` |
| `A05` | `192` | `0.8701` | `53.5` | `24.6` | `247.8 / 4223.6` | `17.0` | `55.3 / 412.0` | `7.5` | `788.0 / 93.7` | `8.4` |
| `A05` | `256` | `2.3112` | `52.3` | `32.7` | `2119.4 / 2119.3` | `1.0` | `2119.4 / 2119.3` | `1.0` | `2119.4 / 2119.3` | `1.0` |
| `A06` | `64` | `1.4963` | `93.6` | `8.3` | `344.7 / 13062.1` | `37.9` | `221.6 / 2380.1` | `10.7` | `739.4 / 209.6` | `3.5` |
| `A06` | `128` | `1.3109` | `63.1` | `16.4` | `334.2 / 8343.6` | `25.0` | `487.9 / 2942.5` | `6.0` | `806.5 / 266.9` | `3.0` |
| `A06` | `192` | `1.4717` | `52.3` | `24.6` | `806.1 / 4132.2` | `5.1` | `300.8 / 2606.4` | `8.7` | `5853.9 / 646.9` | `9.0` |
| `A06` | `256` | `1.4613` | `49.8` | `32.8` | `1213.0 / 13072.7` | `10.8` | `439.6 / 3803.5` | `8.7` | `4608.8 / 561.2` | `8.2` |
| `A07` | `64` | `0.9858` | `79.3` | `8.3` | `267.0 / 19793.5` | `74.1` | `64.4 / 14851.7` | `230.5` | `1158.1 / 2648.9` | `0.4` |
| `A07` | `128` | `0.7827` | `66.1` | `16.4` | `46.5 / 14378.4` | `309.0` | `59.4 / 13580.0` | `228.6` | `4907.0 / 496.3` | `9.9` |
| `A07` | `192` | `0.9353` | `66.7` | `24.6` | `675.4 / 26254.0` | `38.9` | `53.6 / 733.2` | `13.7` | `4467.8 / 195.0` | `22.9` |
| `A07` | `256` | `0.8105` | `44.0` | `32.8` | `52.6 / 8634.1` | `164.2` | `16.0 / 884.4` | `55.1` | `2698.8 / 73.3` | `36.8` |

## Takeaway

- Best validation-loss setting:
  - `A03 @ batch 64`
  - `val_loss=0.7769`
- Best probe-score setting:
  - `A03 @ batch 256`
  - strongest combined broad-contract probe margins
  - but clearly worse validation loss
- Fastest setting:
  - `A07 @ batch 256`
  - `44.0s/epoch`
  - with strong but not best probe behavior

Most important conclusions:
- `A03` remains the strongest broad-contract canonical Presto neighborhood.
- `A05` is unstable at batch-size extremes and should not be treated as a default.
- `A06` gets faster at large batch size but is consistently weaker biologically.
- `A07` benefits from larger batches for throughput and `NFLIKFLLI`, but loses `SLLQHLIGL` tail calibration relative to `A03`.
- There is no single universally best batch size:
  - `64` best for validation fit
  - `256` best for probe score in `A03`
  - `256` also fastest in `A07`

Relative to earlier experiments:
- This family refines the `A03/A07` frontier from:
  - [2026-03-12_1034_codex_broad-frontier-5ep-round1](../2026-03-12_1034_codex_broad-frontier-5ep-round1/)
- It shows that once hardware is fixed to `H100!`, batch size is a real optimization variable and interacts strongly with assay-head topology.

## Artifacts

- [options_vs_perf.md](./options_vs_perf.md)
- [options_vs_perf.json](./options_vs_perf.json)
- [analysis/parsed_metrics.csv](./analysis/parsed_metrics.csv)
- [analysis/parsed_metrics.json](./analysis/parsed_metrics.json)
- [analysis/summary.json](./analysis/summary.json)
- [analysis/val_loss_vs_batch.png](./analysis/val_loss_vs_batch.png)
- [analysis/epoch_time_vs_batch.png](./analysis/epoch_time_vs_batch.png)
- [analysis/probe_score_vs_batch.png](./analysis/probe_score_vs_batch.png)
- [variants.md](./variants.md)
- [manifest.json](./manifest.json)
