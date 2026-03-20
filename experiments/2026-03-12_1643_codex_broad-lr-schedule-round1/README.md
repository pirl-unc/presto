# Broad LR / Schedule Bakeoff

- Date: `2026-03-12`
- Agent: `codex` / `gpt-5`
- Purpose: compare a small set of initial learning rates and scheduler shapes for the two leading broad-contract canonical Presto hypotheses on a fixed 20-epoch budget.
- Source script: `scripts/benchmark_lr_schedule_bakeoff.py`

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

## Training / Pretraining Contract

- Warm start:
  - `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
- Epochs:
  - `20`
- Batch size:
  - `256`
- GPU:
  - requested `H100!`
- Runtime defaults:
  - `allow_tf32=true`
  - `matmul_precision=high`
  - `num_workers=0`
  - `pin_memory=true`
- Assay label -> output mapping:
  - `IC50` -> `assays.IC50_nM`
  - direct `KD` -> direct KD output (`A03` in `log10_100k`, `A07` in `mhcflurry_100k` target space)
  - `KD (~IC50)` / `KD (~EC50)` -> split proxy-KD outputs
  - `EC50` -> `assays.EC50_nM`

## Conditions Tested

Two architecture/target-space families:
- `A03_log10_100k`
  - `shared_base_segment_residual`
  - `split_kd_proxy`
  - target space `log10`, max `100,000 nM`
- `A07_mhcflurry_100k`
  - `shared_base_factorized_context_plus_segment_residual`
  - `split_kd_proxy`
  - target space `mhcflurry`, max `100,000 nM`

Learning rates:
- `1e-4`
- `2.8e-4`
- `8e-4`

Schedulers:
- `constant`
- `warmup_cosine`
- `onecycle`

## Results

### A03_log10_100k

| lr | schedule | best epoch | best val | term val | epoch s | gpu util % | SLL A02 / A24 | SLL ratio | FLR A02 / A24 | FLR ratio | NFL A02 / A24 | NFL ratio | probe score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | ---: |
| `2.8e-4` | `onecycle` | 12 | 0.8336 | 0.8555 | 53.1 | 55.0 | 36.9 / 26993.6 | 730.7 | 28.5 / 2039.0 | 71.5 | 17431.2 / 138.4 | 126.0 | 6.82 |
| `2.8e-4` | `constant` | 11 | 0.8469 | 0.8697 | 56.2 | 51.6 | 60.0 / 23768.1 | 395.9 | 29.7 / 1660.2 | 55.9 | 4522.4 / 41.1 | 110.0 | 6.39 |
| `2.8e-4` | `warmup_cosine` | 12 | 0.8492 | 0.8908 | 53.2 | 59.6 | 32.1 / 40508.7 | 1261.9 | 38.7 / 703.0 | 18.2 | 15915.2 / 79.8 | 199.4 | 6.66 |
| `1e-4` | `warmup_cosine` | 18 | 0.8677 | 0.8691 | 55.7 | 53.0 | 87.6 / 46163.1 | 526.9 | 54.4 / 2098.3 | 38.6 | 25611.7 / 6.4 | 4010.6 | 7.91 |
| `1e-4` | `constant` | 18 | 0.8687 | 0.8766 | 52.7 | 59.3 | 43.5 / 37737.3 | 867.6 | 31.6 / 1577.2 | 50.0 | 18518.4 / 30.8 | 600.7 | 7.42 |
| `1e-4` | `onecycle` | 11 | 0.8703 | 0.9253 | 50.5 | 60.9 | 128.0 / 39114.7 | 305.5 | 30.3 / 623.3 | 20.6 | 13971.9 / 122.4 | 114.1 | 5.86 |
| `8e-4` | `onecycle` | 18 | 1.0513 | 1.0561 | 50.5 | 59.9 | 71.5 / 35657.4 | 499.0 | 25.3 / 165.1 | 6.5 | 8531.8 / 1517.2 | 5.6 | 4.26 |
| `8e-4` | `warmup_cosine` | 20 | 1.0888 | 1.0888 | 56.1 | 54.0 | 45.1 / 48065.1 | 1064.8 | 23.6 / 208.7 | 8.9 | 17085.3 / 290.3 | 58.9 | 5.74 |
| `8e-4` | `constant` | 5 | 1.3152 | 1.5260 | 49.8 | 62.6 | 26.7 / 4578.0 | 171.7 | 29.4 / 2060.2 | 70.1 | 2489.4 / 323.8 | 7.7 | 4.97 |


### A07_mhcflurry_100k

| lr | schedule | best epoch | best val | term val | epoch s | gpu util % | SLL A02 / A24 | SLL ratio | FLR A02 / A24 | FLR ratio | NFL A02 / A24 | NFL ratio | probe score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | ---: |
| `2.8e-4` | `warmup_cosine` | 14 | 0.0269 | 0.0276 | 49.6 | 62.1 | 207.4 / 4440.4 | 21.4 | 157.5 / 568.5 | 3.6 | 5708.3 / 23.2 | 246.2 | 4.28 |
| `2.8e-4` | `onecycle` | 11 | 0.0271 | 0.0275 | 49.0 | 61.7 | 168.9 / 29951.5 | 177.4 | 136.4 / 1112.6 | 8.2 | 5863.7 / 55.0 | 106.6 | 5.19 |
| `2.8e-4` | `constant` | 16 | 0.0279 | 0.0318 | 57.5 | 52.5 | 75.3 / 21754.3 | 289.1 | 81.8 / 806.9 | 9.9 | 2062.1 / 101.5 | 20.3 | 4.76 |
| `1e-4` | `constant` | 20 | 0.0290 | 0.0290 | 48.8 | 62.3 | 118.9 / 11449.7 | 96.3 | 282.7 / 840.8 | 3.0 | 7249.8 / 95.9 | 75.6 | 4.34 |
| `1e-4` | `onecycle` | 13 | 0.0293 | 0.0301 | 53.4 | 57.9 | 253.8 / 5449.8 | 21.5 | 133.7 / 894.9 | 6.7 | 8064.6 / 46.4 | 173.9 | 4.40 |
| `8e-4` | `constant` | 20 | 0.0293 | 0.0293 | 50.6 | 63.3 | 136.0 / 28897.7 | 212.6 | 147.0 / 3057.3 | 20.8 | 3503.8 / 443.7 | 7.9 | 4.54 |
| `1e-4` | `warmup_cosine` | 18 | 0.0308 | 0.0312 | 50.2 | 61.7 | 333.1 / 33092.1 | 99.4 | 512.6 / 1063.3 | 2.1 | 13025.3 / 14.3 | 911.9 | 5.27 |
| `8e-4` | `warmup_cosine` | 15 | 0.0404 | 0.0429 | 52.8 | 57.9 | 46.0 / 4588.5 | 99.7 | 76.2 / 452.0 | 5.9 | 835.0 / 233.4 | 3.6 | 3.33 |
| `8e-4` | `onecycle` | 20 | 0.0423 | 0.0423 | 48.6 | 63.2 | 63.3 / 2586.5 | 40.8 | 130.5 / 730.1 | 5.6 | 630.0 / 318.0 | 2.0 | 2.66 |


## Winner / Preferred Conditions

- Best `A03` by validation loss:
  - `2.8e-4 + onecycle`
  - `best_val=0.8336`
- Best `A03` by probe score:
  - `1e-4 + warmup_cosine`
  - `SLL ratio=526.9`
  - `FLR ratio=38.6`
  - `NFL ratio=4010.6`

- Best `A07` by validation loss:
  - `2.8e-4 + warmup_cosine`
  - `best_val=0.0269` in MHCflurry encoded target space
- Best `A07` by probe score:
  - `1e-4 + warmup_cosine`
  - but still weak on `FLRYLLFGI` relative to `A03`

## Takeaway

- `A03_log10_100k` is the clear broad-contract winner of this sweep.
- The best pure validation-loss condition for `A03` is `2.8e-4 + onecycle`, but the strongest broad probe behavior comes from `1e-4 + warmup_cosine`.
- `A07_mhcflurry_100k` can fit its bounded target well, but even its best probe configurations remain less biologically balanced than the top `A03` settings.
- High LR `8e-4` is consistently worse and should be treated as outside the stable regime for both families.
- No run numerically diverged, but the `8e-4` settings show clear optimization degradation.
- Relative to the earlier 5-epoch frontier:
  - `A03` benefits strongly from longer training and schedule choice on the broad contract.
  - `A07` remains competitive on loss, but not the preferred broad-data biology choice.

## Artifacts

- [options_vs_perf.md](./options_vs_perf.md)
- [options_vs_perf.json](./options_vs_perf.json)
- [analysis/parsed_metrics.csv](./analysis/parsed_metrics.csv)
- [analysis/parsed_metrics.json](./analysis/parsed_metrics.json)
- [analysis/per_epoch_metrics.csv](./analysis/per_epoch_metrics.csv)
- [analysis/per_epoch_metrics.json](./analysis/per_epoch_metrics.json)
- [analysis/summary.json](./analysis/summary.json)
- [analysis/a03_log10_100k_val_loss_by_epoch.png](./analysis/a03_log10_100k_val_loss_by_epoch.png)
- [analysis/a03_log10_100k_probe_score_by_epoch.png](./analysis/a03_log10_100k_probe_score_by_epoch.png)
- [analysis/a07_mhcflurry_100k_val_loss_by_epoch.png](./analysis/a07_mhcflurry_100k_val_loss_by_epoch.png)
- [analysis/a07_mhcflurry_100k_probe_score_by_epoch.png](./analysis/a07_mhcflurry_100k_probe_score_by_epoch.png)
- [analysis/best_val_vs_probe_score.png](./analysis/best_val_vs_probe_score.png)
- [variants.md](./variants.md)
- [manifest.json](./manifest.json)
