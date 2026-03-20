# Modal Hardware Bakeoff
- Agent: `codex`
- Source script: `scripts/benchmark_modal_hardware.py`
- Purpose: determine whether larger broad-contract designs should default to `H100!` instead of `A100`, while keeping batch size fixed.

## Dataset Contract
- 7 class-I alleles: `HLA-A*02:01`, `HLA-A*24:02`, `HLA-A*03:01`, `HLA-A*11:01`, `HLA-A*01:01`, `HLA-B*07:02`, `HLA-B*44:02`
- Broad numeric binding families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- `measurement_profile=numeric_no_qualitative`
- `qualifier_filter=all`
- No synthetics, no ranking

## Training Contract
- `5` epochs
- batch size `140` for every condition
- warm start: `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
- hardware matrix: `A100`, `H100!`, `H200`

## Tested Designs
- `DP00`: directness P00 legacy triple/triple
- `DP01`: directness P01 shared_base_segment_residual triple/triple
- `A03`: shared_base_segment_residual + split_kd_proxy
- `A07`: factorized_context_plus_segment_residual + split_kd_proxy

## Results
| Design | GPU | Success | Setup s | Epoch s | GPU util % | Peak alloc GiB | Val loss | SLL A02 / A24 | SLL ratio | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| DP00 | A100 | no | 61.9 |  |  |  |  |  |  | OOM before epoch 1 |
| DP00 | H100! | yes | 34.1 | 63.6 | 71.8 | 35.47 | 2.8347 | 2159.3 / 2159.3 | 1.0x | completed 5 epochs |
| DP00 | H200 | yes | 58.1 | 67.8 | 60.4 | 35.47 | 2.8200 | 2157.9 / 2157.9 | 1.0x | completed 5 epochs |
| DP01 | A100 | no | 61.1 |  |  |  |  |  |  | OOM before epoch 1 |
| DP01 | H100! | yes | 34.1 | 64.2 | 72.8 | 35.47 | 2.0346 | 2013.9 / 2014.8 | 1.0x | completed 5 epochs |
| DP01 | H200 | yes | 41.5 | 64.9 | 62.3 | 35.47 | 2.1196 | 1975.9 / 1976.9 | 1.0x | completed 5 epochs |
| A03 | A100 | yes | 48.5 | 86.4 | 60.8 | 17.91 | 0.8426 | 215.5 / 2823.9 | 13.1x | completed 5 epochs |
| A03 | H100! | yes | 34.7 | 52.2 | 62.9 | 17.96 | 0.8485 | 31.1 / 10666.0 | 343.4x | completed 5 epochs |
| A03 | H200 | yes | 51.9 | 59.5 | 48.8 | 17.96 | 0.8611 | 24.7 / 2127.0 | 86.0x | completed 5 epochs |
| A07 | A100 | yes | 74.1 | 120.3 | 42.3 | 17.91 | 0.7837 | 237.5 / 2224.5 | 9.4x | completed 5 epochs |
| A07 | H100! | yes | 39.1 | 53.7 | 59.5 | 17.96 | 0.7922 | 197.2 / 3031.5 | 15.4x | completed 5 epochs |
| A07 | H200 | yes | 77.0 | 62.7 | 47.6 | 17.96 | 0.7569 | 41.9 / 2295.5 | 54.8x | completed 5 epochs |

## Takeaway
- `A100` is not a safe default for larger broad-contract designs at fixed batch size `140`; both `DP00` and `DP01` OOM there.
- `H100!` is the best default hardware contract for future Modal experiments in this repo. It avoids the `A100` OOMs and gives the best speed among successful runs.
- `H200` is viable, but in this bakeoff it was not a clear win over `H100!` on wall-clock.
- Among the canonical Presto designs, `A03` improved markedly on `H100!` versus `A100` for `SLLQHLIGL` separation while also cutting epoch time from `86.4s` to `52.2s`.
- `A07` also sped up substantially on `H100!`, but `A03` showed the cleaner biologic gain in this hardware bakeoff.
