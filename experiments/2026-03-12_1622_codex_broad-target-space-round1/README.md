# Broad Target-Space Bakeoff

- Date: `2026-03-12`
- Agent: `codex` / `gpt-5`
- Purpose: compare target-space encodings and affinity caps for the two leading broad-contract canonical Presto hypotheses on a fixed 5-epoch budget.
- Source script: `scripts/benchmark_target_space_bakeoff.py`

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
- Train / val split: `32,855 / 8,194`
- Synthetic data: none
- Ranking / contrastive: none

## Training / Pretraining Contract

- Warm start: `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
- Epochs: `5`
- Batch size: `256`
- GPU: `H100!`
- Runtime defaults:
  - `allow_tf32=true`
  - `matmul_precision=high`
  - `num_workers=0`
  - `pin_memory=true`

## Assay Label -> Output Mapping

- `IC50` -> `assays.IC50_nM`
- direct `KD` -> direct KD output
- `KD (~IC50)` / `KD (~EC50)` -> split proxy-KD outputs when enabled by the design
- `EC50` -> `assays.EC50_nM`

## Conditions Tested

- `A03`:
  - `shared_base_segment_residual`
  - `split_kd_proxy`
- `A07`:
  - `shared_base_factorized_context_plus_segment_residual`
  - `split_kd_proxy`
- target spaces:
  - `log10_50k`
  - `log10_100k`
  - `mhcflurry_50k`
  - `mhcflurry_100k`

## Results

# Broad Target-Space Results

Fixed contract: 7 class-I alleles, broad numeric binding (`IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`), qualifiers `all`, warm start, no synthetics, no ranking, batch size `256`, `5` epochs, `H100!`.

| design | target | best val | epoch s | gpu util % | SLL A02 / A24 | SLL ratio | FLR A02 / A24 | FLR ratio | NFL A02 / A24 | NFL ratio |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |
| `A03` | `log10_100k` | 0.9847 | 50.2 | 58.5 | 8.5 / 19197.1 | 2261.5 | 16.9 / 3268.2 | 193.9 | 1732.7 / 227.8 | 7.6 |
| `A03` | `log10_50k` | 0.9503 | 51.6 | 57.0 | 16.3 / 44739.8 | 2746.2 | 20.5 / 9563.4 | 466.8 | 970.7 / 1036.1 | 0.9 |
| `A03` | `mhcflurry_100k` | 0.0371 | 46.9 | 64.4 | 50.0 / 39178.4 | 784.0 | 13.4 / 3457.2 | 258.5 | 3353.9 / 451.8 | 7.4 |
| `A03` | `mhcflurry_50k` | 0.0410 | 45.3 | 64.1 | 63.0 / 30671.3 | 486.7 | 12.5 / 13700.3 | 1096.8 | 2725.6 / 746.1 | 3.7 |
| `A07` | `log10_100k` | 0.8137 | 50.4 | 57.4 | 110.7 / 7824.1 | 70.7 | 34.9 / 1606.9 | 46.1 | 14411.5 / 156.1 | 92.3 |
| `A07` | `log10_50k` | 0.7580 | 49.8 | 59.5 | 81.3 / 13795.6 | 169.7 | 29.0 / 1192.0 | 41.1 | 6143.9 / 61.5 | 99.9 |
| `A07` | `mhcflurry_100k` | 0.0359 | 49.3 | 60.3 | 61.6 / 30304.8 | 491.8 | 45.5 / 9186.6 | 201.8 | 4741.1 / 39.5 | 119.9 |
| `A07` | `mhcflurry_50k` | 0.0326 | 50.9 | 57.0 | 70.4 / 26920.3 | 382.3 | 20.0 / 3502.2 | 175.1 | 3320.9 / 63.9 | 51.9 |


## Winner / Preferred Conditions

- Best `A03` target-space by probe behavior:
  - `log10_100k`
  - `SLL=8.5 / 19197.1`
  - `FLR=16.9 / 3268.2`
  - `NFL=1732.7 / 227.8`
- Best `A07` target-space by validation loss:
  - `log10_50k`
  - `best_val=0.7580`
- Best `A07` target-space by broad probe balance:
  - `mhcflurry_100k`

## Takeaway

- `A03 log10_100k` is the most compelling broad-contract target-space setting in this family.
  - It yields the strongest combination of weak-tail calibration and probe separation.
- `A07` prefers the simpler `log10` target space more than the bounded MHCflurry target in this fixed 5-epoch comparison.
- Increasing the cap from `50k` to `100k` helps the weak tail for `A03` without obviously breaking the strong binders.
- The MHCflurry-style bounded target can fit its own space, but it did not produce the best broad biologic behavior in this sweep.

## 50k vs 100k Cap Comparison

Question: is `100k` always better than `50k`, regardless of encoding?

Answer: **no**.

Important caveat:
- this experiment saved aggregate/probe metrics only
- it did **not** save per-example validation predictions
- so exact-`IC50` rank correlation and `<=500 nM` accuracy cannot be reconstructed from these artifacts

### Within-encoding cap comparisons

#### `A03`

| encoding | `50k` best val | `100k` best val | `50k` SLL ratio | `100k` SLL ratio | `50k` FLR ratio | `100k` FLR ratio | `50k` NFL ratio | `100k` NFL ratio | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `log10` | `0.9503` | `0.9847` | `2746.2` | `2261.5` | `466.8` | `193.9` | `0.9` | `7.6` | `50k` better `SLL`, `100k` much better `NFL`, `50k` better val |
| `mhcflurry` | `0.0410` | `0.0371` | `486.7` | `784.0` | `1096.8` | `258.5` | `3.7` | `7.4` | `100k` better `SLL`, `100k` better `NFL`, `100k` better val |

#### `A07`

| encoding | `50k` best val | `100k` best val | `50k` SLL ratio | `100k` SLL ratio | `50k` FLR ratio | `100k` FLR ratio | `50k` NFL ratio | `100k` NFL ratio | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `log10` | `0.7580` | `0.8137` | `169.7` | `70.7` | `41.1` | `46.1` | `99.9` | `92.3` | `50k` better `SLL`, `50k` better `NFL`, `50k` better val |
| `mhcflurry` | `0.0326` | `0.0359` | `382.3` | `491.8` | `175.1` | `201.8` | `51.9` | `119.9` | `100k` better `SLL`, `100k` better `NFL`, `50k` better val |

## Artifacts

- [options_vs_perf.md](./options_vs_perf.md)
- [options_vs_perf.json](./options_vs_perf.json)
- [analysis/parsed_metrics.csv](./analysis/parsed_metrics.csv)
- [analysis/parsed_metrics.json](./analysis/parsed_metrics.json)
- [analysis/cap_comparison.md](./analysis/cap_comparison.md)
- [analysis/cap_comparison_summary.json](./analysis/cap_comparison_summary.json)
- [analysis/a03_50k_vs_100k_probe_ratios.png](./analysis/a03_50k_vs_100k_probe_ratios.png)
- [analysis/a07_50k_vs_100k_probe_ratios.png](./analysis/a07_50k_vs_100k_probe_ratios.png)
- [analysis/best_val_by_design_encoding_cap.png](./analysis/best_val_by_design_encoding_cap.png)
- [analysis/sll_ratio_by_design_encoding_cap.png](./analysis/sll_ratio_by_design_encoding_cap.png)
- [variants.md](./variants.md)
- [manifest.json](./manifest.json)
