# Positional Composition Round 1

- Agent: `codex`
- Source script: `scripts/benchmark_positional_composition.py`
- Date: `2026-03-11`
- Goal: replace the ad hoc additive positional schemes with a more principled bakeoff over explicit start/end/fraction composition, on both canonical Presto and the groove-transformer control, under one matched broad-data contract.

## Dataset Contract

- Source: `data/merged_deduped.tsv`
- Alleles:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `HLA-A*03:01`
  - `HLA-A*11:01`
  - `HLA-A*01:01`
  - `HLA-B*07:02`
  - `HLA-B*44:02`
- Measurement profile: `numeric_no_qualitative`
- Included assay families:
  - `IC50`
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  - `EC50`
- Excluded assay families:
  - `qualitative binding`
- Qualifier policy: `all`
  - exact and inequality rows are included
  - censor-aware loss is used
- Train / val split:
  - `32,855 / 8,194`

## Assay Label -> Output Mapping

- `IC50` -> `assays.IC50_nM`
- `EC50` -> `assays.EC50_nM`
- direct `KD` -> `assays.KD_nM`
- `KD (~IC50)` -> `assays.KD_nM`
- `KD (~EC50)` -> `assays.KD_nM`

Important caveat:
- this experiment keeps the current merged-KD family assumption
- direct `KD` and proxy-KD families are not split here

## Training Contract

- Warm start: `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
- Epochs: `3`
- Synthetics: `off`
- Ranking / contrastive: `off`
- Target space: `log10(nM)` with `50,000 nM` cap
- Canonical Presto batch size: `140`
- Groove-transformer batch size: `256`
- Runtime knobs:
  - `num_workers=0`
  - `pin_memory=true`
  - `persistent_workers=false`
  - `allow_tf32=true`
  - `matmul_precision=high`

## Conditions Tested

Position-composition modes:
1. `start_only`
2. `end_only`
3. `start_plus_end`
4. `concat(start,end)`
5. `concat(start,end,nterm_frac,cterm_frac)`
6. `mlp(concat(start,end))`
7. `mlp(concat(start,end,nterm_frac,cterm_frac))`
8. `triple_baseline`

Model families:
- `P00..P07`: canonical Presto, `P03`-style broad-contract setup
- `G00..G07`: groove-transformer control, same broad-data contract

## Canonical Presto Results

| design | position mode | val loss | setup s | epoch s | GPU util mean | `SLLQHLIGL` A02 / A24 | `FLRYLLFGI` A02 / A24 | `NFLIKFLLI` A02 / A24 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `P00` | `start_only` | `0.9474` | `53.8` | `84.1` | `61.4%` | `11.1 / 6389.2` | `17.6 / 9998.6` | `1046.4 / 317.5` |
| `P01` | `end_only` | `0.9475` | `61.2` | `107.5` | `44.7%` | `21.3 / 483.8` | `15.5 / 80.5` | `274.7 / 173.4` |
| `P02` | `start_plus_end` | `0.9274` | `51.8` | `85.8` | `56.4%` | `12.0 / 13658.0` | `10.5 / 5141.8` | `818.3 / 1023.6` |
| `P03` | `concat(start,end)` | `0.8975` | `73.4` | `117.1` | `42.8%` | `34.7 / 8874.6` | `21.6 / 2233.5` | `1171.1 / 338.1` |
| `P04` | `concat(start,end,frac)` | `0.8785` | `52.8` | `77.4` | `63.0%` | `19.2 / 10540.6` | `40.9 / 981.9` | `153.1 / 210.1` |
| `P05` | `mlp(concat(start,end))` | `0.9411` | `54.1` | `95.9` | `54.4%` | `15.1 / 13072.1` | `58.3 / 7030.5` | `462.1 / 240.4` |
| `P06` | `mlp(concat(start,end,frac))` | `0.9098` | `48.6` | `89.8` | `57.7%` | `21.8 / 4330.8` | `36.5 / 2122.4` | `1195.8 / 202.7` |
| `P07` | `triple_baseline` | `0.8968` | `54.1` | `85.0` | `59.0%` | `29.7 / 14512.4` | `88.1 / 2923.4` | `2606.7 / 633.5` |

## Groove-Transformer Control Results

| design | position mode | val loss | `SLLQHLIGL` A02 / A24 | `FLRYLLFGI` A02 / A24 | `NFLIKFLLI` A02 / A24 |
| --- | --- | ---: | --- | --- | --- |
| `G00` | `start_only` | `0.8616` | `25.1 / 22388.1` | `108.9 / 4761.3` | `459.2 / 3760.8` |
| `G01` | `end_only` | `0.9232` | `11.9 / 19805.8` | `15.1 / 3853.2` | `1759.5 / 349.8` |
| `G02` | `start_plus_end` | `0.8752` | `47.7 / 20536.5` | `142.4 / 2838.1` | `1394.0 / 433.8` |
| `G03` | `concat(start,end)` | `0.8730` | `37.2 / 19219.0` | `128.8 / 2346.1` | `2302.5 / 5125.5` |
| `G04` | `concat(start,end,frac)` | `0.8736` | `56.1 / 21345.3` | `66.2 / 9408.2` | `1384.4 / 651.6` |
| `G05` | `mlp(concat(start,end))` | `0.8552` | `15.7 / 7632.4` | `11.0 / 6071.1` | `5269.6 / 2830.4` |
| `G06` | `mlp(concat(start,end,frac))` | `0.8208` | `21.4 / 27420.4` | `16.8 / 2354.9` | `947.4 / 2692.0` |
| `G07` | `triple_baseline` | `0.8938` | `16.0 / 16171.5` | `19.2 / 7890.8` | `3453.0 / 579.2` |

## Comparison Notes

- `end_only` is clearly weak in both families.
- One-sided position is not sufficient on this broad contract.
- For canonical Presto, the best new loss result is `P04`, but `P07` remains stronger on the full three-probe picture.
- For the groove-transformer control, explicit learned composition of start/end/fraction is clearly better than simple additive baselines.
- `P04` is notable because it improves validation loss *and* is one of the faster canonical Presto variants.
- `P03` remains a strong canonical compromise when the priority is preserving `SLLQHLIGL` and `FLRYLLFGI` together.

## Takeaway

Promote these canonical Presto candidates into the next broad-contract bakeoff:
1. `P04`
   - best validation loss
   - fastest strong canonical variant
2. `P07`
   - strongest carry-forward baseline from the old triple family
3. `P03`
   - best prior broad-contract compromise

Promote this groove-transformer control:
1. `G06`
   - strongest direct-segment positional-composition result

Primary conclusion:
- the next positional step should not be more redundant additive embeddings
- it should be explicit composition of start/end/fraction signals
- especially for the groove-transformer family, and potentially for canonical Presto if `P04` continues to hold up in longer or assay-head-target-space sweeps

## Artifacts

- [launch logs](./launch_logs/)
- expected comparison table path: `options_vs_perf.md`
- expected machine-readable table path: `options_vs_perf.json`
