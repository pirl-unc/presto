# Directness Round 2c

Broad-contract 7-allele class-I bake-off on the same numeric binding contract.

## Contract

- Alleles: `HLA-A*02:01`, `HLA-A*24:02`, `HLA-A*03:01`, `HLA-A*11:01`, `HLA-A*01:01`, `HLA-B*07:02`, `HLA-B*44:02`
- Measurement profile: `numeric_no_qualitative`
- Qualifier filter: `all`
- Warm start: `mhc-pretrain-20260308b`
- Rows: `41049` total, `32855` train, `8194` val
- Numeric assay families used: `IC50`, `KD (~IC50)`, `KD (~EC50)`, direct `KD`, `EC50`
- Measurement counts in train split: `{'dissociation constant KD': 2097, 'dissociation constant KD (~EC50)': 8542, 'dissociation constant KD (~IC50)': 12989, 'half maximal effective concentration (EC50)': 304, 'half maximal inhibitory concentration (IC50)': 8923}`
- Qualifier counts in train split: `{'0': 24028, '1': 8827}`
- No qualitative binding rows in this contract

## Canonical Presto Variants

| design | peptide pos | groove pos | assay residual | params | setup s | val loss | SLL A02/A24 | SLL ratio | FLR A02/A24 | FLR ratio | NFL A02/A24 | NFL ratio |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |
| `P01` | `triple` | `triple` | `shared_base_segment_residual` | 4799364 | 65.7 | 0.8879 | 21.0 / 12487.6 | 593.7x | 115.2 / 2898.3 | 25.2x | 4944.4 / 1611.0 | 3.1x |
| `P02` | `triple` | `triple_plus_abs` | `legacy` | 4799172 | 65.3 | 0.9283 | 26.0 / 6746.2 | 259.1x | 208.9 / 80.1 | 0.4x | 3497.0 / 561.2 | 6.2x |
| `P05` | `triple_plus_abs` | `triple` | `shared_base_segment_residual` | 4799364 | 50.0 | 0.9284 | 27.0 / 9501.6 | 351.7x | 31.7 / 6953.1 | 219.2x | 4953.2 / 671.9 | 7.4x |
| `P03` | `triple` | `triple_plus_abs` | `shared_base_segment_residual` | 4799364 | 49.5 | 0.9323 | 24.6 / 15826.9 | 644.5x | 27.0 / 1806.0 | 66.9x | 3225.3 / 774.5 | 4.2x |
| `P07` | `triple_plus_abs` | `triple_plus_abs` | `shared_base_segment_residual` | 4799364 | 53.0 | 0.9686 | 15.2 / 2093.7 | 138.2x | 21.6 / 5499.1 | 254.4x | 5304.3 / 527.4 | 10.1x |
| `P06` | `triple_plus_abs` | `triple_plus_abs` | `legacy` | 4799172 | 48.2 | 1.0496 | 36.7 / 19628.7 | 534.2x | 104.2 / 922.9 | 8.9x | 109.0 / 2131.4 | 0.1x |
| `P00` | `triple` | `triple` | `legacy` | 4799172 | 49.8 | 1.1270 | 42.1 / 1030.1 | 24.4x | 5835.1 / 266.4 | 0.0x | 373.5 / 2544.3 | 0.1x |
| `P04` | `triple_plus_abs` | `triple` | `legacy` | 4799172 | 48.3 | 1.1681 | 18.2 / 7571.0 | 415.9x | 51.4 / 58.0 | 1.1x | 21.4 / 747.2 | 0.0x |

## Groove Controls

| design | model | ranking | params | val loss | SLL A02/A24 | SLL ratio | FLR A02/A24 | FLR ratio | NFL A02/A24 | NFL ratio |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |
| `G0` | `groove_mlp` | `off` | 26497 | 1.9354 | 0.1 / 0.1 | 1.6x | 0.0 / 0.1 | 1.3x | 0.0 / 0.1 | 0.8x |
| `G0R` | `groove_mlp` | `on` | 26497 | 3.9814 | 0.1 / 0.1 | 1.8x | 0.0 / 0.0 | 1.2x | 0.0 / 0.0 | 0.8x |
| `G1` | `groove_transformer` | `off` | 392705 | 0.8513 | 10.7 / 18265.6 | 1711.7x | 11.3 / 13867.8 | 1230.8x | 904.1 / 4004.3 | 0.2x |
| `G1R` | `groove_transformer` | `on` | 392705 | 1.6430 | 18.8 / 12208.2 | 649.2x | 47.9 / 1205.2 | 25.2x | 317.6 / 82.7 | 3.8x |

## Read

- `P01` has the best validation loss among canonical Presto variants on the broad contract.
- `P03` is still the strongest overall canonical compromise across `SLLQHLIGL`, `FLRYLLFGI`, and `NFLIKFLLI`.
- `P05` remains the strongest `FLRYLLFGI` canonical variant.
- No `abs_only` positional mode was tested in this round; only `triple` and `triple_plus_abs`.
- `G1` (groove transformer, pure regression) is the strongest groove-only control on the broad contract.
- `G1R` improves `FLRYLLFGI` less than `G1` and still does not recover correct `NFLIKFLLI` sign by epoch 3.
- `G0` / `G0R` are not competitive; they collapse into unrealistically tiny affinity predictions despite modest allele ordering.
