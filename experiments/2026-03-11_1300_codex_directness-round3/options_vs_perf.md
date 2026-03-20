# Directness Round 3

Broad-contract 7-allele class-I bake-off on the same numeric binding contract, extending round 2 with `abs_only` positional modes and affinity target/output encodings.

## Contract

- Alleles: `HLA-A*02:01`, `HLA-A*24:02`, `HLA-A*03:01`, `HLA-A*11:01`, `HLA-A*01:01`, `HLA-B*07:02`, `HLA-B*44:02`
- Measurement profile: `numeric_no_qualitative`
- Qualifier filter: `all`
- Warm start: `mhc-pretrain-20260308b`
- Rows: `41049` total, `32855` train, `8194` val
- Numeric assay families used: `IC50`, `KD (~IC50)`, `KD (~EC50)`, direct `KD`, `EC50`
- No qualitative binding rows in this contract

## Positional Sweep (`Q00..Q08`)

| design | peptide pos | groove pos | residual | encoding | cap nM | params | setup s | epoch s | val loss | SLL A02/A24 | SLL ratio | FLR A02/A24 | FLR ratio | NFL A02/A24 | NFL ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |
| `Q00` | `triple` | `triple` | `shared_base_segment_residual` | `log10` | 50000 | 4799364 | 63.8 | 108.6 | 0.8945 | 14.5 / 7840.2 | 539.3x | 43.3 / 48.5 | 1.1x | 5098.7 / 1056.1 | 4.8x |
| `Q01` | `triple` | `abs_only` | `shared_base_segment_residual` | `log10` | 50000 | 4799364 | 98.1 | 113.9 | 0.9043 | 22.2 / 7439.1 | 334.7x | 20.5 / 4510.2 | 219.6x | 1055.9 / 501.1 | 2.1x |
| `Q02` | `triple` | `triple_plus_abs` | `shared_base_segment_residual` | `log10` | 50000 | 4799364 | 78.1 | 110.1 | 0.8804 | 25.0 / 3307.3 | 132.1x | 27.4 / 1026.9 | 37.5x | 2683.1 / 683.5 | 3.9x |
| `Q03` | `abs_only` | `triple` | `shared_base_segment_residual` | `log10` | 50000 | 4799364 | 76.3 | 113.7 | 0.9122 | 23.5 / 2634.7 | 112.3x | 22.0 / 115.2 | 5.2x | 225.3 / 232.7 | 1.0x |
| `Q04` | `abs_only` | `abs_only` | `shared_base_segment_residual` | `log10` | 50000 | 4799364 | 62.8 | 105.2 | 0.9725 | 21.8 / 8103.4 | 372.0x | 45.0 / 2684.8 | 59.7x | 891.3 / 252.3 | 3.5x |
| `Q05` | `abs_only` | `triple_plus_abs` | `shared_base_segment_residual` | `log10` | 50000 | 4799364 | 56.4 | 89.9 | 0.9587 | 16.4 / 9848.3 | 599.3x | 20.0 / 1269.4 | 63.4x | 432.4 / 188.6 | 2.3x |
| `Q06` | `triple_plus_abs` | `triple` | `shared_base_segment_residual` | `log10` | 50000 | 4799364 | 112.1 | 115.7 | 0.9530 | 18.3 / 3098.5 | 169.5x | 31.5 / 270.3 | 8.6x | 2777.4 / 755.9 | 3.7x |
| `Q07` | `triple_plus_abs` | `abs_only` | `shared_base_segment_residual` | `log10` | 50000 | 4799364 | 61.8 | 90.9 | 0.9624 | 17.9 / 9970.2 | 558.2x | 39.2 / 7290.4 | 185.9x | 864.5 / 1597.7 | 0.5x |
| `Q08` | `triple_plus_abs` | `triple_plus_abs` | `shared_base_segment_residual` | `log10` | 50000 | 4799364 | 62.5 | 107.2 | 0.9425 | 21.9 / 11389.0 | 520.5x | 158.0 / 3600.0 | 22.8x | 2660.1 / 444.0 | 6.0x |

## Encoding Sweep (`E00..E03`)

| design | peptide pos | groove pos | residual | encoding | cap nM | params | setup s | epoch s | val loss | SLL A02/A24 | SLL ratio | FLR A02/A24 | FLR ratio | NFL A02/A24 | NFL ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |
| `E00` | `triple` | `triple_plus_abs` | `shared_base_segment_residual` | `log10` | 50000 | 4799364 | 71.9 | 114.1 | 0.9395 | 14.8 / 2177.5 | 147.2x | 26.1 / 467.8 | 17.9x | 5971.8 / 348.3 | 17.1x |
| `E01` | `triple` | `triple_plus_abs` | `shared_base_segment_residual` | `log10` | 100000 | 4799364 | 67.1 | 110.4 | 1.0345 | 21.8 / 14686.5 | 674.5x | 29.5 / 6120.7 | 207.6x | 3011.8 / 1478.2 | 2.0x |
| `E02` | `triple` | `triple_plus_abs` | `shared_base_segment_residual` | `mhcflurry` | 50000 | 4799364 | 63.1 | 108.2 | 0.0441 | 26.6 / 8684.2 | 326.4x | 167.7 / 301.4 | 1.8x | 487.3 / 68.6 | 7.1x |
| `E03` | `triple` | `triple_plus_abs` | `shared_base_segment_residual` | `mhcflurry` | 100000 | 4799364 | 75.8 | 110.3 | 0.0364 | 24.0 / 11793.3 | 490.7x | 27.8 / 831.5 | 29.9x | 6939.4 / 796.8 | 8.7x |

## Read

- Best positional variant by validation loss: `Q02` (`val_loss=0.8804`)
- Best encoding variant by validation loss: `E03` (`val_loss=0.0364`)
- Strongest `SLLQHLIGL` separation in this round: `E01` (21.8 / 14686.5 nM, 674.5x)
