# Runtime M1 Rerun Summary

Fixed semantic contract: `legacy_m1`, 7-allele exact `IC50`, warm start, no synthetics, no ranking.

| variant | epoch_wall_s | data_wait_s | forward_s | backward_s | val_loss | SLL A02 nM | SLL A24 nM | SLL ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V00` | 46.68 | 1.28 | 22.08 | 5.73 | 1.0639 | 99.1 | 26083.1 | 263.2x |
| `V10` | 49.42 | 1.12 | 27.17 | 5.87 | 1.0797 | 37.1 | 26738.5 | 721.5x |
| `V06` | 49.75 | 1.12 | 27.70 | 6.01 | 1.1262 | 16.7 | 16942.5 | 1014.7x |
| `V09` | 49.83 | 1.48 | 27.46 | 5.99 | 1.0878 | 18.9 | 3683.6 | 194.5x |
| `V08` | 50.05 | 1.14 | 28.00 | 6.00 | 1.1653 | 54.9 | 33419.2 | 608.2x |
| `V11` | 51.45 | 1.58 | 28.26 | 5.93 | 1.1403 | 35.6 | 16234.9 | 455.4x |
| `V07` | 52.13 | 1.45 | 29.68 | 6.04 | 1.0529 | 44.9 | 29042.4 | 647.1x |
| `V01` | 52.75 | 0.92 | 27.84 | 5.92 | 1.0578 | 93.2 | 26977.6 | 289.6x |
| `V02` | 54.37 | 1.09 | 29.29 | 5.82 | 1.0579 | 143.1 | 28564.9 | 199.7x |
| `V03` | 55.25 | 1.44 | 29.70 | 5.91 | 1.0906 | 53.1 | 12320.7 | 232.0x |
| `V05` | 58.83 | 2.41 | 32.17 | 6.63 | 1.0639 | 139.6 | 34573.8 | 247.6x |

## Notes

- `V04` did not produce usable epoch output in the first rerun and is being relaunched.
- `V09` completed training; its local launch log ended with a Modal client timeout after epoch output was already written.
- Lower `epoch_wall_s` is better; probe preservation is tracked via the `SLLQHLIGL` ratio and should be judged together with `val_loss`.
