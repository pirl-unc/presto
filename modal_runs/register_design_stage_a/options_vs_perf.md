# Stage-A Options vs Performance

Date: 2026-03-08

Dataset / recipe:
- class-I exact `IC50`
- alleles:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `HLA-A*03:01`
  - `HLA-A*11:01`
  - `HLA-A*01:01`
  - `HLA-B*07:02`
  - `HLA-B*44:02`
- warm start: `mhc-pretrain-20260308b`
- peptide ranking on
- allele ranking off
- no synthetic negatives
- best checkpoint chosen by minimum validation loss

## Table

| option | design | status | runs | >=1.5x correct | any correct | mean log10 margin | best/mean val loss | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `M0` | `G0 + R0 + C1` | provisional | `1 debug epoch` | `2/3` | `2/3` | `0.207` | `1.9017` | only a 1-epoch debug run produced artifacts; detached 12-epoch baseline launches failed to publish checkpoints |
| `M1` | `G1 + R1 + C1` | complete | `3 seeds` | `3.00` | `3.00` | `1.799` | `1.2977` | current Stage-A winner |
| `M2` | `G1 + R2 + C2` | complete | `3 seeds` | `3.00` | `3.00` | `1.684` | `1.2591` | slightly better mean val loss, weaker probe margins than `M1` |

## Best runs

| option | best run | best epoch | val loss | `SLLQHLIGL` A0201/A2402 | `FLRYLLFGI` A0201/A2402 | `NFLIKFLLI` A2402/A0201 |
| --- | --- | ---: | ---: | --- | --- | --- |
| `M0` | `register-m0-debug-20260308` | `1` | `1.9017` | `419 / 850 nM` | `303 / 692 nM` | `1102 / 994 nM` |
| `M1` | `register-stagea-20260308b-m1-seed42` | `12` | `1.2918` | `77 / 2894 nM` | `153 / 4619 nM` | `7.7 / 3338 nM` |
| `M2` | `register-stagea-20260308b-m2-seed43` | `12` | `1.2794` | derived from leaderboard JSON | derived from leaderboard JSON | derived from leaderboard JSON |

## Read

- `M1` and `M2` both preserve class-I behavior well on this panel.
- `M1` wins on the biologic probe criterion:
  - larger average log-ratio margins
  - strongest best run on the held-out probe panel
- `M2` is competitive and slightly better on average validation loss, but the probe margin is smaller.
- `M0` is not a fair head-to-head baseline yet because the full detached run path did not produce artifacts.

