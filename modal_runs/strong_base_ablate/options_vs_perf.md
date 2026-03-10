# Strong-Base Affinity Ablations

Matched base contract:
- 7-allele class-I panel:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `HLA-A*03:01`
  - `HLA-A*11:01`
  - `HLA-A*01:01`
  - `HLA-B*07:02`
  - `HLA-B*44:02`
- exact `IC50`
- warm-start checkpoint `mhc-pretrain-20260308b`
- probe panel:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`

Completed comparisons:

| option | run_id | best epoch | `SLLQHLIGL` A02/A24 | `FLRYLLFGI` A02/A24 | `NFLIKFLLI` A02/A24 | notes |
| --- | --- | ---: | --- | --- | --- | --- |
| `legacy_baseline` | `class1-panel-ic50-exact-warmstart-legacy-e006parity-20260309c` | `12` | `37.2 / 28405.4` | `35.5 / 3562.1` | `5789.9 / 5.64` | restored strong baseline |
| `score_context` | `class1-panel-ic50-exact-scorecontext-e006match-20260309c` | `8` | `445.6 / 11699.7` | `195.2 / 5638.7` | `6604.9 / 10.40` | materially weaker than `legacy` |
| `legacy_peprank` | `class1-panel-ic50-exact-legacy-peprank-e006match-20260309c` | `11` | `47.9 / 25307.1` | `38.5 / 18079.7` | `14451.4 / 27.35` | mixed; improves some tails, hurts others |
| `legacy_m1` | `class1-panel-ic50-exact-legacy-m1-e006match-20260309c` | `10` | `37.3 / 29588.2` | `32.8 / 24346.2` | `5749.3 / 6.10` | current best overall |
| `legacy_m1_peprank` | `class1-panel-ic50-exact-legacy-m1-peprank-e006match-20260309c` | `10` | `28.9 / 24178.9` | `101.7 / 6589.4` | `16584.9 / 2.21` | stronger on `NFLIKFLLI`, worse on `FLRYLLFGI` |

Next runs from the promoted base (`legacy_m1`):

| option | run_id | status | notes |
| --- | --- | --- | --- |
| `legacy_m1_none` | `class1-panel-ic50-exact-legacy-m1-e006match-20260309c` | complete | promoted benchmark base |
| `legacy_m1_synth_peptide_random` | `class1-panel-ic50-exact-legacy-m1-synth-peptide-random-20260309d` | pending | refreshed each epoch, fraction `0.25` |
| `legacy_m1_synth_no_mhc_alpha` | `class1-panel-ic50-exact-legacy-m1-synth-no-mhc-alpha-20260309d` | pending | refreshed each epoch, fraction `0.25` |
| `legacy_m1_synth_no_mhc_beta` | `class1-panel-ic50-exact-legacy-m1-synth-no-mhc-beta-20260309d` | pending | refreshed each epoch, fraction `0.25` |
| `legacy_m1_ic50_censored` | `class1-panel-ic50-allqual-legacy-m1-20260309d` | pending | `qualifier_filter=all`, censor-aware loss |
