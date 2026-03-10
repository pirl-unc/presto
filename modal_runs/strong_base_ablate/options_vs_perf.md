# Strong-Base Affinity Ablations

Base contract:
- broader class-I panel
- exact `IC50`
- warm-start checkpoint `mhc-pretrain-20260308b`
- `affinity_assay_mode=legacy`
- `binding_contrastive_weight=0`
- probe panel:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`

| option | run_id | status | best epoch | best val loss | SLLQHLIGL (A0201/A2402 nM) | FLRYLLFGI (A0201/A2402 nM) | NFLIKFLLI (A0201/A2402 nM) | notes |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| baseline | `class1-panel-ic50-exact-warmstart-legacy-20260309b` | running |  |  |  |  |  | parity target vs `E006` |
| peptide ranking | `class1-panel-ic50-exact-warmstart-legacy-peprank-20260309b` | running |  |  |  |  |  | no synthetics |
| synth `peptide_random` | `class1-panel-ic50-exact-warmstart-legacy-synth-peptide-random-20260309b` | running |  |  |  |  |  | refreshed each epoch, fraction `0.25` |
| synth `no_mhc_alpha` | `class1-panel-ic50-exact-warmstart-legacy-synth-no-mhc-alpha-20260309b` | running |  |  |  |  |  | refreshed each epoch, fraction `0.25` |
| synth `no_mhc_beta` | `class1-panel-ic50-exact-warmstart-legacy-synth-no-mhc-beta-20260309b` | running |  |  |  |  |  | refreshed each epoch, fraction `0.25` |
