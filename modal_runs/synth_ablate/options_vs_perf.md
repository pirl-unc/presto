| run | synthetic mode(s) | best epoch | best val loss | SLLQHLIGL A0201 | SLLQHLIGL A2402 | FLRYLLFGI ordering | NFLIKFLLI ordering | groove audit | verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `synth-ablate-none-20260309a` | none | `9` | `0.6882` | `110.0 nM` | `2382.4 nM` | correct (`104.8 < 1342.7`) | correct (`8.2 < 69.2`) | real rows only, clean `91:93` | current biologic baseline |
| `synth-ablate-all-20260309a` | all six modes | `9` | `0.6768` | `20867.7 nM` | `43182.3 nM` | barely correct (`21534.9 < 23838.5`) | correct (`4857.2 < 33225.5`) | mixed bundle includes fallback-heavy `mhc_scramble` | lower val loss, worse calibration |
| `synth-ablate-peptide_scramble-20260309a` | `peptide_scramble` | `9` | `0.6599` | `26647.3 nM` | `27514.0 nM` | wrong-sign (`20751.0 > 16238.1`) | correct (`8975.4 < 41394.1`) | clean groove inputs | rejects A0201-specific ranking |
| `synth-ablate-peptide_random-20260309b` | `peptide_random` | `8` | `0.6961` | `32915.2 nM` | `46817.0 nM` | barely correct (`37859.7 < 40158.8`) | wrong-sign (`40345.3 > 44217.2`) | clean groove inputs | degrades probe quality |
| `synth-ablate-mhc_scramble-20260309b` | `mhc_scramble` | `5` | `0.7136` | `619.1 nM` | `1103.7 nM` | correct (`208.8 < 272.4`) | wrong-sign (`269.1 > 209.5`) | fallback on `54/128` sampled synthetic rows; many noncanonical groove lengths | structurally unsafe |
| `synth-ablate-mhc_random-20260309c` | `mhc_random` | pending | pending | pending | pending | pending | pending | pending | waiting on Modal |
| `synth-ablate-no_mhc_alpha-20260309c` | `no_mhc_alpha` | pending | pending | pending | pending | pending | pending | pending | waiting on Modal |
| `synth-ablate-no_mhc_beta-20260309c` | `no_mhc_beta` | pending | pending | pending | pending | pending | pending | pending | waiting on Modal |

## Current read

- Validation loss is not ranking the biologically correct runs.
- The best biologic behavior is still the no-synthetic baseline.
- `mhc_scramble` is the clearest bad mode:
  - it frequently breaks groove parsing on synthetic rows
  - it should not remain in the default synthetic bundle
- `peptide_scramble` and `peptide_random` preserve groove inputs, but they still distort quantitative calibration enough to hurt probe behavior on the current assay path.
- Any future synthetic-negative reintroduction should start from:
  - lower ratios than `0.25`
  - no `mhc_scramble`
  - explicit probe checks, not validation loss alone
