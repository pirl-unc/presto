# Experiment Log

Date started: 2026-03-09

Purpose:
- keep a durable, context-independent record of binding-focused experiments
- record architecture, data slice, training contract, best checkpoint, and key
  probe results
- make it obvious which changes improved class-I allele separation and which
  hurt it

Conventions:
- probe ratios are reported as `weaker_allele_nM / stronger_allele_nM`
- for `SLLQHLIGL` and `FLRYLLFGI`, stronger means `HLA-A*02:01`
- for `NFLIKFLLI`, stronger means `HLA-A*24:02`
- "best checkpoint" means minimum validation loss unless stated otherwise

## Architecture shorthand

- `Warm start`
  - 1 epoch MHC-only pretraining on groove sequences for class/species
- `Peptide ranking`
  - same-allele / different-peptide ranking loss
- `Allele ranking`
  - same-peptide / different-allele ranking loss
- `Synthetic negatives`
  - binding negatives added through the synthetic augmentation path
- `M0`
  - `G0 + R0 + C1`
  - groove positions: sequential per half
  - core: fixed contiguous 9-mer window
  - class sharing: shared proposal with class-specific calibration
- `M1`
  - `G1 + R1 + C1`
  - groove positions: start-distance + end-distance + fractional-position MLP
  - core: variable contiguous windows `8,9,10,11`
  - class sharing: shared proposal with class-specific calibration
- `M2`
  - `G1 + R2 + C2`
  - groove positions: same as `M1`
  - core: variable contiguous windows `8,9,10,11`
  - class sharing: shared proposal with class-specific refinement and calibration

## Record format

For each run:
- `ID`
- `Goal`
- `Architecture`
- `Data`
- `Training contract`
- `Best checkpoint`
- `Probe results`
- `Interpretation`
- `Artifacts`

---

## E001

- `ID`: `modal_runs/a0201-a2402-ic50-exact-allrows-ic50only-20260308h`
- `Goal`: clean 2-allele exact-`IC50` baseline without warm start
- `Architecture`:
  - affinity-only focused trainer
  - no warm start
  - no peptide ranking
  - no allele ranking
  - no synthetic negatives
- `Data`:
  - exact `IC50`
  - alleles: `HLA-A*02:01`, `HLA-A*24:02`
  - dataset rows: `7793`
  - train: `6234`
  - val: `1559`
- `Training contract`:
  - strict per-batch allele balance
  - real `IC50_nM` supervision
- `Best checkpoint`:
  - epoch `9`
  - val loss `0.9378`
- `Probe results`:
  - `SLLQHLIGL`: `61.5` vs `123.8 nM`, ratio `2.01`
  - `FLRYLLFGI`: `51.8` vs `61.6 nM`, ratio `1.19`
  - `NFLIKFLLI`: `116.7` vs `250.3 nM`, ratio `2.15`
- `Interpretation`:
  - first clean run with correct signs
  - still too weak on `A*24:02` weak-binder calibration
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/a0201-a2402-ic50-exact-allrows-ic50only-20260308h/summary.json)

## E002

- `ID`: `modal_runs/a0201-a2402-ic50-exact-allrows-ic50only-warmstart-20260308i`
- `Goal`: test whether MHC-only warm start improves exact-`IC50` 2-allele affinity learning
- `Architecture`:
  - same as `E001`
  - plus `Warm start`
- `Data`:
  - same as `E001`
- `Training contract`:
  - same as `E001`
- `Best checkpoint`:
  - epoch `7`
  - val loss `0.9097`
- `Probe results`:
  - `SLLQHLIGL`: `35.3` vs `5677.7 nM`, ratio `160.64`
  - `FLRYLLFGI`: `31.8` vs `134.0 nM`, ratio `4.21`
  - `NFLIKFLLI`: `77.3` vs `2465.0 nM`, ratio `31.90`
- `Interpretation`:
  - warm start was a major improvement
  - sign and separation both improved materially
  - still under-calibrated on `A*24:02` for `SLLQHLIGL` relative to the desired `>10,000 nM`
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/a0201-a2402-ic50-exact-allrows-ic50only-warmstart-20260308i/summary.json)

## E003

- `ID`: `modal_runs/a0201-a2402-ic50-exact-warmstart-peptide-rank-20260308b`
- `Goal`: test peptide-ranking on top of the warm-start exact-`IC50` baseline
- `Architecture`:
  - `Warm start`
  - `Peptide ranking`
  - no synthetic negatives
  - no allele ranking
- `Data`:
  - same 2-allele exact `IC50` slice as `E001`
- `Training contract`:
  - `binding_peptide_contrastive_weight=0.5`
- `Best checkpoint`:
  - epoch `7`
  - val loss `1.1392`
- `Probe results`:
  - `SLLQHLIGL`: `42.3` vs `4445.0 nM`, ratio `105.18`
  - `FLRYLLFGI`: `71.0` vs `2268.1 nM`, ratio `31.96`
  - `NFLIKFLLI`: `165.0` vs `2992.8 nM`, ratio `18.13`
- `Interpretation`:
  - peptide ranking improved some probe margins but hurt validation loss
  - weaker overall than warm start alone for `SLLQHLIGL`
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/a0201-a2402-ic50-exact-warmstart-peptide-rank-20260308b/summary.json)

## E004

- `ID`: `modal_runs/a0201-a2402-ic50-exact-warmstart-synth-20260308b`
- `Goal`: test synthetic weak negatives on top of the warm-start exact-`IC50` baseline
- `Architecture`:
  - `Warm start`
  - `Synthetic negatives`
  - no peptide ranking
  - no allele ranking
- `Data`:
  - exact `IC50`
  - 2 alleles
  - train rows doubled by synthetics: `12468`
  - val: `1559`
- `Training contract`:
  - synthetic negatives only in training
- `Best checkpoint`:
  - epoch `8`
  - val loss `0.8666`
- `Probe results`:
  - `SLLQHLIGL`: `43.4` vs `20231.1 nM`, ratio `466.20`
  - `FLRYLLFGI`: `14.5` vs `436.2 nM`, ratio `30.18`
  - `NFLIKFLLI`: `112.7` vs `17971.0 nM`, ratio `159.41`
- `Interpretation`:
  - strongest weak-binder calibration on the 2-allele setup
  - likely the best path if the immediate target is to push `A*24:02` toward `>10k nM`
  - caveat: synthetic negatives can distort sparsely supported peptide families
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/a0201-a2402-ic50-exact-warmstart-synth-20260308b/summary.json)

## E005

- `ID`: `modal_runs/a0201-a2402-ic50-exact-warmstart-allele-rank-20260308b`
- `Goal`: test same-peptide allele-ranking on the warm-start exact-`IC50` baseline
- `Architecture`:
  - `Warm start`
  - `Allele ranking`
  - no peptide ranking
  - no synthetic negatives
- `Data`:
  - same 2-allele exact `IC50` slice
- `Training contract`:
  - `binding_contrastive_weight=1.0`
- `Best checkpoint`:
  - epoch `12`
  - val loss `1.3685`
- `Probe results`:
  - `SLLQHLIGL`: `88.5` vs `11300.9 nM`, ratio `127.73`
  - `FLRYLLFGI`: `23.9` vs `207.6 nM`, ratio `8.70`
  - `NFLIKFLLI`: `72.6` vs `86.3 nM`, ratio `1.19`
- `Interpretation`:
  - allele ranking hurt overall
  - especially bad on `NFLIKFLLI`
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/a0201-a2402-ic50-exact-warmstart-allele-rank-20260308b/summary.json)

## E006

- `ID`: `modal_runs/class1-panel-ic50-exact-warmstart-20260308a`
- `Goal`: test whether adding more motif-diverse class-I alleles improves groove learning and probe generalization
- `Architecture`:
  - focused affinity trainer
  - `Warm start`
  - no peptide ranking
  - no allele ranking
  - no synthetic negatives
- `Data`:
  - class-I exact `IC50`
  - alleles:
    - `HLA-A*02:01`
    - `HLA-A*24:02`
    - `HLA-A*03:01`
    - `HLA-A*11:01`
    - `HLA-A*01:01`
    - `HLA-B*07:02`
    - `HLA-B*44:02`
  - dataset rows: `10193`
  - train: `8166`
  - val: `2027`
- `Best checkpoint`:
  - epoch `12`
  - val loss `1.0316`
- `Probe results`:
  - `SLLQHLIGL`: `42.2` vs `16590.1 nM`, ratio `393.48`
  - `FLRYLLFGI`: `89.1` vs `10640.2 nM`, ratio `119.36`
  - `NFLIKFLLI`: `6.1` vs `11501.8 nM`, ratio `1896.07`
- `Interpretation`:
  - adding more class-I alleles helped biology a lot
  - this was one of the strongest pure-MHC-I, no-synthetic runs
  - aggregate val loss was worse than the 2-allele baseline, but probe behavior was much better
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/class1-panel-ic50-exact-warmstart-20260308a/summary.json)

## E007

- `ID`: `artifacts/class1_quant_warmstart_smoke2`
- `Goal`: broad class-I all-alleles smoke after warm start
- `Architecture`:
  - focused trainer generalized to `train_all_alleles=true`
  - `Warm start`
  - no ranking
  - no synthetic negatives
- `Data`:
  - class-I quantitative smoke subset
  - dataset rows: `1971`
- `Best checkpoint`:
  - epoch `1`
  - val loss `2.7116`
- `Probe results`:
  - `SLLQHLIGL`: `4457.5` vs `4458.0 nM`, ratio `1.0001`
- `Interpretation`:
  - broad all-alleles smoke without the later architecture/design changes did not separate the key probe
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/artifacts/class1_quant_warmstart_smoke2/summary.json)

## E008

- `ID`: `modal_runs/register_design_stage_a/register-stagea-20260308b-m1-seed42`
- `Goal`: Stage-A benchmark winner among executable class-I-preserving design variants
- `Architecture`:
  - `M1 = G1 + R1 + C1`
  - `Warm start`
  - `Peptide ranking`
  - no allele ranking
  - no synthetic negatives
- `Data`:
  - class-I exact `IC50`
  - same 7-allele panel as `E006`
- `Best checkpoint`:
  - epoch `12`
  - val loss `1.2918`
- `Probe results`:
  - `SLLQHLIGL`: `77.5` vs `2894.3 nM`, ratio `37.36`
  - `FLRYLLFGI`: `152.6` vs `4619.0 nM`, ratio `30.27`
  - `NFLIKFLLI`: `7.7` vs `3338.4 nM`, ratio `432.85`
- `Interpretation`:
  - strongest Stage-A design on biologic probe margins
  - current architecture winner to carry forward
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/register_design_stage_a/register-stagea-20260308b-m1-seed42/summary.json)

## E009

- `ID`: `modal_runs/register_design_stage_a/register-stagea-20260308b-m2-seed43`
- `Goal`: Stage-A comparison of class-specific refinement against `M1`
- `Architecture`:
  - `M2 = G1 + R2 + C2`
  - `Warm start`
  - `Peptide ranking`
  - no allele ranking
  - no synthetic negatives
- `Data`:
  - class-I exact `IC50`
  - same 7-allele panel as `E006`
- `Best checkpoint`:
  - epoch `12`
  - val loss `1.2794`
- `Probe results`:
  - see leaderboard JSON / summary pull
  - aggregate mean probe log10 margin across seeds: `1.684`
- `Interpretation`:
  - viable alternate
  - slightly better mean val loss than `M1`
  - weaker probe margins than `M1`
- `Artifacts`:
  - [leaderboard.json](/Users/iskander/code/presto/modal_runs/register_design_stage_a/leaderboard.json)

## E010

- `ID`: `modal_runs/register_design_stage_a/register-m0-debug-20260308`
- `Goal`: baseline debug run for `M0`
- `Architecture`:
  - `M0 = G0 + R0 + C1`
  - no warm start change relative to Stage-A contract
- `Data`:
  - class-I exact `IC50`
  - same 7-allele panel as `E006`
- `Best checkpoint`:
  - epoch `1` only
  - val loss `1.9017`
- `Probe results`:
  - `SLLQHLIGL`: `419.4` vs `849.7 nM`, ratio `2.03`
  - `FLRYLLFGI`: `303.2` vs `692.5 nM`, ratio `2.28`
  - `NFLIKFLLI`: `1102.3` vs `993.7 nM`, ratio `0.90`
- `Interpretation`:
  - baseline is clearly weaker than `M1/M2`
  - full detached 12-epoch `M0` comparison still needs a clean rerun
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/register_design_stage_a/register-m0-debug-20260308/summary.json)

## E011

- `ID`: `modal_runs/mhc-pretrain-20260308b`
- `Goal`: MHC encoder warm start
- `Architecture`:
  - MHC-only groove encoder pretraining
  - targets:
    - MHC class
    - species category
- `Data`:
  - indexed groove sequences
- `Best checkpoint`:
  - epoch `1`
- `Results`:
  - class accuracy: `1.0000` val
  - species accuracy: `0.9706` val
- `Interpretation`:
  - warm start is justified and repeatedly improved downstream affinity runs
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/mhc-pretrain-20260308b/summary.json)

---

## Current conclusions

1. Best pure class-I probe separations so far:
   - strongest no-synthetic multi-allele run:
     - `E006`
   - strongest Stage-A benchmark design:
     - `E008` (`M1`)
   - strongest weak-binder calibration on the 2-allele setup:
     - `E004` (warm start + synthetic negatives)
2. Changes that helped most:
   - MHC warm start
   - adding more motif-diverse class-I alleles
   - improved groove positional encoding (`G1`)
   - variable core lengths (`R1`)
3. Change that consistently hurt:
   - same-peptide allele ranking
4. Open question:
   - whether joint class-I + class-II quantitative affinity training preserves
     the class-I probe gains under the corrected Stage-B launch contract

## E012

- `ID`: `modal_runs/scorepath_bench/pulls/e004-scorepathb-20260309a`
- `Goal`: verify the narrowed score-fed assay path against the old `E004` synthetic benchmark
- `Architecture`:
  - current affinity path with scalar score outputs exposed
  - affinity score kept in the canonical assay path through the existing mixed-KD route
  - stability score fed into stability assays
  - `Warm start`
  - `Synthetic negatives`
  - no peptide ranking
  - no allele ranking
- `Data`:
  - exact `IC50`
  - 2 alleles:
    - `HLA-A*02:01`
    - `HLA-A*24:02`
- `Best checkpoint`:
  - final epoch `12` only was pulled reliably during the live comparison
- `Probe results`:
  - `SLLQHLIGL`: `45611.9` vs `49494.2 nM`, ratio `1.09`
  - `FLRYLLFGI`: `40116.0` vs `43358.8 nM`, ratio `1.08`
  - `NFLIKFLLI`: `30544.3` vs `48119.1 nM`, ratio `1.58`
  - probe-head-only readout still separated somewhat:
    - `SLLQHLIGL`: `5.0` vs `166.6 nM`
    - `NFLIKFLLI`: `0.70` vs `5.09 nM`
- `Interpretation`:
  - this is a regression versus old `E004`
  - under the synthetic-negative contract, the current assay-specific `IC50` outputs collapse toward the weak tail and lose useful probe calibration
  - synthetic negatives are therefore not trustworthy on the current architecture without additional redesign
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/scorepath_bench/pulls/e004-scorepathb-20260309a/summary.json)

## E013

- `ID`: `modal_runs/scorepath_bench/pulls/e006-scorepathb-20260309a`
- `Goal`: verify the narrowed score-fed assay path against the old `E006` broader real-data benchmark
- `Architecture`:
  - same narrowed score-fed path as `E012`
  - `Warm start`
  - no synthetic negatives
  - no peptide ranking
  - no allele ranking
- `Data`:
  - class-I exact `IC50`
  - same 7-allele panel as `E006`
- `Best checkpoint`:
  - final epoch `12` pulled from the completed run
- `Probe results`:
  - `SLLQHLIGL`: `131.5` vs `1818.4 nM`, ratio `13.83`
  - `FLRYLLFGI`: `214.7` vs `3372.0 nM`, ratio `15.71`
  - `NFLIKFLLI`: `3.9` vs `434.1 nM`, ratio `110.88`
- `Interpretation`:
  - broader real-data class-I training still works on the narrowed score-fed path
  - weaker than old `E006` on `SLLQHLIGL` and `FLRYLLFGI`, but still biologically correct on all three probes
  - this is the current stable launch point for the next single-factor ablations
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/scorepath_bench/pulls/e006-scorepathb-20260309a/summary.json)

## E014

- `ID`: `modal_runs/synth_ablate/synth-ablate-none-20260309a`
- `Goal`: clean control for the refreshed focused synthetic contract
- `Architecture`:
  - current score-fed assay path
  - `Warm start`
  - no synthetic negatives
  - no peptide ranking
  - no allele ranking
- `Data`:
  - exact `IC50`
  - 2 alleles:
    - `HLA-A*02:01`
    - `HLA-A*24:02`
  - train: `6234`
  - val: `1559`
- `Training contract`:
  - strict per-batch allele balance
  - real-only batches
- `Best checkpoint`:
  - epoch `9`
  - val loss `0.6882`
- `Probe results`:
  - `SLLQHLIGL`: `110.0` vs `2382.4 nM`, ratio `21.66`
  - `FLRYLLFGI`: `104.8` vs `1342.7 nM`, ratio `12.81`
  - `NFLIKFLLI`: `8.2` vs `69.2 nM`, ratio `8.43`
- `Interpretation`:
  - current best biologic behavior under the refreshed focused contract
  - serves as the reference row for the 2026-03-09 synthetic-mode ablations
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/synth_ablate/synth-ablate-none-20260309a/summary.json)

## E015

- `ID`: `modal_runs/synth_ablate/synth-ablate-all-20260309a`
- `Goal`: test refreshed low-ratio mixed synthetic negatives against `E014`
- `Architecture`:
  - same as `E014`
  - plus `Synthetic negatives`
- `Data`:
  - same real split as `E014`
  - train synthetics per epoch: `1558`
  - batch synthetic fraction: `0.25`
  - modes:
    - `peptide_scramble`
    - `peptide_random`
    - `mhc_scramble`
    - `mhc_random`
    - `no_mhc_alpha`
    - `no_mhc_beta`
- `Training contract`:
  - epoch-refreshed synthetic generation
  - explicit per-batch real:synth balancing
- `Best checkpoint`:
  - epoch `9`
  - val loss `0.6768`
- `Probe results`:
  - `SLLQHLIGL`: `20867.7` vs `43182.3 nM`, ratio `2.07`
  - `FLRYLLFGI`: `21534.9` vs `23838.5 nM`, ratio `1.11`
  - `NFLIKFLLI`: `4857.2` vs `33225.5 nM`, ratio `6.84`
- `Interpretation`:
  - better validation loss than `E014`
  - worse biologic calibration and much weaker A0201-specific separation
  - mixed synthetic bundle is not a safe default on the current assay path
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/synth_ablate/synth-ablate-all-20260309a/summary.json)

## E016

- `ID`: `modal_runs/synth_ablate/synth-ablate-peptide_scramble-20260309a`
- `Goal`: isolate refreshed `peptide_scramble`
- `Architecture`:
  - same as `E014`
  - synthetic mode: `peptide_scramble` only
- `Data`:
  - same real split as `E014`
  - train synthetics per epoch: `1558`
  - batch synthetic fraction: `0.25`
- `Best checkpoint`:
  - epoch `9`
  - val loss `0.6599`
- `Probe results`:
  - `SLLQHLIGL`: `26647.3` vs `27514.0 nM`, ratio `1.03`
  - `FLRYLLFGI`: wrong-sign (`20751.0` vs `16238.1 nM`)
  - `NFLIKFLLI`: `8975.4` vs `41394.1 nM`, ratio `4.61`
- `Interpretation`:
  - lowest validation loss among the completed synthetic ablations
  - destroys the A0201-favored `FLRYLLFGI` ordering
  - lower val loss here is not tracking biologic quality
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/synth_ablate/synth-ablate-peptide_scramble-20260309a/summary.json)

## E017

- `ID`: `modal_runs/synth_ablate/synth-ablate-peptide_random-20260309b`
- `Goal`: isolate refreshed `peptide_random`
- `Architecture`:
  - same as `E014`
  - synthetic mode: `peptide_random` only
- `Data`:
  - same real split as `E014`
  - train synthetics per epoch: `1558`
  - batch synthetic fraction: `0.25`
- `Best checkpoint`:
  - epoch `8`
  - val loss `0.6961`
- `Probe results`:
  - `SLLQHLIGL`: `32915.2` vs `46817.0 nM`, ratio `1.42`
  - `FLRYLLFGI`: `37859.7` vs `40158.8 nM`, ratio `1.06`
  - `NFLIKFLLI`: wrong-sign (`40345.3` vs `44217.2 nM`)
- `Interpretation`:
  - preserves groove construction perfectly
  - still degrades probe calibration badly
  - random peptide negatives alone are not enough to explain the earlier mixed-mode collapse
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/synth_ablate/synth-ablate-peptide_random-20260309b/summary.json)

## E018

- `ID`: `modal_runs/synth_ablate/synth-ablate-mhc_scramble-20260309b`
- `Goal`: isolate refreshed `mhc_scramble`
- `Architecture`:
  - same as `E014`
  - synthetic mode: `mhc_scramble` only
- `Data`:
  - same real split as `E014`
  - train synthetics per epoch: `1558`
  - batch synthetic fraction: `0.25`
- `Best checkpoint`:
  - epoch `5`
  - val loss `0.7136`
- `Probe results`:
  - `SLLQHLIGL`: `619.1` vs `1103.7 nM`, ratio `1.78`
  - `FLRYLLFGI`: `208.8` vs `272.4 nM`, ratio `1.30`
  - `NFLIKFLLI`: wrong-sign (`269.1` vs `209.5 nM`)
- `Interpretation`:
  - groove audit is clearly pathological:
    - fallback used on `54/128` sampled synthetic rows at the best epoch
    - common statuses: `no_cys_pairs`, `alpha3_fallback`, `no_alpha2_pair`
    - noncanonical groove lengths are common
  - this mode should not be in the default synthetic mix
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/synth_ablate/synth-ablate-mhc_scramble-20260309b/summary.json)

## E019

- `ID`: `modal_runs/class1-panel-ic50-exact-warmstart-synth-peprank-20260309a`
- `Goal`: test the combined recipe on the broader class-I panel
- `Architecture`:
  - current score-fed assay path
  - `Warm start`
  - `Synthetic negatives`
  - `Peptide ranking`
  - no allele ranking
- `Data`:
  - class-I exact `IC50`
  - 7-allele panel from `E006`
- `Best checkpoint`:
  - epoch `10`
  - val loss `1.1837`
- `Probe results`:
  - `SLLQHLIGL`: `44575.0` vs `48802.8 nM`, ratio `1.09`
  - `FLRYLLFGI`: `40236.5` vs `48192.2 nM`, ratio `1.20`
  - `NFLIKFLLI`: `18603.0` vs `47433.0 nM`, ratio `2.55`
- `Interpretation`:
  - the combined recipe collapses the broader class-I assay outputs toward the weak tail
  - on the current score-fed path, adding synthetics plus peptide ranking on top of the broader panel is not competitive with `E013`
  - the synthetic redesign needs to be validated mode-by-mode before this combined recipe is revisited
- `Artifacts`:
  - [summary.json](/Users/iskander/code/presto/modal_runs/class1-panel-ic50-exact-warmstart-synth-peprank-20260309a/summary.json)

## E020

- `ID`: `modal_runs/strong_base_ablate/class1-panel-ic50-exact-warmstart-legacy-20260309b`
- `Goal`: restore the strongest broader class-I baseline on the legacy assay path before any new ablations.
- `Architecture`:
  - broader class-I panel baseline
  - `affinity_assay_mode=legacy`
  - warm start from `mhc-pretrain-20260308b`
  - no synthetics
  - no peptide ranking
  - no allele ranking
- `Data`:
  - exact `IC50`
  - alleles:
    - `HLA-A*02:01`
    - `HLA-A*24:02`
    - `HLA-A*03:01`
    - `HLA-A*11:01`
    - `HLA-A*01:01`
    - `HLA-B*07:02`
    - `HLA-B*44:02`
- `Status`: invalidated before interpretation
- `Note`:
  - mistakenly launched with `--train-all-alleles --train-mhc-class-filter I`
  - live setup showed `rows=24515`, so this was not the original 7-allele `E006` contract
  - replaced by a corrected parity rerun
- `Interpretation target`:
  - parity against `E006`
  - this is the new base for all follow-on ablations if probe behavior lands back in the old strong regime

## E021

- `ID`: `modal_runs/strong_base_ablate/class1-panel-ic50-exact-warmstart-legacy-peprank-20260309b`
- `Goal`: isolate peptide ranking on top of the restored strongest baseline.
- `Architecture`:
  - same as `E020`
  - `binding_peptide_contrastive_weight=0.5`
- `Status`: invalidated before interpretation
- `Note`: same flag error as `E020`; do not compare against `E006`

## E022

- `ID`: `modal_runs/strong_base_ablate/class1-panel-ic50-exact-warmstart-legacy-synth-peptide-random-20260309b`
- `Goal`: test refreshed `peptide_random` negatives on the restored strongest baseline.
- `Architecture`:
  - same as `E020`
  - refreshed train-only synthetics
  - `synthetic_modes=peptide_random`
  - `batch_synthetic_fraction=0.25`
- `Status`: invalidated before interpretation
- `Note`: same flag error as `E020`; do not compare against `E006`

## E023

- `ID`: `modal_runs/strong_base_ablate/class1-panel-ic50-exact-warmstart-legacy-synth-no-mhc-alpha-20260309b`
- `Goal`: test missing-alpha synthetics on the restored strongest baseline.
- `Architecture`:
  - same as `E020`
  - refreshed train-only synthetics
  - `synthetic_modes=no_mhc_alpha`
  - `batch_synthetic_fraction=0.25`
- `Status`: invalidated before interpretation
- `Note`: same flag error as `E020`; do not compare against `E006`

## E024

- `ID`: `modal_runs/strong_base_ablate/class1-panel-ic50-exact-warmstart-legacy-synth-no-mhc-beta-20260309b`
- `Goal`: test missing-beta synthetics on the restored strongest baseline.
- `Architecture`:
  - same as `E020`
  - refreshed train-only synthetics
  - `synthetic_modes=no_mhc_beta`
  - `batch_synthetic_fraction=0.25`
- `Status`: invalidated before interpretation
- `Note`: same flag error as `E020`; do not compare against `E006`

---

## Groove Baseline Experiments

These use a standalone minimalist model (`scripts/groove_baseline_probe.py`) to test
whether groove sequences alone can predict allele-specific binding, independent of
the full Presto architecture. The groove baseline model uses mean-pooled AA embeddings
of peptide + groove_half_1 + groove_half_2, fed through a small transformer encoder
(~106K params). It outputs `log10(IC50_nM)` directly — no attention cross-talk, no
kinetics, no assay heads.

### Architecture shorthand (groove baseline)

- `GrooveTransformer`: 2-layer transformer encoder, embed_dim=64, hidden_dim=128, 4 heads
- Inputs: peptide tokens + groove_half_1 tokens + groove_half_2 tokens (all mean-pooled)
- Output: scalar `log10(IC50_nM)` with smooth range bound

---

## G001 — 6-allele matrix (5 experiments)

- `ID`: `groove-6allele-matrix-20260309`
- `Goal`: systematic test of inequality data, contrastive loss, synthetics, and curriculum on groove baseline
- `Architecture`: `GrooveTransformer`, 106K params (107K for curriculum variant with classify head)
- `Data`:
  - alleles: `HLA-A*02:01`, `HLA-A*24:02`, `HLA-B*07:02`, `HLA-B*27:05`, `HLA-A*03:01`, `HLA-A*01:01`
  - `qualifier-filter=all` (includes inequality ">" measurements)
  - train: `10,180`, val: `2,498`
  - probe peptides: `SLLQHLIGL`, `YLISIFLHL`, `QFLSFASLF`
- `Training`: 20 epochs, batch_size=256, lr=1e-3

| Sub-exp | Config | train_loss | val_loss | SLLQHLIGL ratio | Dyn range |
|---------|--------|-----------|---------|----------------|-----------|
| A | regress only | 0.80 | 1.09 | **2.2x** | 3.59 log |
| B | + contrastive (wt=1.0) | 2.36 | 2.50 | 4.5x | 2.05 log |
| C | + pep synthetics (0.25) | 0.55 | 0.72 | 9.1x | 4.14 log |
| D | synth + contrastive | 1.42 | 1.75 | 8.0x | 3.09 log |
| E | curriculum (5:classify, 10:regress, 5:regress+synth+contrastive) | 1.67 | 1.83 | **21.7x** | 2.67 log |

- `Interpretation`:
  - all experiments show weak allele discrimination (2–22x) despite verified groove correctness
  - curriculum achieves best ratio (21.7x) but still far from target >50x
  - synthetics help regression loss but don't dramatically improve allele separation
  - contrastive loss inflates total loss without proportionate benefit
  - **root cause identified later**: insufficient data volume and allele diversity (see G002)

---

## G002 — 7-allele panel (5 experiments)

- `ID`: `modal_runs/groove_7allele/groove-7allele-exp{A..E}-*-20260309a`
- `Goal`: repeat G001 matrix with the E006 class-I panel for more allele diversity and data volume
- `Architecture`: same `GrooveTransformer`, 106K params (107K for curriculum)
- `Data`:
  - alleles: `HLA-A*02:01`, `HLA-A*24:02`, `HLA-A*03:01`, `HLA-A*11:01`, `HLA-A*01:01`, `HLA-B*07:02`, `HLA-B*44:02`
  - `qualifier-filter=all`, `train-mhc-class-filter=I`
  - train: `32,855`, val: `8,194` (3.2x more than G001)
  - probe peptides: `SLLQHLIGL`, `IMLEGETKL`, `NFLIKFLLI`
- `Training`: 20 epochs, batch_size=256, lr=1e-3

### SLLQHLIGL (HLA-A\*02:01 binder)

| Sub-exp | Config | train_loss | val_loss | A\*02:01 nM | A\*24:02 nM | **Ratio** |
|---------|--------|-----------|---------|------------|------------|---------|
| A | regress only | 0.30 | 0.82 | 34.8 | 30,785 | **884x** |
| B | + contrastive | 0.57 | 1.34 | 71.6 | 30,078 | **420x** |
| C | + pep synthetics | 0.30 | 0.84 | 45.8 | 38,383 | **838x** |
| D | synth + contrastive | 0.49 | 1.23 | 10.8 | 32,693 | **3,020x** |
| E | curriculum | 0.63 | 1.36 | 48.4 | 13,366 | **276x** |

### NFLIKFLLI (HLA-A\*24:02 binder)

| Sub-exp | A\*24:02 nM | A\*02:01 nM | **Ratio** |
|---------|------------|------------|---------|
| A | 10.4 | 16,020 | **1,540x** |
| B | 8.1 | 7,286 | **899x** |
| C | 65.6 | 32,971 | **503x** |
| D | 1,337 | 22,882 | **17x** |
| E | 37.3 | 5,979 | **160x** |

- `Interpretation`:
  - **allele diversity was the dominant factor**: 884x vs 2.2x for the same baseline config
  - pure regression (A) is the most reliable: strong on both probe peptides (884x + 1,540x)
  - synth+contrastive (D) excels on SLLQHLIGL (3,020x) but fails on NFLIKFLLI (17x) — the two losses compete
  - synthetics compress the binder end (C: NFLIKFLLI 65.6 nM vs A: 10.4 nM)
  - curriculum (E) underperforms the baseline on both probes
  - **B\*44:02 acts as a natural hard negative** — consistently gets the highest IC50 predictions (46–49K nM), likely because its acidic anchor preference (D/E at P2) is most unlike the hydrophobic-anchor peptides used as probes
- `Artifacts`:
  - [expA summary](modal_runs/groove_7allele/groove-7allele-expA-baseline-20260309a/summary.json)
  - [expB summary](modal_runs/groove_7allele/groove-7allele-expB-contrastive-20260309a/summary.json)
  - [expC summary](modal_runs/groove_7allele/groove-7allele-expC-synth-20260309a/summary.json)
  - [expD summary](modal_runs/groove_7allele/groove-7allele-expD-synth-contrastive-20260309a/summary.json)
  - [expE summary](modal_runs/groove_7allele/groove-7allele-expE-curriculum-20260309a/summary.json)

---

## Key Insight: Why Allele Diversity Matters So Much

The jump from G001 (6 alleles, 2.2x) to G002 (7 alleles, 884x) is driven by three factors:

1. **3x more training data**: The E006 panel alleles are the most heavily studied in IEDB.
   A\*02:01 alone contributes 18,195 rows. Total: 32,855 train vs 10,180.

2. **More contrastive allele pairs**: 7 alleles = 21 unique pairs (vs 15 with 6). The added
   alleles span diverse binding motifs: A\*11:01 prefers basic C-terminal anchors (K/R),
   B\*44:02 uniquely prefers acidic anchors (D/E at P2). Each pair teaches the model
   "these groove differences → these binding differences."

3. **Shortcut breaking**: With few alleles, the model can learn a crude "which allele?"
   binary feature. With 7 alleles sharing overlapping groove features (A\*03:01 and A\*11:01
   have similar alpha-1 helices but different alpha-2), the model must learn position-specific
   groove residue features — exactly what drives real peptide-MHC binding specificity.

**Implication for Presto**: see G003 for fair comparison — the groove baseline's
884x advantage was inflated by 4x more data from broader assay types.

---

## G003 — Architecture ablations and fair comparison

- `ID`: `modal_runs/groove_7allele/groove-7allele-{mlp,larger,exact-ic50}-*`
- `Goal`: determine (a) whether attention matters, (b) whether capacity matters,
  (c) how the groove baseline performs on the same data contract as Codex E006

### G003a: MLP (no self-attention)

- `Architecture`: `GrooveBaselineMLP`, 26.5K params, embed_dim=64, hidden_dim=128
- `Data`: same broad contract as G002-A (32,855 train)
- `Result`: **total collapse** — all IC50 predictions ~0.001 nM, no allele discrimination
- `Interpretation`:
  - mean-pooling without attention loses all positional information
  - the model cannot learn groove-peptide interactions from amino acid composition alone
  - **self-attention is essential** for this task

### G003b: Larger Transformer (embed=128, hidden=256)

- `Architecture`: `GrooveTransformer`, 393K params (3.7x more than G002-A)
- `Data`: same broad contract as G002-A (32,855 train)
- `Final`: train_loss=0.23, val_loss=0.81 (overfitting)
- `Probe results`:
  - `SLLQHLIGL`: A\*02:01=7.9 nM, A\*24:02=25,173 nM, ratio **3,170x**
  - `NFLIKFLLI`: A\*24:02=38.5 nM, A\*02:01=16,308 nM, ratio **424x**
  - `IMLEGETKL`: A\*02:01=29.5 nM, A\*24:02=30,917 nM, ratio **1,048x**
- `Interpretation`:
  - more capacity = better discrimination (3,170x vs 884x on SLLQHLIGL)
  - clear overfitting: train 0.23 vs val 0.81
  - suggests the 106K model was capacity-limited; real allele signal is even stronger

### G003c: Exact IC50 Only (fair comparison to Codex E006)

- `Architecture`: same as G002-A (106K params)
- `Data`:
  - `measurement_profile=direct_affinity_only`
  - `measurement_type_filter=ic50`
  - `qualifier_filter=exact`
  - train: `8,166`, val: `2,027` — **same data contract as Codex E006**
- `Final`: train_loss=0.17, val_loss=1.08
- `Probe results`:
  - `SLLQHLIGL`: A\*02:01=61.2 nM, A\*24:02=7,561 nM, ratio **124x**
  - `NFLIKFLLI`: A\*24:02=27.4 nM, A\*02:01=1,988 nM, ratio **73x**
  - `FLRYLLFGI`: A\*02:01=282 nM, A\*24:02=18,933 nM, ratio **67x**
- `Interpretation`:
  - **Presto E006 (393x) beats the groove baseline (124x) on the same data**
  - the groove baseline's apparent 884x advantage was inflated by 4x more data
  - Presto's deeper encoder and core-window cross-attention extract more allele
    signal per data point than the groove baseline's 2-layer transformer
- `Artifacts`:
  - [mlp summary](modal_runs/groove_7allele/groove-7allele-mlp-20260309a/summary.json)
  - [larger summary](modal_runs/groove_7allele/groove-7allele-larger-20260309a/summary.json)
  - [exact-ic50 summary](modal_runs/groove_7allele/groove-7allele-exact-ic50-20260310a/summary.json)

---

## Key Insight: Data Volume vs Architecture

| Model | Params | Data contract | Train rows | SLLQHLIGL ratio |
|-------|--------|--------------|-----------|-----------------|
| Groove baseline | 106K | broad (all numeric) | 32,855 | 884x |
| Groove baseline | 106K | exact IC50 only | 8,166 | 124x |
| Groove larger | 393K | broad (all numeric) | 32,855 | 3,170x |
| Presto (E006) | 4.5M | exact IC50 only | 8,166 | 393x |

1. **Data volume is the dominant factor** for the groove baseline: 884x → 124x when
   restricted to exact IC50 (7x drop).
2. **Presto beats the groove baseline on matched data** (393x vs 124x), validating that
   deeper cross-attention and core-window enumeration extract more per data point.
3. **The larger groove baseline (393K) with broad data is the overall winner** (3,170x)
   — more capacity + more data = best result.
4. **Next experiment**: run Presto on the broad data contract to see if it also benefits
   from the 4x data increase.

### Data Contract Details

The jump from 10K to 44K rows when switching `measurement_profile` from `direct_affinity_only`
to `numeric_no_qualitative` is driven by:

| Measurement Type | Count | In `direct_affinity_only`? | In `numeric_no_qualitative`? |
|-----------------|-------|-----|-----|
| IC50 | 11,211 | yes | yes |
| KD (~IC50) | 16,275 | **no** | yes |
| KD (~EC50) | 10,641 | **no** | yes |
| Direct KD | 2,562 | yes | yes |
| EC50 | 360 | yes | yes |
| Qualitative binding | 2,932 | no | **no** |
| 3D structure | 436 | no | **no** |

The "KD (~IC50)" rows are KD values derived from competitive IC50 assays — reliable but indirect.
The "KD (~EC50)" rows are derived from EC50 — less direct. Together they add ~27K rows.

---

## G004: Presto on broad data contract (7 alleles)

**Date**: 2026-03-10
**Run ID**: `presto-7allele-broad-20260310c`
**Question**: Does Presto match the groove baseline's 884x (or larger's 3,170x) on the same broad data contract?

### Config
- Model: Full Presto (4.5M params), warm start from MHC pretrain
- Alleles: HLA-A*02:01, A*24:02, A*03:01, A*11:01, A*01:01, B*07:02, B*44:02
- Data: `numeric_no_qualitative` profile, `qualifier_filter=all` → 32,855 train / 8,194 val
- Loss: `affinity_loss_mode=full` (all measurement types contribute)
- No synthetic negatives, no contrastive losses
- Balanced batches (20 slots per allele)
- 12 epochs, batch size 140

### Loss trajectory
| Epoch | Train | Val |
|-------|-------|-----|
| 1 | 1.469 | 1.221 |
| 3 | 0.712 | 0.935 |
| 6 | 0.557 | 0.805 |
| 9 | 0.471 | 0.789 |  ← val minimum
| 12 | 0.409 | 0.842 |

### Probe results (epoch 12)

| Peptide | A*02:01 (nM) | A*24:02 (nM) | Ratio |
|---------|-------------|-------------|-------|
| SLLQHLIGL | 15.2 | 20,229 | **1,335x** |
| IMLEGETKL | 584.5 | 20,600 | 35x |
| NFLIKFLLI | 13,981 | 243.9 | **57x** (reversed) |

### Key result

**SLLQHLIGL ratio: 1,335x** — Presto dramatically improves from 393x (exact IC50) to 1,335x with the broad data contract.

### Complete comparison table

| Model | Params | Data contract | Train rows | SLLQHLIGL ratio |
|-------|--------|--------------|-----------|-----------------|
| Groove baseline | 106K | exact IC50 | 8,166 | 124x |
| Presto (E006) | 4.5M | exact IC50 | 8,166 | 393x |
| Groove baseline | 106K | broad | 32,855 | 884x |
| Groove larger | 393K | broad | 32,855 | 3,170x |
| **Presto (G004)** | **4.5M** | **broad** | **32,855** | **1,335x** |

### Analysis

1. **Broad data helps Presto enormously**: 393x → 1,335x (3.4x improvement). The additional KD(~IC50), KD(~EC50), and inequality-qualified rows provide critical signal.

2. **Presto beats groove baseline (884x) but loses to groove larger (3,170x)**: Despite having 11.5x more parameters (4.5M vs 393K), Presto achieves only 42% of the groove larger transformer's discrimination ratio. This confirms the groove information bottleneck hypothesis — the full Presto architecture loses some allele signal through its processing pipeline.

3. **Bottleneck diagnosis**: The groove larger transformer processes groove sequences through dedicated 2-layer self-attention → direct MLP to log10(IC50). Presto instead compresses groove through:
   - Single learned query → 256-D groove_vec
   - Concatenation into 1536-D binding_affinity input → projection back to 256-D
   - Two-path blend (probe 75% / kinetic 25%) where the kinetic path uses interaction_vec, not groove information directly

4. **Val loss suggests overfitting**: Best val loss at epoch 9 (0.789) with divergence to 0.842 by epoch 12. An LR scheduler or early stopping at epoch 9 could further improve results.

5. **NFLIKFLLI shows correct reversed preference**: A*24:02 (243.9 nM) is correctly predicted as much stronger than A*02:01 (13,981 nM), with 57x ratio. This confirms groove information is flowing through — it's just attenuated.

### Next steps

1. **Wider groove bottleneck**: Try multiple groove queries (4-8 instead of 1) to retain more groove information
2. **Feed groove_vec to kinetic path**: Currently kinetic path uses interaction_vec (which lacks direct groove signal)
3. **Adjust probe/kinetic blend**: 75/25 means probe path dominates; try 50/50 or making it trainable
4. **LR scheduler**: Add cosine annealing or ReduceLROnPlateau to prevent overfitting
5. **Add contrastive losses to Presto**: These helped groove baseline; may help Presto too
