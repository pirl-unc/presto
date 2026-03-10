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
