# Runtime Sweep Plan (2026-03-09)

## Spec

- Goal: benchmark 12 runtime-only variants of the current `legacy_m1` class-I affinity baseline on a fixed 10-epoch contract, improving wall-clock without changing training semantics or losing probe accuracy.
- Fixed semantic baseline:
  - explicit 7-allele class-I panel:
    - `HLA-A*02:01`
    - `HLA-A*24:02`
    - `HLA-A*03:01`
    - `HLA-A*11:01`
    - `HLA-A*01:01`
    - `HLA-B*07:02`
    - `HLA-B*44:02`
  - `measurement_profile=direct_affinity_only`
  - `measurement_type_filter=ic50`
  - `qualifier_filter=exact`
  - warm start from the MHC class/species checkpoint
  - `affinity_assay_mode=legacy`
  - `groove_pos_mode=triple`
  - `binding_core_lengths=8,9,10,11`
  - `binding_core_refinement=shared`
  - no synthetics
  - no peptide ranking
  - no allele ranking
- Runtime constraints:
  - keep model/data semantics fixed
  - compare only runtime-oriented knobs:
    - dataloader workers
    - pin-memory
    - persistent workers
    - prefetch factor
    - TF32 / matmul precision
    - `torch.compile`
  - run all variants for the same fixed epoch count (`10`)
  - record enough timing breakdown to explain wins/losses

## Variant Set

- `V00`: baseline `num_workers=0`, `pin_memory=false`
- `V01`: `num_workers=2`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=2`
- `V02`: `num_workers=4`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=2`
- `V03`: `num_workers=8`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=2`
- `V04`: `num_workers=4`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=4`
- `V05`: `num_workers=8`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=4`
- `V06`: `V02 + allow_tf32=true + matmul_precision=high`
- `V07`: `V03 + allow_tf32=true + matmul_precision=high`
- `V08`: `V04 + allow_tf32=true + matmul_precision=high`
- `V09`: `V05 + allow_tf32=true + matmul_precision=high`
- `V10`: `V02 + allow_tf32=true + matmul_precision=high + persistent_workers=false`
- `V11`: `V03 + allow_tf32=true + matmul_precision=high + persistent_workers=false`

## Execution order

- [x] Phase 1: expose runtime-only knobs in the focused runner
  - add CLI flags for:
    - `num_workers`
    - `pin_memory`
    - `persistent_workers`
    - `prefetch_factor`
    - `allow_tf32`
    - `matmul_precision`
    - `torch_compile`
  - plumb those flags through train and validation loaders
  - compile the model only after warm-start loading, if enabled
- [x] Phase 2: add timing instrumentation
  - record:
    - setup wall-clock
    - per-epoch wall-clock
    - per-epoch data wait
    - per-epoch forward/loss time
    - per-epoch backward time
    - per-epoch optimizer time
    - per-epoch validation time
    - per-epoch probe evaluation time
    - per-epoch artifact write time
  - surface those in structured logs and `summary.json`
- [x] Phase 3: add a benchmark launcher
  - create a local script that launches all 12 variants on Modal against the fixed `legacy_m1` contract
  - write a manifest with variant id, run id, app id, and exact args
- [x] Phase 4: verify locally before launch
  - targeted tests for parser/runtime knob wiring
  - one local focused smoke on the baseline and one runtime-tuned variant
- [x] Phase 5: launch all 12 Modal runs
  - use fixed `10` epochs
  - write results into a single benchmark ledger / table
- [ ] Phase 6: summarize winners
  - rank by:
    - total wall-clock
    - epoch wall-clock
    - probe correctness / margins
    - validation loss
  - recommend the fastest variant that preserves `legacy_m1` biology
- [ ] Phase 7: inspect obvious forward/loss bottlenecks in code
  - review the focused affinity training loop and `_affinity_only_loss(...)`
  - check whether focused affinity still computes unnecessary full-model outputs or repeated expensive diagnostics inside the measured forward/loss region
  - inspect pair-mining / contrastive bookkeeping for Python-heavy per-batch work even when weights are zero
  - inspect core-window / register code for batch loops or repeated tensor materialization
  - document the most likely structural bottlenecks before running another benchmark

## Review

- Known bottlenecks already observed:
  - focused runner hardcodes `num_workers=0`, `pin_memory=false`
  - collator tokenization is still on the data path
  - older GPU perf logs show `~38%` data wait
  - repeated Modal app startup exists, but this sweep keeps app shape fixed and focuses first on within-run training/runtime savings
- Current semantic base to preserve:
  - `legacy_m1`
  - best checkpoint on matched 7-allele exact-`IC50` benchmark:
    - `SLLQHLIGL`: `37.3` vs `29588.2 nM`
    - `FLRYLLFGI`: `32.8` vs `24346.2 nM`
    - `NFLIKFLLI`: `5749.3` vs `6.10 nM`
- Implementation/verification:
  - `scripts/focused_binding_probe.py` now exposes:
    - `--num-workers`
    - `--pin-memory`
    - `--persistent-workers`
    - `--prefetch-factor`
    - `--allow-tf32`
    - `--matmul-precision`
    - `--torch-compile`
  - per-epoch summaries now record:
    - `epoch_wall_s`
    - `train_data_wait_s`
    - `train_forward_loss_s`
    - `train_backward_s`
    - `train_optimizer_s`
    - `val_wall_s`
    - `probe_eval_wall_s`
    - `summary_write_wall_s`
  - benchmark launcher added:
    - `scripts/benchmark_runtime_variants.py`
  - verification:
    - `python -m py_compile scripts/focused_binding_probe.py scripts/benchmark_runtime_variants.py`
    - `pytest -q tests/test_focused_probe.py tests/test_presto.py tests/test_train_iedb.py`
    - result: `130 passed`
- Launch status:
  - 12 fixed-epoch (`10`) runtime variants launched on Modal under:
    - [runtime_m1_bench/variants.md](/Users/iskander/code/presto/modal_runs/runtime_m1_bench/variants.md)
    - [runtime_m1_bench/manifest.json](/Users/iskander/code/presto/modal_runs/runtime_m1_bench/manifest.json)
- Failure / relaunch status:
  - first launch failed before epoch 1:
    - `V00`-`V09`: `_build_epoch_train_state(...)` signature mismatch for the new loader args
    - `V10`-`V11`: `torch.compile` / Inductor failure on Modal
  - fix applied on `2026-03-10`:
    - `_build_epoch_train_state(...)` now accepts and forwards runtime loader args
    - compile variants replaced with non-compile `persistent_workers=false` variants
    - re-verified:
      - `python -m py_compile scripts/focused_binding_probe.py scripts/benchmark_runtime_variants.py`
      - `pytest -q tests/test_focused_probe.py tests/test_presto.py tests/test_train_iedb.py`
      - result: `130 passed in 3.35s`
- Scope correction from user feedback:
  - this runtime sweep is only a lower-bound benchmark for a static contract
  - it is **not** the canonical production runtime benchmark because it disables dynamic augmentation / pair mining
  - the production-relevant runtime benchmark must include the augmentation regime we expect to keep:
    - at minimum, per-epoch pair mining for ranking / contrastive objectives
    - optionally per-epoch synthetic refresh if that remains in the chosen training contract
  - follow-up benchmark plan:
    - rerun runtime profiling on the strongest accuracy-oriented contract we actually plan to keep
    - include per-epoch dynamic train-state rebuild costs in the measured workload instead of optimizing them away

# Multi-Allele Runtime/Training Sweep (2026-03-10)

## Spec

- Goal: run a fixed-epoch (`3`) 16-variant Modal benchmark on the canonical shared model path using the ~44k multi-allele class-I binding dataset, with pairwise pMHC ranking active in every run and censor-aware assay losses on `KD/IC50/EC50`.
- Fixed semantic contract for all 16 variants:
  - explicit 7-allele class-I panel:
    - `HLA-A*02:01`
    - `HLA-A*24:02`
    - `HLA-A*03:01`
    - `HLA-A*11:01`
    - `HLA-A*01:01`
    - `HLA-B*07:02`
    - `HLA-B*44:02`
  - `measurement_profile=all_binding_rows`
  - `measurement_type_filter=""`
  - `qualifier_filter=all`
  - dataset size after filtering expected to be `44,417` rows
  - warm start from the MHC class/species checkpoint
  - `affinity_assay_mode=legacy`
  - `groove_pos_mode=triple`
  - `binding_core_lengths=8,9,10,11`
  - `binding_core_refinement=shared`
  - no synthetic negatives in this sweep
    - keep dynamic per-epoch pair mining active
  - pairwise ranking always on:
    - `binding_contrastive_weight > 0`
    - `binding_peptide_contrastive_weight > 0`
  - assay losses supervised headwise with censor-aware loss:
    - `KD_nM` only from `binding_kd`
    - `IC50_nM` only from `binding_ic50`
    - `EC50_nM` only from `binding_ec50`
    - no generic all-binding loss on `KD_nM`
  - qualitative / structure rows may remain in the dataset but should only influence pairwise ranking if no assay-head target exists
- Required outputs per run:
  - startup/setup wall-clock
  - per-epoch wall-clock
  - train data wait / forward-loss / backward / optimizer timings
  - validation wall-clock
  - probe-eval wall-clock
  - GPU peak memory
  - GPU utilization / occupancy proxy sampled during training
  - loss metrics
  - probe affinities including at least:
    - `SLLQHLIGL`
    - `FLRYLLFGI`
    - `NFLIKFLLI`
- Deliverables:
  - one manifest of the 16 variants
  - one result table with runtime + accuracy
  - one recommendation for fastest variant that still trains effectively

## Variant Set

- `R00`: baseline `num_workers=0`, `pin_memory=false`, TF32 off
- `R01`: `num_workers=0`, `pin_memory=true`, TF32 off
- `R02`: `num_workers=0`, `pin_memory=false`, `allow_tf32=true`, `matmul_precision=high`
- `R03`: `num_workers=0`, `pin_memory=true`, `allow_tf32=true`, `matmul_precision=high`
- `R04`: `num_workers=2`, `pin_memory=true`, `persistent_workers=false`, `prefetch_factor=2`, TF32 off
- `R05`: `num_workers=4`, `pin_memory=true`, `persistent_workers=false`, `prefetch_factor=2`, TF32 off
- `R06`: `num_workers=8`, `pin_memory=true`, `persistent_workers=false`, `prefetch_factor=2`, TF32 off
- `R07`: `num_workers=2`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=2`, TF32 off
- `R08`: `num_workers=4`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=2`, TF32 off
- `R09`: `num_workers=8`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=2`, TF32 off
- `R10`: `num_workers=2`, `pin_memory=true`, `persistent_workers=false`, `prefetch_factor=2`, `allow_tf32=true`, `matmul_precision=high`
- `R11`: `num_workers=4`, `pin_memory=true`, `persistent_workers=false`, `prefetch_factor=2`, `allow_tf32=true`, `matmul_precision=high`
- `R12`: `num_workers=8`, `pin_memory=true`, `persistent_workers=false`, `prefetch_factor=2`, `allow_tf32=true`, `matmul_precision=high`
- `R13`: `num_workers=2`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=2`, `allow_tf32=true`, `matmul_precision=high`
- `R14`: `num_workers=4`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=2`, `allow_tf32=true`, `matmul_precision=high`
- `R15`: `num_workers=8`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=2`, `allow_tf32=true`, `matmul_precision=high`

## Execution order

- [ ] Phase 1: add an assay-head-only censored affinity loss mode
  - keep the shared model path unchanged
  - add a focused-run loss mode that supervises:
    - `binding_kd`
    - `binding_ic50`
    - `binding_ec50`
  - keep pairwise ranking active independently of assay-head supervision
  - ensure qualitative rows are not forced onto numeric heads if they lack assay-head targets
- [ ] Phase 2: add GPU telemetry
  - per-epoch peak allocated / reserved memory
  - sampled GPU utilization / memory-utilization proxy
  - clear labeling that utilization is a proxy, not Nsight SM occupancy
- [ ] Phase 3: add a 16-variant launcher on the fixed ~44k contract
  - write manifest rows with exact args
  - launch all 16 variants on Modal in parallel
- [ ] Phase 4: collect results
  - pull/parse `summary.json` for all completed runs
  - write a single table with:
    - setup wall-clock
    - epoch wall-clock
    - forward/backward/data-wait split
    - GPU telemetry
    - validation loss
    - probe affinities / ratios
- [ ] Phase 5: rank and recommend
  - choose the fastest variant that preserves effective training behavior
  - explain which knobs actually matter and which do not

## Review

- Local contract check:
  - 7-allele panel with `measurement_profile=all_binding_rows` and `qualifier_filter=all` yields `44,417` rows
  - measurement mix:
    - `KD (~IC50)`: `16,275`
    - `IC50`: `11,211`
    - `KD (~EC50)`: `10,641`
    - `qualitative binding`: `2,932`
    - `KD`: `2,562`
    - `3D structure`: `436`
    - `EC50`: `360`
  - qualifier mix:
    - exact (`0`): `33,474`
    - right-censored (`1`): `10,943`
- The existing focused runner already exposes:
  - runtime knobs (`num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, `allow_tf32`, `matmul_precision`, `torch_compile`)
  - timing fields (`setup_wall_s`, `epoch_wall_s`, `train_data_wait_s`, `train_forward_loss_s`, `train_backward_s`, `train_optimizer_s`, `val_wall_s`, `probe_eval_wall_s`)
- The current gap before launch:
  - add GPU telemetry
  - add the headwise mixed-assay loss mode
  - create the 16-variant launcher and result collector
- Implementation / verification:
  - added `assay_heads_only` loss mode in `scripts/focused_binding_probe.py`
  - added GPU telemetry sampling via `nvidia-smi` in `scripts/focused_binding_probe.py`
  - added 16-variant launcher / collector:
    - [benchmark_runtime_multiallele.py](/Users/iskander/code/presto/scripts/benchmark_runtime_multiallele.py)
  - verification:
    - `python -m py_compile scripts/focused_binding_probe.py scripts/benchmark_runtime_multiallele.py`
    - `pytest -q tests/test_focused_probe.py tests/test_tasks.py tests/test_training_e2e.py tests/test_presto.py`
    - result: `112 passed`
- Modal sweep result source:
  - the checkpoint-volume collector did not find `summary.json` artifacts for these detached runs
  - fallback collector parsed structured `focused_binding_setup` / `focused_binding_epoch` JSON directly from `modal app logs`
  - summarized outputs:
    - [runtime table](/Users/iskander/code/presto/modal_runs/runtime_multiallele_44k/options_vs_perf_from_logs.md)
    - [runtime summary json](/Users/iskander/code/presto/modal_runs/runtime_multiallele_44k/collected_from_app_logs.json)
- 16-run runtime sweep result:
  - all `16/16` variants produced usable setup + epoch metrics in app logs
  - slowest cost centers were not data wait:
    - `train_forward_loss_s_mean`: roughly `35s` to `113s`
    - `train_backward_s_mean`: roughly `12s` to `30s`
    - `train_data_wait_s_mean`: roughly `2s` to `7s`
  - fastest raw epoch wall-clock:
    - `R02` = `82.3s/epoch`
    - config:
      - `num_workers=0`
      - `pin_memory=false`
      - `allow_tf32=true`
      - `matmul_precision=high`
    - downside:
      - weakest biology of the fast variants
      - `SLLQHLIGL` only `31.1 / 195.7 nM`
  - best speed / effectiveness tradeoff:
    - `R03` = `108.2s/epoch`
    - config:
      - `num_workers=0`
      - `pin_memory=true`
      - `allow_tf32=true`
      - `matmul_precision=high`
    - results:
      - `best_val_loss = 2.5061`
      - `SLLQHLIGL = 57.9 / 2283.2 nM`
      - `FLRYLLFGI = 25.2 / 1100.3 nM`
      - `NFLIKFLLI = 2107.2 / 3655.9 nM`
  - best validation loss:
    - `R11` = `131.1s/epoch`
    - config:
      - `num_workers=4`
      - `pin_memory=true`
      - `persistent_workers=false`
      - `prefetch_factor=2`
      - `allow_tf32=true`
      - `matmul_precision=high`
    - `best_val_loss = 2.3625`
    - but probe behavior is mixed and slower than `R03`
- Runtime conclusion:
  - on the ~44k mixed-assay contract with pairwise ranking active, the main wall-clock cost is model compute, not dataloader wait
  - `allow_tf32=true` + `matmul_precision=high` is the clearest wall-clock win
  - higher `num_workers` / `persistent_workers` do not pay off on the current code path
  - the next runtime wins should come from reducing forward/loss work, not from adding more loader workers
  - promote `R03` as the focused-runner runtime baseline:
    - `num_workers=0`
    - `pin_memory=true`
    - `persistent_workers=false`
    - `allow_tf32=true`
    - `matmul_precision=high`
  - hardware follow-up to test once software/runtime path is stabilized:
    - compare `A100` vs `H100!` on the exact same 44k contract
    - use `H100!` rather than `H100` for apples-to-apples benchmarking because Modal may auto-upgrade plain `H100` requests to `H200`

## Startup / Loading Diagnosis (2026-03-10)

### Spec

- Goal: explain the slow pre-epoch setup/startup path before changing architecture again.
- Scope:
  - focused runner / current `legacy_m1` benchmarking path
  - one-run startup cost before epoch 1
  - loading/setup only, not forward/backward kernel profiling

### Findings

- The startup path makes multiple whole-file passes over `data/merged_deduped.tsv`:
  - `_load_binding_records_from_merged_tsv(...)` scans the full TSV once to build binding rows.
  - `_audit_probe_support(...)` scans the full TSV again once per probe peptide.
  - This is avoidable repeated I/O and CSV parsing before training even begins.
- `PrestoDataset(...)` eagerly materializes every sample into a `PrestoSample`.
  - For each record it resolves MHC class/allele handling and calls `prepare_mhc_input(...)`.
  - That means groove extraction/validation is repeated per sample, not per unique allele or unique pMHC.
  - On multi-allele binding runs this is expensive and redundant because the same allele appears thousands of times.
- The focused runner constructs both:
  - `real_train_dataset`
  - `val_dataset`
  - and then immediately builds a fresh epoch train state in `_build_epoch_train_state(...)`
  - so there is significant object construction before epoch 1.
- `_build_epoch_train_state(...)` also performs groove audits:
  - `_audit_record_groove_preparation(...)`
  - `_audit_dataset_groove_inputs(...)`
  - These are useful for correctness, but they add more pre-epoch CPU work.
- The startup path also resolves probe-support summaries up front:
  - this is helpful diagnostically, but it is not required for actual training.
- Runtime sweeps already showed that loader wait is modest relative to model compute.
  - So startup slowness is more likely dominated by Python-side preprocessing and repeated scans than by GPU starvation.

### Likely highest-value startup fixes

- Cache filtered/split row subsets by dataset-contract hash.
- Cache probe-support summaries instead of rescanning the merged TSV per probe.
- Cache groove-prepared MHC segments per unique allele / sequence, not per record.
- Separate optional diagnostics (probe audits / groove audits) from the critical training path so they can be skipped or deferred.
- Detailed implementation plan:
  - [runtime_speedup_plan.md](/Users/iskander/code/presto/tasks/runtime_speedup_plan.md)

# Architecture / Curriculum Experiments (2026-03-10)

## Spec

- Goal: test architectural and pretraining changes that could improve binding quality and convergence without fragmenting the model into separate specialized architectures.
- Constraints:
  - preserve one canonical Presto model
  - all experiments must be benchmarked against the current `legacy_m1` class-I base
  - new curricula must be explicitly specifiable / reproducible in config

## Candidate experiments

- [ ] `d_model` sweep on the current `legacy_m1` base
  - compare at least:
    - `128`
    - `192`
    - `256` (current)
  - hold:
    - 7-allele exact-`IC50` contract
    - warm start
    - no allele ranking
  - measure:
    - probe panel
    - val loss
    - wall-clock
    - params
- [ ] `pmhc_interaction` field-of-view audit
  - explicitly log how much attention mass each interaction layer/query places on:
    - peptide tokens
    - groove half 1
    - groove half 2
  - use the probe panel and a few fit-supported peptide families
  - confirm whether groove information is actually being used where expected
- [ ] MS detectability simplification
  - current `ms_detectability` latent is peptide-only, but still derived from token-level latent attention
  - test a stricter head that uses only a mean-pooled peptide representation
  - compare against current design on elution/presentation-linked settings later
- [ ] Peptide physicochemical pretraining head
  - add simple peptide-only auxiliaries for warm start / curriculum:
    - GRAVY
    - net charge / basic-acidic balance
    - aromatic fraction
    - aliphatic fraction
    - simple helix / sheet propensity scores from a fixed rubric
  - keep these cheap and deterministic from sequence
- [ ] Configurable pretraining curriculum
  - support an explicit stage schedule such as:
    - epoch 0:
      - MHC groove sequences -> class / species
      - peptides -> physicochemical properties
    - epoch 1+:
      - continue MHC class / species
      - switch peptides -> species of origin
  - curriculum should be declarative in config / CLI, not hard-coded

## Review

- Current relevant architectural facts from code:
  - `d_model` default is `256` in [presto.py](/Users/iskander/code/presto/models/presto.py)
  - `pmhc_interaction` attention is bidirectional peptide<->MHC and the MHC view is the concatenation of groove half 1 and groove half 2
  - candidate core binding queries also attend over:
    - core tokens
    - concatenated groove halves
    - latent dependency tokens
  - `ms_detectability` is peptide-only in the latent DAG, but currently not constrained to a simple mean-pooled peptide vector

# MHCflurry-Style Modular Refactor (2026-03-07)

## Spec

- Goal: refactor `Presto` into a shared pMHC trunk plus explicit MHCflurry-like submodules that can be trained and sanity-checked on task-appropriate subsets while still supporting end-to-end gradients in the full model.
- Required modules:
  - `AffinityPredictor`
  - `Class1ProcessingPredictor`
  - `Class2ProcessingPredictor`
  - `Class1PresentationPredictor`
  - `Class2PresentationPredictor`
- Architectural constraints:
  - keep a shared peptide + groove-half MHC trunk
  - keep current biologic latent flow: `recognition -> immunogenicity` stays canonical
  - submodules must consume explicit trunk tensors, not reach back into raw tokens
  - full `Presto.forward()` must still produce the current output contract
  - add standalone forward paths for at least affinity-only and presentation-only use
  - standalone affinity training must use quantitative affinity rows only
  - treat `SLLQHLIGL` as a generalization probe unless direct quantitative supervision is present

## Execution order

- [x] Phase 0: specify tensor contracts for the modular trunk
  - define the shared trunk state the submodules consume:
    - `processing_vec`
    - `interaction_vec`
    - `binding_affinity_vec`
    - `binding_stability_vec`
    - `presentation_vec`
    - `recognition_vec`
    - `immunogenicity_vec`
    - `pmhc_vec`
    - `class_probs`
    - `groove_vec`
  - decide which outputs belong to each submodule vs. the trunk
- [x] Phase 1: implement submodule classes
  - add explicit module classes for affinity, class-I processing, class-II processing, class-I presentation, class-II presentation
  - move the current output logic into those classes without changing numerical behavior
- [x] Phase 1 verification
  - `Presto.forward()` still returns the current binding / processing / presentation outputs
  - broad model tests still pass
- [x] Phase 2: expose standalone submodule execution paths
  - add a shared trunk method that returns a typed/dict feature bundle
  - add `forward_affinity_only(...)`
  - add `forward_presentation_only(...)`
  - keep those paths weight-sharing with the full model
- [x] Phase 2 verification
  - affinity-only forward returns the same affinity outputs as full forward on the same input
  - presentation-only forward returns the same presentation outputs as full forward on the same input
- [x] Phase 3: refactor focused diagnostics to use the standalone affinity path
  - update `scripts/focused_binding_probe.py` to train affinity-only on binding rows
  - optimize only the affinity-relevant parameters plus shared trunk
  - log probe-support metadata so fit-supported vs generalization-only peptides stay explicit
- [x] Phase 3 verification
  - focused affinity-only diagnostic runs end to end
  - `SLLQHLIGL` is reported as a generalization probe
  - at least one quantitatively supervised A0201/A2402 peptide family is reported alongside it
- [ ] Phase 4: sanity check affinity behavior
  - run a short affinity-only sanity check on the focused A0201/A2402 subset
  - inspect whether the supervised peptide probes separate cleanly and whether `SLLQHLIGL` generalizes in the correct direction

## Review

- Rationale:
  - the current model already has the right ingredients, but the task boundaries are implicit and hard to test
  - modular heads make it possible to separate “architecture failure” from “wrong supervision mixture”
  - the key debugging requirement is a true affinity-only path, because `SLLQHLIGL` has no direct quantitative binding supervision for `A*02:01` vs `A*24:02`
- Progress:
  - added explicit `AffinityPredictor`, `ClassProcessingPredictor`, and `ClassPresentationPredictor` modules in `models/presto_modules.py`
  - kept `Presto.forward()` numerically aligned by routing processing, affinity, and presentation outputs through those modules
  - added compatibility aliases plus checkpoint key remapping so old attribute names and saved checkpoints still load
  - added `forward_affinity_only(...)` and `forward_presentation_only(...)`
  - updated `scripts/focused_binding_probe.py` to use an affinity-only censor-aware loss instead of the full multitask trainer
- Verification:
  - `pytest -q tests/test_presto.py tests/test_checkpointing.py tests/test_train_synthetic.py tests/test_train_iedb.py` -> `115 passed`
  - local smoke:
    - `python -m presto.scripts.focused_binding_probe --data-dir data --out-dir artifacts/focused_binding_probe_smoke --alleles 'HLA-A*02:01,HLA-A*24:02' --measurement-profile direct_affinity_only --epochs 1 --batch-size 128 --max-records 512 --synthetic-negatives --negative-ratio 0.5`
    - completed end to end
    - `SLLQHLIGL` probe stayed correct-sign but nearly tied:
      - `A*02:01 ≈ 189.17 nM`
      - `A*24:02 ≈ 189.20 nM`
  - Modal focused runs launched:
    - `ap-lEYlh8Sf0MOr0lEMZmacmH` direct-affinity, no synthetic negatives
    - `ap-aPSBVsmXZX1yN1RTXjbfh4` numeric affinity profile with synthetic negatives
  - Modal affinity sanity findings:
    - direct-only focused run (`direct_affinity_only`, no synthetic negatives, `10005` rows, 10 epochs on Modal GPU):
      - `SLLQHLIGL` finished correct-sign but with a negligible gap:
        - epoch 10: `A*02:01 ≈ 168.94 nM`
        - epoch 10: `A*24:02 ≈ 169.51 nM`
      - this run did not learn strong peptide-specific allele discrimination reliably:
        - `FLRYLLFGI` is strongly A0201-favored in the real rows (`~2.49 nM` vs `~8249 nM`) but the model ended nearly tied / slightly wrong-sign
        - `NFLIKFLLI` is A2402-favored in the real rows (`~62.4 nM` vs `~307011.9 nM`) but the model ended wrong-sign
    - numeric+synth focused run (`numeric_no_qualitative`, synthetic negatives on, `38728` rows):
      - early trajectory is materially better on the key probe:
        - epoch 1: `SLLQHLIGL` `A0201 ≈ 5233 nM`, `A2402 ≈ 7052 nM`
        - epoch 2: `SLLQHLIGL` `A0201 ≈ 39.94 nM`, `A2402 ≈ 41.90 nM`
        - epoch 3: `SLLQHLIGL` `A0201 ≈ 20.74 nM`, `A2402 ≈ 25.06 nM`
      - `FLRYLLFGI` also stays correct-sign by epoch 3
      - caveat: synthetic negatives can distort under-supported probe families, so this is not a free lunch
- Current judgment:
  - the modular affinity refactor is successful as a debugging tool and training contract
  - the standalone affinity path is numerically healthy
  - direct quantitative-only A0201/A2402 training is still too weak to learn robust allele-specific anchor preferences
  - adding synthetic weak negatives helps `SLLQHLIGL` generalization quickly, but needs tighter control to avoid over-imposing incorrect pair structure on sparsely supported peptides

# Focused Binding Experiment Corrections (2026-03-08)

## Spec

- Goal: make the focused `HLA-A*02:01` vs `HLA-A*24:02` binding experiment measure peptide-specific binding learning cleanly before touching the full architecture again.
- Problems to fix:
  - current `--max-per-allele` balancing is raw-row downsampling and can discard the paired peptide families that actually teach allele discrimination
  - current focused train/val split is row-random after augmentation, which leaks peptide families across splits
  - current class-I anchor-opposite synthetic negatives are wired into the global augmentation helper instead of being an explicit focused-experiment choice
- Constraints:
  - preserve shared peptide families across the target alleles when balancing
  - split real records by peptide before synthetic augmentation
  - keep anchor-aware synthetic negatives opt-in for the focused experiment, not silently global

## Execution order

- [x] Phase 1: fix focused allele balancing
  - replace raw row capping with peptide-family-aware balancing
  - preserve at least one representative row per shared peptide family for each target allele
  - fill remaining capacity with stratified sampling over assay/affinity bins
  - emit richer `balance_stats` including shared-peptide retention
- [x] Phase 2: fix focused train/val split
  - split real records by peptide before synthetic augmentation
  - keep validation real-only so synthetic negatives do not contaminate the diagnostic split
  - log split stats in the summary artifact
- [x] Phase 3: scope anchor-aware negatives correctly
  - make class-I anchor-aware scrambling an explicit augmentation strategy parameter
  - default the global training helper to no anchor-opposite forcing
  - expose the strategy only in the focused probe script for now
- [x] Phase 4: verify locally
  - add/adjust unit tests for peptide-family-aware balancing, peptide-group split, and explicit anchor strategy
  - run targeted pytest for focused probe + binding augmentation + train scripts
- [ ] Phase 5: rerun focused Modal diagnostics
  - run balanced direct-only A0201/A2402 affinity-only training
  - run balanced numeric+synth affinity-only training with explicit anchor-aware negatives
  - evaluate at least:
    - `SLLQHLIGL` as generalization probe
    - `FLRYLLFGI` as A0201-supported probe
    - `NFLIKFLLI` as A2402-supported probe

## Review

- Intended decision rule:
  - if balanced direct-only still fails on the fit-supported probes, the next fix should be in the binding objective or representation
  - if balanced direct-only improves materially, then the previous experiment was mostly data-contract noise rather than model failure
- Implemented corrections:
  - `scripts/focused_binding_probe.py` now does peptide-family-aware allele balancing instead of raw row capping.
  - real records are split by peptide before any augmentation.
  - synthetic negatives are train-only; validation stays real-only by design.
  - the focused script now fails fast if either target allele disappears after source/measurement filtering.
  - focused summaries now include split and balance diagnostics plus post-filter allele counts.
- Verification:
  - `pytest -q tests/test_focused_probe.py tests/test_train_iedb.py` -> `61 passed`
  - local smoke:
    - `python -m presto.scripts.focused_binding_probe --data-dir data --out-dir artifacts/focused_binding_probe_smoke3 --alleles 'HLA-A*02:01,HLA-A*24:02' --measurement-profile direct_affinity_only --epochs 1 --batch-size 128 --max-records 512 --max-per-allele 0 --synthetic-negatives --class-i-anchor-strategy property_opposite --negative-ratio 0.5`
    - setup emitted balanced real rows after filtering: `22` A0201 / `22` A2402
    - split emitted real-only validation: `train_rows=54`, `val_rows=8`, summary `synthetic_stats.val = {"added": 0, "reason": "validation_real_only"}`

# Focused Affinity Objective Parity (2026-03-08)

## Spec

- Goal: make the focused affinity-only diagnostic use the same allele-discrimination pressure as the main binding trainer, so A0201/A2402 subset runs are not underconstrained by construction.
- Evidence:
  - balanced direct-only Modal run (`a0201-a2402-direct-balanced-20260308a`) still ends wrong-sign on fit-supported probes (`FLRYLLFGI`, `NFLIKFLLI`) and on the generalization probe (`SLLQHLIGL`)
  - the focused script currently trains only absolute censor-aware affinity losses and omits the same-peptide/different-allele ranking loss already present in the main trainer
- Constraints:
  - keep the focused path lightweight and affinity-only
  - do not add new speculative objectives before restoring parity with the existing main binding contract

## Execution order

- [ ] Phase 1: add binding contrastive parity to the focused affinity loss
  - reuse the same-peptide/different-allele ranking logic and defaults as the main training path
  - surface the ranking hyperparameters in `scripts/focused_binding_probe.py`
- [ ] Phase 2: verify locally
  - add/update targeted tests for the focused loss config path
  - run focused smoke after the loss change
- [ ] Phase 3: rerun focused Modal diagnostics
  - balanced direct-only with contrastive ranking enabled
  - balanced numeric+synth with contrastive ranking enabled
  - compare against `a0201-a2402-direct-balanced-20260308a` and `a0201-a2402-numeric-synth-balanced-20260308a`

## Review

- Current verdict before this fix:
  - the balanced data contract is better, but the focused trainer is still weaker than the main binding path
  - direct-only training is therefore not yet a fair architectural test
- Verification:
  - `pytest -q tests/test_focused_probe.py tests/test_train_iedb.py` -> `62 passed`
  - corrected focused Modal runs:
    - direct-only + allele-ranking parity: `a0201-a2402-direct-balanced-contrastive-20260308b`
    - numeric+synth + allele-ranking parity: `a0201-a2402-numeric-synth-balanced-contrastive-20260308b`
- Result:
  - restoring same-peptide / different-allele ranking into the focused trainer did not help
  - on the direct-only run it made the wrong-sign gaps materially worse:
    - `FLRYLLFGI`: `A24 - A02 ≈ -312 nM`
    - `NFLIKFLLI`: `A24 - A02 ≈ -309 nM`
    - `SLLQHLIGL`: `A24 - A02 ≈ -224 nM`
  - interpretation: cross-allele ranking alone amplifies the corpus-level allele prior instead of peptide specificity

# Focused Peptide-Specific Ranking (2026-03-08)

## Spec

- Goal: force the focused affinity predictor to learn peptide specificity within each allele, instead of converging on a mostly global A24-over-A02 prior.
- Evidence:
  - balanced direct-only + same-peptide allele contrastive still ends wrong-sign on all tracked probes
  - adding only same-peptide/different-allele ranking makes the wrong-sign gap larger, which means the loss is amplifying the shared-peptide corpus prior instead of anchor-specific peptide effects
- Constraints:
  - implement this in the focused affinity path first, not the full trainer
  - keep the existing same-peptide/different-allele loss, but complement it with a within-allele / different-peptide ranking loss
  - if this works on the focused A0201/A2402 runs, then port the objective to the full architecture

## Execution order

# Minimal Affinity Baseline + Direct Affinity Input Repair (2026-03-08)

## Spec

- Goal: determine whether the A0201/A2402 exact-IC50 slice is learnable with a minimal model and, if so, repair the main Presto affinity path to expose the same information.
- Evidence collected:
  - exact paired IC50 signal is sparse but real (`366` paired peptides, `151` with `>=1.0` log10 gap)
  - motif frequencies match expected anchor preferences for both alleles
  - direct Presto affinity runs still predict near ties
  - a tiny peptide+allele baseline learns the expected direction on real IC50 data, while an allele-only baseline does not
- Likely misspecification:
  - the direct affinity probe is fed from `binding_affinity_readout_proj(interaction_vec)` only
  - this omits the explicit peptide and raw groove/MHC summaries that the simple baseline relies on

## Execution order

- [ ] Phase 1: record the minimal baseline result
  - capture the allele-only vs peptide+allele baseline finding in the review log
  - state clearly that the data are learnable and the current affinity path is under-specified
- [ ] Phase 2: repair the direct affinity input path
  - build `binding_affinity_vec` from `interaction_vec + pep_vec + mhc_a_vec + mhc_b_vec + groove_vec`
  - keep output names/contracts stable
  - keep binding stability on the old interaction-only path unless evidence says otherwise
- [ ] Phase 3: verify numerics
  - run focused unit tests
  - run a local forward/backward smoke on the IC50-focused subset
- [ ] Phase 4: rerun minimal focused diagnostics
  - rerun A0201/A2402 IC50-only `probe_only` runs on Modal
  - compare against the near-tie prior runs
- [ ] Phase 5: switch the focused experiment to the real IC50 contract
  - add strict per-batch allele balancing for the two-allele focused trainer
  - use all exact A0201/A2402 IC50 rows, not row-capped subsets
  - add an `IC50_nM`-only focused loss mode so the real assay output is the primary target
  - rerun the focused Modal experiment on the full exact IC50 slice

## Review

- Minimal baseline result:
  - `allele_only` cannot learn peptide-specific ordering and predicts the same value for every peptide per allele
  - `pep_plus_allele` does learn the expected directions on real IC50 data:
    - unclipped:
      - `SLLQHLIGL`: `A0201 ~177.7 nM`, `A2402 ~2767.7 nM`
      - `FLRYLLFGI`: `A0201 ~43.3 nM`, `A2402 ~644.4 nM`
      - `NFLIKFLLI`: `A0201 ~2312.5 nM`, `A2402 ~284.3 nM`
  - conclusion: the dataset and target scaling are not fundamentally broken
  - implication: the current Presto direct affinity path is hiding or smoothing away useful peptide+allele information
- Implementation:
  - patched `models/presto.py` so the direct affinity probe now consumes
    `interaction_vec + pep_vec + mhc_a_vec + mhc_b_vec + groove_vec`
    instead of `interaction_vec` alone
  - added `--qualifier-filter` to `scripts/focused_binding_probe.py` so the
    focused Presto run can match the exact-only baseline contract
- Verification:
  - `pytest -q tests/test_focused_probe.py tests/test_presto.py tests/test_train_iedb.py` -> `114 passed`
  - target-path audit:
    - raw exact `IC50` values arrive in `bind_target` unchanged
    - qualifiers arrive in `bind_qual` unchanged
    - normalization is correct log10(nM) with the current 50k ceiling
  - local exact-only focused run after the affinity-input repair:
    - rows after filter/balance: `141` A0201, `141` A2402
    - epoch 5 `probe_kd_nM`:
      - `SLLQHLIGL`: `A0201 ~2789.3`, `A2402 ~2827.7`
      - `TNFLIKFLL`: `A0201 ~2760.3`, `A2402 ~2798.0`
    - this is still a small gap, but it is a real move away from the previous near-tie/wrong-sign behavior
  - Modal runs launched:
    - `a0201-a2402-ic50-exact-probe-only-20260308g`
    - `a0201-a2402-ic50-exact-probe-peptide-rank-20260308g`

# Focused IC50 Improvement Plan (2026-03-08)

## Spec

- Goal: move the focused binding model from "correct sign, weak separation" toward realistic quantitative allele discrimination, then scale outward without losing interpretability.
- Current state:
  - best focused run trains on all exact A0201/A2402 `IC50` rows with strict per-batch allele balance and no synthetics
  - it now gets the right directions for `SLLQHLIGL`, `FLRYLLFGI`, and `NFLIKFLLI`
  - however, the quantitative gap is still too small for some probes, especially the A2402-weak generalization case
- Key architectural fact:
  - the model consumes groove segments, not learned allele-ID embeddings
  - `primary_allele` is currently used for sampling / diagnostics, not as a learned embedding input

## Hypotheses to test in order

- [ ] H1: the model is now optimization-limited, not data-contract-limited
  - evidence needed: stable focused `IC50_nM` training over multiple seeds with the new exact-only, all-rows, strict-balance contract
- [ ] H2: two-allele training is too narrow to force general groove/anchor learning
  - evidence needed: a small multi-allele class-I panel improves held-out peptide discrimination
- [ ] H3: synthetic negatives and cross-allele ranking help only after the exact-assay baseline is already strong
  - evidence needed: one-at-a-time ablations improve quantitative fit without collapsing fit-supported probes
- [ ] H4: remaining under-separation is due to an overly smooth affinity trunk/readout, not parsing or labels
  - evidence needed: the exact-assay baseline plateaus despite clean data and balanced batching

## Execution order

- [ ] Phase 1: lock the clean baseline
  - patch focused probe artifacts to log `IC50_nM` directly, not only `KD_nM`
  - rerun the exact all-rows A0201/A2402 `IC50_nM` experiment once more for confirmation
  - evaluate:
    - `SLLQHLIGL`
    - `FLRYLLFGI`
    - `NFLIKFLLI`
  - success criterion:
    - correct direction on all three under the real `IC50_nM` output

- [ ] Phase 2: seed stability and calibration
  - rerun the same focused exact-IC50 experiment with 3 seeds
  - report median and spread for probe predictions and validation loss
  - inspect whether the 50k clip is hiding useful gradient on weak binders
  - ablate:
    - `max_affinity_nM = 50k`
    - `max_affinity_nM = 500k`
    - unclipped exact values in the focused path

- [ ] Phase 3: expand to a small multi-allele panel
  - keep exact `IC50` only, no synthetics, no ranking losses initially
  - candidate panel: abundant, motif-distinct class-I alleles with usable exact IC50 support
  - require strict per-batch balancing across alleles
  - primary purpose:
    - test whether groove-based learning generalizes beyond the two-allele pair
    - reduce the chance the model is just learning a narrow A02-vs-A24 separator

- [ ] Phase 4: add one extra pressure at a time
  - only after Phase 3 baseline is stable
  - ablation order:
    1. same-allele / different-peptide ranking
    2. carefully scoped synthetic negatives
    3. same-peptide / different-allele ranking, only if the prior two help
  - never combine new pressures before testing them individually

- [ ] Phase 5: synthetic negatives, but only conservative ones
  - keep validation real-only
  - introduce only after exact-assay baseline is clearly working
  - start with:
    - anchor-broken decoys from strong exact binders
  - avoid broad random-scramble synthetics until exact-fit behavior is strong

- [ ] Phase 6: architecture escalation only if the cleaner training path still plateaus
  - make the affinity head consume more explicit peptide-to-groove interaction structure
  - possible options:
    - pocket-aware peptide/groove readout
    - explicit anchor-position interaction features
    - less smooth pooling inside the affinity trunk

- [ ] Phase 7: scale back toward unified Presto
  - once focused affinity is stable:
    - add more alleles
    - add stability
    - add processing
    - add presentation
    - add immunogenicity last
  - introduce one component at a time and verify it does not damage affinity

## Review

- Current recommendation:
  - yes, add more alleles, but only after the exact-assay A0201/A2402 baseline is stable
  - yes, test synthetic negatives and contrastive terms one at a time
  - no, do not jump back to full unified training yet
- detailed staged plan:
  - see [tasks/focused_affinity_improvement_plan.md](/Users/iskander/code/presto/tasks/focused_affinity_improvement_plan.md)
- Why more alleles help:
  - with only two alleles, the model can still solve the task with a narrow separator
  - a small motif-diverse panel is a better test of whether groove-segment learning is actually working
- Why one-at-a-time additions matter:
  - same-peptide cross-allele ranking was previously harmful
  - synthetic negatives previously improved some probes but distorted others
  - the only way to attribute improvement cleanly is single-factor ablation

- [ ] Phase 1: add within-allele peptide ranking to the focused affinity loss
  - collect same-allele / different-peptide pairs with sufficient target gap
  - cap pair counts and log support metrics separately from the allele-ranking loss
- [ ] Phase 2: verify locally
  - add focused tests for the new pair collector
  - run a short focused smoke
- [ ] Phase 3: rerun focused Modal diagnostics
  - direct-only + allele ranking + peptide ranking
  - numeric+synth + allele ranking + peptide ranking
  - compare against the previous `20260308a` and `20260308b` runs

## Review

- Current verdict before this fix:
  - balanced direct-only without ranking was weak and wrong-sign
  - balanced direct-only with same-peptide allele ranking became more wrong-sign
  - that pattern strongly suggests missing peptide-specific pressure, not a dead encoder
- Verification:
  - `pytest -q tests/test_focused_probe.py tests/test_train_iedb.py` -> `63 passed`
  - local focused smoke (`artifacts/focused_binding_probe_smoke5`) showed the new loss is active on realistic batches:
    - `out_binding_same_allele_rankable_pairs = 9558`
    - `out_binding_same_allele_pairs_used = 128`
    - same-peptide cross-allele pairs remained sparse (`5` used)
- Modal result:
  - direct-only + allele-ranking + peptide-ranking: `a0201-a2402-direct-balanced-peptide-rank-20260308c`
  - numeric+synth + allele-ranking + peptide-ranking: `a0201-a2402-numeric-synth-balanced-peptide-rank-20260308c`
  - within-allele peptide ranking helped relative to allele-ranking alone, but not enough:
    - `FLRYLLFGI`: wrong-sign gap improved from `-312 nM` to `-46 nM`
    - `NFLIKFLLI`: wrong-sign gap improved from `-309 nM` to `-52 nM`
    - `SLLQHLIGL`: still wrong-sign at `-59 nM`
  - interpretation: peptide-specific pressure is the right direction, but the same-peptide allele-ranking loss is still fighting it

# Minimal Binding Affinity Debug (2026-03-08)

## Spec

- Goal: make the smallest real binding-affinity-only A0201/A2402 experiment work before scaling back up.
- Principle:
  - minimize assumptions
  - keep only real binding-affinity data first
  - keep the dataset balanced between alleles
  - trace gradients and losses explicitly
  - add complexity back stepwise only after the simpler stage works
- Questions to answer:
  - is the failure caused by mixed assay supervision (`KD` / `IC50` / `EC50`)?
  - is the failure caused by using the full kinetic output path instead of the direct affinity head?
  - is the failure caused by the pairwise objectives overpowering the regression objective?
  - what is the smallest dataset slice on which the affinity predictor actually learns peptide-specific allele preferences?

## Execution order

- [ ] Phase 1: inspect the real A0201/A2402 binding corpus in the exact focused path
  - quantify counts by assay type, qualifier, shared-peptide family size, and replicate support
  - identify same-assay paired subsets (`IC50`-only, `KD`-only, `EC50`-only, pooled direct)
  - identify high-support fit peptides instead of relying only on sparse probes
- [ ] Phase 2: simplify the focused trainer
  - add explicit modes for:
    - direct affinity probe head only
    - full binding path only
    - combined
  - allow assay-type filtering in the focused script
  - log per-loss magnitudes and gradient norms for trunk vs affinity head
- [ ] Phase 3: run minimal ablations on Modal
  - balanced `IC50`-only, real data, direct affinity head only
  - balanced pooled direct (`KD/IC50/EC50`), real data, direct affinity head only
  - balanced pooled direct, real data, direct affinity head only + peptide ranking
  - only if needed: full binding path on the same slices
- [ ] Phase 4: decide the winning minimal contract
  - if the direct affinity head works on a minimal real slice, port that contract upward
  - if it does not, inspect gradient flow / representation path before adding more data or tasks
- [ ] Phase 5: scale back up stepwise
  - add more alleles
  - then stability
  - then processing/presentation
  - stop and diagnose at the first regression

## Review

- Current working hypothesis:
  - the model is not learning `A*24:02` preferences cleanly because the current focused objective still mixes too much:
    - mixed assay semantics
    - full kinetic path plus direct affinity path
    - ranking objectives that can reinforce allele priors
  - the next clean test is a real-data-only, balanced, direct-affinity-head-only experiment on the smallest same-assay slice with enough paired peptides
- Next loop:
  - audit exact paired `IC50` gap distribution on the shared-peptide slice
  - inspect whether trained A0201 vs A2402 representations are nearly identical at the binding-head input
  - run a tiny overfit experiment on the strongest paired peptides to determine whether the direct affinity path can learn peptide-specific allele separation at all

# Binding Training Dynamics Recovery (2026-03-07)

## Spec

- Goal: make the post-refactor pMHC stack learn binding and presentation quickly enough to show clear allele discrimination on short diagnostics, then validate on a 1-epoch Modal run before any longer training.
- Primary probe: `SLLQHLIGL` should separate in the correct direction for `HLA-A*02:01` vs `HLA-A*24:02`, with the gap widening rather than collapsing during early training.
- Constraints:
  - keep the groove-native MHC representation and receptor-free canonical path
  - run training and focused diagnostics on Modal rather than on the local laptop unless the task is a sub-minute CPU check
  - prefer fixes that improve the real end-to-end objective before adding new pretraining subsystems
  - do not trust head-capped corpus slices for probe diagnostics; use reservoir sampling or explicitly constructed probe-supporting subsets
  - do not claim training dynamics from Modal until the bf16 indexed-write bug is fixed or AMP is disabled
  - preserve allele, MHC class, and host-species diversity within each batch even when using a curriculum or task-biased batch construction
  - distinguish fitting probes from generalization probes: verify whether the probe peptide has direct quantitative binding supervision before using it to judge the binding head

## Current evidence

- Corrected local reservoir diagnostics show the model is not dead:
  - 30-batch run: `A0201 KD≈762.7 nM`, `A2402 KD≈795.7 nM`
  - 100-batch run: `A0201 KD≈689.7 nM`, `A2402 KD≈755.3 nM`
  - the sign is correct and the gap widens with more updates, but separation remains weak
- Earlier wrong-sign probe results were invalid because `head` capping removed `A*24:02` binding/elution support entirely from the diagnostic slice.
- The detached Modal reservoir diagnostic (`ap-6tjbpNx2OTXldvGAfOvcXZ`) did not complete an epoch:
  - it died at batch `0` on CUDA bf16 with an indexed-write dtype mismatch
  - there is no usable remote loss curve or probe trajectory yet
- Current code-level bottlenecks worth acting on:
  - `interaction_vec -> BindingModule -> {log_koff, log_kon_intrinsic, log_kon_chaperone}` is a narrow kinetic bottleneck
  - the direct `binding_affinity_probe_kd` bypass exists but has `base_weight=0.3`
  - `sample_weighted` supervised aggregation amplifies large-support tasks even with balanced batching
  - `BindingModule` still uses hard clamps while assay heads already use smooth bounds
  - there is no within-batch allele-discrimination/ranking loss for binding
- Current code-level hypotheses that are not primary blockers:
  - input MHC resolution/tokenization for `A*02:01` vs `A*24:02` is working
  - the multi-query interaction architecture can overfit a two-sample allele-discrimination toy
  - blindly reducing `tcr_evidence` and MHC augmentation did not improve the short local diagnostic
  - a strong MHC-attention sparsity prior made the small discrimination toy worse, not better

## Execution order

- [ ] Phase A: instrumentation and numeric unblock
  - add explicit diagnostics for per-batch supervised support by task, effective supervised task weights after aggregation, and binding clamp saturation rates
  - add a small same-peptide/different-allele pair counter so we know whether the batch actually supports allele-ranking losses
  - audit activation functions and saturation points in the binding/presentation path, especially `tanh` and hard clamp usage
  - fix the remaining CUDA bf16 indexed-write path or run the next Modal diagnostic with AMP disabled to get a real trajectory immediately
- [ ] Phase A verification
  - one local diagnostic run emits batch-composition metrics and clamp-saturation metrics
  - one Modal 1-epoch diagnostic completes far enough to produce epoch/probe outputs
- [ ] Phase B: low-risk binding optimization fixes
  - raise `binding_affinity_probe` weight from `0.3` to `1.0`
  - switch diagnostic/binding-focused training from `sample_weighted` to `task_mean`
  - replace hard clamps in `BindingModule` with the existing smooth bound functions used by assay heads
  - add a direct logging path for agreement/divergence between `binding_affinity_probe_kd` and kinetic-derived KD
- [ ] Phase B verification
  - short local reservoir runs improve the `A0201` vs `A2402` gap faster than the current baseline
  - no exploding or NaN kinetics under CPU fp32 and CUDA bf16/fp32
- [ ] Phase C: explicit allele-discrimination objective
  - add a same-peptide/different-allele ranking loss for binding within a batch
  - restrict it to pairs with usable binding supervision so unlabeled comparisons do not dominate
  - log pair counts and loss contribution separately from the main binding losses
- [ ] Phase C.1: pair-aware binding batch construction
  - extend the balanced sampler so each binding-heavy batch reserves a small subquota for quantitative peptide families with multi-allele support
  - prefer drawing rankable partner examples from already-seeded peptide families before falling back to generic binding draws
  - keep allele, MHC class, and species diversity scoring active for those family draws instead of turning them into a monoculture
- [ ] Phase C verification
  - the ranking loss is active on real batches
  - the probe gap widens earlier than the Phase B-only baseline
- [ ] Phase D: curriculum and batch-composition tuning
  - bias early training more strongly toward binding, kinetics, stability, and elution while keeping auxiliary tasks present but not dominant
  - maintain within-batch diversity across allele, MHC class, and species as a hard sampler goal, not just a corpus-level weighting preference
  - if needed, add a short warmup schedule for existing MHC-only auxiliary heads (species/class/chain/domain) rather than a separate pretrain-and-merge pipeline
  - if needed, add a peptide-origin warmup only after binding-focused fixes are measured
- [ ] Phase D verification
  - early-epoch probe separation and binding validation improve without harming presentation/elution trends
- [ ] Phase E: full 1-epoch Modal diagnosis and critique
  - run one full-epoch Modal diagnostic on the best Phase D config
  - analyze trajectory, gradients, latent statistics, task balance, probe outputs, and MHC-specific separation
  - decide whether the model is strong enough for a long run or whether another training-dynamics revision is needed
- [ ] Phase F: Modal-only focused binding diagnostics
  - add a dedicated Modal entrypoint for allele-panel binding-only diagnostics using `load_binding_records_for_alleles_from_merged_tsv(...)`
  - support strict quantitative-only A0201/A2402 runs with configurable synthetic-negative enablement
  - emit compact per-epoch probe summaries plus artifact files in the checkpoints volume
  - keep `SLLQHLIGL` in the report, but also track at least one peptide family with direct quantitative A0201/A2402 supervision
- [ ] Phase F verification
  - one strict A0201/A2402 binding-only run completes on Modal
  - one synthetic-augmented variant completes on Modal
  - result summary explicitly separates `fit-supported` probes from `generalization-only` probes

## Review

- External diagnosis synthesis:
  - correct and worth acting on:
    - binding bottleneck through three scalar kinetics latents
    - loss dilution from `sample_weighted`
    - missing allele-discrimination objective
    - hard clamps in `BindingModule`
    - raw corpus imbalance against binding
  - real but not first-order for the `SLLQHLIGL` probe:
    - unresolved-MHC filtering
    - groove fallback degradation
  - not supported as primary explanations by the current evidence:
    - tokenization/resolution bug for the probe alleles
    - architecture being incapable of allele discrimination
    - auxiliary-task removal as the first fix
- Pretraining assessment:
  - full standalone pretraining on MHC species/class or peptide species-of-origin is not the first move
  - the model already reads enough MHC signal to get the sign right once sampling is corrected
  - if end-to-end binding still learns too slowly after Phases A-C, a short warmup curriculum using existing auxiliary heads is a cleaner next step than building separate subnetworks and merging weights
- Activation-path assessment:
  - re-audit on 2026-03-07 shows no remaining `tanh` on the active canonical model path
  - current bounded nonlinearities on the pMHC path are:
    - smooth lower/upper/range bounds in binding/assay heads
    - `softsign` on the assay-bias residual in `models/presto.py`
    - probability/denominator safety clamps around softmax-like normalizations and `logit`
  - the main remaining gradient-flow risk is now missing pair structure in batches, not a hidden `tanh`
  - latest diagnostic update:
    - pair-aware binding batching is now active on real canary batches
    - trusted canary with probe-family bootstrap reached:
      - `batch_support_binding ≈ 49`
      - `out_binding_same_peptide_rankable_pairs ≈ 12.6`
      - `out_binding_contrastive_pred_gap_mean ≈ 0.23`
      - `out_binding_contrastive_required_gap_mean ≈ 1.57`
      - `out_binding_probe_core_kd_l1_mean ≈ 1.30`
    - despite that, `SLLQHLIGL` still goes the wrong way for `A*02:01` vs `A*24:02`
    - 3-epoch trusted canary showed stable optimization but continued wrong-sign probe drift
    - strongest remaining hypothesis: the model is still too MHC-prior-dominant and needs stronger peptide-specific discrimination pressure, not more generic allele-pair supervision
- Probe semantics update:
  - the focused A0201/A2402 binding subset already recovered the correct `SLLQHLIGL` sign once the task was concentrated
  - however, `SLLQHLIGL` may not have direct quantitative binding supervision in the merged corpus
  - if confirmed, it must be treated as a generalization probe for peptide-anchor learning, not as a memorization target

# Integrated MHC Processing + Receptor Removal (2026-03-06)

## Spec

- Goal: implement the new groove-centric MHC representation and remove receptor sequences from canonical Presto without causing backwards progress across the recently refactored learning architecture.
- Canonical planning inputs:
  - `tasks/mhc_processing_plan.md`
  - `tasks/receptor_removal_plan.md`
  - `tasks/tcr_evidence_spec.md`
- Design constraints from user clarifications:
  - keep `recognition` as the canonical repertoire-level latent upstream of `immunogenicity`
  - remove TCR/BCR sequences as model inputs and canonical training features
  - keep a pMHC-only `tcr_evidence` output family derived from curated receptor databases
  - do not couple `tcr_evidence` to inference-time assay/context embeddings
  - stage interface changes so the same loader/collate/model surfaces are not churned twice

## Integrated execution order

- [x] Phase 0: baseline current pMHC-only training path with a short no-AMP run and record output keys, task composition, and minibatch behavior.
- [x] Phase 1: implement the groove extractor and augmented MHC index additively without changing the runtime sample/batch contract yet.
- [x] Phase 1 verification: add parser/index tests and prove groove extraction stats on representative reference alleles and failure modes.
- [x] Phase 2: add pMHC-only TCR-evidence record parsing and metadata preservation in VDJdb/McPAS/cross-source code, still without changing the model API yet.
- [x] Phase 2 verification: audit counts for `tcr_evidence` and selected method bins from live loaders, and prove canonical source summaries can expose those records without receptor sequences.
- [ ] Phase 3: finish the unified sample/batch contract rewrite in `data/loaders.py` and `data/collate.py`:
  - groove-half MHC inputs are already live
  - add `tcr_evidence` targets/masks
  - remove receptor sequences and chain auxiliary labels/tensors from canonical `PrestoSample` / `PrestoBatch`
- [ ] Phase 3 verification: `PrestoSample` / `PrestoBatch` contain only pMHC inputs, assay context, and pMHC-level targets; no `tcr_*` or receptor-chain tensors remain.
- [ ] Phase 4: finish the canonical `models/presto.py` surgery for the receptor-free forward path:
  - groove-half segment semantics are already live
  - remove TCR tower / matcher / chain-head branches entirely
  - add pMHC-only `tcr_evidence` head(s)
- [ ] Phase 4 verification: forward/backward finite on mixed real batches; output contract updated; no `enable_tcr`, `tcr_vec`, `match_logit`, `classify_chain`, or chain outputs remain in canonical `Presto`.
- [ ] Phase 5: update training, predictor, CLI, evaluation, and task registries to consume the new batch/model contract and drop receptor-conditioned entrypoints.
- [ ] Phase 5 verification: canonical training no longer loads or advertises VDJdb/10x receptor sequences, TCR retrieval, or chain classification, but does train/evaluate `tcr_evidence`.
- [ ] Phase 6: run local sanity training on the refactored stack, then run a 1-epoch Modal diagnostic without AMP if needed to bypass the remaining bf16 blocker.
- [ ] Phase 6 review: inspect loss curves, gradients, probe peptides including `SLLQHLIGL`, allele separation (`A*02:01` vs `A*24:02`), batch composition, and latent/output behavior before deciding on a full run.

## Focused fix: groove dispatch + species semantics (2026-03-06)

- [x] Fix `extract_groove()` so class-II dispatch does not silently default to alpha when `chain` is omitted.
- [x] Add explicit class-II chain inference from sequence and return an explicit failure/ambiguity status when inference is not defensible.
- [x] Add regression tests covering omitted-chain class-II alpha, omitted-chain class-II beta, and ambiguous/failed inference paths.
- [x] Audit whether class-II allele-only default pairing is implemented anywhere in the user-facing resolution path and document the exact current gap.
- [x] Separate fine-grained MHC species identity used by parsing/indexing from the coarse species buckets used by network classification.
- [x] Make `mhcgnomes` the required allele parser everywhere relevant and remove remaining heuristic parser fallbacks for class/species normalization.
- [x] Fix the local editable/namespace-package import case so `mhcgnomes` resolution works identically under `python` and `pytest`.
- [x] Verify targeted groove/parser/index tests and summarize the exact behavior change for the user.

## Focused audit: mhcgnomes failing cases (2026-03-06)

- [ ] Enumerate every allele-bearing source in the workspace that should be parseable by `mhcgnomes`.
- [ ] Run a full parse audit over normalized and raw allele strings, keeping source, field, count, and example rows.
- [ ] Cluster failures by pattern so library fixes can be implemented at the parser rule level rather than as one-off aliases.
- [ ] Distinguish true parser gaps from bad/non-allelic data tokens.
- [ ] Summarize exact failure classes and the highest-value fixes for the `mhcgnomes` repo.

## Focused fix: DR alpha defaults (2026-03-06)

- [x] Audit DRB/DRA coverage in the local MHC index and merged data to determine which species actually need beta-only DR default pairing.
- [x] Select native default DRA alleles from the local index for every species where both DRB and DRA are present locally.
- [x] Expose a canonical species-to-default-DRA dictionary in the resolver layer.
- [x] Update loader and predictor chain assembly so class-II DR beta-only inputs become `(DRA, DRB)` instead of `(DRB, empty)`.
- [x] Make loader-side `mhc_b` resolution accept allele names, not only literal amino-acid sequences.
- [x] Verify that the selected default DRA alleles exist in `data/mhc_index.csv` and that class-II DR beta-only samples assemble correctly.

## Focused fix: protein-resolution allele normalization (2026-03-06)

- [x] Make `mhcgnomes`-based protein-resolution normalization the shared contract for user-facing allele names.
- [x] Refactor parsing so compact/raw user inputs can still be parsed without `normalize_allele_name()` recursing back into itself.
- [x] Normalize default DR alpha mappings to compact protein-level names where sequence-unique, while preserving higher resolution for prefixes whose local index entries remain protein-ambiguous at two fields.
- [x] Re-key the canonical DR alpha default map by MHC prefix (`HLA`, `SLA`, `Mamu`, etc.) and keep species-name aliases as a compatibility layer only.
- [x] Add tests covering two-field normalization, suffix preservation, prefix-keyed DR defaults, and index-backed existence checks.
- [x] Verify resolver/index/loader/predictor tests against the new contract and summarize the corpus-level collision findings.

## Review

- `normalize_allele_name()` is now the shared user-facing protein-resolution normalizer:
  - `HLA-DRA*01:01:01:01 -> HLA-DRA*01:01`
  - `HLA-DRA*01:01:01:01N -> HLA-DRA*01:01N`
  - `H2-Kb -> H2-K*b`
  - `A2 -> HLA-A*02`
- Parsing no longer recurses through `normalize_allele_name()`. `parse_allele_name()` now tries a lightly coerced allele token first, then the raw token, which preserves compact input support without circular normalization.
- Canonical DR defaults are now keyed by stable MHC prefixes instead of free-text species names.
- Important corpus finding:
  - blindly collapsing the local index to two-field names is wrong for this corpus
  - audit over `data/mhc_index.csv` found `1,217` cases where two-field collapse merged distinct protein sequences
  - so the index and internal resolver storage keep full distinguishing allele names, while user-facing/default aliases stay compact where that is biologically safe
- Default DR alpha map outcome:
  - most prefixes now use compact two-field names (`HLA-DRA*01:01`, `SLA-DRA*01:01`, `Mamu-DRA*01:01`, etc.)
  - two prefixes retain higher resolution because two-field collapse is protein-ambiguous in the local index:
    - `Chsa -> Chsa-DRA*01:01:01`
    - `Mafa -> Mafa-DRA*01:01:01:01`
- Loader and predictor custom allele-sequence maps now normalize lookup keys, so two-field caller inputs and higher-resolution stored alleles resolve against each other correctly.
- Verification:
  - `python -m py_compile data/allele_resolver.py data/mhc_index.py data/loaders.py inference/predictor.py tests/test_allele_resolver.py tests/test_loaders.py tests/test_predictor.py tests/test_mhc_index.py`
  - `pytest -q tests/test_allele_resolver.py tests/test_loaders.py tests/test_predictor.py tests/test_mhc_index.py tests/test_tasks.py` -> `188 passed`

## Focused fix: two-field alias representative selection (2026-03-06)

- [x] Audit all two-field alias collisions into:
  - nested fragment/full cases where the longest sequence safely dominates
  - ambiguous cases where sequences differ beyond pure subsequence nesting
- [x] Update MHC index alias selection so two-field aliases resolve to the longest representative only in the safe nested case.
- [x] Make ambiguous two-field aliases explicit in the index resolution layer instead of silently picking one record.
- [x] Keep full-resolution index records unchanged; only adjust alias collapse behavior.
- [x] Verify that augmented groove fields for safe two-field aliases come from the chosen longest representative sequence.
- [x] Produce an enumeration of the ambiguous non-subsequence collisions and diagnose whether they reflect null/questionable modifiers, mixed fragments, or true amino-acid differences.

## Review

- Two-field collision audit over `data/mhc_index.csv`:
  - `1,217` total collided two-field groups
  - `1,155` safe nested groups where every shorter sequence is contained in one unique longest sequence
  - `62` ambiguous groups where two-field collapse should not silently choose a representative
- Ambiguous-group breakdown:
  - `52` `non_nested_seq_conflict`
  - `10` `same_length_diff_content`
- Suffix/modifier diagnosis:
  - only `2` ambiguous groups involve any suffix at all, both `N`
  - `60/62` ambiguous groups have no suffix modifiers
  - so these are not mainly `N/Q/PS` annotation issues
- Main causes in the local corpus:
  - mixed completeness records that are not clean subsequences of one longest sequence
  - a small non-human IPD-MHC subset with true same-length amino-acid disagreements inside one two-field family
  - HLA ambiguous cases in this corpus are still mixed-completeness conflicts, not same-length full-protein disagreements
- Runtime behavior now:
  - safe two-field aliases resolve to the longest representative record
  - ambiguous two-field aliases return an explicit unresolved/ambiguous result with candidate metadata
  - full-resolution record keys are unchanged
- Shared lookup behavior:
  - predictor, CLI prediction, and probe-training index sequence lookups now use the same collision-aware alias logic as `resolve_alleles`
- Artifacts:
  - full ambiguous enumeration written to `artifacts/two_field_ambiguous_collisions.tsv`
- Verification:
  - `python -m py_compile data/mhc_index.py inference/predictor.py cli/predict.py scripts/probe_training.py tests/test_mhc_index.py`
  - `pytest -q tests/test_mhc_index.py tests/test_predictor.py tests/test_allele_resolver.py tests/test_loaders.py` -> `145 passed`
  - `pytest -q tests/test_tasks.py tests/test_data_cli.py tests/test_predict_cli.py` -> `70 passed`

## Focused refactor: groove-native MHC runtime path (2026-03-06)

- [x] Replace the current two-field alias policy with a runtime representative policy:
  - nested fragments -> longest representative
  - exact/consistent overlap assemblies with >=100aa overlap -> assembled representative
  - multiple full records with identical extracted groove halves -> deterministic exemplar
  - groove-disagreeing families -> keep two-field alias ambiguous
- [x] Materialize representative metadata for alias resolution so `resolve_alleles()`, predictor lookup, and training all use the same sequence/groove decision.
- [x] Ensure any assembled representative is re-parsed through groove extraction and only accepted when the resulting chain looks structurally valid for its class.
- [x] Switch loader-side MHC preparation from full-chain `(mhc_a, mhc_b)` semantics to groove-half semantics while keeping the existing two-segment tensor interface stable.
- [x] Replace class-I `(alpha, B2M)` runtime inputs with `(alpha1, alpha2)` groove halves and class-II `(alpha, beta)` full-chain inputs with `(alpha1, beta1)` groove halves.
- [x] Update runtime fallback behavior for direct sequences and unknown alleles to use `prepare_mhc_input()` / live groove extraction rather than raw full-chain truncation.
- [x] Reduce collator MHC max length to groove-scale lengths and keep MIL bag collation aligned with the new groove halves.
- [x] Update `Presto` segment embeddings/position embeddings to explicit groove-half semantics and delete the learned groove-bias path that previously downweighted non-groove residues.
- [x] Keep the public forward signature stable (`mhc_a_tok`, `mhc_b_tok`) for now, but make those tensors unambiguously mean `groove_half_1` and `groove_half_2`.
- [x] Update inference and probe-training utilities to resolve and tokenize groove halves from the same runtime representative/index logic.
- [x] Add tests for:
  - overlap assembly rescue on ambiguous two-field families
  - groove-equivalent same-two-field families
  - ambiguous groove-disagreeing families remaining unresolved
  - loader/predictor class-I and class-II groove-half assembly
  - collate/model forward with groove-scale segment lengths
- [x] Run targeted sanity checks:
  - parser/index/predictor/loaders tests
  - model/collate tests
  - a short mini-batch training loop on real data with groove-half inputs

## Staging rationale

- Groove extraction and index augmentation are additive and can be validated in isolation. Doing them first reduces risk and avoids mixing biologic parsing failures with model-contract regressions.
- TCR-evidence schema work is also additive. Parsing/preserving method metadata before the batch/model rewrite prevents a second pass through the data sources.
- The largest API churn is in loader/collate/model contracts. MHC groove migration and receptor-input removal both hit those same interfaces, so they should land together in one coordinated patch, not as two sequential rewrites.
- Predictor/CLI/training cleanup should come only after the data and model contracts are stable; otherwise the same public commands/tests will fail for moving reasons.
- Training diagnostics belong at the end. Running them before the canonical contract is coherent would not answer the user's question about whether the new design works.

## Review targets

- Canonical model after this sequence should be:
  - pMHC and assay-context inputs only
  - groove-half MHC representation
  - `recognition -> immunogenicity` latent path intact
  - no receptor-sequence-conditioned branches
  - optional pMHC-only `tcr_evidence` auxiliary outputs
- Canonical training should use receptor databases only as pMHC evidence supervision, not as receptor-token sources.
- Modal analysis should be interpreted only after the no-AMP/local sanity path is proven and the batch/model contract has stopped changing.

## Review

- Phase 0 baseline completed on a real pMHC-only slice built from raw IEDB head subsets plus index-resolved MHC sequences:
  - loaded records: binding `128`, kinetics `30`, stability `32`, elution `64`, tcell `57`, processing `0`
  - dataset size after resolved-only filtering: `311`
  - MHC resolution on that slice: `28/33` alleles resolved from index, `0` invalid sequences dropped, unresolved filter dropped `2` kinetics rows and `7` tcell rows
  - balanced batch composition on the first three batches was stable and mixed:
    - binding_ic50 `6`
    - binding_kinetics `2`
    - binding_stability `2`
    - elution_ms `3`
    - tcell_response `3`
- Five repeated CPU training steps on one collated balanced batch were finite and improved rapidly:
  - loss trace: `23.81 -> 3.96 -> 3.29 -> 3.20 -> 2.49`
- Current forward path still carries substantial non-canonical baggage:
  - receptor/chain outputs remain in the canonical output dictionary (`mhc_*_type_logits`, `mhc_*_species_logits`, `tcell_context_logits`, compatibility scaffolding, etc.)
  - pMHC-only training works, but the output contract is much broader than the desired post-refactor design
- Interpretation:
  - the current stack is trainable enough to support a before/after comparison
  - the next work should focus on contract cleanup and MHC representation, not emergency numerical stabilization
- Phase 1 groove extraction/index layer completed.
- New additive surfaces:
  - [data/groove.py](/Users/iskander/code/presto/data/groove.py)
  - augmented index support in [data/mhc_index.py](/Users/iskander/code/presto/data/mhc_index.py)
  - CLI wiring in [cli/data.py](/Users/iskander/code/presto/cli/data.py) and [cli/main.py](/Users/iskander/code/presto/cli/main.py)
- Verification:
  - `python -m py_compile data/groove.py data/mhc_index.py cli/data.py cli/main.py tests/test_groove.py tests/test_mhc_index.py tests/test_data_cli.py`
  - `pytest -q tests/test_groove.py tests/test_mhc_index.py tests/test_data_cli.py` -> `49 passed`
- Species-specific audit results from the live index:
  - `Gallus gallus` class I: `27 ok`
  - `Oncorhynchus mykiss`: class I `48 ok`; class II alpha `3 ok`; class II beta `20 ok`, `2 beta1_only_fallback`
  - `Salmo salar`: class I `47 ok`, `1 alpha3_fallback`; class II alpha `18 ok`, `4 fragment_fallback`; class II beta `26 ok`, `16 fragment_fallback`
- Important biological/index observation:
  - `fragment_fallback` is not just a fish edge case. The local index contains many class-II beta groove-domain fragments directly, including large human `DRB1/DQB1/DPB1` slices at ~`79-89 aa`.
  - Accepting those directly is consistent with the project’s groove-fragment policy and avoids misclassifying valid groove-only records as failures.
- Bug caught during verification:
  - a class-II fragment fallback path briefly leaked into the class-I parser on `no_cys_pairs`; fixed before landing the phase.
- Additional non-model validation:
  - added explicit live-index regression tests for `Gaga-BF2*002:01:01` (chicken class I) and `Onmy-DAB*16:01` (trout class-II beta `beta1_only_fallback`) in `tests/test_groove.py`.
  - verified the species-specific counts with the class-II chain chosen from gene semantics rather than the generic `extract_groove()` default:
    - `Gallus gallus`: `BF1 12 ok`, `BF2 15 ok`
    - `Oncorhynchus mykiss`: `UBA 48 ok`, `DAA 3 ok`, `DAB 20 ok + 2 beta1_only_fallback`
    - `Salmo salar`: `UBA 47 ok + 1 alpha3_fallback`, `DAA 18 ok + 4 fragment_fallback`, `DAB 26 ok + 16 fragment_fallback`
  - one earlier ad hoc audit briefly under-counted fish class-II beta because `extract_groove()` defaults class-II dispatch to alpha when no chain is passed; the parser/index pipeline itself is unaffected because it already routes by gene.
- Phase 2 TCR-evidence schema layer completed.
- New additive surfaces:
  - `TcrEvidenceRecord`, `load_vdjdb_tcr_evidence()`, `load_mcpas_tcr_evidence()` in [data/loaders.py](/Users/iskander/code/presto/data/loaders.py)
  - preserved receptor-evidence metadata in [data/cross_source_dedup.py](/Users/iskander/code/presto/data/cross_source_dedup.py)
- Verification:
  - `python -m py_compile data/groove.py data/mhc_index.py data/loaders.py data/cross_source_dedup.py tests/test_groove.py tests/test_mhc_index.py tests/test_data_cli.py tests/test_loaders.py tests/test_cross_source_dedup.py`
  - `pytest -q tests/test_groove.py tests/test_mhc_index.py tests/test_data_cli.py tests/test_loaders.py tests/test_cross_source_dedup.py` -> `94 passed`
- Phase 2 notes:
  - VDJdb method bins are now derived from both `method.identification` and `method.verification`, which is necessary to recover `target_cell_functional` labels from rows where the identification method is only multimer-based.
  - per-assay cross-source CSVs now preserve the new receptor-evidence metadata columns instead of silently dropping them during serialization.
- Groove-native runtime path completed across index resolution, loader/collate, model embeddings, predictor, and training utilities.
- Alias-resolution behavior now distinguishes:
  - `nested_longest_unique`
  - `assembled_overlap`
  - `groove_equivalent_exemplar`
  - explicit unresolved ambiguous families
- `resolve_alleles()` now reports the concrete representative full allele in `resolved`, while preserving alias-level metadata in `representative_allele` / `representative_policy`.
- Runtime MHC contract is now:
  - class I -> `(alpha1_groove, alpha2_groove)`
  - class II -> `(alpha1_groove, beta1_groove)`
  - public tensor names remain `mhc_a_tok` / `mhc_b_tok`
- Collation and model input scale changed from full-chain MHC lengths to groove-scale lengths:
  - collator default `max_mhc_len=120`
  - `Presto` positional embeddings are now groove-half specific
  - legacy groove-bias masking over non-groove residues was removed
- Verification:
  - `python -m py_compile data/mhc_index.py data/loaders.py inference/predictor.py data/collate.py models/presto.py scripts/train_synthetic.py scripts/train_iedb.py tests/test_loaders.py tests/test_predictor.py tests/test_presto.py`
  - `pytest -q tests/test_mhc_index.py tests/test_loaders.py tests/test_predictor.py tests/test_presto.py tests/test_train_iedb.py tests/test_training_e2e.py` -> `194 passed`
- Mini-batch runtime sanity on real class-I and class-II sequences:
  - class I emitted groove halves of lengths `91` and `93`
  - class II emitted groove halves of lengths `84` and `94`
  - collated batch tensors were `(4, 120)` for both groove segments
  - five CPU optimizer steps on a real class-I batch were finite and improved monotonically:
    - `10.5980 -> 10.5574 -> 10.4994 -> 10.4153 -> 10.3170`
- Focused fix review:
  - `extract_groove()` no longer defaults missing class-II chain identity to alpha. It now uses allele/gene inference first, sequence-only inference second, and otherwise returns explicit `ambiguous_chain` / `chain_inference_failed` statuses.
  - fine-grained species identity is now preserved separately through `infer_species_identity()` while coarse network buckets remain in `normalize_processing_species_label()`.
  - `mhcgnomes` is now the required parser for allele normalization/class/species inference across resolver, index, dedup, and training-task helpers.
  - the local editable namespace-package case is fixed by resolving `mhcgnomes.function_api.parse` when the top-level `mhcgnomes` import does not expose `parse`.
  - targeted verification after the parser hardening: `pytest -q tests/test_allele_resolver.py tests/test_mhc_index.py tests/test_groove.py tests/test_loaders.py tests/test_cross_source_dedup.py tests/test_tasks.py tests/test_predictor.py` -> `208 passed`
- DR alpha default review:
  - native default DRA mapping is now explicit in `data/allele_resolver.py` as `DEFAULT_DR_ALPHA_BY_SPECIES`.
  - scope is intentionally native-only: no cross-species DRA proxy sequences were invented for species missing local DRA entries.
  - merged training data species that actually present DRB-only records are covered natively:
    - `Homo sapiens -> HLA-DRA*01:01:01:01`
    - `Bos sp. -> BoLA-DRA*001:01`
    - `Macaca mulatta -> Mamu-DRA*01:01`
    - `Macaca fascicularis -> Mafa-DRA*01:01:01:01`
    - `Pan troglodytes -> Patr-DRA*01:01:01`
    - `Sus sp. -> SLA-DRA*01:01:01`
  - loader/predictor behavior is corrected for class-II DR beta-only inputs:
    - before: `mhc_a=DRB`, `mhc_b=""`
    - now: `mhc_a=DRA_default`, `mhc_b=DRB`
  - loader `mhc_b` inputs now resolve allele names as sequences, which also fixes VDJdb-style explicit class-II partner alleles.
  - verification:
    - `pytest -q tests/test_allele_resolver.py tests/test_loaders.py tests/test_predictor.py` -> `122 passed`
    - `pytest -q tests/test_mhc_index.py tests/test_tasks.py tests/test_allele_resolver.py tests/test_loaders.py tests/test_predictor.py` -> `184 passed`

# Receptor Removal Planning (2026-03-06)

## Spec

- Goal: define a surgical plan to remove receptor sequences (TCR/BCR) from canonical Presto inputs, training data, and training/inference infrastructure while preserving the pMHC and immune-response model.
- Refined boundary from user clarification:
  - remove receptor sequences as inputs and training infra
  - keep pMHC-only outputs capturing whether a cognate TCR has been observed in receptor databases such as VDJdb
  - where metadata exists, optionally segment that output by assay/evidence method instead of collapsing everything to one scalar
- Canonical references reviewed:
  - `docs/design.md`
  - `docs/tcr_spec.md`
  - `data/loaders.py`
  - `data/collate.py`
  - `models/presto.py`
  - `models/tcr.py`
  - `scripts/train_iedb.py`
  - `scripts/train_synthetic.py`
  - `training/tasks.py`
  - `training/trainer.py`
  - `inference/predictor.py`
  - `cli/main.py`
  - `cli/evaluate.py`
  - `data/cross_source_dedup.py`

## Plan

- [x] Audit the live receptor-specific surface area in dataset ingestion, collate/batch schema, model forward path, predictor/CLI APIs, and legacy training tasks.
- [x] Decide the canonical boundary: keep pMHC + immune-response supervision, remove receptor-sequence-conditioned inputs and objectives.
- [x] Refine that boundary so receptor databases become pMHC-only evidence outputs rather than sequence-conditioned tasks.
- [x] Write the detailed execution plan in `tasks/receptor_removal_plan.md`.
- [x] Audit local VDJdb and McPAS source schemas to determine which TCR-evidence metadata fields actually survive in raw files and current loaders.
- [x] Quantify counts per candidate evidence bin and reject any taxonomy bins that are too sparse or unsupported by the data.
- [x] Write an exact `tcr_evidence` / `tcr_evidence_method` spec with field mappings, normalization rules, target semantics, and recommended loss family.
- [ ] Phase 0: baseline the current pMHC-only path with a no-AMP mini-run so receptor removal can be measured against a stable canonical reference.
- [ ] Phase 1: replace receptor-sequence source ingestion with pMHC-only `tcr_evidence` / `tcr_evidence_method` supervision while keeping raw archival parsers optional.
- [ ] Phase 2: remove receptor fields from `PrestoSample` / `PrestoBatch` and delete receptor-chain auxiliary collation.
- [ ] Phase 3: remove the TCR tower and receptor classifier/matcher outputs from `models/presto.py`, `models/tcr.py`, predictor APIs, and CLI commands.
- [ ] Phase 4: remove legacy TCR/BCR training-task infrastructure (`tcr_pmhc`, receptor-chain typing, TCR pairing, retrieval evaluation) and replace it with pMHC-only TCR-evidence outputs, while preserving pMHC-only MIL contrastive regularization.
- [ ] Phase 5: update docs/tests to reflect that canonical Presto is a pMHC and immune-response model, not a receptor-conditioned model.
- [ ] Verification: prove canonical training no longer ingests `vdjdb`, `10x`, `tcr_pmhc`, or receptor tokens; then run a short pMHC training sanity check and compare against the Phase 0 baseline.

## Review

- Architectural conclusion:
  - receptor sequences do not fit cleanly into the current canonical Presto factorization.
  - the core model is now a pMHC plus immune-response model: peptide processing, MHC binding, presentation, and assay-context-dependent T-cell response.
  - receptor-specific matching is a different problem class: instance-level recognition conditioned on a particular clonotype, not a latent upstream cause of processing/presentation.
- Refined output policy:
  - removing receptor sequences does not require dropping receptor-derived supervision.
  - VDJdb/McPAS-like data can be converted into a pMHC-only output family such as `tcr_evidence_prob` plus optional method-segmented logits.
  - this keeps the fact that a pMHC has attested cognate-TCR evidence, without forcing canonical Presto to ingest clonotype sequences.
- Exact spec completed in `tasks/tcr_evidence_spec.md`.
  - required outputs:
    - `tcr_evidence_logit/prob`
    - `tcr_evidence_method_logits/probs`
  - selected VDJdb assay-family bins:
    - `multimer_binding`: `1,217` exact unique pMHC
    - `target_cell_functional`: `409`
    - `functional_readout`: `291`
  - rejected bins:
    - `culture_stimulation`: `74`
    - `display_selection`: `1` exact unique pMHC despite `59,376` raw rows
    - `other` / `unknown`
  - overall receptor-evidence support:
    - VDJdb: `226,494` rows, `1,962` coarse unique pMHC
    - McPAS: `20,227` rows, `427` coarse unique pMHC
    - coarse union across sources: `2,329` unique pMHC with `60` source overlap
  - method panel must be multi-label:
    - `1,529` exact pMHC with one selected family
    - `188` with two
    - `4` with three
  - McPAS supports the overall `tcr_evidence` label but not trustworthy method segmentation from local metadata.
- Current-code audit:
  - docs already say TCR is future-only (`docs/tcr_spec.md`), but canonical code still carries receptor scaffolding.
  - `models/presto.py` still exposes `enable_tcr`, `tcr_a_tok`, `tcr_b_tok`, `encode_tcr()`, `predict_chain_attributes()`, `TCRpMHCMatcher`, and chain/cell auxiliary heads.
  - `data/loaders.py` and `data/collate.py` still inject receptor sequences and receptor-chain labels into canonical batches via `vdjdb_records`, `sc10x_records`, `tcr_a`, `tcr_b`, and `chain_*` labels.
  - `scripts/train_iedb.py` still loads VDJdb and 10x inputs into the canonical unified trainer even though the design docs say TCR-conditioned training is not canonical.
  - `training/tasks.py`, `training/trainer.py`, `models/tcr.py`, `cli/evaluate.py`, and `inference/predictor.py` still carry legacy receptor matching / chain-classification infrastructure.
  - `VDJdbRecord` and the cross-source VDJdb parser currently discard most assay/method metadata, so method-segmented TCR-evidence outputs require a small schema expansion first.
- Important boundary for implementation:
  - keep pMHC-only contrastive MIL regularization from the learning refactor; it is not the TCR tower and should not be removed.
  - keep T-cell response labels and T-cell assay context features; they remain valid population-level supervision even after receptor sequences are removed.
  - add a new pMHC-only receptor-evidence output family instead of deleting receptor databases outright.
  - B-cell receptor sequence handling is already mostly non-canonical. The live receptor baggage on the training path is primarily TCR plus the generic `chain_aux` route that also covers IGH/IGK/IGL.

# Learning Refactor Execution (2026-03-06)

## Spec

- Canonical analysis: `tasks/training_analysis.md`
- Canonical checklist: `tasks/learning_refactor.md`
- Canonical implementation order: `tasks/learning_refactor_plan.md`
- Execution rule for this pass:
  - implement by dependency phase, not by scattered TODO item
  - prove each phase locally before moving on
  - keep Phase 0 independent so training-dynamics fixes are isolated from the latent-DAG rewrite

## Plan

- [x] Phase 0: implement `E3` weight initialization, `E4` LR scheduler, and `D1` `mhc_class` default/inference cleanup.
- [x] Phase 0 verification: targeted tests for loader/class handling, model initialization sanity, and trainer scheduler behavior.
- [x] Phase 1: implement `A1`-`A4` together as one coupled latent-DAG refactor.
- [x] Phase 1 verification: forward/output contract checks, dependency-flow checks, and regression coverage for the new latent structure.
- [x] Phase 2: implement `B1` coupled core enumeration with explicit PFR representation on top of the new interaction latent.
- [x] Phase 2 verification: core-candidate enumeration sanity, shape/compute checks, and targeted biological edge cases (`8mer`, `9mer`, long class II peptides).
- [x] Phase 3: implement `C1`-`C3` context-token cleanup, `groove_vec`, and `pmhc_vec` signal repair.
- [x] Phase 3 verification: confirm processing isolation, binding/presentation access to groove context, and direct allele signal in `pmhc_vec`.
- [x] Phase 4: implement `D2` class-split MIL bags and `D3` pathway-MIL handling for ambiguous T-cell assays.
- [x] Phase 4 verification: dataset/batch audits showing correct bag construction and class/pathway separation.
- [x] Phase 5: implement `E1` contrastive MIL and `E2` bag sparsity regularization.
- [x] Phase 5 verification: loss wiring tests plus small-run diagnostics that show allele-discrimination pressure is present.
- [ ] End-to-end verification: close the unresolved-MHC default/config gap, rerun targeted parser/trainer regression checks, and prove the full profile enters resolved-only training mode by default.
- [ ] End-to-end verification: run a fresh 1-epoch full unified Modal training job, pull artifacts, and analyze losses, probes, gradients/latents, and allele separation (`SLLQHLIGL`, `HLA-A*02:01` vs `HLA-A*24:02`).
- [ ] End-to-end verification: if the epoch looks healthy, launch a longer full training run; otherwise, document the failure mode, implement the next training-dynamics fix, and re-verify locally before relaunch.

## Review

- Phase 0 completed.
- `D1`:
  - `data/loaders.py` record dataclasses no longer default `mhc_class="I"`; they now preserve `None` for genuinely missing class labels.
  - loader class resolution now normalizes explicit class labels first, then prefers mhcgnomes-backed inference via `data/allele_resolver.py`, with the old heuristic retained only as fallback.
  - `load_iedb_processing()` no longer silently injects class I when both source class and allele are absent.
- `E3`:
  - `models/presto.py` now applies explicit transformer-scale initialization.
  - latent queries use `std=1/sqrt(d_model)`.
  - embeddings use the same scale, while preserving both the padding row and the fixed-zero `X` row semantics.
  - multi-head attention projection weights are explicitly Xavier-initialized with zero biases.
- `E4`:
  - `scripts/train_synthetic.py` now provides `build_warmup_cosine_scheduler()` and steps the scheduler once per successful optimizer step inside `train_epoch()`.
  - `scripts/train_iedb.py` now constructs the same warmup+cosine schedule using the effective per-epoch step count and forwards it through the compatibility wrapper.
  - both training scripts now log current LR per epoch.
- Verification:
  - `python -m py_compile data/allele_resolver.py data/loaders.py models/presto.py scripts/train_synthetic.py scripts/train_iedb.py tests/test_loaders.py tests/test_presto.py tests/test_train_synthetic.py tests/test_train_iedb.py`
  - `pytest -q tests/test_loaders.py tests/test_presto.py tests/test_train_synthetic.py tests/test_train_iedb.py` -> `121 passed`
  - `pytest -q tests/test_allele_resolver.py tests/test_predictor.py` -> `83 passed`
  - scheduler tests emit PyTorch's known `lr_scheduler` deprecation warning from inside `SequentialLR`; behavior is correct and the warning is non-blocking for now.
- Phase 1 completed.
- `A1` + `A2`:
  - `models/presto.py` now uses a single MHC-aware cross-attention latent, `pmhc_interaction`, instead of separate `binding_affinity`/`binding_stability` latents.
  - default interaction query count is now `8`, and the interaction path preserves multi-query detail by flattening projected query outputs instead of mean-pooling them back to one vector.
  - attention-stat diagnostics were updated to average over both heads and query slots so multi-query attention remains measurable.
- `A3`:
  - the scalar bottleneck between binding and presentation was removed from the main information path.
  - binding kinetics remain physics-aware readouts through `BindingModule`, but presentation and immunogenicity now consume the full normalized interaction vector.
  - `pmhc_vec` is now projected from `pmhc_interaction + presentation`, not from the old scalar-bottlenecked latent tuple.
  - added vector normalization on the flattened interaction path plus presentation/immunogenicity vectors to keep optimization stable at the higher-dimensional Phase 1 interface.
- `A4`:
  - the cross-attention DAG now has five canonical latents: `processing`, `ms_detectability`, `species_of_origin`, `pmhc_interaction`, and `recognition`.
  - presentation and immunogenicity are now derived MLP stages, not duplicated latent queries.
  - class-specific output heads are retained only as a compatibility/readout layer (`*_class1`, `*_class2`, `*_cd8`, `*_cd4`) so existing training and inference code continues to run while the internal architecture is unified.
- Phase 1 verification:
  - `pytest -q tests/test_presto.py tests/test_train_synthetic.py tests/test_train_iedb.py` -> `93 passed`
  - `pytest -q tests/test_presto.py tests/test_training_e2e.py` -> `45 passed`
  - `pytest -q tests/test_loaders.py tests/test_presto.py tests/test_train_synthetic.py tests/test_train_iedb.py tests/test_allele_resolver.py tests/test_predictor.py tests/test_training_e2e.py tests/test_pmhc.py tests/test_heads.py` -> `260 passed`
- Phase 2 completed.
- `B1`:
  - `models/presto.py` no longer predicts a global soft core start/width from pooled MHC vectors and no longer injects a shared core-relative position field back into the peptide stream.
  - the `pmhc_interaction` latent now enumerates candidate peptide core windows directly from encoder states, with per-candidate core tokens cross-attending to MHC tokens and explicit N-/C-terminal PFR summaries folded into the candidate interaction vector.
  - 8-mers are handled as a one-candidate special case with width `8`; all longer peptides use the canonical 9-residue binding window, so `11mer -> 3` candidates and `15mer -> 7` candidates without any peptide tiling step.
  - candidate scores are normalized into a posterior over core placements; the downstream interaction vector is the posterior-weighted mixture, so binding and core placement are now coupled in one differentiable path.
  - compatibility outputs remain available: `core_start_logit`, `core_start_prob`, `core_membership_prob`, plus new candidate-level diagnostics (`core_window_mask`, `core_window_start`, `core_window_length`, `core_window_prior_logit`, `core_window_logit`, `core_window_posterior_prob`).
- Phase 2 verification:
  - `python -m py_compile models/presto.py tests/test_presto.py`
  - direct runtime sanity:
    - mixed-length batch forward pass emits finite `core_window_*`, `binding_logit`, and attention-support stats.
    - mixed-length train-mode backward pass stays finite with `core_counts=[1, 3, 7, 7]` and `core_lengths=[8, 9, 9, 9]`.
  - `pytest -q tests/test_presto.py tests/test_training_e2e.py` -> `46 passed`
  - `pytest -q tests/test_presto.py tests/test_train_synthetic.py tests/test_train_iedb.py tests/test_training_e2e.py tests/test_pmhc.py` -> `128 passed`
- Phase 3 completed.
- `C1`:
  - the APC/cell-type conditioning path is now named and surfaced as `apc_cell_type_context`, with the projection accessible via `apc_cell_type_context_proj`.
  - semantics are unchanged: it still carries class/species/chain-compatibility context, not allele identity.
- `C2`:
  - added a learned `groove_vec` summary in `models/presto.py` using a dedicated groove attention query over class-conditional MHC groove masks.
  - Class I groove summary uses early chain-A positions only; Class II groove summary uses early chain-A plus early chain-B positions; uncertain cases mix the two by `class_probs`.
  - `groove_vec` is injected into the `pmhc_interaction` latent as an extra KV token and into the presentation MLP as an explicit input, while processing remains isolated from this allele-specific signal.
- `C3`:
  - `pmhc_vec` now includes `interaction_vec + presentation_vec + pep_vec + mhc_a_vec + mhc_b_vec`.
  - this removes the dead `pep_vec` path and gives downstream TCR matching a direct allele/peptide skip connection instead of relying purely on latent bottlenecks.
- Phase 3 verification:
  - `python -m py_compile models/presto.py tests/test_presto.py`
  - mini-batch trainer sanity:
    - 5 repeated trainer steps on one collated batch stayed finite.
    - losses moved `25.95 -> 10.76 -> 38.46 -> 8.05 -> 10.92`; non-monotone but clearly trainable and numerically stable.
  - `pytest -q tests/test_presto.py tests/test_training_e2e.py` -> `49 passed`
  - `pytest -q tests/test_presto.py tests/test_train_synthetic.py tests/test_train_iedb.py tests/test_training_e2e.py tests/test_pmhc.py` -> `131 passed`
- Phase 4 completed.
- `D2`:
  - `data/collate.py` now splits mixed-class elution MIL bags into separate class-I and class-II bags instead of a single biologically invalid shared Noisy-OR bag.
  - split bags keep explicit `:<class>` suffixes in `mil_bag_sample_ids` so audits can still trace them back to the source sample.
- `D3`:
  - `data/loaders.py` now preserves all parsed restriction alleles on T-cell records and only builds pathway MIL for genuinely ambiguous mixed-class assays.
  - `data/collate.py` suppresses direct T-cell supervision for those ambiguous samples and emits `tcell_mil_*` tensors plus replicated `tcell_mil_context`, so the pathway bags still reach the T-cell assay head with assay metadata.
- Phase 4 verification:
  - `python -m py_compile data/collate.py data/loaders.py tests/test_collate.py tests/test_loaders.py`
  - `pytest -q tests/test_collate.py tests/test_loaders.py` -> `64 passed`
- Phase 5 completed.
- `E1`:
  - `scripts/train_synthetic.py` now applies genotype-substitution contrastive MIL regularization on positive presentation bags by swapping in a sufficiently dissimilar same-class genotype from another bag in the batch.
  - contrastive pairing uses token-level MHC-alpha sequence identity `<0.90` as the in-batch dissimilarity gate.
- `E2`:
  - MIL bag sparsity regularization now penalizes `softplus(sum(p_i) - target_sum)` for elution/presentation/MS bags and pathway-T-cell MIL bags.
  - the trainer now also consumes `tcell_mil_*` bags directly via noisy-OR supervision on `tcell_logit` and `immunogenicity_logit`; before this patch, the new pathway bags would have been ignored.
- Phase 5 verification:
  - `python -m py_compile scripts/train_synthetic.py scripts/train_iedb.py tests/test_train_synthetic.py`
  - `pytest -q tests/test_collate.py tests/test_loaders.py tests/test_train_synthetic.py tests/test_train_iedb.py tests/test_training_e2e.py` -> `120 passed`
  - real-model minibatch sanity after Phase 5:
    - loss trace over 5 optimizer steps on a mixed supervision batch: `2.2402 -> 2.6259 -> 2.0863 -> 1.7115 -> 1.7107`
    - gradient norms stayed finite: `11.84, 7.97, 5.90, 2.52, 6.51`
    - active loss terms included `elution_mil_sparsity`, `presentation_mil_contrastive`, and `immunogenicity_mil`, confirming the new MIL path is live.
  - synthetic CLI smoke run:
    - `python -m presto.scripts.train_synthetic --epochs 1 --batch_size 8 --n_binding 16 --n_elution 16 --n_tcr 8 --d_model 64 --n_layers 1 --n_heads 4 --run-dir /tmp/presto_sanity_refactor`
    - completed successfully with `train_loss=2.3767`, `val_loss=1.0607`
- End-to-end verification in progress:
  - fixed the full-profile config contract so `filter_unresolved_mhc` defaults to `True` consistently in `IEDB_DEFAULTS`, the script parser, the public CLI parser, and `run()`'s programmatic fallback.
  - targeted regression coverage:
    - `python -m py_compile cli/main.py scripts/train_iedb.py tests/test_train_cli.py tests/test_train_iedb.py`
    - `pytest -q tests/test_train_cli.py tests/test_train_iedb.py` -> `60 passed`
    - `pytest -q tests/test_collate.py tests/test_loaders.py tests/test_train_synthetic.py tests/test_train_iedb.py tests/test_training_e2e.py tests/test_train_cli.py` -> `142 passed`
  - local full-pipeline canary on real merged/index data reached live optimization without loader/config errors:
    - scanned `3,163,395` merged rows under canary caps
    - resolved-only filtering dropped the expected unresolved rows and reached `resolved=2117/2117`
    - optimizer entered `train` on 24 batches with probe tracking enabled before the CPU run was stopped for throughput reasons
  - first full Modal epoch retry (`refactor-e1-57725c3a`, app `ap-vH91pB9Ocqs9k7pS04GcNY`) got past the unresolved-MHC gate but failed during dataset construction on `Likely nucleotide sequence loaded for MHC chain: allele=Mamu-A1*001:01, chain=mhc_a, len=1098`.
  - follow-up repair:
    - `scripts/train_iedb.py` now treats nucleotide-like sequences as invalid in three places:
      - resolved-index sequence audit / invalid-allele pruning
      - resolved-row filtering for direct MHC sequences
      - MHC-only augmentation sampling from the index
    - MHC-only augmentation now uses the canonical `70 aa` floor instead of a stale `50 aa` threshold.
  - regression coverage for the nucleotide-like failure mode:
    - `python -m py_compile scripts/train_iedb.py tests/test_train_iedb.py`
    - `pytest -q tests/test_train_iedb.py tests/test_train_cli.py` -> `62 passed`
    - `pytest -q tests/test_collate.py tests/test_loaders.py tests/test_train_synthetic.py tests/test_train_iedb.py tests/test_training_e2e.py tests/test_train_cli.py` -> `144 passed`
  - second full Modal retry (`refactor-e1-dabcd3da`, app `ap-mr0QF2doBlrtj7GPH8bNp4`) reached the first training batch on CUDA/bf16 and then failed with a mixed-dtype indexed write: `Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source.`
  - bf16 follow-up repair:
    - `models/presto.py` now keeps `core_window_posterior` and related mask/scatter tensors dtype-consistent through the `core_start_prob[...] = ...` write and `core_membership.scatter_add_` path inside `_binding_latent_query`.
    - added a CPU bf16-autocast regression in `tests/test_presto.py` that reproduces the exact forward path without requiring a GPU.
  - regression coverage for the bf16 fix:
    - local repro script under `torch.autocast('cpu', dtype=torch.bfloat16)` now completes
    - `pytest -q tests/test_presto.py tests/test_training_e2e.py tests/test_train_synthetic.py tests/test_train_iedb.py` -> `106 passed`
  - third full Modal retry (`refactor-e1-bfa88c7a`, app `ap-aPyOEBVofEDqTmXnvASvsA`) got further but still failed on the first CUDA bf16 batch with the inverse mixed-dtype indexed-write shape (`destination Float`, `source BFloat16`).
  - final bf16 repair in this pass:
    - `models/presto.py` no longer preallocates a float positional buffer and fills slices; `_build_single_stream()` now builds segment positional embeddings independently and concatenates them, eliminating another autocast-sensitive indexed-write pattern.
    - added a focused regression in `tests/test_presto.py` that forces one positional subpath (`pep_frac_mlp`) to emit bf16 while the rest of the stream stays default dtype.
  - regression coverage for the positional bf16 fix:
    - focused local repro with a bf16-wrapped positional submodule now completes
    - `python -m py_compile models/presto.py tests/test_presto.py`
    - `pytest -q tests/test_presto.py tests/test_training_e2e.py tests/test_train_synthetic.py tests/test_train_iedb.py` -> `107 passed`

# Unified Training Failure Audit + Repair (2026-03-06)

## Plan

- [x] Read the canonical docs and inspect the concrete source paths for unified data generation, collation, model forward/loss wiring, and Modal training orchestration.
- [x] Reproduce the "doesn't learn" behavior with targeted local diagnostics: dataset sanity probes, batch/label audits, forward/loss checks, and a short train/probe run.
- [x] Identify the highest-impact bugs or design/implementation mismatches causing failed learning, with file-level evidence.
- [x] Implement the smallest defensible fixes for the most egregious issues in data generation, model architecture, or training wiring.
- [x] Add or update targeted tests and run focused verification plus at least one end-to-end local training sanity check.
- [x] Regenerate unified merged data and confirm the generated output matches the intended semantics.
- [x] Run a fresh 1-epoch unified Modal training job on the regenerated data, collect artifacts/logs, and analyze training dynamics.
- [ ] Commit the full resulting change set and push it to the current branch.

## Review

- Root-cause data bug fixed: `scripts/train_iedb.py` had a stale hard-coded `MIN_MHC_SEQUENCE_LEN = 101` while `data/mhc_index.py` and `data/loaders.py` had already been corrected to accept groove-bearing fragments at `>=70 aa`. This caused the training path to silently drop valid short class II chains even when the index and loader accepted them.
- Canonicalized MHC fragment handling:
  - `data/mhc_index.py`: reject `<70 aa` fragments and nucleotide-like entries during index build/validation.
  - `data/loaders.py`: fail fast on nucleotide-like chains and `<70 aa` sequences.
  - `scripts/train_iedb.py`: import the canonical MHC AA alphabet and minimum chain length from `presto.data.loaders` instead of duplicating its own threshold.
- Sampler/loss repair already in this audit remains in place and validated:
  - synthetic binding negatives no longer form their own top-level task group;
  - MHC-only augmentation samples are explicitly isolated as `mhc_aux`;
  - MHC augmentation sample count is capped by dataset fraction;
  - auxiliary MHC losses and affinity-probe loss use explicit base weights before support aggregation.
- Focused verification:
  - `pytest -q tests/test_mhc_index.py tests/test_loaders.py tests/test_train_synthetic.py tests/test_train_iedb.py` -> `85 passed`.
  - additional regression after the train-path threshold fix:
    - `python -m py_compile scripts/train_iedb.py tests/test_train_iedb.py`
    - `pytest -q tests/test_train_iedb.py tests/test_mhc_index.py tests/test_loaders.py` -> `75 passed`.
- Regenerated index outcome:
  - `data/mhc_index.csv` rebuilt with `lt70=0`, `lt80=421`, `lt100=5883`, `nucleotide_like=0`.
  - `70-99 aa` bucket is dominated by class II, especially human class II, so the old `>100 aa` filter was biologically and statistically incorrect.
- Diagnostic Modal comparison:
  - Pre-fix run `diag-repair-20260306a`:
    - `train_loss=1.1332`, `val_loss=0.7220`
    - `SLLQHLIGL` probe:
      - `HLA-A*02:01`: `KD≈1486 nM`, `bind≈0.2056`, `present≈0.8972`
      - `HLA-A*24:02`: `KD≈1596 nM`, `bind≈0.1916`, `present≈0.8844`
      - delta: binding `+0.0141`, presentation `+0.0128`
    - pMHC flow: binding-logit norms `mhc=1.0817`, `peptide=0.1026`, `interaction=0.1021`
  - Post-fix run `diag-repair-20260306c`:
    - no more `short_len=81/89` invalid-sequence drops in the live training path
    - resolved MHC coverage improved from `44960` to `44974` rows in the diagnostic cap
    - `train_loss=1.1285`, `val_loss=0.8165`
    - `SLLQHLIGL` probe:
      - `HLA-A*02:01`: `KD≈840.8 nM`, `bind≈0.3441`, `present≈0.9376`
      - `HLA-A*24:02`: `KD≈856.1 nM`, `bind≈0.3391`, `present≈0.9360`
      - delta: binding `+0.0050`, presentation `+0.0016`
    - pMHC flow: binding-logit norms `mhc=1.1418`, `peptide=0.3769`, `interaction=0.2672`
- Interpretation:
  - The model is no longer in a "learns nothing" regime. One epoch produces stable monotone loss reduction, correct `A*02:01 > A*24:02` ordering on the probe, and nontrivial joint peptide-MHC interaction.
  - The corrected path increases peptide/interactions entering the binding head substantially, which is a positive mechanistic sign.
  - The one-epoch validation-loss comparison is not apples-to-apples because adding accepted short-fragment rows changes dataset cardinality and thus the random train/val split.
  - Probe discrimination is still weak after one epoch, especially on presentation; this remains the main open modeling concern.
- Long run launched:
  - Detached full-profile Modal run: `full-repair-20260306a`
  - app id: `ap-EWy6JjrqiOA3i3xKzEk5Nk`
  - uses corrected `70 aa` index and strict resolved-only MHC filtering.

# Improve pMHC Binding Latent Architecture

# Unified Merge Rerun + Sanity Debug (2026-03-05)

## Plan

- [x] Confirm rerun completion and capture merge stats JSON/logs.
- [x] Sanity-check regenerated `data/merged_deduped.tsv` for strict allele-set semantics (`cell_hla_allele_set` only allele-like tokens).
- [x] Verify strict retention policy: cellular rows are kept only when allele set is known.
- [x] Identify and patch merge bottlenecks before rerun.
- [x] Re-run targeted tests and re-run merge sanity probes; record results below.

## Review

- Performance fixes applied in `data/cross_source_dedup.py`:
  - `UnifiedRecord` now uses `@dataclass(slots=True)` and `raw_data=None` default (removes per-record empty dict allocation).
  - IEDB/CEDAR parsers now compute `reference_text` only when PMID/DOI is unavailable (preserves fuzzy fallback semantics while removing unnecessary per-row string/regex work).
  - TSV write path switched from `csv.DictWriter` per-row dict materialization to `csv.writer` list rows.
  - tqdm overhead reduced by batching updates and coarser refresh intervals.
- Full merge rerun command:
  - `env TQDM_DISABLE=1 python -m presto data merge --datadir ./data`
- Full merge timings (from run log):
  - `load: 302.52s`
  - `dedup: 66.29s`
  - `cell_hla_lookup: 10.25s`
  - `cell_hla_annotate_filter: 9.37s`
  - `write_tsv: 12.08s`
  - `write_funnel: 3.47s`
  - `total: 405.11s` (~6.75 min)
- Throughput/scale:
  - loaded `12,459,631` records at `41,186.7/s`.
  - final output `3,317,673` rows.
- Sanity results on regenerated `data/merged_deduped.tsv`:
  - `invalid_token_count_total=0` for `cell_hla_allele_set` (only allele-like tokens remain).
  - `missing_cell_hla_by_assay={}` for cellular assays present in final output.
  - final assay composition:
    - `elution_ms: 2,727,347`
    - `binding_affinity: 251,418`
    - `tcell_response: 159,847`
    - `tcr_pmhc: 166,321`
    - kinetic assays: `12,740` total

## Plan

- [x] Add constructor flags to `Presto.__init__` for 4 architecture variants (A-D)
- [x] Implement `_binding_latent_query()` method (Variants A, B, D)
- [x] Implement `_run_pmhc_interaction()` method (Variant C)
- [x] Route binding latents through new methods in `forward()`
- [x] Write benchmark script (`scripts/benchmark_binding_latents.py`)
- [x] Run benchmark across 8 configurations
- [x] Verify: all 36 existing tests pass, processing isolation confirmed, all 8 configs work
- [x] Apply winner as new default: **Variant D** (groove prior) — `use_groove_prior=True`
- [ ] Full training on Modal (running: `groove-prior-v1`, 10 epochs)

## Benchmark Results (500 steps, lr=1e-3, d_model=64, n_layers=1)

### Round 2 (with Class I/II-aware groove bias)

| Config | Params | Train Loss | Val Loss | Pearson r | Efficiency |
|--------|--------|-----------|----------|-----------|------------|
| **D** | 1,346,306 | 0.2264 | 0.5506 | **0.2462** | 0.18 r/M |
| baseline | 1,345,506 | 0.1501 | 0.6829 | 0.1930 | 0.14 r/M |
| A+B4+C | 1,879,906 | 0.0699 | 0.5443 | 0.1879 | 0.10 r/M |
| C | 1,545,442 | 0.0577 | 0.5476 | 0.1064 | 0.07 r/M |
| B4+C | 1,612,898 | 0.0654 | 0.7691 | 0.0375 | 0.02 r/M |
| A | 1,745,378 | 1.6003 | 1.3143 | -0.0000 | - |
| A+C | 1,945,314 | 1.6147 | 1.3140 | -0.0000 | - |
| A+D | 1,746,178 | 1.6003 | 1.3143 | 0.0000 | - |
| A+C+D | 1,946,114 | 1.6147 | 1.3140 | 0.0000 | - |

## Winner Selection: Variant D (groove prior)

- **Best correlation** (r=0.2462) with near-lowest val loss (0.5506)
- **Minimal parameter overhead** (+800 params, 0.06% increase)
- **Biologically principled**: learnable position bias initialized to favor α1+α2 domains (Class I) and β1 domain (Class II)
- **No architectural complexity**: no new modules, just a learnable bias on existing attention

## Biological Audit

- **Processing isolation verified**: Variant C (and all others) do NOT change processing latent outputs
- **Groove prior Class I/II fix**: split into `groove_bias_a` (α chain, favor 0-180) and `groove_bias_b` (β chain, favor 0-90)
- **Bidirectional pMHC interaction (C)**: biologically correct — mutual induced fit
- **Benchmark scoring**: uses real HLA pocket positions (B/F/D pockets) and anchor positions (P2/PΩ)
- **LATENT_DEPS DAG**: all dependencies preserved across all variants

## Verification

- All 36 existing tests pass with groove prior as default
- All 12 configs produce identical output dict structure and tensor shapes
- All configs have finite gradients, no NaN/Inf
- Processing latent isolation confirmed with weight-copying test

---

# Implementation-vs-Design Audit + Latent Flow Critique (2026-03-04)

## Plan

- [x] Extract canonical architecture and latent-flow requirements from docs.
- [x] Inspect implementation paths (`models/presto.py`, encoders/heads, training loss wiring) against those requirements.
- [x] Identify concrete design/implementation mismatches with file-level evidence.
- [x] Evaluate information flow and gradient flow through latent DAG (including activations, residual pathways, bottlenecks).
- [x] Propose targeted architecture improvements ranked by expected impact and implementation risk.
- [x] Record findings in this review section with explicit verdicts.

## Review

- Overall: core latent-DAG wiring is mostly faithful, but there are high-impact mismatches in override routing, processing/MS isolation, and multi-allele architecture implementation depth.
- Design/implementation mismatches found:
  - `mhc_species` override is computed and surfaced in outputs but not used in `context_token` construction; downstream latent path still uses inferred per-chain species probs.
  - core-relative peptide embedding is injected into all non-recognition latents, which leaks MHC-informed core signals into `processing_*` and `ms_detectability` despite the documented isolation contract.
  - `design.md` token layout specifies explicit segment boundary tokens (`[NFLANK]`, `[CLEAVE_N]`, `[MHC_A]`, etc.), but runtime vocabulary/stream builder uses plain AA tokens + segment IDs without those markers.
  - `design.md` section 7.1 still states 11 latents, while code includes 12 (`species_of_origin` added).
  - `design.md` MIL section specifies competition-transformer + attention pooling over allele vectors, while current inference/training uses per-allele forward + external Noisy-OR aggregation.
  - `training_spec.md` lists core-start CE and binding-orthogonality regularization, but canonical loss wiring currently does not include explicit core-start or orthogonality losses.
- Information/gradient-flow assessment:
  - Strong: pre-norm residual latent cross-attention blocks are stable; strict segment-blocked base attention plus peptide-only recognition path prevents obvious shortcut leakage into recognition.
  - Risk: presentation output is dominated by additive logit path from processing/binding logits, so presentation-latent branch gets comparatively weak supervision signal.
  - Risk: MIL instance capping subsamples globally and can drop all instances for some bags, introducing noisy bag losses with no instance-level corrective gradient.
  - Risk: hard clamping of kinetic latents/logits improves numeric safety but can flatten gradients on extreme examples.
- Recommended improvement priorities:
  - P1: route `mhc_species` override into `context_token` and downstream species-conditioned paths.
  - P1: restrict core-relative injection to latents that are allowed to use MHC-informed core context (binding/presentation), excluding processing/MS latents.
  - P1: make MIL cap bag-aware (preserve at least one instance per bag) and consider soft attention MIL to reduce noisy-or saturation.
  - P2: add explicit core-start supervision when labels exist; add binding affinity/stability orthogonality penalty as documented.
  - P2: increase presentation-latent contribution via scheduled/gated residual scaling so presentation supervision meaningfully trains presentation latents.

# Align Presto Code to Design Specification

# Unresolved Alleles Taxonomy + Mouse MHC Coverage

## Spec

User request:

- distinguish unresolved-allele cases explicitly (serotypes, non-classical molecules, murine haplotypes/pairs, etc.), not as one opaque unresolved bucket;
- confirm/strengthen `mhcgnomes` usage for normalization/parsing;
- augment sequence coverage with mouse MHC alleles broadly (not only currently failing examples).

## Plan

- [x] Verify current unresolved-report path and define a deterministic unresolved-case taxonomy.
- [x] Implement unresolved-case classification in strict audit outputs (`unresolved_mhc_alleles.csv` and detail report) with category counts.
- [x] Expand index refresh/download path to include mouse IPD-MHC sequence source by default alongside existing IMGT/IPD inputs.
- [x] Add tests for new unresolved taxonomy fields and mouse-source resolution wiring.
- [x] Run focused tests for `mhc_index`, data CLI refresh, and strict unresolved training audit paths.
- [x] Document review outcomes in this section with concrete before/after behavior.

## Review

- Added `mhcgnomes`-backed unresolved taxonomy via `classify_unresolved_allele` in `data/mhc_index.py`.
  - Categories now distinguish cases like `murine_pair_shorthand`, `murine_haplotype`, `human_serotype`, `human_locus`, `human_nonclassical_gene`, and allele-level missing-sequence buckets.
- Strict unresolved reports now include classification metadata.
  - `unresolved_mhc_alleles.csv` columns now include category/type/normalized/species/class/gene/parse_error.
  - `unresolved_mhc_detail.csv` now includes per-row `category` with `(modality, source, allele, category, count)`.
  - Training logs now print unresolved counts by category.
- MHC index refresh path now prefers the `data/ipd_mhc/` directory when it contains FASTA payloads, so additional overlays (including mouse FASTA files) are included automatically during index build instead of being ignored due to single-file path selection.
- Focused verification:
  - `pytest -q tests/test_mhc_index.py tests/test_data_cli.py tests/test_train_iedb.py` -> `51 passed`.
  - `python -m py_compile data/mhc_index.py scripts/train_iedb.py cli/data.py tests/test_mhc_index.py tests/test_data_cli.py tests/test_train_iedb.py` passed.

---

# Mouse MHC Overlay From IMGT + UniProt (Provenance First)

## Spec

User requirement:

- If a dedicated mouse allele source is unavailable, use curated mouse MHC gene/allele nomenclature from specialized sources (IMGT/MGI-like) and fetch protein sequences from UniProt.
- Keep explicit per-protein provenance so wrong mappings can be traced and fixed later.

Design choices:

- Use IMGT mouse MHC nomenclature as machine-readable gene list source.
- Query UniProt REST for Mus musculus (`organism_id:10090`) per IMGT gene symbol.
- Keep reviewed/unreviewed candidates in a catalog; build FASTA overlay from reviewed entries with deterministic allele-token derivation.
- Emit explicit provenance columns per row (source URL/query/accession/derivation rule).

## Plan

- [x] Add a mouse MHC overlay builder module in `data/`:
  - [x] fetch and parse IMGT mouse MHC nomenclature page into canonical gene symbols/classes;
  - [x] query UniProt for each gene and collect candidate proteins;
  - [x] derive resolvable mouse allele tokens (e.g., `H2-K*b`, `H2-D*b`, `H2-AA*b`) from UniProt protein naming patterns;
  - [x] write provenance catalog CSV + FASTA overlay.
- [x] Add CLI entrypoint under `presto data mhc-index` to build/update the overlay into `data/ipd_mhc/`.
- [x] Add tests for parser/derivation/provenance schema and CLI wiring.
- [x] Update docs with "mouse overlay source provenance" guidance.
- [x] Run focused tests and basic command smoke run.

## Review

- Added `data/mouse_mhc_overlay.py` with a full provenance-first pipeline:
  - IMGT mouse nomenclature extraction (`Mu_MHCnom.html`) -> canonical H2 gene list.
  - per-gene UniProt REST queries (`organism_id:10090`, reviewed by default).
  - deterministic allele-token derivation from UniProt protein naming haplotype suffixes (e.g., `K-B` -> `H2-K*b`).
  - output catalog with explicit source columns and selected-allele FASTA overlay.
- Added CLI command:
  - `python -m presto data mhc-index mouse-overlay --datadir ./data`
  - implemented in `cli/data.py` + wired in `cli/main.py`.
- Added docs:
  - `README.md` section "Mouse MHC Overlay (IMGT + UniProt, Provenance Tracked)".
  - `docs/notes/mouse_mhc_overlay_sources.md` with source URLs + provenance schema.
  - `docs/index.md` and `docs/cli.md` links/usage updates.
- Added tests:
  - new `tests/test_mouse_mhc_overlay.py`.
  - extended `tests/test_data_cli.py` for parser + command success/error paths.
- Verification:
  - `pytest -q tests/test_data_cli.py tests/test_mouse_mhc_overlay.py tests/test_train_cli.py tests/test_mhc_index.py tests/test_train_iedb.py` -> `74 passed`.
  - `python -m py_compile data/mouse_mhc_overlay.py cli/data.py cli/main.py tests/test_data_cli.py tests/test_mouse_mhc_overlay.py` passed.
- Live smoke runs:
  - limited run (`--max-genes 8`): `catalog_rows=17`, `selected_alleles=17`.
  - full IMGT-derived run (`77` genes): `catalog_rows=29`, `selected_alleles=27`.
  - selected alleles include key unresolved targets: `H2-K*b`, `H2-K*k`, `H2-D*b`, `H2-AA*b`, `H2-AB*k`, etc.
  - refreshing index against overlay-only directory yields `mouse_h2: 27`.

## Status: COMPLETE

### Implementation Summary

All 9 phases completed successfully. 483/483 tests pass.

### Phases

- [x] Phase 1: Fix LATENT_SEGMENTS, LATENT_DEPS, extra-token injection (A, B, L, M)
- [x] Phase 2: Add ms_detectability latent (C)
- [x] Phase 3: Convert immunogenicity to MLP (E)
- [x] Phase 4: 2-layer cross-attention per latent (D)
- [x] Phase 5: Fix pmhc_vec computation (F)
- [x] Phase 6: Per-chain MHC inference (G, H, I)
- [x] Phase 7: Segment-specific positional encoding (K)
- [x] Phase 8: Global conditioning embedding (J)
- [x] Phase 9: Update tests and verify

### Files Modified

| File | Changes |
|------|---------|
| `models/presto.py` | All 8 architecture phases |
| `tests/test_presto.py` | 15 new design alignment tests, updated existing tests |

### Deferred

- T-cell assay head compositional upgrade (N): proc_gate, ambiguity_gate, pepfmt, duration
- Vocab minor expansions (O): species naming, T-cell context entries

### Verification

- All 483 tests pass
- 15 new design alignment tests verify each discrepancy fix
- Model forward pass produces all expected output keys

---

# Audit Docs Design Consistency

## Plan

- [x] Inventory design-related docs and extract explicit requirements/assumptions
- [x] Cross-compare requirements across docs to find contradictions or ambiguities
- [x] Assess whether the combined design is coherent as a system
- [x] Record findings and recommended reconciliations in a review section

## Review

Overall design is coherent at a high level (latent DAG, MIL, compositional assay heads), but there are several specification conflicts that should be resolved before strict code-vs-spec validation:

1. TCR:pMHC matching representation is inconsistent across docs (recognition latent vs presentation latent vs `pmhc_vec` anchor).
2. Recognition path is described as peptide-only, but rationale and MIL text also claim allele-specific recognition/context influence.
3. MHC beta chain type categories conflict (`class_II_beta` in inference section vs `class_II_alpha` implied by “same categories” in auxiliary heads table).
4. TCR implementation status is inconsistent (`TODO/not fully wired` vs multiple components marked implemented; contrastive loss listed as canonical in training spec but partial in TCR spec).
5. Missing/sentinel token vocabulary is underspecified/inconsistent (special-token inventory omits `[MISSING_*]` tokens used later).
6. `MINIMAL_EPITOPE` range conflicts (8-11 vs <=14 vs 8-14) within design doc.

---

# Canonicalize Design + Align Code/Tests + Verify Training

## Spec

User decisions to canonicalize:

1. TCR match head representation: use `pmhc_vec` as the pMHC anchor (single canonical path).
2. Recognition stays peptide-only for now (do not add solvent-accessibility modeling yet).
3. Chain-type naming: use class-specific beta naming (`class_I_beta` / `class_II_beta`) consistently.
4. TCR is a future feature: keep design details, but clearly state it is not yet working in canonical training/inference.
5. Missingness: use NLP/LLM-style explicit dedicated missing token (`<MISSING>`) rather than overloading `<UNK>`.
6. `MINIMAL_EPITOPE` length canonicalized to 8-15mer.

## Plan

- [x] Update `tasks/lessons.md` for this correction pattern
- [x] Canonicalize docs (`design.md`, `tcr_spec.md`, `training_spec.md`, `index.md`, `cli.md`) to match the six decisions
- [x] Align model/data code to docs:
  - [x] Add explicit `<MISSING>` token handling and route optional missing segments/chains through it
  - [x] Make TCR path explicitly non-canonical/future (disable active match path in model and training)
  - [x] Normalize chain-type naming in outputs/tests/docs references
  - [x] Apply peptide-format 8-15 minimal-epitope boundary
- [x] Update/repair tests to match canonical behavior and run focused tests first
- [x] Run full relevant tests for touched areas
- [x] Run local small training run and verify successful completion + metrics artifact generation
- [ ] Launch Modal larger run, collect metrics artifacts (`metrics.csv/jsonl`), and analyze:
  - [x] Per-output loss curves (from latest completed large Modal run artifacts)
  - [x] Per-output and per-latent variance trends over time (from latest completed large Modal run artifacts)
  - [x] Convergence assessment (from latest completed large Modal run artifacts)
- [x] Run sanity-check predictions with trained checkpoint:
  - [x] `SLLQHLIGL` + `HLA-A*02:01` (positive control)
  - [x] `SLLQHLIGL` + `HLA-A*24:02` (negative control)
- [x] Record results/review findings in this file

## Review

- Docs were canonicalized for the six user decisions and code/tests were aligned:
  - dedicated `<MISSING>` token now exists and is used for missing optional segments,
  - TCR path is explicitly future/inactive in canonical model/predict APIs,
  - chain-type compositional labels use `class_I_beta` / `class_II_beta`,
  - peptide minimal-epitope boundary is 8-15.
- Verification:
  - `pytest -q tests/test_presto.py tests/test_training_e2e.py tests/test_predictor.py tests/test_tokenizer.py` passed (`96 passed`).
  - `pytest -q tests/test_checkpointing.py` passed (`2 passed`).
- Local training smoke run succeeded:
  - command: unified training on `artifacts/local_runs/merged_small_multimodal_20260224.tsv` with explicit small caps and `1` epoch.
  - artifacts: `artifacts/local_runs/unified_small_20260224/{metrics.csv,metrics.jsonl,config.json,presto_small.pt}`.
  - headline: train loss `98.1489`, val loss `3.9776`, checkpoint saved.
- Modal larger-run status:
  - attempted multiple launches from current workspace code; blocked by repeated cold image rebuild/dependency install cycles in this environment before reaching reliable training completion.
  - therefore convergence analysis used the latest completed large Modal run artifacts already present:
    - `modal_runs/iedb-2k-10ep-20260216i/metrics.csv` and corresponding plots.
- Convergence assessment from completed large Modal run (`iedb-2k-10ep-20260216i`):
  - global loss decreased strongly and smoothly: train `35.0227 -> 4.3950`, val `6.3021 -> 0.7847` across epochs `1..10`.
  - per-task losses improved on train/val (`binding`, `elution`, `t_half`, `tm`); `tcell` improved modestly (`~0.70 -> ~0.64`), suggesting harder/noisier signal.
  - uncertainty log-variances stayed bounded (no divergence/explosion), indicating stable multi-task balancing.
  - overall: this completed run appears convergent without obvious instability.
- Sanity controls (`SLLQHLIGL`) using the newly trained canonical local checkpoint:
  - `HLA-A*02:01`: `presentation_prob=0.9489`, `binding_prob=0.0100`, `processing_prob=0.4867`.
  - `HLA-A*24:02`: `presentation_prob=0.9489`, `binding_prob=0.0100`, `processing_prob=0.4867`.
  - interpretation: no practical separation between the positive/negative control alleles in this small local model; expected control ordering is not yet learned in this smoke checkpoint.

---

# Debug Binding/Presentation Consistency + Allele Conditioning

## Spec

User reported:
- high `presentation_prob` with low `binding_prob` and `processing_prob`,
- identical `binding_prob` across `HLA-A*02:01` and `HLA-A*24:02`,
- suspicion that inference ignores MHC.

## Plan

- [x] Reproduce and isolate the two failure modes from current codepath
  - [x] unresolved allele fallback tokenization path (`HLA-A*02:01` -> AA tokenizer)
  - [x] weak-affinity KD saturation path that collapses `binding_prob`
- [x] Patch model/inference behavior
  - [x] Prevent silent raw-allele token fallback in predictor for unresolved allele names
  - [x] Fix KD upper-cap handling so weak binders do not collapse to one constant log10(KD)
  - [x] Add explicit messaging/guardrails for unresolved allele lookup
- [x] Add/adjust regression tests covering:
  - [x] unresolved allele behavior
  - [x] KD saturation regression
  - [x] A*02:01 vs A*24:02 path using resolved MHC sequences
- [x] Run focused tests and a quick inference sanity check
- [x] Record final findings and remaining model-training limitations in review

## Review

- Root causes confirmed:
  - unresolved allele fallback: `Predictor._get_mhc_sequence` previously tokenized raw allele strings as AA text when lookup was missing.
  - KD saturation: `Presto.forward` hard-clamped KD at max before smooth capping, collapsing weak binders to one constant (`~4.30685` log10 nM).
- Code changes:
  - `inference/predictor.py`: added strict allele-resolution guardrails, optional index-backed lookup (`index_csv`), default index autoload support, and unresolved-allele error path.
  - `models/presto.py`: removed hard KD upper clamp before `derive_affinity_observables`; retained smooth upper bound behavior.
- Regression tests added:
  - `tests/test_predictor.py`: strict unresolved-allele raises, non-strict fallback path.
  - `tests/test_presto.py`: KD soft-cap regression verifies weak-affinity values are no longer collapsed by hard pre-clamp.
- Verification:
  - `pytest -q tests/test_predictor.py tests/test_presto.py` -> `77 passed`.
- Sanity check on local checkpoint (`artifacts/local_runs/unified_small_20260224/presto_small.pt`) after fixes:
  - `SLLQHLIGL + HLA-A*02:01`: `binding_prob=0.001615809`, `processing_prob=0.486672997`, `presentation_prob=0.625911176`.
  - `SLLQHLIGL + HLA-A*24:02`: `binding_prob=0.001615828`, `processing_prob=0.486671925`, `presentation_prob=0.625781000`.
  - unresolved allele in strict mode now errors with actionable message.
- Remaining limitation:
  - A*02:01 vs A*24:02 separation is now non-identical but still tiny in this small local checkpoint, indicating limited allele discrimination from the smoke-trained model rather than pure inference-path collapse.

---

# Class Conditioning Refactor (Default Sequence-Inferred pI/pII)

## Spec

User requirement:
- do not persist an implicit `"unknown"` MHC-class embedding through the network by default;
- default behavior should infer class from MHC chain sequences and use inferred `pI/pII` to mix class-specific presentation terms;
- keep user override simple (`mhc_class` = `I`/`II`) and keep soft `pI/pII` internal only.

## Plan

- [x] Refactor model class-conditioning path
  - [x] Remove implicit `"unknown"` class embedding injection when class is absent
  - [x] Keep explicit class embedding only when class is explicitly provided
- [x] Refactor predictor default behavior
  - [x] Stop forcing string class override into model when user does not explicitly provide class
  - [x] Keep class inference for class-I beta2m assembly only
  - [x] Keep user-facing override as hard class only (`I`/`II`)
- [x] Keep soft class probabilities internal
  - [x] remove user-facing soft `pI/pII` inputs from predictor/CLI
  - [x] keep internal inferred `mhc_class_probs` for class mixing/reporting
- [x] Add regression tests
  - [x] predictor default path uses inferred class probs (no hard class override)
  - [x] predictor hard class override remains supported
  - [x] model path no longer injects unknown class embedding by default
- [x] Run focused tests and record review notes

## Review

- Implemented behavior now matches the simplified requirement:
  - default inference does **not** force class labels from caller/allele names;
  - soft class uncertainty is inferred internally from chain sequences and used in class mixing;
  - user override remains simple and disjoint (`mhc_class="I"` or `"II"`).
- Code changes:
- `models/presto.py`: `_build_single_stream` no longer injects `"unknown"` class embedding when class is absent; it uses zero class-conditioning unless explicit class is provided.
  - `inference/predictor.py`: default path passes `mhc_class=None` to model (class inferred internally), while still using inferred class only for beta2m assembly fallback.
  - `cli/main.py` + `cli/predict.py`: removed user-facing soft class-probability options.
- Regression tests:
  - `tests/test_predictor.py`: validates default model-class inference path and hard class override path.
  - `tests/test_presto.py`: validates no implicit unknown-class embedding when class isn’t provided.
- Verification:
  - `pytest -q tests/test_predictor.py tests/test_presto.py tests/test_predict_cli.py` -> `84 passed`.
- Sanity check on local checkpoint (`SLLQHLIGL`, `HLA-A*02:01`):
  - default inferred class path: `binding_prob=0.001619254`, `processing_prob=0.320803523`, `presentation_prob=0.000294732`.
  - forced class-I path: `binding_prob=0.001615809`, `processing_prob=0.486672997`, `presentation_prob=0.625911176`.
  - this confirms the previous high presentation issue was dominated by hard class forcing, not by inferred-class default.
- Note: this intermediate state was later simplified further by fully removing token-level class embedding in the next section.

---

# Remove Token-Level Class Embedding (Use pI/pII Only)

## Spec

User preference:
- remove global/token-level MHC class embedding entirely;
- infer class uncertainty (`pI/pII`) from chain sequences internally;
- keep optional user class override simple/hard (`I` or `II`);
- class influence should come via class-probability mixing/gating only.

## Plan

- [x] Remove `mhc_class_cond_embed` usage from model stream construction
- [x] Remove `mhc_class_id` plumbing used only for token-level class conditioning
- [x] Update tests/docs that assume presence of class embedding
- [x] Run focused tests (`test_presto`, `test_predictor`, `test_predict_cli`)

## Review

- `models/presto.py`:
  - removed `mhc_class_cond_embed` and all `mhc_class_id` stream-plumbing;
  - global token conditioning now uses species + chain-completeness only;
  - `mhc_class` continues to affect only downstream class-probability override/mixing;
  - added `_load_from_state_dict` shim to drop legacy `mhc_class_cond_embed.weight` for old checkpoint compatibility.
- `tests/test_presto.py`:
  - updated global-conditioning expectations (no class embedding table);
  - replaced unknown-class embedding test with signature-level guard (`_build_single_stream` has no `mhc_class_id`).
- `docs/design.md`:
  - removed class from global conditioning embedding description/code block;
  - clarified `mhc_class` is optional hard override for downstream class-specific mixing, while default class is inferred from chains.
- Verification:
  - `pytest -q tests/test_presto.py tests/test_predictor.py tests/test_predict_cli.py` -> `84 passed`.

---

# Retrain + Re-eval SLLQHLIGL Controls

## Spec

User request:
- retrain after latest class-conditioning simplification;
- re-run sanity checks for `SLLQHLIGL` on:
  - `HLA-A*02:01` (positive control),
  - `HLA-A*24:02` (negative control).

## Plan

- [x] Launch a fresh local unified training run using the same small bounded dataset/config used previously
- [x] Run presentation predictions for both control alleles from the new checkpoint
- [x] Record metrics and control outputs in review

## Review

- Local retrain command (5 epochs, bounded dataset) completed:
  - checkpoint: `artifacts/local_runs/unified_small_20260226_retrain/presto_small.pt`
  - run dir: `artifacts/local_runs/unified_small_20260226_retrain`
  - best validation loss: `3.6451817750930786`
  - epoch losses:
    - epoch 1: train `107.0320`, val `4.0675`
    - epoch 2: train `90.1276`, val `5.2392`
    - epoch 3: train `70.2243`, val `5.6483`
    - epoch 4: train `54.0141`, val `4.7413`
    - epoch 5: train `60.8371`, val `3.6452`
- Re-eval on `SLLQHLIGL` (default class inference, index-backed allele resolution):
  - `HLA-A*02:01`:
    - `processing_prob=0.0394527465`
    - `binding_prob=0.0079197282`
    - `presentation_prob=0.0043406878`
  - `HLA-A*24:02`:
    - `processing_prob=0.0390903875`
    - `binding_prob=0.0079237005`
    - `presentation_prob=0.0042436179`
- Outcome:
  - both controls now have low presentation probability (no longer the earlier high-presentation inconsistency),
  - allele separation remains very small in this bounded local run.

---

# Track A0201/SLLQHLIGL Affinity During Training

## Spec

User question:
- is low predicted affinity due to MHC sequence not being used well?
- does `HLA-A*02:01` + `SLLQHLIGL` affinity improve over training?
- track this over mini-batches/epochs.

## Plan

- [x] Run an instrumented local training trace that logs probe predictions after each mini-batch
- [x] Capture per-epoch train/val loss alongside probe affinity trajectory
- [x] Compare probe trajectory and identify whether affinity improves over time

## Review

- Generated probe artifacts:
  - `artifacts/local_runs/unified_small_20260226_probe/a0201_sllqhligl_trace.csv`
  - `artifacts/local_runs/unified_small_20260226_probe/epoch_losses.csv`
- Setup matched bounded local unified config (same caps; includes 10x records when present).
- Probe trajectory (`SLLQHLIGL` + `HLA-A*02:01`):
  - init: `KD_log10=4.9551` (`~90.2 uM`), `binding_prob=0.001584`
  - epoch 1 end: `4.9465` (`~88.4 uM`), `0.001624`
  - epoch 2 end: `4.8619` (`~72.8 uM`), `0.002066`
  - epoch 3 end: `4.5336` (`~34.2 uM`), `0.005262`
  - epoch 4 end: `4.1123` (`~13.0 uM`), `0.017325` (best in trace)
  - epoch 5 end: `4.3851` (`~24.3 uM`), `0.008023`
- Conclusion from trace:
  - affinity prediction does improve substantially through epoch 4, then regresses by epoch 5;
  - so this probe is train-sensitive, but optimization on the small mixed-task setup is not monotonic for this specific pair.
- Data coverage note:
  - bounded small merged training file used in local runs contains no `SLLQHLIGL` rows,
  - full merged corpus does contain multiple positive `SLLQHLIGL + HLA-A*02:01` elution entries.

---

# Built-In Probe Tracking + 10-Epoch Large Run

## Spec

User request:
- make training track `SLLQHLIGL` against `HLA-A*02:01` and `HLA-A*24:02`;
- produce affinity-over-epoch plot;
- run a larger 10-epoch training run with current architecture (no class embedding) and show behavior.

## Plan

- [x] Add probe tracking to unified training loop with per-epoch metrics logging
- [x] Emit probe artifacts (`probe_affinity_over_epochs.csv/.json/.png`) in run-dir
- [x] Wire CLI flags/defaults for probe tracking and controls
- [x] Run parser/unit checks for touched CLI/training config paths
- [x] Launch 10-epoch larger run and collect probe trajectories
- [ ] Summarize A*02:01 vs A*24:02 affinity behavior over epochs

## Review

- Code changes:
  - `scripts/train_iedb.py` now computes fixed probe metrics each epoch and logs them under split `probe`.
  - default probe config is `SLLQHLIGL` vs `HLA-A*02:01,HLA-A*24:02`.
  - per-run artifacts now include:
    - `probe_affinity_over_epochs.csv`
    - `probe_affinity_over_epochs.json`
    - `probe_affinity_over_epochs.png` (when matplotlib is available)
  - `cli/main.py` + `scripts/train_iedb.py` expose probe flags:
    - `--track-probe-affinity` / `--no-track-probe-affinity`
    - `--probe-peptide`
    - `--probe-alleles`
    - `--probe-plot-file`
  - packaging fix: `pyproject.toml` now includes `tqdm` runtime dependency for Modal/clean installs.
- Validation:
  - `pytest -q tests/test_train_cli.py` -> `12 passed`
  - `pytest -q` targeted `test_train_iedb` arg-resolution tests -> `4 passed`
  - local probe-smoke run produced expected artifacts in `artifacts/local_runs/probe_smoke_20260226/`.
- Larger-run execution status:
  - Modal large canonical dataset (`/data/merged_canonical_large_20260224a.tsv`, ~146k rows) starts correctly with probe tracking.
  - full 10-epoch default settings are multi-hour in this environment (thousands of batches/epoch with synthetic augmentation), so final 10-epoch trajectory capture is still pending.

### 2026-02-26 Active 10-Epoch Full Run (No Class Embedding)

- Active run details:
  - Modal app: `ap-JrMONfejUNu35iff9AeRja`
  - run-id: `unified-full10ep-probe-fast64-20260226d`
  - checkpoint: `presto_unified_full10ep_probe_fast64.pt`
  - dataset: `/data/merged_canonical_large_20260224a.tsv` (146,110 merged rows scanned)
  - index: `/data/mhc_index.csv`
  - probe tracking: `SLLQHLIGL` vs `HLA-A*02:01,HLA-A*24:02`
  - synthetic augmentation: disabled (all synthetic ratios set to 0)
  - model/runtime profile: `d_model=64, n_layers=1, n_heads=4, batch_size=1024`
- Observed runtime from startup logs:
  - `Total samples: 156819`
  - `Train batches: 123`, `Val batches: 31`
  - early throughput ~`145` samples/sec, implying roughly multi-hour wall-clock for full 10 epochs.

---

# Modal 20M Sweep + Live Monitoring

## Spec

User request:
- run the 20M sweep on Modal;
- provide richer real-time monitoring, ideally visible on modal.com / app logs;
- clarify whether Modal has a W&B-like equivalent.

## Plan

- [x] Extend Modal training launcher with a native 20M sweep function
- [x] Add structured per-epoch metric logs emitted during remote training
- [x] Launch a live Modal sweep smoke run and verify structured logging appears
- [x] Launch a second sweep run with safer memory settings to compute real drop metrics
- [x] Record run IDs/artifact paths and monitoring workflow

## Review

- Implemented in `scripts/train_modal.py`:
  - Added `sweep_20m_runs(...)` Modal function to run near-20M candidate sweeps directly on Modal.
  - Added structured epoch log emitter in `_run_command(...)` that parses `metrics.csv` and prints:
    - `MODAL_METRIC {...}` with `epoch`, `train_loss`, `val_loss`, and selected probe metrics.
  - Added sweep summary artifact writers:
    - `/checkpoints/<sweep_id>/summary.json`
    - `/checkpoints/<sweep_id>/summary.csv`
- Validation:
  - `python -m py_compile scripts/train_modal.py scripts/sweep_20m_models.py`
  - `pytest -q tests/test_sweep_20m_models.py` -> `3 passed`
- Modal sweep runs:
  - Run 1 app: `ap-U8lo52okmEITSa9LysAXFt`
    - sweep id: `sweep20m-live-20260226-20260226T171152Z`
    - candidate `d224_l4_h8_p20388418` failed OOM at `batch_size=256`
    - candidate `d256_l2_h8_p21826466` completed (1 epoch), and emitted `MODAL_METRIC {...}`.
    - downloaded summary artifacts to:
      - `modal_runs/sweep20m-live-20260226-20260226T171152Z/summary.json`
      - `modal_runs/sweep20m-live-20260226-20260226T171152Z/summary.csv`
  - Run 2 app: `ap-zrOed9A4fuloeoxEsdDK1J`
    - sweep id: `sweep20m-live2-20260226-20260226T171416Z`
    - rerun with `batch_size=128`, `epochs=2` to avoid OOM and enable non-NaN drop-rate ranking.
    - confirmed live `MODAL_METRIC {...}` lines appear after epoch boundaries.

---

# 20M Hyperparameter Sweep Script (Loss-Drop Speed)

## Spec

User request:
- write a script that creates ~20M-parameter model configurations across different hyperparameters;
- train each configuration for short runs;
- visualize which configurations show the fastest loss decrease.

## Plan

- [x] Add `scripts/sweep_20m_models.py` with:
  - parameter-count based candidate generation around a target param budget;
  - subprocess training launcher for `presto train unified`;
  - metrics aggregation from `metrics.csv` per run;
  - ranking by loss-drop speed and plot generation.
- [x] Add outputs:
  - per-run raw metadata (`runs.json`);
  - summarized ranking table (`summary.csv`);
  - plots (`val_loss_curves.png`, `loss_drop_speed.png`, `params_vs_speed.png`).
- [x] Add a focused unit test for:
  - candidate selection near target params;
  - loss-speed summarization behavior.
- [x] Run targeted tests and record results.

## Review

- Added `scripts/sweep_20m_models.py`:
  - discovers candidate architectures in a parameter band around `--target-params` (default 20M);
  - computes trainable parameter counts directly from `Presto`;
  - launches per-candidate `presto train unified` runs with configurable shared training args;
  - parses `metrics.csv` and computes per-run speed summaries (`drop_per_epoch`, linear `slope`, `speed`);
  - ranks runs by selected metric (`--ranking-metric`, default `val_drop_per_epoch`);
  - writes sweep artifacts:
    - `summary.csv`
    - `runs.json`
    - `val_loss_curves.png`, `loss_drop_speed.png`, `params_vs_speed.png` (when matplotlib + successful runs).
- Added tests in `tests/test_sweep_20m_models.py`:
  - candidate generation respects param band and target-distance ordering;
  - loss-series summarization reports expected drop/slope/speed;
  - empty-series summary returns NaN metrics safely.
- Verification:
  - `pytest -q tests/test_sweep_20m_models.py` -> `3 passed`
  - dry-run smoke:
    - `python -m presto.scripts.sweep_20m_models --dry-run --max-candidates 3 --layer-min 3 --layer-max 4 --d-models 192,224,256 --heads 4,8,16 --sweep-dir artifacts/local_sweeps/sweep20m_dryrun_test`
    - script produced expected planning outputs and artifact files (`summary.csv`, `runs.json`).

---

# 10x Naming Canonicalization (User Correction)

## Spec

User request:
- always refer to this modality as `10x` and never `tenx` in user-facing surfaces.

Scope:
- CLI args/help text and config keys used by active training paths;
- dataset sample IDs / source tags that can appear in logs and metrics;
- tests that assert old naming;
- remove legacy `tenx` terminology from active source paths.

## Plan

- [x] Update argument/config resolution to canonical `10x` names.
- [x] Update loader sample metadata IDs/tags from `tenx_*` to `10x_*` where user-visible.
- [x] Update tests to assert canonical `10x` naming.
- [x] Run targeted tests for CLI, loader, and training arg resolution.
- [x] Add review notes and caveats.

## Review

- Canonicalized unified training flags and config keys to:
  - `--10x-file` / `sc10x_file`
  - `--max-10x` / `max_10x`
- Updated both parser entry points:
  - `scripts/train_iedb.py`
  - `cli/main.py`
- Updated loader-facing naming:
  - record type renamed to `Sc10xVDJRecord`
  - dataset argument renamed to `sc10x_records`
  - sample IDs now use `10x_*` prefixes
- Updated config + tests:
  - `conf/train/default.yaml` now uses `max_10x`
  - tests in `test_train_cli.py`, `test_train_iedb.py`, `test_loaders.py`, `test_downloaders.py` updated to canonical names
- Validation:
  - `pytest -q tests/test_train_cli.py tests/test_train_iedb.py tests/test_loaders.py tests/test_downloaders.py` -> `88 passed`

---

# Sampler Reweight + Synthetic Negative Audit

## Spec

User request:
- verify whether binding KD entries are truly scarce in source/merged data;
- avoid overweighting rare assay tasks in balanced batches;
- prefer proportional per-task batch composition with a minimum of one sample per task;
- confirm synthetic negatives are always on and enumerate all negative types currently generated.

## Plan

- [x] Measure KD counts in merged source data (full + capped training settings).
- [x] Update `BalancedMiniBatchSampler` task quota policy to proportional-with-min-one.
- [x] Add/adjust tests to lock in new quota behavior.
- [x] Run targeted loader/training parser tests.
- [x] Summarize synthetic-negative generation modes and sources.

## Review

- KD scarcity confirmed from merged corpus (`modal_runs/merged_canonical_large_20260224a.tsv`):
  - full load (`max_* = 0`): `binding=60,000`, with strict `"dissociation constant kd"` only `34` records.
  - capped run profile (`max_binding=4000,...`): strict `"dissociation constant kd"` only `3` records.
  - after train/val split under this capped run, train observed `binding_kd=2` (consistent with prior logs).
- Updated `BalancedMiniBatchSampler` to use proportional task quotas with minimum 1 per task per batch:
  - old behavior: near-equal quotas by task when `n_tasks <= batch_size`;
  - new behavior: `min 1` per task + remaining slots allocated by task prevalence.
- Added a new loader test to lock quota behavior:
  - rare task gets one slot while dominant task gets remaining slots in a batch.
- Synthetic negatives are now left enabled by default for 20M Modal sweeps:
  - `scripts/train_modal.py::sweep_20m_runs(... synthetic_negatives=True)`.
- Renamed sweep synthetic toggle to positive form:
  - `synthetic_negatives` (default true) replaces `disable_synthetic_negatives`.
  - compatibility bridge kept for existing callers in Modal/sweep scripts.
- Binding synthetic mode labels now use explicit names:
  - `peptide_scramble`
  - `peptide_random`
  - `mhc_scramble`
  - `mhc_random`
  - `no_mhc_alpha`
  - `no_mhc_beta`
- Processing synthetic negatives now use context-corruption modes (instead of pure random-only):
  - `flank_shuffle`
  - `peptide_scramble`
- Validation:
  - `pytest -q tests/test_loaders.py tests/test_train_iedb.py tests/test_train_cli.py tests/test_downloaders.py` -> `88 passed`.
  - `pytest -q tests/test_sweep_20m_models.py tests/test_train_iedb.py tests/test_loaders.py tests/test_train_cli.py tests/test_downloaders.py` -> `92 passed`.

---

# Synthetic Negative Naming/Mode Clarity Cleanup

## Spec

User request:
- avoid negative-form control naming like `disable_synthetic_negatives`;
- use clear, explicit synthetic-negative mode labels and semantics;
- avoid cryptic abbreviations (`rp_mr`) and clarify `random` vs `scramble`;
- keep processing negatives biologically aligned with processing (not just blanket no-binding constructs).

## Plan

- [x] Replace remaining public `no_b2m` synthetic-negative option naming with canonical `no_mhc_beta` terminology while preserving hidden backward-compatible aliases.
- [x] Keep `synthetic_negatives` as the public positive-form sweep toggle; remove public parser exposure of legacy disable/enable aliases.
- [x] Update docs to enumerate explicit binding/elution/processing synthetic modes and random-vs-scramble semantics.
- [x] Update tests for renamed function parameters/options where needed.
- [x] Run focused tests for synthetic augmentation and CLI parser wiring.

## Review

- Updated canonical public option naming:
  - `--synthetic-class-i-no-mhc-beta-negative-ratio` is now the primary CLI/config key.
  - legacy `--synthetic-class-i-no-b2m-negative-ratio` remains as hidden alias for compatibility.
- Updated sweep controls:
  - public toggle remains positive form (`--synthetic-negatives` / `--no-synthetic-negatives`);
  - removed public exposure of old disable/enable alias flags from sweep CLI;
  - Modal sweep internals still accept legacy `disable_synthetic_negatives` only as compatibility keyword.
- Updated docs (`docs/training_spec.md`) to explicit mode names and semantics:
  - binding: `peptide_scramble`, `peptide_random`, `mhc_scramble`, `mhc_random`, `no_mhc_alpha`, `no_mhc_beta`;
  - elution: `peptide_random_mhc_real`, `peptide_real_mhc_random`, `peptide_random_mhc_random`;
  - processing: `flank_shuffle`, `peptide_scramble`;
  - clarified `random` (de novo generation/sampling) vs `scramble` (permutation).
- Validation:
  - `pytest -q tests/test_train_iedb.py tests/test_loaders.py tests/test_train_cli.py tests/test_sweep_20m_models.py` -> `60 passed`.

---

# Remove Legacy Synthetic-Negative Compatibility Paths

## Spec

User correction:
- remove legacy compatibility code instead of keeping hidden aliases/keyword bridges;
- canonical names should be the only supported path for synthetic-negative controls.

## Plan

- [x] Remove legacy keyword bridge `disable_synthetic_negatives` from Modal sweep entrypoint.
- [x] Remove legacy CLI/config aliases for class-I no-beta synthetic ratio (`*_no_b2m_*`) and keep only `*_no_mhc_beta_*`.
- [x] Remove backward-compatible synthetic augmentation stats alias key `no_b2m`; keep only `no_mhc_beta`.
- [x] Update/replace tests that currently assert legacy alias behavior.
- [x] Run focused tests for train CLI + synthetic augmentation + loaders/sweep.

## Review

- Removed compatibility-only sweep keyword handling:
  - deleted `disable_synthetic_negatives` bridge code in `scripts/train_modal.py::sweep_20m_runs`.
- Removed synthetic no-beta legacy aliases:
  - deleted hidden CLI alias `--synthetic-class-i-no-b2m-negative-ratio` from `scripts/train_iedb.py` and `cli/main.py`;
  - deleted config-key migration from `synthetic_class_i_no_b2m_negative_ratio` to `synthetic_class_i_no_mhc_beta_negative_ratio` in `scripts/train_iedb.py`.
- Removed backward-compatible binding augmentation stats alias:
  - `augment_binding_records_with_synthetic_negatives` now emits only `no_mhc_beta` (no `no_b2m` key).
- Tests updated:
  - removed assertions depending on `stats[\"no_b2m\"]` in `tests/test_train_iedb.py`;
  - removed legacy alias parser test in `tests/test_train_cli.py`.
- Verification:
  - `pytest -q tests/test_train_iedb.py tests/test_loaders.py tests/test_train_cli.py tests/test_sweep_20m_models.py` -> `59 passed`.
  - `python -m py_compile scripts/train_modal.py scripts/train_iedb.py cli/main.py tests/test_train_iedb.py tests/test_train_cli.py` passed.

---

# Synthetic-Negative Visibility + 20M Modal Sweep Refresh

## Spec

User request:
- confirm all synthetic-negative categories are enabled;
- add an easy-to-find table enumerating all synthetic-negative categories/modes;
- rerun Modal 20M parameter sweep to compare convergence speed;
- provide A*02:01 vs A*24:02 (`SLLQHLIGL`) affinity-trajectory plots for best and worst hyperparameter runs.

## Plan

- [x] Add visible synthetic-negative table to `README.md` (with category, modes, controls/defaults, supervision target).
- [x] Add parser-default test that synthetic-negative category ratios are all enabled (>0) in unified training defaults.
- [x] Run focused tests for touched docs/tests and training parser.
- [x] Launch fresh Modal sweep (`sweep_20m_runs`) with explicit synthetic-negative ratios and probe tracking enabled.
- [x] Fetch sweep artifacts from Modal volume, identify best/worst by ranking metric, and collect paths to `probe_affinity_over_epochs` plots.
- [x] Summarize convergence ranking and report best/worst A*02:01 vs A*24:02 probe behavior vs expected KD ranges.

## Review

- Added an easy-to-find synthetic-negative table to `README.md` with:
  - all categories (`binding`, `elution`, `processing`, `cascade`),
  - explicit mode names,
  - default unified flags/ratios,
  - and random-vs-scramble semantics.
- Added parser default guard in `tests/test_train_cli.py`:
  - verifies all synthetic-negative category ratios are enabled (`> 0`) by default.
- Focused verification:
  - `pytest -q tests/test_train_cli.py tests/test_train_iedb.py tests/test_loaders.py tests/test_sweep_20m_models.py` -> `59 passed`.
- Modal sweep rerun:
  - app: `ap-O0EDRf4QPUGaMD80T1Aws6`
  - sweep dir: `sweep20m-snfull-bs64-20260226-20260226T193011Z`
  - explicit synthetic-negative args passed for all categories:
    - binding/no-mhc-beta/elution/processing/cascade-elution/cascade-tcell all non-zero.
  - initial attempt with `batch_size=256` failed OOM; rerun at `batch_size=64` completed.
- Convergence ranking (`val_drop_per_epoch`, successful runs):
  - best: `d224_l5_h4_p21598466` (`0.04573`)
  - 2nd: `d224_l4_h8_p20388418` (`0.03116`)
  - 3rd: `d224_l5_h8_p21598466` (`0.00881`)
  - worst successful: `d224_l4_h4_p20388418` (`0.00288`)
  - failed: `d224_l4_h16_p20388418`, `d224_l5_h16_p21598466` (CUDA OOM).
- Best/worst probe affinity artifacts (local):
  - sweep summary:
    - `modal_runs/sweep20m-snfull-bs64-20260226-20260226T193011Z/summary.csv`
    - `modal_runs/sweep20m-snfull-bs64-20260226-20260226T193011Z/summary.json`
  - raw probe trajectories:
    - best: `04_d224_l5_h4_p21598466_probe_affinity_over_epochs.csv`
    - worst: `01_d224_l4_h4_p20388418_probe_affinity_over_epochs.csv`
  - generated plots:
    - `analysis/convergence_ranking_val_drop_per_epoch.png`
    - `analysis/best_d224_l5_h4_probe_affinity.png`
    - `analysis/worst_d224_l4_h4_probe_affinity.png`
- Observed probe KD behavior vs expectation:
  - expected: A*02:01 ~8-10 nM; A*24:02 >10 uM (10-25 uM acceptable).
  - observed (best/worst runs): both alleles remain ~6.5-19 uM and very close to each other (small/no separation).
  - interpretation: current 20M sweep settings improve loss speed differences, but do not yet achieve the desired allele-discriminative affinity calibration for this control pair.

---

# Synthetic-Negative Simplification + GPU Memory Sizing

## Spec

User request:
- simplify synthetic-negative configuration so users do not need to reason about separate binding/elution/processing/cascade knobs for non-binding consequences;
- keep processing negatives explicit (processing can be independent of pMHC binding);
- quantify practical GPU capacity in terms of maximum batch rows and approximate memory per row for the 20M architecture sweep context;
- clarify how `max-*` dataset caps relate to GPU memory and epoch size.

Design decisions for this implementation:
- expose one primary synthetic-negative rate knob for non-binding pMHC negatives and downstream propagation;
- keep one explicit processing-negative knob;
- preserve existing synthetic negative mode generation, but derive downstream negative synthesis rates from the primary pMHC knob so users do not set multiple downstream ratios;
- update docs/CLI help/examples to reflect simplified controls and remove redundant user-facing flags.

## Plan

- [x] Introduce simplified synthetic-negative CLI/API surface in `scripts/train_iedb.py` and `cli/main.py`:
  - [x] add canonical `--synthetic-pmhc-negative-ratio` (primary);
  - [x] keep explicit `--synthetic-processing-negative-ratio`;
  - [x] remove user-facing independent `--synthetic-elution-negative-ratio`, `--synthetic-cascade-elution-negative-ratio`, `--synthetic-cascade-tcell-negative-ratio` knobs and compute them from the primary ratio.
- [x] Update internal augmentation wiring to derive downstream negative generation from `synthetic_pmhc_negative_ratio` and document mapping in logs.
- [x] Update tests (`tests/test_train_cli.py`, `tests/test_train_iedb.py`, `tests/test_sweep_20m_models.py`) to the simplified knobs and default expectations.
- [x] Update documentation (`README.md`, `docs/training_spec.md`) with the simplified synthetic-negative specification and explicit category table.
- [x] Run targeted unit tests for modified parser/augmentation/sweep code.
- [x] Run a GPU memory probe on Modal (20M model) sweeping batch size until OOM and record:
  - [x] largest stable batch size;
  - [x] OOM threshold;
  - [x] peak allocated/reserved memory and approximate bytes-per-row.
- [x] Summarize results:
  - [x] explain why full dataset size does not need to fit in GPU memory;
  - [x] provide concrete guidance for `max-*` caps and full-corpus epoch training.

## Review

- Simplified synthetic-negative interface implemented:
  - new primary flag `--synthetic-pmhc-negative-ratio` replaces separate user-facing binding/elution/cascade ratio flags;
  - `--synthetic-processing-negative-ratio` remains explicit;
  - downstream elution/cascade synthetic rates are derived internally (`0.5x` each) from pMHC ratio;
  - class-I no-beta negatives remain explicit via `--synthetic-class-i-no-mhc-beta-negative-ratio`.
- Updated callsites and defaults:
  - parsers: `scripts/train_iedb.py`, `cli/main.py`;
  - sweep launchers: `scripts/train_modal.py`, `scripts/sweep_20m_models.py`;
  - config: `conf/train/default.yaml`;
  - docs: `README.md`, `docs/training_spec.md`.
- Added runtime GPU memory instrumentation in `scripts/train_iedb.py`:
  - per-epoch logs now report `gpu_peak_allocated_gib`, `gpu_peak_reserved_gib`, and `gpu_peak_allocated_bytes_per_batch_row`.
- Modal memory probe results (`d_model=224, n_layers=5, n_heads=4`, canary profile, `/data/merged_canonical_large_20260224a.tsv`):
  - successful: batch sizes `64, 96, 112, 128, 160, 176`;
  - OOM: batch sizes `180, 192`;
  - largest stable observed batch size: `176`;
  - average allocated memory per batch row across successful runs: `~221.86 MiB/row`;
  - GPU capacity inferred from OOM traces: `~39.49 GiB` total.
- Data-size context:
  - local canonical merged corpus file: `data/merged_deduped.tsv` has `5,593,162` lines (~`5,593,161` data rows);
  - probe subset file used in Modal memory sweep: `modal_runs/merged_canonical_large_20260224a.tsv` has `146,111` lines.
- Verification:
  - `pytest -q tests/test_train_cli.py tests/test_train_iedb.py tests/test_sweep_20m_models.py` -> `41 passed`.
  - `python -m py_compile scripts/train_iedb.py cli/main.py scripts/train_modal.py scripts/sweep_20m_models.py` passed.

---

# Full-Data Throughput Bottleneck Profiling + Training Speedup

## Spec

User request:
- add performance logging that identifies where full-data training time is spent;
- use that information to speed up training.

Observed issue:
- full-data Modal run (`ap-c1W5Ig8zeYUfeE0b81fthN`) is using CUDA but progresses at ~394 sec/batch.

## Plan

- [x] Add epoch-level performance instrumentation in training loop:
  - [x] dataloader wait time per batch;
  - [x] compute-loss/forward time breakdown (main forward, MIL forward/loss, regularization);
  - [x] backward + optimizer time;
  - [x] derived percentages and per-batch averages.
- [x] Optimize large-dataset balanced sampling path:
  - [x] avoid scanning million-scale candidate pools per draw;
  - [x] sample bounded candidate subsets for weighted/batch-balanced selection.
- [x] Expose/enable DataLoader throughput controls for unified training:
  - [x] `--num-workers`;
  - [x] `--pin-memory` / `--no-pin-memory`.
- [x] Update parser/default wiring in both CLI entrypoints.
- [x] Add/adjust focused tests for parser + sampler behavior.
- [x] Run focused tests and static checks.
- [x] Relaunch full-data run with probe tracking enabled and collect first performance metrics snapshot.
- [x] Add real-time in-epoch perf logging (rolling window) so bottleneck can be monitored without waiting for epoch end.
- [x] Add non-blocking batch transfer path (`pin_memory` + `tensor.to(..., non_blocking=True)`) and benchmark impact.
- [x] Re-run short Modal GPU probe and record throughput + bottleneck readout.
- [x] Record review summary and recommended run settings.

## Review

- Code changes completed:
  - `scripts/train_synthetic.py`: added rolling perf window logs (`perf_log_interval_batches`), non-blocking batch transfer plumbing, and per-window inner compute breakdown (forward/MIL/regularization).
  - `data/collate.py`: `PrestoBatch.to(device, non_blocking=...)` support.
  - `scripts/train_iedb.py` + `cli/main.py` + `conf/train/default.yaml`: new `--perf-log-interval-batches` flag/default and train-loop wiring for non-blocking transfers.
  - `tests/test_train_cli.py`: updated defaults/flags assertions.
- Verification:
  - `python -m py_compile data/collate.py scripts/train_synthetic.py scripts/train_iedb.py cli/main.py tests/test_train_cli.py` passed.
  - `pytest -q tests/test_train_cli.py tests/test_train_iedb.py tests/test_loaders.py` passed (`57 passed`).
- Throughput improvement evidence:
  - baseline old full-data run (`ap-c1W5Ig8zeYUfeE0b81fthN`): ~`394 sec/batch`, `sps~0.3`.
  - optimized full-data run (`ap-Ul5YAGkcwXaZqpOUmwvWqI`) early progress snapshot: ~`1.0 sec/batch`, `sps~121` by batch ~400.
  - implied speedup: roughly `390x` faster batches versus the prior run.
- Bottleneck attribution from new live perf logs (Modal canary run `ap-ZuG991bMw1MX1IpHB7xA1C`, GPU, batch 128, workers 8):
  - epoch averages: `wait=0.045s`, `compute=0.214s`, `backward=0.107s`, `optim=0.010s` per batch.
  - epoch percentages: wait `9.6%`, compute `45.3%`, backward `22.5%`, optim `2.1%`.
  - rolling windows stabilized around compute-forward dominating (`compute ~49%`, backward ~27%, wait low single digits after warm-up).
  - conclusion: after sampler/loader fixes, training is compute-bound rather than data-loader-bound.
- GPU/memory snapshot from same canary:
  - `train_samples_per_sec=270.5` (epoch average).
  - `gpu_peak_allocated_gib=10.40`, `gpu_peak_reserved_gib=10.53`.
  - `gpu_peak_allocated_bytes_per_batch_row=87,271,848` (~`83.23 MiB/row`).
- Operational note:
  - full-profile detached run with the new logging (`ap-RWDgkjm5qQuZ3DSQHjx4DN`) was interrupted by Modal preemption before a stable long-window snapshot; stopped after confirming restart behavior.

---

# Retrain + Epoch Curves (Loss + SLLQHLIGL Affinity)

## Spec

User request:
- retrain with the current optimized pipeline;
- check whether additional bottlenecks remain;
- if not, provide epoch-vs-loss and epoch-vs-affinity curves for `SLLQHLIGL` (`HLA-A*02:01` and `HLA-A*24:02`).

## Plan

- [x] Launch a fresh Modal retraining run with:
  - [x] probe tracking enabled for `SLLQHLIGL` on `HLA-A*02:01,HLA-A*24:02`;
  - [x] perf profiling enabled (`profile_performance=true`, rolling perf windows).
- [x] Download run artifacts (`metrics.csv`, `probe_affinity_over_epochs.csv/json`) from checkpoint volume.
- [x] Quantify bottlenecks from per-epoch perf metrics:
  - [x] data wait vs compute vs backward vs optimizer.
- [x] Generate local plots:
  - [x] epoch vs train/val loss;
  - [x] epoch vs probe KD nM for `A*02:01` and `A*24:02`.
- [x] Summarize whether any new bottleneck is exposed and provide curve file paths/results.

## Review

- Retrain run:
  - Modal app: `ap-5iEbPRF7DOV7VJlaSoFl3y`
  - run id: `retrain-canary10-20260228`
  - profile: `canary`, epochs: `10`, batch size: `128`, `num_workers=8`, `pin_memory=true`, rolling perf windows every `5` batches.
- Artifacts downloaded:
  - `/tmp/retrain_canary10_20260228_metrics.csv`
  - `/tmp/retrain_canary10_20260228_probe.csv`
  - `/tmp/retrain_canary10_20260228_config.json`
- Curve outputs generated locally:
  - `artifacts/analysis/retrain_canary10_20260228/epoch_vs_loss.png`
  - `artifacts/analysis/retrain_canary10_20260228/epoch_vs_sllqhligl_affinity_kd_nM.png`
  - (supporting) `artifacts/analysis/retrain_canary10_20260228/epoch_vs_perf_breakdown_pct.png`
  - summary table: `artifacts/analysis/retrain_canary10_20260228/epoch_summary.csv`
- Bottleneck check (10-epoch averages):
  - wait `14.12%`, compute `45.69%`, backward `23.75%`, optimizer `1.77%`.
  - no new bottleneck detected; training remains compute-dominant.
- Loss trend:
  - train loss: `80.02 -> 22.54`
  - val loss: `3.043 -> 2.486`
- SLLQHLIGL probe KD trend:
  - `HLA-A*02:01`: `66.6 uM -> 44.1 uM`
  - `HLA-A*24:02`: `66.6 uM -> 44.2 uM`
  - both improved similarly; allele separation remained small throughout this run.

---

# pMHC Information-Flow Diagnostics During Training

## Spec

User request:
- track information flow during training to determine whether peptide-MHC interactions are being learned;
- if not, identify why not.

## Plan

- [x] Add per-epoch pMHC shuffle diagnostics on validation batches:
  - [x] baseline real-pair score statistics;
  - [x] MHC-shuffled metrics (same peptides, permuted MHC);
  - [x] peptide-shuffled metrics (same MHC, permuted peptides);
  - [x] both-shuffled metrics;
  - [x] interaction signal metric from differential decomposition.
- [x] Log diagnostics into epoch metrics (`metrics.csv/jsonl`) and console.
- [x] Add CLI controls:
  - [x] `--track-pmhc-flow` / `--no-track-pmhc-flow`
  - [x] `--pmhc-flow-batches`
  - [x] `--pmhc-flow-max-samples`
- [x] Add parser tests for the new flags/defaults.
- [x] Run focused tests.
- [x] Run a short training pass with real merged data and interpret diagnostics:
  - [x] whether MHC shuffle materially degrades binding metrics;
  - [x] whether peptide+MHC interaction term is positive and growing across epochs.

## Review

- Implemented `scripts/train_iedb.py::_evaluate_pmhc_information_flow`:
  - per-epoch counterfactual diagnostics over validation batches for:
    - real pairs,
    - MHC-shuffled pairs,
    - peptide-shuffled pairs,
    - both shuffled;
  - reports per-head metrics (`binding_logit/prob`, `presentation_logit/prob`, `processing_logit`, `KD_nM(log10)`):
    - absolute deltas for MHC/peptide/both perturbations,
    - interaction term magnitude (`real - mhc - pep + both`),
    - normalized variants and simple diagnostic status code.
- Integrated diagnostics into training epoch loop:
  - console summary each epoch with interpreted status,
  - persisted under `split=pmhc_flow` in `metrics.csv/jsonl`.
- Added CLI/config wiring:
  - `--track-pmhc-flow` / `--no-track-pmhc-flow`,
  - `--pmhc-flow-batches`,
  - `--pmhc-flow-max-samples`,
  - defaults added in `IEDB_DEFAULTS`, `cli/main.py`, and `conf/train/default.yaml`.
- Added tests:
  - parser defaults/overrides in `tests/test_train_cli.py`,
  - flow-diagnostic behavior tests in `tests/test_train_iedb.py` (peptide-dominant vs joint interaction dummy models).
- Verification:
  - `python -m py_compile scripts/train_iedb.py cli/main.py tests/test_train_cli.py tests/test_train_iedb.py`
  - `pytest -q tests/test_train_cli.py tests/test_train_iedb.py` -> `42 passed`.
- Short real-data diagnostic run:
  - command used real merged subset (`artifacts/local_runs/merged_small_multimodal_20260224.tsv`) with bounded per-assay caps and `3` epochs;
  - run dir: `artifacts/analysis/pmhc_flow_short_20260228e/run`;
  - extracted analysis:
    - summary table: `artifacts/analysis/pmhc_flow_short_20260228e/epoch_pmhc_flow_summary.csv`
    - loss curve: `artifacts/analysis/pmhc_flow_short_20260228e/epoch_vs_loss.png`
    - flow curve: `artifacts/analysis/pmhc_flow_short_20260228e/epoch_vs_pmhc_flow_binding_norms.png`
    - probe KD curve: `artifacts/analysis/pmhc_flow_short_20260228e/epoch_vs_probe_kd.png`
  - observed (binding-logit flow norms):
    - epoch 1: `mhc=1.0396`, `peptide=0.1080`, `interaction=0.0716`, `status=3`
    - epoch 2: `mhc=0.6554`, `peptide=0.1079`, `interaction=0.0960`, `status=3`
    - epoch 3: `mhc=0.6890`, `peptide=0.1089`, `interaction=0.0938`, `status=3`
  - interpretation:
    - MHC shuffle impact is consistently much larger than peptide shuffle impact in this run, indicating the model is sensitive to MHC context (not ignoring MHC);
    - interaction term remains positive/non-trivial across epochs, indicating learned joint peptide-MHC effects;
    - despite this, probe allele separation for `SLLQHLIGL` remains minimal over 3 epochs (A0201 and A2402 KD stay nearly equal in the weak-binding regime), so allele-specific discrimination for this control is still not learned to target levels.

---

# Strict MHC Sequence Resolution (No Allele-String Fallback)

## Spec

User request:
- unresolved MHC alleles in training must be a hard error;
- never use allele names as sequence inputs;
- enumerate unresolved edge cases explicitly instead of silently falling back;
- clarify numeric/allele-token behavior and continue with the broader training plan.

## Plan

- [x] Remove dataset fallback that returns raw allele names as `mhc_a` sequence.
- [x] Add strict unresolved-MHC validation before dataset construction:
  - [x] compute unresolved allele counts by modality/source;
  - [x] write unresolved report artifact when `run_dir` is set;
  - [x] raise with actionable summary by default.
- [x] Improve MHC index resolution aliases for common shorthand forms (e.g., low-field HLA, murine case variants) to reduce false unresolveds.
- [ ] Add optional explicit unresolved allowlist plumbing (empty by default) for future enumerated edge cases, while preserving strict default behavior.
- [x] Update CLI/config parser defaults and tests for strict behavior.
- [x] Re-run focused tests and verify strict mode errors on unresolved data.
- [x] Re-run a short training pass under strict mode and continue plan steps from that baseline.

## Review

- Fallback removal:
  - `PrestoDataset._get_mhc_sequence` now raises on unresolved allele when strict mode is enabled and never returns raw allele strings as sequence content.
  - Predictor non-strict path also no longer tokenizes unresolved raw allele strings; unresolved now maps to empty MHC sequence unless strict mode raises.
- Strict unresolved audit is now enforced in unified training:
  - `scripts/train_iedb.run` audits unresolved alleles before dataset construction and raises `RuntimeError` in strict mode.
  - unresolved reports are written to run dir: `unresolved_mhc_alleles.csv`, `unresolved_mhc_detail.csv`.
- CLI/config wiring:
  - Added `--strict-mhc-resolution` (default true) and `--allow-unresolved-mhc` in both `cli/main.py` and `scripts/train_iedb.py`.
  - Added `strict_mhc_resolution: true` in `conf/train/default.yaml`.
- Verification:
  - `pytest -q tests/test_train_cli.py tests/test_loaders.py tests/test_train_iedb.py tests/test_mhc_index.py tests/test_predictor.py` -> `119 passed`.
  - Strict smoke run on merged corpus with low caps:
    - command used `data/merged_deduped.tsv` + `data/mhc_index.csv` + strict defaults.
    - failed before training as intended with: `163 unresolved rows` and report artifacts under `artifacts/analysis/strict_mhc_smoke_20260228/`.
    - unresolved alleles in this smoke subset were concentrated in coarse/non-specific labels (e.g., `H2-AA*b/AB*b`, `H2-K*b`, `H2-D*b`, `H2-b class I`, `HLA-DR`, `HLA-A68`).
  - Added triage note with explicit unresolved buckets: `docs/notes/mhc_unresolved_edge_cases.md`.

---

# Strict Tokenization + X Zero Vector

## Spec

User preference:
- avoid silently mapping unfamiliar sequence characters to `<UNK>`;
- prefer hard errors on unfamiliar tokens in canonical paths;
- if ambiguity is present, use explicit `X` amino acid handling and keep it simple.

## Plan

- [x] Update tokenizer API to support explicit unknown-token policy, defaulting to strict error.
- [x] Keep compatibility mode for `<UNK>` mapping only when explicitly requested.
- [x] Set `X` amino-acid embedding row to fixed zero vector in canonical Presto model.
- [x] Add/adjust tests for strict tokenizer behavior, legacy compatibility, and zero-`X` embedding behavior.
- [x] Run focused tests for tokenizer/model/collate/inference codepaths.

## Review

- Tokenization behavior:
  - `Tokenizer` now defaults to `unknown_policy=\"error\"`; unfamiliar characters raise `ValueError`.
  - Legacy fallback remains available with `Tokenizer(unknown_policy=\"unk\")`.
- X-token representation:
  - `Presto` keeps `X` embedding fixed to zero (`AA_TO_IDX[\"X\"]`) at init and after checkpoint load.
  - Gradient hook zeros the `X` row gradient to keep the row fixed.
- Test updates:
  - tokenizer tests now assert strict-error default and legacy-UNK mode explicitly.
  - collate/predictor fixtures that used non-AA placeholders (`B2M`, `B2MSEQ`, protein with `J`) were updated to valid amino-acid strings.
  - added Presto regression test for fixed-zero `X` embedding row and zero gradient row.
- Verification:
  - `pytest -q tests/test_train_cli.py tests/test_loaders.py tests/test_train_iedb.py tests/test_mhc_index.py tests/test_tokenizer.py tests/test_collate.py tests/test_predictor.py tests/test_presto.py` -> `204 passed`.

# MHC Coverage Audit + Diagnostic Training Mode + Convergence/Flow Analysis

## Spec

User request:

- quantify training-data MHC sequence coverage after mouse-allele augmentation;
- report resolved vs missing MHC fractions and species distributions using the project's coarsened species categories;
- run training/analysis to verify epoch-wise learning, peptide-MHC information mixing, and SLLQHLIGL probe movement toward expected A0201 vs A2402 behavior;
- use a distinct diagnostic training mode to isolate data/architecture breakage;
- if diagnostic run reveals issues, fix root cause, then proceed to fuller training and report output-head + latent diagnostics.

## Plan

- [x] Add reusable MHC coverage reporting utilities in training pipeline:
  - [x] compute per-row resolved vs missing MHC across all relevant modalities;
  - [x] compute resolved/missing species distributions with coarse buckets (`human`, `murine`, `nhp`, `other`);
  - [x] emit machine-readable artifacts (`mhc_sequence_coverage.json`, `mhc_sequence_coverage.csv`) in run dir and concise console summary.
- [x] Add a distinct `diagnostic` unified training profile:
  - [x] wire parser/profile resolution (`--profile diagnostic`);
  - [x] enable stricter observability defaults (probe affinity + pmhc flow + perf profiling);
  - [x] keep strict MHC resolution defaulted on and document expected use.
- [x] Add/adjust tests:
  - [x] profile resolution tests for diagnostic profile;
  - [x] coverage report unit tests (fractions + coarse species buckets + artifact writing).
- [x] Run verification:
  - [x] focused tests for modified modules;
  - [x] local diagnostic training run with probe/pmhc flow tracking enabled and capture metrics artifacts.
- [x] Run extended training analysis:
  - [x] run a longer training job on Modal (`10` epochs, diagnostic profile, capped corpus) with probe + pmhc flow + output/latent stats;
  - [x] summarize epoch-vs-loss, epoch-vs-SLLQHLIGL probe affinities, pmhc_flow metrics over time;
  - [x] summarize per-output/latent trend indicators from logged metrics.
- [x] If a clear failure mode is detected, implement fix + re-run diagnostics; otherwise continue to fuller run and report final findings.

## Review

- Implemented coverage reporting + diagnostic mode in training code and CLI:
  - `scripts/train_iedb.py`: coverage audit/report writers, resolved-only filtering, diagnostic profile defaults, output/latent diagnostics.
  - `cli/main.py`: diagnostic profile + new filter/stats flags.
- Added tests and verified:
  - `pytest -q tests/test_train_iedb.py tests/test_train_cli.py` -> `52 passed`.
- Full-corpus coverage audit (local, merged corpus):
  - run dir: `artifacts/local_runs/coverage_diag_20260228b`.
  - pre-filter coverage:
    - rows considered: `5,000,531`
    - resolved: `2,942,582` (`58.85%`)
    - missing: `2,057,949` (`41.15%`)
  - pre-filter species by state:
    - resolved: human `2,647,570`, murine `266,724`, nhp `9,027`, other `19,261`
    - missing: human `1,887,267`, murine `125,932`, nhp `2,702`, other `42,048`
  - post-filter (`--filter-unresolved-mhc`) coverage: `100%` resolved over `2,942,582` rows.
- 10-epoch diagnostic Modal run completed:
  - run id: `diaggpu-small-20260228`
  - profile/config highlights: `diagnostic`, strict resolution on, unresolved filtering on, batch size `256`, probe + pmhc flow + output/latent stats enabled.
  - artifact copy: `modal_runs/diaggpu-small-20260228`.
- Analysis artifacts generated:
  - `artifacts/reports/diagnostic_20260228/diaggpu_small_epoch_trajectory.csv`
  - `artifacts/reports/diagnostic_20260228/diaggpu_small_summary.json`
  - plots:
    - `loss_vs_epoch.png`
    - `probe_kd_vs_epoch.png`
    - `probe_deltas_vs_epoch.png`
    - `pmhc_flow_binding_norms.png`
    - `pmhc_flow_presentation_norms.png`
    - `output_logit_variances.png`
    - `pmhc_latent_stats.png`
- Key findings from diagnostic run:
  - Loss decreases overall (`train 39.88 -> 17.02`, `val 1.30 -> 1.23`).
  - Probe affinity target not met:
    - `SLLQHLIGL + HLA-A*02:01`: `~4.15 uM -> ~4.34 uM` by epoch 10.
    - `SLLQHLIGL + HLA-A*24:02`: `~4.15 uM -> ~4.36 uM` by epoch 10.
    - allele separation remains tiny (`delta log10 KD ~ -0.0014` at epoch 10).
  - pMHC flow indicates interaction signal exists but binding head is dominated by MHC-side perturbation:
    - mean binding interaction ratio: `~0.017`
    - mean presentation interaction ratio: `~0.176`
  - Output/latent variances stay bounded (no catastrophic divergence), but the probe-control behavior indicates insufficient allele-specific discrimination in this training regime.

---

# Binding Learning Failure Analysis + MHC Quality Guards

## Spec

User request:

- show top missing alleles (top 20);
- analyze why binding (especially A0201 vs A2402) is not being learned from trajectories/latents/probe/input mix/gradients;
- add MHC quality guards:
  - enforce MHC chain length assertions (`>100 aa`, with class-I B2M exception for shorter beta chains),
  - check/report non-canonical residues and `X` usage in loaded MHC sequences;
- add a regularization option encouraging sparse MHC-residue usage for binding (targeting ~30-60 effective residues);
- evaluate whether allele-motif auxiliary supervision (positionwise motif distributions) is useful or overkill.

## Plan

- [x] Produce analysis artifacts for current diagnostic run:
  - [x] top-20 unresolved allele table from full-corpus unresolved report;
  - [x] capped-vs-full binding data composition (A0201/A2402 counts, motif coverage, probe peptide presence);
  - [x] binding probe dynamics + pMHC flow interpretation summary.
- [x] Implement MHC sequence quality checks:
  - [x] validate loaded MHC characters (`ACDEFGHIKLMNPQRSTVWY` + `X`) and fail fast on non-canonical residues;
  - [x] log/report counts of sequences containing `X`;
  - [x] enforce min-length assertions for MHC chains (`>100 aa`) with explicit class-I beta/B2M exception.
- [x] Implement optional binding MHC-attention sparsity regularization:
  - [x] expose binding latent attention weights when requested;
  - [x] compute effective attended MHC support (inverse-simpson style) and penalize outside [30, 60];
  - [x] wire regularization weight/config in train loop and CLI defaults (off by default).
- [x] Add/adjust tests for:
  - [x] MHC sequence validation behavior;
  - [x] attention-regularization plumbing and no-op behavior when disabled.
- [x] Run focused tests + short diagnostic smoke to verify no regressions.
- [x] Document findings and concrete root-cause diagnosis in review.

## Review

- Top-20 unresolved alleles from full-corpus strict audit (`artifacts/local_runs/coverage_diag_20260228/unresolved_mhc_alleles.csv`) were extracted; largest unresolved buckets are coarse class/locus labels (`human class I`, `human class II`, `HLA-DR`) plus murine shorthand/haplotype forms.
- Added and validated MHC sequence quality guards:
  - `data/loaders.py`: fail-fast on non-canonical residues, strict chain length floor (`>100 aa`) with explicit class-I beta/B2M short-chain exception, plus runtime `X` ambiguity warnings with allele examples.
  - `scripts/train_iedb.py`: `audit_loaded_mhc_sequence_quality` now blocks training if index-loaded MHC sequences contain non-canonical residues or short chains and logs aggregate `X` usage.
- Added optional binding-attention sparsity regularization:
  - `models/presto.py`: optional `return_binding_attention` path + binding MHC attention support stats.
  - `scripts/train_synthetic.py`/`scripts/train_iedb.py`/`cli/main.py`: config + CLI flags for `--mhc-attention-sparsity-*`.
  - Fixed an implementation bug where latent dependency tokens extended attention `K`; attention support now trims to base stream tokens before MHC masking.
- Added tests:
  - `tests/test_loaders.py`: non-canonical residue rejection, short-chain rejection, `X` warning behavior, and updated biologically valid mock MHC sequences.
  - `tests/test_train_iedb.py`: quality-audit regression for non-canonical/`X`/short-sequence detection.
  - `tests/test_train_synthetic.py`: sparsity-penalty active/in-range no-op coverage + valid AA mock sequences.
  - `tests/test_train_cli.py` + `tests/test_presto.py`: parser coverage for new flags and attention-stat emission coverage.
- Verification:
  - `pytest -q tests/test_loaders.py tests/test_train_iedb.py tests/test_train_synthetic.py tests/test_train_cli.py tests/test_presto.py` -> `119 passed`.
  - `python -m py_compile models/presto.py data/loaders.py scripts/train_iedb.py scripts/train_synthetic.py tests/test_loaders.py tests/test_train_iedb.py tests/test_train_synthetic.py tests/test_train_cli.py tests/test_presto.py` passed.
- Root-cause diagnostic bundle generated at `artifacts/reports/binding_failure_20260301/diagnosis_summary.json`:
  - training loss falls but allele probe remains nearly identical (`SLLQHLIGL`: A0201 `4341.9 nM` vs A2402 `4355.9 nM`, `Δlog10=-0.0014`);
  - pMHC flow shows strong MHC sensitivity but weak binding interaction term (`binding_interaction_ratio~0.016`);
  - capped+augmented train composition remains highly skewed (e.g., `binding_affinity_A2402=0` and only `24` sampled `binding_ic50` draws for A2402 vs `511` for A0201 in one epoch), which is sufficient to explain weak allele discrimination.

---

# Full Merge Data-Quality Report + 40k Binding Retrain

## Spec

User request:

- generate a report for full-data merge/filter/MHC sequence quality across assay sources;
- sanity check species/allele diversity and quantify filtering/drop impact;
- clarify the effective size of the prior capped dataset that produced `A0201=760`;
- run a new training attempt with `--max-binding 40000` and check whether behavior improves.

## Plan

- [x] Build a reproducible data-quality report artifact from merged full corpus:
  - [x] source-level record composition and assay-type composition;
  - [x] MHC resolution coverage before/after strict filtering;
  - [x] drop counts by modality and unresolved-category summaries;
  - [x] species distribution and allele diversity among resolved rows.
- [x] Write report outputs under `artifacts/reports/` (`.json` + concise `.md` summary).
- [x] Sanity-check report and flag whether filtering appears acceptable or excessive.
- [x] Restate effective size of prior `--max-binding 4000` slice (before/after filtering/augmentation).
- [x] Launch a new run with `--max-binding 40000` (keeping probe tracking on) and gather:
  - [x] epoch losses,
  - [x] `SLLQHLIGL` A0201/A2402 probe curves,
  - [x] A0201 vs A2402 binding separation deltas.
- [x] Compare `40000` run against prior `4000` diagnostic and summarize whether it is better (epoch-1/2 readout; epoch-3 still running).

## Review

- Full-merge report generated:
  - `artifacts/reports/data_quality_merge_20260301/report.json`
  - `artifacts/reports/data_quality_merge_20260301/report.md`
- Full merged corpus composition:
  - total rows: `5,593,161`;
  - record types: binding `4,131,910`, bcell `816,468`, tcell `478,464`, tcr `166,319`;
  - top sources: IEDB `5,336,154`, VDJdb `152,583`, CEDAR `90,688`, McPAS `13,736`.
- Full-corpus MHC resolution sanity check:
  - unique alleles seen: `1,108`;
  - resolved via index: `766/1,108` (`69.13%`);
  - resolved sequence quality: noncanonical `0`, with `X` `0`, short `<101aa` `3`.
- Full training coverage impact (strict audit artifacts):
  - considered rows: `5,000,531`;
  - resolved rows: `2,942,582` (`58.85%`);
  - missing rows: `2,057,949` (`41.15%`);
  - missing is dominated by coarse labels (`human class I/II`, `HLA-DR`), not by many canonical high-resolution alleles.
- Species diversity in resolved full-data rows:
  - human `2,647,570`, murine `266,724`, nhp `9,027`, other `19,261`.
- Exact provenance of prior `760/49` slice (`--max-binding 4000` config) reproduced by loader path:
  - pre-filter loaded: binding `4000`, kinetics `106`, stability `1000`, elution `4000`, tcell `3000`, vdjdb `1000`;
  - post-resolved-only filter: binding `3279`, kinetics `101`, stability `965`, elution `3645`, tcell `2489`, vdjdb `515`;
  - post-filter binding allele counts: `HLA-A*02:01=760`, `HLA-A*24:02=49`.
- New `--max-binding 40000` Modal run:
  - run id: `diag-max40k-20260301` (Modal app `ap-YURXHcVaiKO3XCXJvTkIBq`);
  - data composition: binding `40000`, kinetics `106`, stability `1000`, elution `4000`, tcell `3000`, vdjdb `1000`;
  - post-filter coverage: `46,518` resolved rows (`100%`), species buckets: human `38,110`, murine `4,073`, nhp `4,294`, other `41`.
- `max-binding 40000` training metrics so far:
  - epoch 1: train `34.9598`, val `1.1783`; probe KD `~74,968 nM` for both A0201/A2402 (no separation);
  - epoch 2: train `17.8746`, val `1.2605`; probe KD `74,650 nM` (A0201) vs `74,586 nM` (A2402), bind probs `0.002002` vs `0.002004` (still no practical separation).
- Comparison vs prior `max-binding 4000` diagnostic:
  - prior run had KD in low-µM regime (`~4,152 nM` at epoch 1) but still near-zero A0201/A2402 delta;
  - new `40000` run (through epoch 2) improves train loss faster but has not improved allele separation and currently predicts weaker absolute binding on the probe.

---

# 40k Binding-Motif Recovery + Full Modal Dynamics Analysis

## Spec

User request:

- keep iterating on the 40k-binding setup until the model learns at least binding motifs;
- after motif learning appears, run a full-data Modal training run;
- analyze dynamics comprehensively (loss curves, assay outputs, latents, gradients/flow, probe behavior) and determine whether learning is real.

## Plan

- [ ] Diagnose the current 40k failure mode from artifacts and data composition.
  - [x] quantify per-allele/per-assay composition in the 40k capped slice;
  - [x] verify whether first-N cap loading is creating order bias;
  - [x] verify quantitative-affinity supervision density in the slice.
- [ ] Implement targeted fixes to improve motif learnability.
  - [x] make capped merged loading representative (reservoir/randomized sampling for capped modalities);
  - [x] change supervised multi-task loss aggregation to reduce overweighting of sparse assays;
  - [x] expose/train with configurable supervised loss weighting mode (default to sample-support weighted);
  - [ ] add optional binding-focused sampling controls without dropping other assays;
  - [x] add motif diagnostics (anchor mutational scan + per-epoch motif/probe summaries).
- [ ] Run iterative 40k diagnostic Modal runs and stop only when motif signal is detectable.
  - [ ] track A0201 vs A2402 probe deltas over epochs;
  - [ ] track motif-directionality from anchor scans (A0201 expected anchors favored).
- [ ] Launch a full-data Modal run with the validated setup.
- [ ] Deliver deep analysis artifacts/report for training dynamics and learning verdict.

---

# Make Binding Work — Learn Contact Residues from Full MHC Sequences

## Completed Steps

- [x] **Step 1**: Enable MHC attention sparsity loss (weight=0.1, band=25-45 residues)
- [x] **Step 2**: Expand probe diagnostics (10 alleles, 9 positions) + discrimination/motif metrics
- [x] **Step 3**: Binding latent auxiliary probe (shortcut gradient path, 0.3 weight)
- [x] **Step 4**: MHC sequence augmentation (mixed training with MHC-only samples)
- [x] **Step 5**: Expand MHC chain type granularity (3→6 fine types with backward compat)

## Files Modified

| File | Changes |
|------|---------|
| `scripts/train_iedb.py` | Sparsity defaults, expanded probes, discrimination/motif metrics, MHC augmentation |
| `scripts/train_synthetic.py` | Sparsity defaults, binding probe loss spec, fine type loss specs + target generation |
| `models/presto.py` | Binding affinity probe head, 6-class chain type heads, updated compositional formula |
| `conf/train/default.yaml` | Sparsity weight and band defaults |
| `cli/main.py` | `--mhc-augmentation-samples` CLI arg |
| `data/vocab.py` | `MHC_CHAIN_FINE_TYPES` (6-class label set) |
| `data/mhc_index.py` | `infer_fine_chain_type()` function |
| `data/collate.py` | `primary_alleles` field on PrestoBatch |
| `training/checkpointing.py` | 3→6 class checkpoint migration |
| `tests/test_presto.py` | Updated type logit shape assertion |

## Verification

- All 536 tests pass (0 failures)
- binding_affinity_probe_kd output: shape (B, 1), clamped [-3.0, 8.0]
- Fine chain type inference: all gene patterns correctly classified
- Discrimination/motif metric functions produce correct outputs

---

# A100 Training Performance Plan (Correctness/Design Preserving)

## Spec

User request:

- training on Modal A100 is slow per epoch with low observed GPU occupancy;
- provide a performance plan that preserves model correctness and current design intent (no objective/architecture changes unless explicitly opted in).

Non-goals for this plan:

- no silent target/loss definition changes;
- no removal of biologic constraints or diagnostics by default.

## Findings (Current Evidence)

- Recent diagnostic runs show **data wait dominates epoch time**:
  - `diaggpu-small-20260228`: `perf_data_wait_pct_epoch` mean `38.55%` (compute `26.79%`, backward `14.88%`).
  - `diag-max40k-resv-motif-20260301b`: data wait mean `61.56%` (compute `17.51%`, backward `9.12%`).
  - `diag-max40k-resv-motif-sw-20260301e`: data wait mean `48.30%` (compute `20.53%`, backward `13.12%`).
- Data pipeline is Python-heavy by design:
  - per-batch tokenization and MIL expansion inside `PrestoCollator.__call__` (`data/collate.py`).
  - fully dynamic per-batch balancing logic in `BalancedMiniBatchSampler.__iter__` (`data/loaders.py`).
  - `PrestoBatch.to()` reconstructs a new dataclass and walks all tensors/dicts each step (`data/collate.py`).
- Training loop currently uses full precision path (no AMP/autocast path in `compute_loss` / `train_epoch` in `scripts/train_synthetic.py`).
- Diagnostic profile defaults add substantial extra epoch work (`probe`, `probe_motif`, `pmhc_flow`, `output_latent`) in `scripts/train_iedb.py`.

## Plan

- [ ] **Phase 1: Reproducible perf baseline and guards**
  - add a stable perf benchmark command pair (`diagnostic` and `full`) with fixed seed/run caps;
  - report mean/std for: wait/compute/backward sec-per-batch, train samples/sec, eval sec, epoch wall time.
- [ ] **Phase 2: Data path acceleration (highest priority, semantics-preserving)**
  - add optional pretokenized dataset path to eliminate per-batch string tokenization while preserving identical token IDs;
  - optimize collator hot path to skip repeated Python list/dict work when fields are absent;
  - add in-place device transfer method for `PrestoBatch` to avoid full dataclass reconstruction each step;
  - keep existing path as fallback and verify tensor equality between paths.
- [ ] **Phase 3: Balanced sampler cost reduction (preserve balance policy intent)**
  - precompute epoch index plans in larger chunks rather than per-sample nested Python loops;
  - add a lower-overhead balancing mode (`task+label`) as an explicit option while retaining existing full mode as default for strict parity;
  - benchmark `balanced=true/false/mode=task_label` on same data slice.
- [ ] **Phase 4: GPU kernel efficiency (opt-in, correctness-preserving numerics)**
  - add optional `--amp bf16` autocast for A100;
  - enable TF32 matmul path for FP32 fallback;
  - switch `optimizer.zero_grad(set_to_none=True)` in non-PCGrad path;
  - gate optional `torch.compile` behind explicit flag and warmup guard.
- [ ] **Phase 5: Diagnostic scheduling without signal loss**
  - keep diagnostics enabled but add per-diagnostic cadence controls (e.g., every N epochs);
  - default diagnostic cadence: lightweight every epoch, heavy scans every 3-5 epochs in long runs;
  - preserve existing outputs/columns when diagnostics run.
- [ ] **Phase 6: Verification and acceptance**
  - correctness: fixed-seed parity checks for losses/metrics over short run (within tolerance for AMP mode);
  - design: verify all output keys/artifacts remain present for enabled diagnostics;
  - performance target: reduce data-wait share to `<25%` and improve train samples/sec by `>=1.6x` on A100 diagnostic profile.

## Review (Planning Stage Complete)

- [x] Collected bottleneck evidence from existing Modal artifacts/metrics.
- [x] Mapped bottlenecks to concrete code hotspots and rank-ordered interventions.
- [x] Produced a staged implementation and verification plan focused on correctness/design preservation.

---

# Stage And Commit Model/Test Code + Config

## Plan

- [x] Define commit scope: include only model/test code and configuration paths; exclude datasets, artifacts, and docs-only files.
- [x] Stage scoped files and review staged diff for correctness.
- [x] Run focused verification tests against staged model/test code.
- [x] Commit with a clear message and push to GitHub.

## Review

- [x] Scoped file set documented and validated.
- [x] Verification command and outcome recorded.
  - `pytest -q tests/test_presto.py tests/test_predictor.py tests/test_trainer.py tests/test_training_e2e.py tests/test_train_iedb.py tests/test_data_cli.py tests/test_train_cli.py tests/test_predict_cli.py tests/test_evaluate_cli.py tests/test_weights_cli.py tests/test_mhc_index.py tests/test_mouse_mhc_overlay.py`
  - Result: `189 passed in 14.20s`
- [x] Commit SHA and push target recorded.
  - Commit: `5e26e2f` (`Add model and test code/configuration updates`)
  - Push: `origin/main` (`fb2ecaf..5e26e2f`)

---

# Stage And Commit Documentation Updates

## Plan

- [x] Define docs scope and stage only documentation-related files.
- [x] Validate staged diff is docs-only and coherent.
- [x] Commit and push docs changes to GitHub.

## Review

- [x] Staged docs file list recorded.
  - `README.md`
  - `mkdocs.yml`
  - `docs/architecture.md` (deleted)
  - `docs/cli.md`
  - `docs/curriculum.md` (deleted)
  - `docs/design.md`
  - `docs/index.md`
  - `docs/notes/ablation_backlog.md`
  - `docs/notes/mhc_unresolved_edge_cases.md`
  - `docs/notes/mouse_mhc_overlay_sources.md`
  - `docs/tcr_spec.md`
  - `docs/training_spec.md`
- [x] Commit SHA and push target recorded.
  - Verification: `python -m mkdocs build --strict -q` (passed)
  - Commit: `324e613` (`Add documentation updates and specs`)
  - Push: `origin/main` (`5e26e2f..324e613`)

---

# Audit Implementation vs Design (2026-03-04)

## Plan

- [x] Extract canonical requirements from `docs/design.md` and `docs/training_spec.md` that impact architecture, information flow, and trainability.
- [x] Trace requirement-to-code mapping across `models/`, `training/`, and data collation paths.
- [x] Identify defects/risk points for gradient flow, masking, detached paths, loss wiring, and target leakage/shape mismatches.
- [x] Run focused tests covering model forward, losses, trainer loop, and representative e2e training smoke.
- [x] Document findings with severity, file/line evidence, and concrete fixes in a review section.

## Review

- Focused verification completed:
  - `pytest -q tests/test_presto.py tests/test_losses.py tests/test_trainer.py tests/test_train_iedb.py tests/test_training_e2e.py tests/test_training_logging.py tests/test_collate.py` -> `131 passed`.
  - `python -m py_compile models/presto.py models/heads.py training/losses.py training/trainer.py scripts/train_synthetic.py scripts/train_iedb.py data/collate.py data/loaders.py` passed.
- Key findings from design-vs-code audit:
  - **High**: global conditioning has cross-sample leakage and loses per-sample missingness flags.
    - `models/presto.py` computes chain completeness bits with batch-level `.any(...).any()` and applies one bitfield to all rows.
    - This causes one sample's optional inputs to alter other samples' embeddings in the same batch.
  - **High**: per-sample species conditioning is ignored for list inputs in canonical training path.
    - `scripts/train_synthetic.py`/`scripts/train_iedb.py` pass `species=batch.processing_species` (list), but `models/presto.py` only maps `species_id` when `species` is a single `str`.
    - Result: global species embedding defaults to index 0 for all samples unless a scalar string is passed.
  - **High**: species probability dimensionality is inconsistent (12-class inferred vs 6-class override path).
    - Per-chain heads use `N_ORGANISM_CATEGORIES` (12) in `models/presto.py`, while `_species_probs_from_input` in `models/pmhc.py` returns `len(PROCESSING_SPECIES_BUCKETS)` (6).
    - This yields inconsistent `outputs["species_probs"]` shape depending on whether `species` is provided.
  - **Medium**: presentation latent branch starts with zero gradient to latent query/head parameters.
    - `models/presto.py` initializes `w_presentation_class{1,2}_latent` to `0.0`; first-step gradients only update these scalars, not the latent head/query parameters.
    - This delays learning in the presentation latent pathway and can weaken early DAG supervision.
  - **Medium**: core-context implementation is simplified relative to spec and drops width-aware soft membership.
    - `core_context_vec` is computed from `core_start_prob` only; the design describes a soft window using both start distribution and width plus core-relative positional injection.
    - Expected core/PFR information flow into `presentation_class2`/`recognition_cd4` is therefore reduced.
  - **Medium**: latent DAG differs from spec for recognition path.
    - Implementation adds `species_of_origin` latent and `foreignness` dependency into recognition latents.
    - Design describes recognition as token-only (plus optional core context for CD4) with no upstream latent dependency.
  - **Medium**: T-cell context implementation omits peptide-format and culture-duration channels described in spec.
    - `models/heads.py` and `data/collate.py` currently use method/readout/APC/culture/stim only.
  - **Medium**: uncertainty-weighting task cardinality is inconsistent between training entrypoints and loss task registry.
    - `scripts/train_iedb.py` allocates log-variance params for `IEDB_LOSS_TASK_NAMES` (18 tasks), while `scripts/train_synthetic.py` computes supervised losses over a larger `LOSS_TASK_SPECS` set.
    - Extra tasks fall back to fixed weighting silently.

---

# Correctness + Flow Fix Pass (2026-03-04)

## Plan

- [x] Fix per-sample global conditioning in `Presto`:
  - [x] compute chain-completeness bitfield per sample (not batch-global),
  - [x] support per-sample species lists/tuples/tensors for `species_id` conditioning.
- [x] Align MHC species dimensionality:
  - [x] make per-chain species heads and conditioning paths use `N_MHC_SPECIES` (6),
  - [x] keep output shapes consistent regardless of whether user overrides species.
- [x] Restore early gradient flow for presentation latent residual path:
  - [x] change latent gate initialization to small positive values,
  - [x] add a focused gradient-flow unit test.
- [x] Improve core-context information flow:
  - [x] compute width-aware soft core membership from start-prob + width,
  - [x] use this soft membership for `core_context_vec`.
- [x] Align recognition latent DAG to design:
  - [x] remove `foreignness` dependency from recognition latents.
- [x] Align uncertainty weighting task cardinality:
  - [x] remove hard-coded IEDB task list for UW sizing/metrics,
  - [x] reuse canonical task registry from `train_synthetic`.
- [x] Revisit assay-presentation regularizer:
  - [x] ensure it does not penalize intended `presentation + ms_detectability` decomposition.
- [x] Add missing T-cell context channels from design:
  - [x] add peptide-format channel (`peptide_format_idx`) through collator -> model head,
  - [x] add culture-duration channel (`culture_duration_hours`) with learned defaults.
- [x] Verification after each code change:
  - [x] run focused tests and a synthetic probe for the changed behavior,
  - [x] run an end-to-end focused test set at the end.

## Review

- Implemented fixes:
  - `models/presto.py`: per-sample chain completeness bits, per-sample species-id mapping (incl list/tuple/tensor), MHC species heads switched to `N_MHC_SPECIES`, non-zero presentation latent gates, width-aware `core_membership_prob`, core-relative positional injection into peptide states, recognition latent deps cleared, and new T-cell context channels threaded through `forward`.
  - `scripts/train_iedb.py`: uncertainty-weighting task registry now mirrors canonical `LOSS_TASK_NAMES`.
  - `scripts/train_synthetic.py`: assay-presentation consistency term now enforces additive decomposition (`elution ~= presentation + ms_detectability`) with compatibility fallback.
  - `models/heads.py`, `data/collate.py`, `data/vocab.py`, `data/__init__.py`: added peptide-format and culture-duration context features for T-cell assay conditioning.
  - `tests/test_presto.py`: updated presentation additive-path expectation, recognition-dependency expectation, and added explicit first-step gradient-flow test.
- Verification:
  - focused checks run after each patch with synthetic probes (batch-isolation, species-list conditioning, latent-gradient flow, recognition-foreignness decoupling, decomposition regularizer behavior, T-cell context channel effect, core-relative ablation sensitivity);
  - final suite: `pytest -q tests/test_presto.py tests/test_losses.py tests/test_trainer.py tests/test_train_iedb.py tests/test_training_e2e.py tests/test_training_logging.py tests/test_collate.py tests/test_heads.py tests/test_train_synthetic.py` -> `161 passed`;
  - compile check: `python -m py_compile ...` (touched model/data/training files) passed.

---

# Foreignness-Recognition Contract + T-Cell Panel Refactor (2026-03-04)

## Plan

- [x] Reinstate recognition latent dependency on `foreignness` while preserving strict information-flow boundaries.
  - [x] `recognition_cd8` / `recognition_cd4` depend on `foreignness`.
  - [x] recognition receives no MHC/flank/core/TCR extra tokens.
  - [x] recognition uses peptide states without core-relative injection.
- [x] Add explicit override APIs for latent-fixing side information.
  - [x] add `mhc_species`, `immune_species`, `species_of_origin` forward overrides.
  - [x] keep compatibility alias behavior for `species` and `peptide_species`.
- [x] Refactor T-cell output to panel-style shared embedding prediction.
  - [x] keep `tcell_logit` for observed context.
  - [x] emit `tcell_panel_logits` / `tcell_context_logits` over categorized assay axes.
- [x] Update design docs to be enumerative about inputs, latent variables (with overrideability), and outputs.
  - [x] update recognition/foreignness DAG prose.
  - [x] add explicit canonical IO/latent/output contract table(s).
- [x] Run focused and end-to-end regressions after changes.

## Review

- Code:
  - `models/presto.py`:
    - recognition deps reset to `["foreignness"]`;
    - strict routing enforced (no context/core/TCR token injection into recognition);
    - side-info overrides added: `mhc_species`, `immune_species`, `species_of_origin` (plus existing compatibility aliases);
    - `species_of_origin` override now cleanly controls downstream `foreignness`;
    - peptide-token conditioning stripped of side-info/completeness injection to prevent leakage into peptide-only paths.
  - `models/heads.py`:
    - `TCellAssayHead` refactored to shared context-embedding composition;
    - added `predict_panel(...)` that returns per-axis assay-outcome panels:
      `assay_method`, `assay_readout`, `apc_type`, `culture_context`,
      `stim_context`, `peptide_format`.
  - `tests/test_presto.py`:
    - recognition dependency assertion updated;
    - added tests for recognition isolation and species-of-origin override effects;
    - added T-cell panel output shape checks.
- Docs:
  - `docs/design.md`:
    - enumerated input/latent/output contract section added;
    - explicit override contract added (`mhc_class`, `mhc_species`, `immune_species`, `species_of_origin`, `species` alias);
    - recognition sections/DAG/table updated to peptide+foreignness-only contract.
  - `docs/tcr_spec.md` and `docs/training_spec.md` updated for peptide+foreignness canonical recognition wording.
- Verification:
  - `pytest -q tests/test_heads.py tests/test_presto.py` -> `62 passed`;
  - `pytest -q tests/test_train_synthetic.py tests/test_training_e2e.py tests/test_collate.py tests/test_train_iedb.py` -> `74 passed`;
  - final focused regression:
    - `pytest -q tests/test_presto.py tests/test_heads.py tests/test_losses.py tests/test_trainer.py tests/test_train_iedb.py tests/test_training_e2e.py tests/test_training_logging.py tests/test_collate.py tests/test_train_synthetic.py`
    - result: `164 passed`;
  - `python -m py_compile models/presto.py models/heads.py scripts/train_synthetic.py scripts/train_iedb.py data/collate.py data/vocab.py` passed.

---

# Mixed Split Outputs + Predictor/CLI Overrides (2026-03-04)

## Plan

- [x] Add mixed outputs for all class-split / CD4-CD8 split branches.
  - [x] expose mixed latent vectors (class-prob weighted),
  - [x] expose mixed logits/prob aliases for split output heads.
- [x] Add predictor-level named overrides for latent-fixing side information.
  - [x] `mhc_species`, `immune_species`, `species_of_origin` on predictor APIs.
  - [x] ensure pass-through into model forward.
- [x] Add CLI flags for the same overrides.
  - [x] `predict presentation`
  - [x] `predict tile`
  - [x] parser support for `predict recognition` placeholder command.
- [x] Update/extend tests for new parser and predictor wiring.
- [x] Run focused regression and compile checks.

## Review

- Code updates:
  - `models/presto.py`:
    - added mixed latent vectors in `outputs["latent_vecs"]`:
      `processing_mixed`, `presentation_mixed`, `recognition_mixed`, `immunogenicity_mixed`;
    - added mixed output aliases:
      `processing_mixed_*`, `binding_mixed_*`, `presentation_mixed_*`,
      `recognition_mixed_*`, `immunogenicity_mixed_*`;
    - retained existing canonical outputs for backward compatibility.
  - `inference/predictor.py`:
    - added named override args on predictor APIs:
      `mhc_species`, `immune_species`, `species_of_origin`;
    - pass-through wiring to model forward for `predict_presentation`,
      `predict_tiled_presentation`, `predict_presentation_multi_allele`,
      and `embed_pmhc`.
  - `cli/main.py` + `cli/predict.py`:
    - added/forwarded CLI flags:
      `--mhc-species`, `--immune-species`, `--species-of-origin`
      for `predict presentation` and `predict tile` (and parser support on `predict recognition`).
  - `scripts/train_synthetic.py`:
    - added CE task specs for T-cell panel axes (`tcell_assay_method`,
      `tcell_assay_readout`, `tcell_apc_type`, `tcell_culture_context`,
      `tcell_stim_context`, `tcell_peptide_format`) using panel logits.
- Tests/docs:
  - `tests/test_presto.py`: mixed-output presence + mixture correctness tests.
  - `tests/test_predictor.py`: predictor override pass-through test.
  - `tests/test_predict_cli.py`: parser + CLI forwarding assertions for new flags.
  - `tests/test_collate.py`: new T-cell context keys/targets assertions.
  - `docs/design.md`: output contract includes split-mixture convenience outputs.
- Verification:
  - focused regression:
    - `pytest -q tests/test_presto.py tests/test_heads.py tests/test_predictor.py tests/test_predict_cli.py tests/test_train_synthetic.py tests/test_collate.py`
    - result: `152 passed`;
  - full relevant regression:
    - `pytest -q tests/test_presto.py tests/test_heads.py tests/test_losses.py tests/test_trainer.py tests/test_train_iedb.py tests/test_training_e2e.py tests/test_training_logging.py tests/test_collate.py tests/test_train_synthetic.py tests/test_predictor.py tests/test_predict_cli.py`
    - result: `217 passed`;
  - compile checks passed for touched inference/cli/model/training files.

---

# Close Remaining Design/Flow Gaps (2026-03-05)

## Spec

Goal: resolve all remaining mismatches identified in the latest implementation-vs-design audit, including latent conditioning, information/gradient-flow issues, and docs drift.

## Plan

- [x] Wire `mhc_species` override into downstream latent conditioning (`context_token`) instead of inferred per-chain species when override is present.
- [x] Prevent core-relative peptide injection from leaking into processing/MS latents; keep it only on intended pathways.
- [x] Make MIL capping bag-aware so instance subsampling preserves at least one instance per bag.
- [x] Add canonical core-start CE supervision path (optional when labels exist) in collation + training loss wiring.
- [x] Add binding affinity/stability orthogonality regularization and expose a training knob for it.
- [x] Improve presentation latent gradient utilization by strengthening latent residual contribution with stable initialization.
- [x] Replace remaining hard clamp points in assay heads with smooth bounds where feasible to preserve extreme-example gradients.
- [x] Align predictor AA validation with the new 26-token AA vocabulary (remove B/Z/O/U acceptance).
- [x] Update design/training docs to match implemented token layout and training objective details.
- [x] Add/update focused tests for each fix and run verification suite.

## Review

- `models/presto.py`
  - `mhc_species` override now conditions `context_token` directly (replicated into both chain-species slots when override is active).
  - Core-relative peptide embedding no longer leaks into processing/MS latents; only binding/presentation pathways use core-aware states.
  - Presentation latent residual branch now uses positive `softplus`-gated scale with stronger initialization (`softplus(-1.5) ~= 0.20`).
- `scripts/train_synthetic.py`
  - Added `core_start` CE task wiring (`core_start_logit` vs `targets["core_start"]`).
  - Added bag-aware MIL cap helper preserving at least one instance per bag even when capped.
  - Added binding orthogonality regularization (`consistency_binding_orthogonality`) with configurable `binding_orthogonality_weight` (default `0.01`).
- `data/collate.py`
  - Added optional `PrestoSample.core_start` and collation into `targets["core_start"]` + mask.
- `models/heads.py`
  - Replaced several hard clamps with smooth lower/range bounds to preserve extreme-gradient flow.
- `inference/predictor.py`
  - AA sequence validator updated to match 26-token vocabulary (removed B/Z/O/U acceptance).
- Docs updated
  - `docs/design.md`: token-stream layout now reflects residue-only runtime layout with segment IDs (no explicit boundary tokens).
  - `docs/training_spec.md`: explicit notes for optional core-start CE activation and orthogonality knob/default.
- Verification
  - `python -m py_compile models/presto.py models/heads.py data/collate.py scripts/train_synthetic.py inference/predictor.py tests/test_presto.py tests/test_train_synthetic.py tests/test_collate.py tests/test_predictor.py`
  - `pytest -q tests/test_presto.py tests/test_train_synthetic.py tests/test_collate.py tests/test_predictor.py tests/test_heads.py`
  - Result: `155 passed`.

---

# Modal Small-Run Trajectory Analysis (2026-03-05)

## Spec

Goal: run a short Modal unified-training job and analyze trajectory signals across epochs for:
- losses,
- latent/output diagnostics,
- gradient/backprop magnitude proxies from logged train metrics,
- SLLQHLIGL probe predictions (A*02:01 vs A*24:02 discrimination).

## Plan

- [x] Launch a small capped Modal unified run with probe + latent diagnostics enabled.
- [x] Pull run artifacts (`metrics.csv`, probe artifacts) from Modal volume.
- [x] Parse epoch trajectories for train/val loss and key per-task losses.
- [x] Parse latent/output diagnostics trajectory (`output_latent` split metrics).
- [x] Parse gradient-magnitude proxies from training logs (backward/optimizer timing and any grad-like metrics if present).
- [x] Parse SLLQHLIGL probe trajectory and discrimination deltas across epochs.
- [x] Summarize findings with concrete trend calls and caveats.

## Review

- First attempt (`traj-small-20260305T090347`) failed before epoch 1 due strict unresolved-MHC enforcement (`176` unresolved rows). This was a data-resolution strictness gate, not a model runtime bug.
- Successful run: `traj-small-20260305T090854-filt` (`--filter-unresolved-mhc`, `6` epochs, `--max-batches 80`, `--max-val-batches 20`, probe + pMHC flow + output-latent diagnostics enabled).
- Artifacts downloaded to:
  - `modal_runs/traj-small-20260305T090854-filt/traj-small-20260305T090854-filt/metrics.csv`
  - `modal_runs/traj-small-20260305T090854-filt/traj-small-20260305T090854-filt/probe_affinity_over_epochs.csv`
- Loss trajectory:
  - train loss dropped `0.6374 -> 0.1984` (`-68.9%`) by epoch 6;
  - validation loss was best at epoch 1 (`0.19865`) and rose afterward (`0.31338` at epoch 6), indicating early overfitting/shift under this tiny capped run.
- Latent/output trajectory (selected diagnostics):
  - `diag_pmhc_vec_feature_var_mean`: `0.146 -> 0.976` (monotonic growth);
  - `diag_binding_logit_var`: `0.078 -> 12.048` (strong spread increase);
  - `diag_presentation_logit_var`: `4.365 -> 23.003` (peaked at `37.802` on epoch 5);
  - `diag_latent_presentation_mixed_norm_mean`: `12.455 -> 42.576`;
  - `diag_latent_binding_affinity_norm_mean`: `18.819 -> 20.533`.
- Gradient-magnitude signals:
  - no direct gradient-norm metric is logged in this pipeline;
  - used timing proxies: `perf_backward_sec_per_batch` remained stable (`~0.094–0.100s`), optimizer step stable (`~0.0048–0.0059s`), and throughput improved (`train_samples_per_sec 134.4 -> 145.7`).
- SLLQHLIGL probe trajectory:
  - `HLA-A*02:01` KD: `51960 nM -> 167.5 nM` (about `310x` tighter);
  - `HLA-A*24:02` KD: `52028 nM -> 167.7 nM` (about `310x` tighter);
  - inter-allele separation remained near zero across all epochs (`delta_log10` around `0`, final `-0.000408`; final KD delta `-0.158 nM`).
- pMHC flow signal:
  - MHC-shuffle effect remained dominant (`~0.95–1.01` normalized),
  - peptide and interaction terms increased by epoch 6 (`peptide_norm 0.0097`, `interaction_norm 0.0088`, `interaction_ratio 0.0090`), but are still much smaller than the MHC term.

---

# Modal 1-Epoch Trajectory Re-Run (2026-03-05)

## Spec

Goal: run a fresh 1-epoch Modal unified training pass and repeat the same analysis slices (losses, latent/output diagnostics, gradient proxies, SLLQHLIGL probe metrics) on this 1-epoch run.

## Plan

- [x] Launch 1-epoch Modal unified run with the same diagnostics enabled.
- [x] Pull run artifacts (`metrics.csv`, `probe_affinity_over_epochs.csv`) from Modal volume.
- [x] Recompute loss/latent/perf/probe tables from this new run only.
- [x] Summarize findings and compare against prior 6-epoch behavior.

## Review

- Successful run: `traj-1ep-20260305T115512-filt` (`--epochs 1`, `--max-batches 80`, `--max-val-batches 20`, `--filter-unresolved-mhc`).
- Artifacts downloaded to:
  - `modal_runs/traj-1ep-20260305T115512-filt/traj-1ep-20260305T115512-filt/metrics.csv`
  - `modal_runs/traj-1ep-20260305T115512-filt/traj-1ep-20260305T115512-filt/probe_affinity_over_epochs.csv`
  - `modal_runs/traj-1ep-20260305T115512-filt/traj-1ep-20260305T115512-filt/probe_affinity_over_epochs.png`
- 1-epoch metrics snapshot:
  - loss: `train=0.7989`, `val=0.3451`; binding-loss `train=3.6782`, `val=1.4399`; presentation-loss `train=0.4006`, `val=0.5989`.
  - perf proxies: wait `0.262s/batch` (34.1%), compute `0.297s/batch`, backward `0.168s/batch` (21.8%), optimizer `0.010s/batch` (1.3%), throughput `83.17 samples/s`.
  - latent/output diagnostics: `diag_pmhc_vec_feature_var_mean=0.0877`, `diag_binding_logit_var=0.00191`, `diag_presentation_logit_var=7.3963`, `diag_latent_presentation_mixed_norm_mean=11.0755`.
  - pMHC flow: `mhc_norm=0.9125`, `peptide_norm=0.00460`, `interaction_norm=0.00553`, `interaction_ratio=0.00642`.
  - probe: `SLLQHLIGL + A*02:01 KD=85597.3 nM`, `SLLQHLIGL + A*24:02 KD=85590.9 nM`; separation remains effectively zero (`delta_log10=3.24e-05`, `delta_binding_prob=-1.56e-07`).
- Comparison to prior 6-epoch run:
  - this 1-epoch run is early-training only and remains in weak-binding regime for both probe alleles;
  - like the 6-epoch run, allele discrimination is still absent at this stage.

---

# Full-Data 1-Epoch Preflight (2026-03-05)

## Spec

Goal: before launching a true full-data 1-epoch Modal run, estimate:
- expected mini-batch count,
- whether epoch sampling uses all real + synthetic rows,
- per-mini-batch sample-type balance under the run configuration.

## Plan

- [ ] Compute full-profile post-augmentation sample counts (real + synthetic + augmentation) with the exact train_iedb logic.
- [ ] Evaluate epoch coverage semantics for balanced vs non-balanced loaders and pick the setting that guarantees full data usage.
- [ ] Produce per-mini-batch sample-type balance summary (expected and empirical) for the chosen setting.
- [ ] Confirm final launch command for the full-data 1-epoch Modal run and proceed.

## Review

- Probe input path is correct:
  - `HLA-A*02:01` and `HLA-A*24:02` resolve to distinct 365 aa representatives and distinct groove-half token streams.
  - class-I groove halves differ at `11` positions in half 1 and `9` positions in half 2 for the control probe path.
- Minimal architecture sanity:
  - a 2-sample binding-only fit (`SLLQHLIGL` against A0201 vs A2402 with opposite KD targets) does overfit cleanly on the current multi-query binding architecture.
  - it does **not** overfit with `binding_n_queries=1`, so the multi-query binding path is carrying real signal.
  - `use_pmhc_interaction_block=True` is not the key determinant on this toy; query multiplicity matters more.
- Binding-attention diagnosis:
  - at initialization, binding attention mass is mostly on MHC (`~0.93`) but spread nearly uniformly over all `184` groove tokens (`effective residues ~183`), so the model begins with almost no pocket focus.
  - a strong explicit MHC-attention sparsity prior can force attention to sharpen, but in toy fits it also collapsed allele separation if applied too aggressively.
- Training-control bugs found:
  - `canary` profile was using `cap_sampling='head'`; in the resulting probe slice, `HLA-A*24:02` had `0` binding samples and `0` elution samples, so the earlier A0201/A2402 comparison was not meaningful.
  - profile presets could overwrite explicit CLI flags if the user-provided value matched the full-profile default.
  - helper gene inference for auxiliary labels crashed on coarse shorthands like `HLA-DR*03`.
- Fixes implemented:
  - `infer_gene(...)` now degrades to heuristic gene extraction when `mhcgnomes` cannot parse a coarse shorthand, instead of crashing training.
  - `canary` profile now defaults to `reservoir` sampling.
  - explicit CLI options are now preserved when profile presets are applied.
  - `scripts/sanity_check_modal.py` now uses `reservoir` sampling.
  - `scripts/probe_training.py` no longer applies the overly strong MHC-attention sparsity prior.
- Verification:
  - `pytest -q tests/test_allele_resolver.py tests/test_train_iedb.py` -> `86 passed`
  - `python -m py_compile scripts/train_iedb.py scripts/sanity_check_modal.py scripts/probe_training.py data/allele_resolver.py tests/test_train_iedb.py tests/test_allele_resolver.py`
  - `python scripts/probe_training.py --batches 1 --batch-size 64` completed successfully after the fallback fix.
- Short local reservoir diagnostic (30 train batches, 8 val batches, explicit merged/index inputs):
  - without uncertainty weighting:
    - `train_loss=1.4216`
    - `val_loss=0.3761`
    - probe:
      - `A0201 KD≈762.7 nM`, `bind≈0.3719`
      - `A2402 KD≈795.7 nM`, `bind≈0.3597`
  - with uncertainty weighting:
    - `train_loss=1.4197`
    - `val_loss=0.3761`
    - probe:
      - `A0201 KD≈766.8 nM`, `bind≈0.3704`
      - `A2402 KD≈800.1 nM`, `bind≈0.3582`
- Interpretation:
  - representative sampling was the main reason the earlier canary pointed the probe in the wrong direction.
  - uncertainty weighting is not the primary bottleneck on this short run.
  - the corrected setup now gives the right sign (`A0201 > A2402`) but the separation is still too small, so the model is improving but not yet where it needs to be.

---

# Merge Enrichment + Funnel Artifacts (2026-03-05)

## Spec

Goal: extend `presto data merge` so merged output includes cell-level HLA context for cellular assays, requires MS rows to have resolvable allele-set context, and emits funnel artifacts that show per-assay filtering stages into the unified dataset.

## Plan

- [x] Add cell-context fields to unified records/TSV schema and parse ligand-cell metadata from IEDB/CEDAR ligand streams.
- [x] Build `cell_context -> allele_set` map from detected elution-like ligand rows and annotate merged records with `cell_hla_allele_set`.
- [x] Filter out MS/elution rows lacking resolvable allele-set context and report retained/dropped counts and fractions.
- [x] Emit merge funnel artifacts (TSV + PNG) summarizing per-assay raw→loaded→deduped→post-filter counts.
- [x] Add/update tests for schema fields, annotation/filter behavior, and funnel artifact generation.
- [x] Run focused test suite for merge/dedup path.

## Review

- Implemented in `data/cross_source_dedup.py` with new unified fields (`cell_hla_allele_set`, `cell_hla_n_alleles`), ligand APC parsing, lookup/annotation/filter stats, and funnel artifact emission (`*_funnel.tsv`, `*_funnel.png` when matplotlib exists).
- Added/updated tests in `tests/test_cross_source_dedup.py`; focused suites passed (`tests/test_cross_source_dedup.py`, `tests/test_data_cli.py`, `tests/test_train_iedb.py`).

---

# Merge Streamlining + Dedup/Allele-Set Optimization (2026-03-05)

## Spec

Goal: streamline `presto data merge` runtime by optimizing the expensive dedup and elution cell-HLA allele-set stages, while adding high-visibility progress instrumentation (tqdm + detailed stage logging) directly in the unification path.

## Plan

- [x] Profile/trim hot loops in cross-source dedup logic while preserving dedup semantics.
- [x] Optimize cell-HLA allele-set generation with caching and a single-pass annotate+filter path.
- [x] Add generous per-stage logging (counts, rates, elapsed time) and tqdm progress bars for long-running merge stages.
- [x] Keep funnel artifact generation in the streamlined path and surface outputs clearly in logs/stats.
- [x] Add/adjust tests for optimized helpers and instrumentation-compatible behavior.
- [x] Run focused merge-related tests and a local merge smoke benchmark to validate speed and correctness.

## Review

- Core optimizations:
  - dedup now partitions by sample-signature buckets and uses indexed candidate matching per bucket (`pmid`, `doi`, no-ref payload), avoiding worst-case pairwise scans;
  - assay bucket classification now caches per record (`_assay_bucket_cache`) to reduce repeated string parsing;
  - allele-set generation now uses cached cell-context normalization and cached informative-allele tokenization;
  - replaced two passes (`annotate` + `filter`) with a single pass helper (`_annotate_and_filter_cell_hla`) in `deduplicate_all`.
- Instrumentation added:
  - tqdm bars for grouping, dedup groups, cell-HLA lookup, annotate+filter;
  - detailed stage timing/rate logging and `stats["timing_sec"]` output.
- Validation:
  - `pytest -q tests/test_cross_source_dedup.py tests/test_data_cli.py tests/test_train_iedb.py` -> `59 passed`.
  - full local merge smoke benchmark completed on full local data set and emitted stage timings:
    - `load=886.73s`, `dedup=727.28s`, `cell_hla_lookup=76.06s`, `cell_hla_annotate_filter=65.17s`, `write_tsv=142.47s`, `write_assay_csvs=137.42s`, `total=2059.64s`.
    - final elution retention with resolvable cell-HLA set: `3441254 / 4091824` (`84.10%`).

---

# Merge CLI Output Defaults + Artifact Cleanup (2026-03-05)

## Spec

Goal: switch `presto data merge` so per-assay CSV materialization is opt-in (`--per-assay-csv`) instead of opt-out, remove stale generated assay artifacts, and audit why `cell_hla_allele_set` dominates merged text volume with concrete value examples.

## Plan

- [x] Replace `--no-assay-csv` with `--per-assay-csv` and make no-assay output the default in CLI wiring.
- [x] Update merge command implementation to respect the new flag semantics.
- [x] Update parser/CLI/docs tests and user docs for the new default/flag.
- [x] Remove stale generated per-assay artifacts from recent local runs.
- [x] Extract example values for `cell_hla_allele_set` and `reference_text` (including longest rows) and summarize root cause of size skew.
- [x] Run focused CLI + merge tests.

## Review

- CLI changes:
  - `cli/main.py`: removed `--no-assay-csv`; added opt-in `--per-assay-csv`.
  - `cli/data.py`: default now writes only merged TSV; per-assay CSVs only when `args.per_assay_csv` is true.
- Tests/docs:
  - `tests/test_data_cli.py` updated for new flag semantics and added default-no-assay regression.
  - `docs/cli.md` and `docs/training_spec.md` updated to state per-assay CSV output is optional/opt-in.
- Artifact cleanup completed:
  - removed stale `data/merged_assays` and local scratch outputs from `/tmp` (`merged_assays_opt`, `merged_deduped_opt*.{tsv,png}`, CSV microbench outputs).
- Field-value audit summary:
  - `cell_hla_allele_set` can become very long because it stores the **union** of alleles for broad cell-context labels (e.g., `B cell`, `PBMC`, `splenocyte`) across many studies; one mapping example reached `len=2699` (`n=240` alleles).
  - `reference_text` is long but less dominant (longest seen around `len=2368` in this corpus).
  - sample `cell_hla_allele_set` values from artifacts include short singletons (e.g., `HLA-DR`) and very large semicolon-delimited unions for generic contexts.
- Follow-up fix:
  - tightened allele-set token filtering so `cell_hla_allele_set` only keeps allele-like identifiers (canonical `*` alleles and murine `H2-*` shorthand), excluding generic tokens like `HLA class I`, `H2 class II`, `MHC class I`, serotype/mutant-style labels.
  - switched lookup from broad `cell_context` unions to strict `(pmid, cell_context)` keys to avoid cross-study allele inflation.
  - single-pass cellular handling now keeps cellular rows only when an allele set is truly known (direct informative `mhc_allele` or strict lookup hit).
  - added tests in `tests/test_cross_source_dedup.py` for the stricter token contract and mixed-token lookup behavior.
- Verification:
  - `pytest -q tests/test_cross_source_dedup.py tests/test_data_cli.py tests/test_train_cli.py` -> `49 passed`.

---

# Receptor-Free Canonical Path + Training Sanity (2026-03-06)

## Spec

Goal: finish the receptor-sequence removal on the canonical runtime path, keep pMHC-only `tcr_evidence` supervision, verify the resulting stack still trains, and surface the next real bottlenecks before a longer Modal run.

## Plan

- [x] Remove receptor tensors from canonical `PrestoSample` / `PrestoBatch`, predictor APIs, trainer entrypoints, and default task registry.
- [x] Keep pMHC-only `tcr_evidence` targets and method-panel supervision wired through loader, collator, model outputs, and loss computation.
- [x] Remove 10x receptor supervision from canonical train/CLI paths and convert merged `tcr` rows to `tcr_evidence`.
- [x] Rewrite stale tests around the receptor-free contract and verify the focused suite.
- [x] Run a real local `train_iedb` no-AMP canary and fix the first runtime regressions it exposes.
- [ ] Finish the remote Modal diagnostic and inspect probe separation / epoch trajectory before deciding on a longer run.

## Review

- Canonical runtime is now receptor-free:
  - `data/collate.py`, `data/loaders.py`, `models/presto.py`, `training/trainer.py`, `inference/predictor.py`, `cli/main.py`, `cli/predict.py`, `cli/evaluate.py`, `scripts/train_synthetic.py`, `scripts/train_iedb.py`
  - `tcr_a_tok` / `tcr_b_tok`, chain aux labels, TCR matcher outputs, and `predict_chain` / `embed_tcr` entrypoints are gone from the active path.
- `tcr_evidence` remains as pMHC-only supervision:
  - VDJdb rows now feed `tcr_evidence_label` + 3-bin method panel targets.
  - merged `record_type="tcr"` now classifies as `tcr_evidence` rather than `tcr_pmhc`.
- Canonical training/CLI no longer advertises or loads 10x receptor-sequence supervision.
- Focused verification passed:
  - `pytest -q tests/test_loaders.py tests/test_tasks.py tests/test_trainer.py tests/test_train_iedb.py tests/test_train_cli.py tests/test_cross_source_dedup.py tests/test_collate.py tests/test_predictor.py tests/test_presto.py tests/test_training_e2e.py`
  - result: `271 passed`
- Two real runtime bugs were found and fixed by an actual local training run:
  - merged TSV loading was using strict `infer_mhc_class(...)` on coarse aliases like `HLA-DR1`; now it uses the optional path.
  - DR beta-only rows correctly defaulted to `DRA` at dataset assembly, but the index preload did not include that synthetic/default alpha chain; `_collect_unique_alleles(...)` now adds default `DRA` alleles for DRB inputs.
- Local no-AMP canary on the canonical `train_iedb` path completed successfully:
  - config: `profile=canary`, `batch_size=32`, `mhc_augmentation=0`, modest caps on each modality
  - dataset after filtering/augmentation: `652` samples, `17` train batches, `5` val batches
  - epoch result: `train_loss=1.7780`, `val_loss=1.3381`
  - probe after 1 epoch:
    - `SLLQHLIGL + HLA-A*02:01`: `KD≈3496 nM`, `bind≈0.0822`
    - `SLLQHLIGL + HLA-A*24:02`: `KD≈3306 nM`, `bind≈0.0876`
  - interpretation:
    - training is numerically healthy and the refactored stack is genuinely learning
    - allele discrimination for this probe is still not correct after a tiny local canary
- Modal findings so far:
  - a true full-data 1-epoch run starts cleanly now, but preprocessing is dominated by full-data synthetic-negative expansion and resolved-only filtering (`1.71M` elution rows dropped, `217k` T-cell rows dropped, `1.07M` synthetic elution negatives added), which makes it a poor interactive diagnostic loop.
  - switched to the smaller dedicated Modal sanity diagnostic to get a faster answer on `SLLQHLIGL` allele separation.

---

# Rapid Probe Triage + Throughput Audit (2026-03-07)

## Spec

Goal: get `SLLQHLIGL` to separate correctly between `HLA-A*02:01` and `HLA-A*24:02` as quickly as possible, while also finding the largest training-throughput bugs and determining the safest numeric mode for GPU training.

## Plan

- [ ] Verify the raw biological inputs end-to-end for the probe pair:
  - resolved allele representative chosen for `HLA-A*02:01` and `HLA-A*24:02`
  - extracted groove halves and tokenization
  - number/location of amino-acid differences reaching the model
- [ ] Run micro-overfit checks that isolate architecture vs. optimizer/data issues:
  - 2-allele probe-only overfit
  - tiny real-binding subset overfit
  - compare no-MHC / swapped-MHC / probe-paired logits and gradients
- [ ] Inspect internal signal flow for the probe and for real batches:
  - MHC encoder similarity for A0201 vs A2402
  - `pmhc_vec`, binding latents, and gradient norms by module/head
  - confirm MHC changes measurably perturb binding outputs early in training
- [ ] Audit real training composition and supervision pressure:
  - per-batch task counts and positive/negative balance
  - per-task loss magnitudes and gradient contributions
  - binding-label distribution by allele and class
- [ ] Enumerate numeric-mode and performance hypotheses:
  - bf16 vs fp32 safety on current code paths
  - whether fp16 is plausible at all
  - data-loader / preprocessing hotspots
  - synthetic-negative materialization cost
- [ ] Implement the highest-leverage fixes for:
  - faster diagnosis/training iteration
  - stronger early binding/MHC discrimination
  - numeric stability when using AMP
- [ ] Re-run short local and Modal diagnostics after each fix cluster until the probe is clearly moving in the right direction.

## Review

- In progress.
# Binding Specificity Recovery: Peptide Negatives + 50k Ceiling + A0201/A2402 Subset (2026-03-07)

## Spec

- Goal: strengthen peptide-specific binding learning quickly enough that short diagnostics stop collapsing into broad MHC priors, while keeping allele/class/species diversity intact in every batch.
- Primary questions:
  - does adding same-allele / different-peptide pressure improve `SLLQHLIGL` separation for `HLA-A*02:01` vs `HLA-A*24:02`?
  - does shifting the weak-binding ceiling from `100k nM` to `50k nM` make synthetic and censored weak binders better aligned with common baselines like MHCflurry?
  - can a larger binding-only A0201/A2402 subset show the architecture learning the right direction when trained on enough directly relevant data?
- Constraints:
  - keep the recently added diversity-aware batch sampler guarantees across allele, class, and species
  - prefer minimally coupled changes that can be verified independently
  - do not trust tiny or head-capped slices for this diagnostic

## Execution order

- [x] Phase 1: weak-binding target calibration + protected peptide scrambles
  - change the shared maximum affinity ceiling from `100k nM` to `50k nM`
  - make synthetic binding negatives default to fixed `50k nM` targets instead of sampling over a wide weak range
  - make peptide-scramble negatives force changes at peptide positions 1, 2, and the last residue
  - update any tests and target normalization assumptions that depend on the old ceiling
- [x] Phase 1 verification
  - affinity conversion tests and synthetic-negative tests pass
  - local diagnostics remain finite and numerically stable
- [x] Phase 2: focused A0201/A2402 binding diagnostic subset
  - add a helper that builds a larger binding-only subset centered on `HLA-A*02:01` and `HLA-A*24:02`
  - include quantitative binding rows for those alleles plus the existing synthetic negatives path
  - run a focused training sanity check and track the `SLLQHLIGL` probe trajectory
- [x] Phase 2 verification
  - subset summary reports sample counts, peptides, allele support, and same-peptide overlap
  - short training run completes and is analyzed against the probe
- [ ] Phase 3: decide whether scale-first is sufficient before adding another peptide-specific ranking loss
  - if the focused larger subset still fails to learn the probe direction, add a same-allele / different-peptide objective
  - otherwise prefer scale and cleanup over another new loss term

## Review

- Implemented:
  - shared affinity ceiling reduced to `50k nM`
  - synthetic binding negatives now default to fixed `50k nM`
  - protected peptide scramble for synthetic negatives now forces anchor changes at `P1`, `P2`, and `PΩ`
  - added `load_binding_records_for_alleles_from_merged_tsv(...)` for focused binding diagnostics
  - removed dead canonical-model modules from `Presto` and added checkpoint-time dropping of their legacy keys
- Verification:
  - `pytest -q tests/test_loaders.py tests/test_affinity.py tests/test_checkpointing.py tests/test_train_iedb.py tests/test_presto.py tests/test_train_synthetic.py tests/test_train_cli.py tests/test_data_cli.py`
  - result: `192 passed`
- Focused A0201/A2402 IEDB-only binding run:
  - real quantitative rows: `19,597`
  - selected panel rows before source filter helper: `21,568`
  - unique peptides:
    - `A*02:01`: `11,657`
    - `A*24:02`: `2,615`
  - shared peptides: `734`
  - with synthetic binding negatives the dataset reached `48,528` samples
  - 3-epoch CPU run on the focused subset produced the correct `SLLQHLIGL` sign:
    - epoch 1: `A0201 ≈ 192.98 nM`, `A2402 ≈ 194.38 nM`
    - epoch 2: `A0201 ≈ 280.17 nM`, `A2402 ≈ 280.31 nM`
    - epoch 3: `A0201 ≈ 260.19 nM`, `A2402 ≈ 260.62 nM`
  - interpretation:
    - the larger relevant dataset is enough to recover the correct direction
    - the gap is still tiny, so scale helps but peptide-specific discrimination is still weak
    - this argues for `scale-first` before adding another new binding objective

# MHC Warm-Start + Exact-IC50 Focused Binding (2026-03-08)

## Spec

- Goal: execute the new staged plan end to end:
  - one epoch of MHC-only warm-start pretraining on indexed groove sequences
  - warm-start the focused `HLA-A*02:01` vs `HLA-A*24:02` exact-IC50 affinity run from that checkpoint
  - measure whether the warm start improves real `IC50_nM` learning and probe separation without adding synthetic assumptions yet
- Constraints:
  - use all exact `IC50` rows for `HLA-A*02:01` and `HLA-A*24:02`
  - keep strict per-batch allele balance in the focused run
  - use `IC50_nM` as the primary supervised target
  - log `IC50_nM` directly in probe artifacts so training target and diagnostics match
  - keep this step synthetic-free and ranking-free unless a verification failure requires a correction

## Execution order

- [ ] Phase 1: verify and harden the new warm-start code path
  - confirm `forward_mhc_only(...)` works on the shared encoder path
  - confirm `scripts/pretrain_mhc_encoder.py` builds valid group-balanced datasets and saves a compatible checkpoint
  - confirm `scripts/focused_binding_probe.py` can warm-start from a checkpoint and logs `IC50_nM` in probe artifacts
- [ ] Phase 2: add/adjust targeted tests
  - add a `forward_mhc_only(...)` smoke test
  - add focused probe regression coverage for exact-only filtering / warm-start load path / IC50 artifact logging where practical
- [ ] Phase 3: local verification
  - run `py_compile` on the touched scripts/model files
  - run targeted `pytest`
  - run a small local MHC pretrain smoke
  - run a small local warm-started focused exact-IC50 smoke
- [ ] Phase 4: Modal execution
  - run one-epoch full MHC pretrain on Modal
  - run the full exact-IC50 focused A0201/A2402 experiment with that checkpoint as initialization
- [ ] Phase 5: analysis
  - compare against the prior non-warm-start exact-IC50 baseline
  - inspect whether `SLLQHLIGL`, `FLRYLLFGI`, and `NFLIKFLLI` move in the right direction
  - decide whether the next step is multi-allele expansion or more binding-specific objective changes

## Review

- Verification completed:
  - `python -m py_compile models/presto.py scripts/focused_binding_probe.py scripts/pretrain_mhc_encoder.py scripts/train_modal.py tests/test_presto.py tests/test_focused_probe.py tests/test_pretrain_mhc_encoder.py`
  - `pytest -q tests/test_pretrain_mhc_encoder.py tests/test_presto.py tests/test_focused_probe.py tests/test_train_iedb.py` -> `118 passed`
  - local MHC pretrain smoke ran clean and emitted a compatible checkpoint
  - local warm-started focused exact-IC50 smoke ran clean and emitted `ic50_nM` probe artifacts
- Modal MHC warm-start run:
  - run id: `mhc-pretrain-20260308b`
  - one epoch on the full indexed groove corpus
  - final metrics:
    - `train_loss=0.1094`
    - `val_loss=0.0716`
    - `train_class_acc=0.9987`
    - `val_class_acc=1.0000`
    - `train_species_acc=0.9273`
    - `val_species_acc=0.9706`
- Modal focused exact-IC50 warm-start run:
  - run id: `a0201-a2402-ic50-exact-allrows-ic50only-warmstart-20260308i`
  - contract:
    - all exact `IC50` rows for `HLA-A*02:01` and `HLA-A*24:02`
    - no synthetic negatives
    - no ranking losses
    - strict per-batch allele balance
    - trained on `IC50_nM`
    - initialized from `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
  - result vs prior non-warm-start baseline `a0201-a2402-ic50-exact-allrows-ic50only-20260308h`:
    - best val loss improved `0.9378 -> 0.9097`
    - final val loss improved `1.0156 -> 0.9420`
    - final train loss improved `0.5703 -> 0.5271`
  - final epoch 12 `IC50_nM` probes:
    - `SLLQHLIGL`: `A*02:01 ≈ 17.0 nM`, `A*24:02 ≈ 2333.9 nM`
    - `FLRYLLFGI`: `A*02:01 ≈ 53.3 nM`, `A*24:02 ≈ 588.3 nM`
    - `NFLIKFLLI`: `A*02:01 ≈ 3820.0 nM`, `A*24:02 ≈ 21.4 nM`
  - interpretation:
    - the MHC warm start materially improved the focused affinity run
    - the focused binding predictor now learns the expected direction for the key A0201-vs-A2402 probes on the real `IC50` output
    - `SLLQHLIGL` remains too optimistic for `A*24:02` versus the desired `>10,000 nM`, but it is now in the correct qualitative regime and far better separated than before
- Bug found and fixed during execution:
  - the MHC warm-start builder was admitting groove halves with invalid `?` tokens and crashing tokenization on Modal
  - fixed by validating extracted groove segments against the same allowed-AA contract before sample construction

# Warm-Start Trajectory + Multi-Allele + Incremental Mix-In (2026-03-08)

## Spec

- Goal: extend the successful warm-started exact-IC50 A0201/A2402 run in a controlled sequence:
  - first, make the per-epoch `SLLQHLIGL` `IC50_nM` trajectory explicit and plot it
  - second, test whether the improvement survives expansion to a small motif-diverse class-I panel
  - third, add one extra element from the older fuller training design at a time and measure whether it helps or hurts
- Constraints:
  - keep the clean warm-start exact-IC50 A0201/A2402 run as the reference baseline
  - do not add more than one extra training element per ablation
  - judge comparisons on the real `IC50_nM` output, not only KD/probe internals
  - keep reporting at least `SLLQHLIGL`, `FLRYLLFGI`, and `NFLIKFLLI`

## Execution order

- [ ] Phase 1: trajectory extraction and visualization
  - extract per-epoch `IC50_nM` for `SLLQHLIGL` on `HLA-A*02:01` and `HLA-A*24:02`
  - generate a compact plot and save it with the experiment artifacts
- [ ] Phase 2: multi-allele exact-IC50 panel
  - run a warm-started exact-IC50 experiment on a small class-I panel with solid data support
  - compare validation loss and tracked probe behavior against the A0201/A2402 baseline
- [ ] Phase 3: one-at-a-time old-design mix-ins
  - start from the clean warm-start exact-IC50 baseline
  - add one extra element per run, likely in this order:
    - same-allele peptide ranking
    - same-peptide allele ranking
    - synthetic negatives
    - broader direct-affinity assay mix beyond exact IC50 if still needed
  - compare each run directly against the clean baseline and the multi-allele baseline

## Review

- Phase 1 completed:
  - extracted per-epoch `IC50_nM` probe values for `SLLQHLIGL` from `a0201-a2402-ic50-exact-allrows-ic50only-warmstart-20260308i`
  - generated plot: `modal_runs/a0201-a2402-ic50-exact-allrows-ic50only-warmstart-20260308i/sllqhligl_ic50_over_epochs.png`
- Phase 2 completed:
  - warm-started multi-allele exact-IC50 panel run `class1-panel-ic50-exact-warmstart-20260308a`
  - adding motif-diverse class-I alleles improved biologic separation on tracked probes, especially `SLLQHLIGL` and `FLRYLLFGI`, though aggregate validation loss was worse than the 2-allele baseline
- Phase 3 completed for three one-factor mix-ins:
  - peptide-ranking only: improved tracked-probe biologic separation but worsened aggregate validation loss
  - allele-ranking only: harmful; degraded `NFLIKFLLI` and gave the worst validation loss
  - synthetic-negatives only: strongest overall one-factor addition so far, improving both tracked-probe separation and best validation loss
- Current best ingredients on exact `IC50_nM` binding are:
  - MHC warm start
  - strict per-batch allele balance
  - no allele-ranking term
  - synthetic negatives as the first extra mix-in to keep
- Recommended next run sequence:
  - combine multi-allele panel + warm start + synthetic negatives, still with allele-ranking off
  - if that holds, compare with adding peptide-ranking on top as the next single additional factor

# Class-I Quantitative Affinity Expansion (2026-03-08)

## Spec

- Goal: extend the successful warm-started A0201/A2402 exact-IC50 binding setup into a binding-only class-I quantitative affinity trainer over the full merged class-I corpus.
- Required data contract:
  - include all class-I quantitative affinity rows from `merged_deduped.tsv`
  - keep distinct supervised outputs for `KD_nM`, `IC50_nM`, and `EC50_nM`
  - drop same-peptide allele-ranking entirely
  - allow censored quantitative rows through the censor-aware loss
  - preserve allele diversity at batch time instead of row-capping the dataset
- Required model contract:
  - affinity outputs must be conditionable on binding assay metadata, not just pMHC features
  - carry binding assay metadata through `BindingRecord -> PrestoSample -> PrestoBatch -> forward_affinity_only`
  - use assay/context embeddings on the affinity path so residual assay heads can adapt by assay type / method / conditions
- Current corpus sizing:
  - class-I quantitative binding rows in merged TSV: `150,743`
  - source: all `IEDB`
  - qualifiers: `103,644 exact`, `47,099 greater-than`
  - measurement mix:
    - `KD (~IC50)`: `64,696`
    - `KD (~EC50)`: `38,072`
    - `IC50`: `27,616`
    - `KD`: `18,962`
    - `EC50`: `1,397`
- Success criteria:
  - training runs cleanly on Modal with no row cap
  - tracked probe biology stays correct on `SLLQHLIGL`, `FLRYLLFGI`, `NFLIKFLLI`
  - aggregate validation on the broader class-I quantitative task is stable
  - if it fails, produce a concrete recovery plan tied to observed diagnostics

## Execution order

- [ ] Phase 1: add binding assay-context plumbing
  - extend `BindingRecord` to retain assay metadata needed for affinity context
  - extend `PrestoSample` / `PrestoBatch` and collate binding assay context tensors
  - keep the context contract parallel to the full model style: assay type / method / culture-like conditions
- [ ] Phase 2: add affinity assay-context embeddings
  - add a small affinity-context encoder in the affinity path
  - use it in KD bias and IC50/EC50 residual heads without changing the public output names
  - keep the path optional so old checkpoints can still load or remap cleanly
- [ ] Phase 3: generalize the focused binding trainer to class-I quantitative panels
  - load all class-I quantitative rows from the merged TSV
  - support batch-balanced training across many alleles without dropping most of the dataset
  - keep `binding_contrastive_weight=0`
  - keep distinct `KD/IC50/EC50` supervision active
- [ ] Phase 4: local verification
  - add targeted tests for new binding context collation and affinity-context forward path
  - run focused/unit tests covering the new binding-only class-I mode
- [ ] Phase 5: Modal experiment
  - run MHC warm start if needed
  - run the class-I quantitative affinity trainer on Modal with no row cap and no allele-ranking
  - save summaries and probe plots
- [ ] Phase 6: analysis / recovery
  - compare against the two-allele baseline and the multi-allele exact-IC50 panel
  - if broad class-I quantitative training hurts probe behavior, write a recovery plan before further architectural changes
- [ ] Phase 7: class-II compatibility check
  - once the class-I quantitative path is stable, add class-II quantitative affinity rows with the same assay-context contract
  - evaluate whether joint class-I/class-II affinity training degrades the class-I probe panel
  - if class-II hurts class-I, decide between:
    - class-specific affinity heads over a shared groove trunk
    - staged pretraining / finetuning by class
    - separate batch balancing / curriculum by MHC class

## Review

- Phase 1 completed:
  - binding assay metadata now flows through `BindingRecord -> PrestoSample -> PrestoBatch -> Presto.forward_affinity_only`
- Phase 2 completed:
  - affinity path now consumes assay-context embeddings for assay-specific calibration
- Phase 3 completed locally:
  - focused trainer now supports `--train-all-alleles` with class filtering and global balanced batching
- Phase 4 completed:
  - targeted suites passed: `161 passed`
- Phase 5 in progress:
  - launched broad Modal run `class1-quant-affinity-warmstart-20260308b`
  - contract:
    - all class-I quantitative binding rows
    - warm-start init from `mhc-pretrain-20260308b`
    - assay-context affinity path enabled
    - `affinity_loss_mode=full`
    - no synthetic negatives
    - no allele-ranking
  - current artifact snapshot:
    - dataset size `150,395`
    - train size `150,380`
    - val size `15`
    - epoch 1 probe `IC50_nM`:
      - `SLLQHLIGL`: `A*02:01 992.96`, `A*24:02 859.54`
      - `NFLIKFLLI`: `A*02:01 2213.02`, `A*24:02 1993.73`
  - note:
    - this run is invalid for model judgement because `train_all_alleles` hit a peptide-split bug and collapsed validation to `15` rows
    - bug fixed by treating an empty allele panel as “use all observed alleles” inside `_split_records_by_peptide(...)`
    - local verification after the fix:
      - targeted tests: `18 passed`
      - broad smoke split on a `1971`-row sample: `1577 train / 394 val`, instead of `1970 / 1`
    - relaunched corrected Modal run: `class1-quant-affinity-warmstart-20260308c`
- Phase 7 early technical check completed:
  - class-II quantitative corpus size: `91,408`
  - local all-alleles class-II smoke with `max_records=2048` completed cleanly
  - post-fix split stats: `1604 train / 404 val`
  - one class-II affinity epoch ran end-to-end with no resolution, collation, or forward-path failure
  - remaining question is behavioral: whether joint class-I + class-II affinity training degrades the class-I probe panel

# Class-II Core/PFR Design Options (2026-03-08)

## Current state

- The active binding latent already uses a single uniform core enumerator for both classes:
  - fixed `core_window_size = 9`
  - contiguous window start enumeration
  - N- and C-terminal PFR summaries
  - class-conditioned prior features
- This is biologically closer to class II than class I:
  - class II: contiguous 9-mer core plus terminal PFRs is a good first-order model
  - class I: 8-11mer binders often occupy the full groove, and longer binders are better described by bulges / register changes than by terminal PFRs

## Options

- [ ] Option 1: keep separate class-specific binding modules
  - class I:
    - full-peptide / 8-11mer binding module
    - no explicit PFR model
  - class II:
    - contiguous 9-mer core + PFR enumerator
  - combination:
    - shared encoder trunk
    - class-specific affinity heads mixed by explicit `mhc_class` or inferred `class_probs`
  - pros:
    - lowest risk to class-I performance
    - easiest to debug
  - cons:
    - not a uniform mechanism
    - duplicates logic

- [ ] Option 2: uniform contiguous window enumerator with class-conditioned priors
  - enumerate `(start, core_len)` for both classes
  - class I:
    - allow `core_len in {8,9,10,11}`
    - strong prior toward covering the full peptide for 8-11mers
    - strong penalty on long terminal PFRs / partial cores
  - class II:
    - strong prior toward `core_len = 9`
    - variable terminal PFRs allowed
  - combination:
    - one shared candidate lattice and one shared posterior
    - separate class-I / class-II calibration heads downstream
  - pros:
    - simplest uniform extension from the current code
    - matches the user's desire to do core enumeration for class I too
  - cons:
    - still mis-specifies class-I bulged ligands as terminal-overhang cases

- [ ] Option 3: uniform groove-contact register model
  - represent binding as assignment of peptide positions to a fixed set of groove-contact slots
  - allow monotonic alignments with:
    - terminal flanks as PFRs
    - optional skipped peptide positions / insertions for bulges
  - class I:
    - priors favor near-full occupancy of groove slots with very short flanks
    - insertions/bulges allowed in central positions
  - class II:
    - priors favor contiguous 9-mer register with longer terminal flanks
  - combination:
    - one shared register lattice
    - class-specific priors and readout calibration
  - pros:
    - best biological unification
    - handles both class-II PFRs and class-I bulges in one formalism
  - cons:
    - biggest implementation jump
    - harder optimization

- [ ] Option 4: soft slot-attention / monotonic alignment
  - learn 9 groove slots that attend to peptide positions
  - add regularization for:
    - monotonic order
    - compactness
    - class-I full-occupancy vs class-II flank tolerance
  - combination:
    - one shared differentiable alignment mechanism
    - class-specific priors / penalties only
  - pros:
    - flexible and uniform
    - avoids discrete search
  - cons:
    - harder to interpret
    - easier for the model to learn degenerate soft alignments

- [ ] Option 5: hybrid proposal-refinement model
  - coarse uniform enumerator proposes registers/windows
  - class-specific refinement heads rescore candidates
  - combination:
    - shared proposal lattice
    - separate class-I / class-II refinement and calibration
  - pros:
    - pragmatic compromise
    - easier migration path from the current model
  - cons:
    - still partially duplicates class-specific logic

## Recommended direction

- Recommend `Option 5` as the near-term path:
  - keep a single shared candidate-generation mechanism
  - extend it from fixed `9`-mer windows to a richer class-aware register lattice
  - use separate class-I and class-II refinement/calibration heads
- If we need the fastest low-risk integration first, use `Option 2` temporarily:
  - variable `core_len`
  - strong class-I penalties on partial cores / long terminal PFRs
  - then upgrade to a true register model if class-I long-peptide behavior remains wrong

## Acceptance criteria for adding class II

- The class-I probe panel must not materially regress after joint training:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
- Class-II diagnostics should include:
  - DR-focused exact-IC50 probes with known distinct registers
  - posterior over core start / register, not just scalar affinity
- If joint class-I + class-II training harms class-I:
  - do not force one scalar affinity head
  - split calibration/refinement by class while keeping the groove trunk shared

## Benchmark handoff

- Detailed benchmark/spec file:
  - `tasks/class2_register_design_benchmark.md`
- Active benchmark matrix is now:
  - `M0 = G0 + R0 + C1`
  - `M1 = G1 + R1 + C1`
  - `M2 = G1 + R2 + C2`
  - `M3 = G3 + R3 + C2`
  - `M4 = G3 + R4 + C2`
  - `M5 = G2 + G4 + R4 + C2`
- Evaluation is staged:
  - Stage A: class-I preservation on the small motif-diverse class-I panel
  - Stage B: joint class-I + class-II quantitative affinity on Stage-A survivors only

## Immediate execution plan

- [ ] Phase 8: implement executable Stage-A variants
  - `M0 = G0 + R0 + C1`
  - `M1 = G1 + R1 + C1`
  - `M2 = G1 + R2 + C2`
  - defer `M3+` until the explicit slot/register mechanism exists
- [ ] Phase 9: add a benchmark harness
  - one script to launch and summarize Stage-A runs
  - one Modal sweep entrypoint
  - one local summarizer that updates the current best model as artifacts arrive
- [ ] Phase 10: run Stage-A Modal sweep
  - class-I exact-IC50 panel first
  - warm-start on
  - peptide-ranking on
  - allele-ranking off
  - no synthetic negatives initially
- [ ] Phase 11: summarize Stage-A results and pick survivors
  - rank by class-I preservation metrics and probe behavior
  - only then expand to Stage B with class-II quantitative affinity

## Stage-A sweep contract

- executable designs:
  - `M0 = groove_pos_mode=sequential, core_lengths=9, core_refinement=shared`
  - `M1 = groove_pos_mode=triple, core_lengths=8,9,10,11, core_refinement=shared`
  - `M2 = groove_pos_mode=triple, core_lengths=8,9,10,11, core_refinement=class_specific`
- seeds:
  - `41`
  - `42`
  - `43`
- class-I exact-IC50 panel:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `HLA-A*03:01`
  - `HLA-A*11:01`
  - `HLA-A*01:01`
  - `HLA-B*07:02`
  - `HLA-B*44:02`
- warm start:
  - `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
- probes:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
- fixed recipe:
  - `measurement_profile=direct_affinity_only`
  - `measurement_type_filter=ic50`
  - `qualifier_filter=exact`
  - `affinity_loss_mode=full`
  - `binding_peptide_contrastive_weight=0.5`
  - `binding_contrastive_weight=0.0`
  - `synthetic_negatives=false`
  - `batch_size=128`
  - `epochs=12`

## Stage-A review criteria

- best checkpoint is the epoch with minimum validation loss
- primary ranking metric:
  - probe ordering count with `>=1.5x` separation
- secondary:
  - average log-ratio margin across the three probes
- tertiary:
  - best validation loss

## Stage-A review

- `M1` and `M2` both completed all `3` seeds on Modal and both achieved `3/3`
  correct probe orderings with `>=1.5x` separation at their best checkpoints.
- Current completed ranking:
  - `M1`:
    - mean best val loss `1.2977`
    - mean probe log10 margin `1.799`
    - best run `register-stagea-20260308b-m1-seed42`
  - `M2`:
    - mean best val loss `1.2591`
    - mean probe log10 margin `1.684`
    - best run `register-stagea-20260308b-m2-seed43`
- Interpretation:
  - `M1` is the current winner for class-I preservation because it produces
    stronger biologic probe separation while staying close on validation loss.
  - `M2` remains a viable alternate because its validation loss is slightly
    better on average.
- `M0` baseline is still incomplete:
  - a direct debug run (`register-m0-debug-20260308`) produced valid 1-epoch
    artifacts and reached `2/3` correct probe orderings
  - the detached 12-epoch baseline launches did not publish checkpoint
    artifacts, so the baseline row is still provisional
- Published summary table:
  - `modal_runs/register_design_stage_a/options_vs_perf.md`
- Durable experiment ledger:
  - `tasks/experiment_log.md`
- Follow-up affinity experiment plan:
  - `tasks/affinity_followup_plan.md`

## Stage-B immediate contract

- entrants:
  - `M1`
  - `M2`
- objective:
  - test whether adding class-II quantitative affinity breaks class-I probe
    behavior
- recipe:
  - `train_all_alleles=true`
  - `train_mhc_class_filter=all`
  - `measurement_profile=direct_affinity_only`
  - `measurement_type_filter=ic50`
  - `qualifier_filter=exact`
  - same warm start, peptide ranking, and no synthetic negatives
- evaluation:
  - keep the class-I probe panel fixed:
    - `SLLQHLIGL`
    - `FLRYLLFGI`
    - `NFLIKFLLI`
  - compare against the Stage-A best checkpoints for class-I regression
  - if class-I probe ordering degrades materially, class-II joint training is
    not safe under that design

## Combined affinity ablation

- [ ] Phase 12: run the combined focused-affinity experiment the user asked for
  - 1 epoch MHC class/species warm start
  - synthetic negatives on
  - peptide ranking on
  - allele ranking off
  - quantitative affinity only for the main experiment
- [ ] Phase 13: make the run contract explicit in the review
  - specify whether this is the 2-allele or broader class-I panel
  - log the exact synthetic modes used
  - log whether the anchor-aware strategy is `none` or `property_opposite`
  - log the exact measurement profile / type filter / qualifier filter
- [ ] Phase 14: audit qualitative binding labels
  - count the qualitative label vocabulary in the merged corpus
  - determine whether labels are only binary or include ordinal levels like
    `Strong Positive > Positive > Weak Positive > Negative`
  - decide whether the current label space is suitable for pairwise ranking
- [ ] Phase 15: summarize whether qualitative binding should enter the focused
  affinity trainer
  - if the label vocabulary is mostly ordinal and consistent, propose a clean
    pairwise/ordinal contract
  - if the label space is noisy or mostly binary/free text, keep it out of the
    focused quantitative runs

## Combined affinity ablation review contract

- main experiment:
  - broader class-I panel if possible, otherwise explicit 2-allele fallback
  - warm start on
  - peptide ranking on
  - allele ranking off
  - synthetic negatives on
  - quantitative affinity supervision only
- report:
  - exact dataset slice
  - synthetic negative modes and counts
  - whether anchor-aware synthetic negatives were enabled
  - best checkpoint probe values for:
    - `SLLQHLIGL`
    - `FLRYLLFGI`
    - `NFLIKFLLI`
  - comparison against:
    - `E004` warm start + synth
    - `E006` broader class-I panel
    - `M1` Stage-A winner

## Combined affinity ablation review

- active run:
  - `class1-panel-ic50-exact-warmstart-synth-peprank-20260309a`
- launch contract:
  - 1 epoch MHC class/species warm start via
    `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
  - explicit class-I panel:
    - `HLA-A*02:01`
    - `HLA-A*24:02`
    - `HLA-A*03:01`
    - `HLA-A*11:01`
    - `HLA-A*01:01`
    - `HLA-B*07:02`
    - `HLA-B*44:02`
  - `measurement_profile=direct_affinity_only`
  - `measurement_type_filter=ic50`
  - `qualifier_filter=exact`
  - `affinity_loss_mode=full`
  - `synthetic_negatives=true`
  - `negative_ratio=1.0`
  - `class_i_anchor_strategy=property_opposite`
  - `binding_peptide_contrastive_weight=0.5`
  - `binding_contrastive_weight=0.0`

## Immediate model cleanup

- [ ] Phase 16: thread scalar affinity/stability scores into assay outputs
  - expose `binding_affinity_score` and `binding_stability_score` as explicit outputs
  - feed `binding_affinity_score` into the quantitative affinity assay path
  - feed `binding_stability_score` into the stability assay path
  - keep compatibility aliases for existing probe outputs
- [ ] Phase 17: verify direct-only score wiring
  - update focused/full forward tests for the new score outputs
  - confirm `forward_affinity_only()` and full `forward()` agree on the score tensors

## Score-to-assay review

- wiring verification:
  - `pytest -q tests/test_presto.py tests/test_checkpointing.py tests/test_focused_probe.py`
  - result: `73 passed`
- first implementation was too aggressive:
  - adding affinity score redundantly into multiple assay residual/bias paths regressed the old synthetic benchmark contract
- narrowed implementation:
  - keeps affinity score on the canonical mixed-KD route
  - keeps stability-score conditioning on stability assays
  - removes the extra duplicate score injection into KD-bias / assay-residual paths
- benchmark result:
  - `E012` (`E004`-like warm start + synth): failed, `IC50` outputs collapsed toward the weak tail
  - `E013` (`E006`-like warm start + broader class-I panel): passed, all three class-I probes remained correct-sign
- current launch point:
  - use the broader real-data class-I panel as the next ablation base
  - do not trust the current synthetic-negative recipe as a default path until it is redesigned
# Epoch-Refreshed Synthetic Negatives For Focused Affinity (2026-03-09)

## Spec

- Goal: fix the focused affinity synthetic-negative contract so synthetic rows do not become a static, over-reused calibration artifact.
- Root cause confirmed from artifact-backed runs:
  - `E004` and `E012` used the same real split and the same synthetic counts.
  - The failure was not a data split change; it was the interaction between:
    - one fixed synthetic pool generated before training
    - exact per-batch allele balancing
    - heavy reuse of the minority-allele synthetic rows across every epoch
- Required contract changes:
  - regenerate synthetic negatives at every epoch from the fixed real-train split
  - make the real:synthetic ratio explicit at batch construction time, not only via dataset-level `negative_ratio`
  - support synthetic-mode isolation (`peptide_scramble` only, `mhc_scramble` only, etc.) for ablations
  - audit groove preparation separately for real rows and each synthetic mode
- Constraints:
  - validation stays real-only
  - same peptide-family real train/val split as the current focused pipeline
  - preserve strict per-batch allele balance
  - do not silently change the broad trainer yet; fix and benchmark the focused path first

## Execution order

- [x] Phase 1: implement epoch-refreshed synthetic generation
  - add focused-run controls for:
    - `synthetic_refresh_each_epoch`
    - synthetic mode selection / subsets
    - explicit batch synthetic fraction
  - rebuild the train dataset and loader at each epoch from:
    - fixed real-train rows
    - freshly generated synthetic negatives using an epoch-dependent seed
- [x] Phase 2: implement explicit real:synth batch balancing
  - replace the focused allele-only sampler with a sampler that balances by:
    - target allele
    - real vs synthetic
  - keep exact target-allele slot balance per batch
  - make synthetic fraction deterministic per batch when synthetic rows are enabled
- [x] Phase 3: add synthetic-mode isolation support
  - allow running the focused trainer with:
    - all current synthetic modes
    - any single mode in isolation
    - selected subsets of modes
  - log the exact mode set and per-epoch synthetic counts into `summary.json`
- [x] Phase 4: add groove audits for real and synthetic rows
  - audit record-level `prepare_mhc_input(...)` status / fallback usage by source bucket
  - audit dataset-level resulting groove-half lengths / missing-chain behavior by source bucket
  - explicitly report whether any synthetic mode produces pathological MHC inputs
- [x] Phase 5: verification
  - targeted pytest for focused sampler / epoch refresh / synthetic mode parsing / groove audit helpers
  - local smoke proving:
    - epoch 1 and epoch 2 synthetic rows differ
    - train batches hit the configured real:synth ratio
    - validation remains real-only
    - groove audits pass on both real and synthetic rows
- [ ] Phase 6: Modal ablations
  - rerun the focused warm-start exact-`IC50` two-allele experiment with:
    - no synthetics
    - refreshed all-mode synthetics at low ratio
    - refreshed single-mode ablations
  - compare at minimum:
    - `SLLQHLIGL`
    - `FLRYLLFGI`
    - `NFLIKFLLI`
  - identify which synthetic modes preserve or destroy known-good probe behavior

## Review

- Confirmed from existing artifacts:
  - `E004` and `E012` both used:
    - `6234` real train rows
    - `6234` synthetic train rows
    - `1559` real validation rows
    - the same six synthetic modes, `1039` rows each
  - train batches were only allele-balanced, not real:synth balanced
  - expected batch composition was roughly `50%` synthetic, but not guaranteed
  - per epoch, the minority allele pool was reused about `8.46x`
  - synthetic rows were generated once before training and then reused for all epochs
- Early groove audit on the existing generator:
  - `peptide_*`, `no_mhc_*`, and sampled `mhc_random` rows produced valid groove inputs in the focused audit
  - `mhc_scramble` showed meaningful fallback usage and needs explicit isolation benchmarking
- Decision:
  - fix the focused synthetic contract first
  - then benchmark synthetic modes one by one before reintroducing them into broader affinity training
- Implementation completed:
  - `scripts/focused_binding_probe.py` now supports:
    - epoch-refreshed synthetic generation
    - explicit `batch_synthetic_fraction`
    - isolated synthetic mode subsets
    - per-epoch groove audits for real and synthetic sources
  - `StrictAlleleBalancedBatchSampler` now balances:
    - allele
    - real vs synthetic slots
  - validation stays real-only
  - `pytest -q tests/test_focused_probe.py tests/test_train_iedb.py` -> `76 passed`
- Local smoke verified:
  - epoch 1 and epoch 2 synthetic seeds differ
  - real rows stay groove-clean
  - `peptide_scramble` stays groove-clean
  - `mhc_scramble` shows fallback-heavy pathological groove generation
- Modal synthetic-mode ablations completed so far:
  - `none`:
    - run: `synth-ablate-none-20260309a`
    - best epoch `9`, val loss `0.6882`
    - `SLLQHLIGL`: `110.0` vs `2382.4 nM`
    - `FLRYLLFGI`: `104.8` vs `1342.7 nM`
    - `NFLIKFLLI`: `8.2` vs `69.2 nM`
  - `all`:
    - run: `synth-ablate-all-20260309a`
    - best epoch `9`, val loss `0.6768`
    - `SLLQHLIGL`: `20867.7` vs `43182.3 nM`
    - `FLRYLLFGI`: `21534.9` vs `23838.5 nM`
    - `NFLIKFLLI`: `4857.2` vs `33225.5 nM`
    - interpretation: better val loss, much worse biologic calibration
  - `peptide_scramble` only:
    - run: `synth-ablate-peptide_scramble-20260309a`
    - best epoch `9`, val loss `0.6599`
    - `SLLQHLIGL`: `26647.3` vs `27514.0 nM`
    - `FLRYLLFGI`: wrong-sign (`20751.0` vs `16238.1 nM`)
    - `NFLIKFLLI`: `8975.4` vs `41394.1 nM`
    - interpretation: lower val loss, destroys A0201-specific ranking
  - `peptide_random` only:
    - run: `synth-ablate-peptide_random-20260309b`
    - best epoch `8`, val loss `0.6961`
    - `SLLQHLIGL`: `32915.2` vs `46817.0 nM`
    - `FLRYLLFGI`: `37859.7` vs `40158.8 nM`
    - `NFLIKFLLI`: wrong-sign (`40345.3` vs `44217.2 nM`)
    - interpretation: preserves groove structure but still collapses useful probe ranking
  - `mhc_scramble` only:
    - run: `synth-ablate-mhc_scramble-20260309b`
    - best epoch `5`, val loss `0.7136`
    - `SLLQHLIGL`: `619.1` vs `1103.7 nM`
    - `FLRYLLFGI`: `208.8` vs `272.4 nM`
    - `NFLIKFLLI`: wrong-sign (`269.1` vs `209.5 nM`)
    - groove audit:
      - sampled synthetic rows used fallback `54/128`
      - statuses: `no_cys_pairs`, `alpha3_fallback`, `no_alpha2_pair`
      - noncanonical groove lengths were common
    - interpretation: structurally invalid enough that it should not be a default negative mode
- Current best conclusion:
  - refreshed synthetics plus explicit real:synth balancing fixed the experimental contract
  - but on the current score-fed `IC50` path, the no-synthetic baseline is still the best biologic behavior
  - `peptide_scramble` and `peptide_random` are not automatically safe
  - `mhc_scramble` is actively suspect because it degrades groove construction on synthetic rows


# Re-Baselining Synthetic Ablations On The Strongest Affinity Path (2026-03-09)

## Spec

- Goal: stop evaluating synthetic and auxiliary changes on the already-regressed score-fed assay path, and instead re-run them on the strongest known class-I baseline contract.
- Problem confirmed from experiment log:
  - old broader class-I baseline `E006` gave much stronger biologic separation than the current score-fed analog `E013`
  - current synthetic ablations (`synth-ablate-*`) therefore answer only which modes harm the weaker path, not which modes improve the strongest path
- Required work:
  - identify the exact architectural difference between old `E006` and current `E013`
  - make that stronger path selectable again without reverting unrelated code
  - verify local parity against `E006`-like behavior as closely as possible
  - then rerun synthetic / auxiliary experiments on top of that strongest baseline
- Constraints:
  - do not silently change the default model path until the benchmark table is complete
  - keep probe panel fixed: `SLLQHLIGL`, `FLRYLLFGI`, `NFLIKFLLI`
  - keep warm start and broader class-I panel in the baseline

## Execution order

- [ ] Phase 1: identify and isolate the regression source
  - diff the current score-fed assay path against the older stronger affinity path
  - decide whether the right fix is:
    - a config switch restoring the old path
    - or a code change making the old path canonical again
- [ ] Phase 2: restore a selectable strongest baseline
  - implement the minimal change needed so the old stronger affinity path can be benchmarked in the current tree
  - keep outputs/checkpoints compatible where practical
- [ ] Phase 3: verify baseline parity
  - local smoke on the broader class-I exact `IC50` panel
  - Modal rerun of an `E006`-like baseline
  - record whether `SLLQHLIGL` returns to the old stronger regime
- [ ] Phase 4: rerun experiments on top of the strongest baseline
  - no synthetics
  - peptide ranking only
  - single-mode synthetic ablations, starting with the structurally safest modes
  - explicitly exclude `mhc_scramble` unless a later design justifies it
- [ ] Phase 5: update tables and conclusions
  - separate:
    - results on the weaker score-fed path
    - results on the restored strongest baseline
  - recommend next default training contract based on the stronger-path table

## Review

- Trigger for this re-baselining:
  - current synthetic-ablation table is valid but not decision-grade for the main model because it is built on a regressed base (`E013`-like behavior)
  - next comparisons must be run on the strongest known broader class-I baseline instead
- Progress:
  - regression source isolated:
    - stronger older path = legacy assay wiring
    - weaker newer path = `score_context` assay wiring
  - implementation:
    - `models/presto_modules.py` now supports `affinity_assay_mode = legacy|score_context`
    - `models/presto.py` and `scripts/focused_binding_probe.py` pass the mode through explicitly
    - focused benchmark CLI defaults to `legacy` so benchmark work now starts from the stronger path
  - verification:
    - `python -m py_compile models/presto.py models/presto_modules.py scripts/focused_binding_probe.py tests/test_presto.py`
    - `pytest -q tests/test_presto.py tests/test_focused_probe.py tests/test_train_iedb.py`
    - result: `128 passed`
  - launched parity rerun on the restored strong base:
    - Modal app: `ap-liwR5X7Dztff4gasmao1KB`
    - run id: `class1-panel-ic50-exact-warmstart-legacy-20260309b`
    - contract:
      - broader class-I panel
      - exact `IC50`
      - warm-start checkpoint `mhc-pretrain-20260308b`
      - no synthetics
      - no peptide ranking
      - no allele ranking
      - `affinity_assay_mode=legacy`
  - next ablation order after parity is confirmed:
    - `none`
    - peptide ranking only
    - `peptide_random` only
    - `no_mhc_alpha` only
    - `no_mhc_beta` only
    - keep `mhc_scramble` excluded unless later evidence justifies it
  - launched on the restored strong base:
    - `class1-panel-ic50-exact-warmstart-legacy-peprank-20260309b`
    - `class1-panel-ic50-exact-warmstart-legacy-synth-peptide-random-20260309b`
    - `class1-panel-ic50-exact-warmstart-legacy-synth-no-mhc-alpha-20260309b`
    - `class1-panel-ic50-exact-warmstart-legacy-synth-no-mhc-beta-20260309b`
  - correction:
    - these launches were not true `E006` parity runs because they incorrectly included `--train-all-alleles --train-mhc-class-filter I`
    - live setup showed `rows=24515`, proving the run had broadened beyond the original 7-allele panel
    - exact `E006` contract should have only the explicit 7 probe/train alleles and about `10193` exact `IC50` rows after filtering
  - immediate fix:
    - stop the misconfigured all-class-I runs
    - relaunch exact 7-allele `E006`-parity baseline on `affinity_assay_mode=legacy`
    - only then launch ablations on top of that exact baseline
  - parallel experiments to run while corrected parity completes:
    - exact same 7-allele contract on `affinity_assay_mode=score_context`
    - exact same 7-allele contract on `legacy + peptide ranking`
    - exact same 7-allele contract on `M1` (`groove_pos_mode=triple`, `binding_core_lengths=8,9,10,11`, shared refinement) with `legacy`
    - exact same 7-allele contract on `M1 + peptide ranking` with `legacy`
  - rationale:
    - `score_context` run measures whether the restored regression diagnosis is correct on a truly matched dataset
    - `legacy + peptide ranking` tests whether ranking helps once the base is correct
    - `M1` tests whether the stronger class-I design still helps under the exact `E006` data contract
  - completed matched-run result:
    - `legacy_m1` is the current best class-I affinity design on the exact 7-allele `E006` contract
    - probe best-checkpoint values:
      - `SLLQHLIGL`: `37.3` vs `29588.2 nM`
      - `FLRYLLFGI`: `32.8` vs `24346.2 nM`
      - `NFLIKFLLI`: `5749.3` vs `6.10 nM`
    - `score_context` remains materially weaker on the same data slice
    - peptide ranking is mixed and should not be the default on top of `legacy_m1`
  - next sequence approved:
    - promote `legacy_m1` as the class-I benchmark base
    - safe synthetic ablations on top of `legacy_m1`:
      - `none`
      - `peptide_random`
      - `no_mhc_alpha`
      - `no_mhc_beta`
    - then add censored quantitative rows with `qualifier_filter=all`
    - only after that test MHC-II entry against `legacy_m1`
- [ ] Inspect current TCR-evidence path and confirm whether it consumes receptor input or only pMHC features
- [ ] Audit obvious dead-code / wasted-compute paths in Presto forward DAG and focused binding trainer
- [ ] Review current pair-mining implementation and identify concrete ways to speed it up without changing future semantics

## Review
- Claude cleanup review:
  - `models/presto.py` / `models/presto_modules.py` simplification is coherent.
  - It removes redundant downstream vectors and dead parameters, but it does not materially change the startup path in `scripts/focused_binding_probe.py`.
  - The runtime speedup plan still applies almost unchanged for startup/setup work.
- Validation after cleanup:
  - `pytest -q tests/test_presto.py tests/test_checkpointing.py tests/test_predictor.py`
  - result: `99 passed`
- Focused runner correction:
  - fixed a real bug where epoch summaries and artifact writes had drifted outside the epoch loop in `scripts/focused_binding_probe.py`.
  - verification:
    - `python -m py_compile scripts/focused_binding_probe.py`
    - `pytest -q tests/test_focused_probe.py tests/test_presto.py tests/test_checkpointing.py tests/test_predictor.py`
    - result: `127 passed`
- Phase 1 startup-cache execution:
  - implemented contract-hash caching in `scripts/focused_binding_probe.py` using:
    - `DatasetContract`
    - `PreparedBindingState`
    - `_prepare_real_binding_state(...)`
  - cache stores:
    - filtered real binding rows
    - train/val split
    - MHC sequence resolution map
    - probe-support summaries for explicit probe peptides
  - local miss/hit smoke on the same contract:
    - run 1 setup: `35.43s`, `cache_hit=false`
    - run 2 setup: `5.53s`, `cache_hit=true`
  - so the prepared-state cache is real and cuts setup by about `6.4x` locally on the same focused contract
- New bottleneck surfaced by the smoke:
  - summary artifact writing still imports Matplotlib on the critical path
  - first local run spent about `4.15s` in `summary_write_wall_s`
  - even the cache-hit run still spent about `3.39s`
  - logs show temporary Matplotlib/font-cache initialization, so plotting/import overhead is now one of the next obvious startup/runtime targets
- Follow-up runtime fix:
  - `scripts/focused_binding_probe.py` now supports `--probe-plot-frequency {epoch,final,off}`
  - default is now `final`
  - local cached 2-epoch smoke with `probe_plot_frequency=final`:
    - epoch 1 `summary_write_wall_s`: `0.0017s`
    - epoch 2 `summary_write_wall_s`: `3.86s`
  - so the Matplotlib/font-cache hit moved off the per-epoch hot path and only lands on the final epoch artifact write
- Setup-stage timing instrumentation:
  - focused setup logs now include:
    - `prepare_real_binding_state_s`
    - `dataset_build_s`
    - `val_loader_build_s`
    - `model_init_and_warm_start_s`
    - `probe_setup_s`
  - cached 1-epoch smoke with plotting off:
    - `prepare_real_binding_state_s`: `0.0033s`
    - `dataset_build_s`: `0.5881s`
    - `val_loader_build_s`: `0.0001s`
    - `model_init_and_warm_start_s`: `0.0298s`
    - `probe_setup_s`: `4.3340s`
    - total `setup_wall_s`: `5.8957s`
  - current startup bottleneck on a cache hit is now clearly `probe_setup_s`, not row filtering/splitting
- Probe-setup fix:
  - probe evaluation now reuses `prepared_state.mhc_sequences` and only falls back to loading the full MHC index if a requested probe allele is missing
  - cached 1-epoch smoke with plotting off after this change:
    - `prepare_real_binding_state_s`: `0.0013s`
    - `dataset_build_s`: `0.4658s`
    - `val_loader_build_s`: `0.0001s`
    - `model_init_and_warm_start_s`: `0.0294s`
    - `probe_setup_s`: `0.0006s`
    - total `setup_wall_s`: `1.2663s`
  - so the cache-hit setup path improved again from `5.90s` to `1.27s`
  - remaining startup cost on the cached path is now mostly `PrestoDataset` construction
  - larger 7-allele exact-`IC50` cold-start smoke on the same code path:
    - `cache_hit=false`
    - `prepare_real_binding_state_s`: `29.7984s`
    - `probe_setup_s`: `23.0579s`
    - `total setup_wall_s`: `53.8008s`
  - implication:
    - repeated-run startup is now cheap on a cache hit
    - cold starts on the real multi-allele contracts are still dominated by contract preparation and probe setup
- `PrestoDataset` MHC-pair memoization:
  - added an internal `_resolved_mhc_pair_cache` in `data/loaders.py` keyed by `(mhc_class, species, allele, direct_seq, mhc_b, allow_default_class_i_beta)`
  - goal: avoid repeated `prepare_mhc_input(...)` / groove-half preparation for repeated alleles within one dataset build
  - verification:
    - `pytest -q tests/test_focused_probe.py tests/test_presto.py tests/test_checkpointing.py tests/test_predictor.py tests/test_loaders.py`
    - result: `163 passed`
  - measured impact on the tiny cached focused smoke was neutral (`dataset_build_s ~0.47s -> ~0.51s`), so this needs a larger repeated-allele benchmark before calling it a real win

- [x] Inspect current TCR-evidence path and confirm whether it consumes receptor input or only pMHC features
- [x] Audit obvious dead-code / wasted-compute paths in Presto forward DAG and focused binding trainer
- [x] Review current pair-mining implementation and identify concrete ways to speed it up without changing future semantics

## Review
- TCR evidence is currently a pMHC-only auxiliary output from `pmhc_vec`; no receptor tokens are consumed in the active model path.
- Biggest runtime wastes identified from code inspection: `forward_affinity_only()` still executes full `forward()`, unconditional Python-heavy pair mining runs even when weights are zero, and strict allele balancing inflates an 8k-row train split into ~39k sampled examples per epoch.
- There is still legacy receptor-related code outside the active affinity path (`training/tasks.py` TCR pairing/pMHC tasks, predictor compatibility APIs), plus full forward computes T-cell and TCR-evidence heads even for affinity-only benchmarks.
- Pair mining can keep current semantics but should move to precomputed batch metadata / index tensors and avoid per-batch GPU->CPU `.tolist()` conversions and nested Python loops.


# Runtime/Dataflow Refactor (2026-03-10)

## Spec

- Goal: remove stale receptor-sequence training task code and prepare the focused affinity trainer for a fixed-dataset, index-driven augmentation/ranking pipeline.
- Constraints:
  - keep the active pMHC-only `tcr_evidence` head and data path
  - remove only unreachable receptor-sequence task code (`tcr_pairing`, `tcr_pmhc`) and their private helpers/exports
  - do not change current training semantics for active affinity benchmarks in the same patch unless verified
- Follow-on runtime design to preserve semantics:
  - keep the real base dataset fixed for a run
  - represent ranking/contrastive pairs as index tensors over that fixed dataset
  - generate epoch-specific synthetic samples as an append-only synthetic index space beyond the fixed dataset size
  - move pair selection / synthetic scheduling off the hot forward path and into epoch-state preparation

## Execution order
- [x] Phase 1: remove stale receptor-sequence training tasks
  - delete `TCRPairingTask` and `TCRpMHCMatchingTask`
  - delete private helpers used only by those tasks
  - remove stale exports from `training/__init__.py`
  - keep `TcrEvidenceTask` intact
- [x] Phase 1 verification
  - `pytest -q tests/test_tasks.py tests/test_training_e2e.py`
  - targeted import smoke for `training`
- [ ] Phase 2: write the fixed-dataset/index-pair refactor plan into `tasks/todo.md`
  - specify dataset IDs / peptide IDs / allele IDs carried in batch metadata
  - specify epoch-state object for synthetic append space + pair-index tensors
  - specify async producer boundary for next-epoch state generation

- Phase 1 verification: `pytest -q tests/test_tasks.py tests/test_training_e2e.py` -> `33 passed`


# Indexed Pair Mining Refactor (2026-03-10)

## Spec

- Goal: keep the shared canonical model path, but refactor focused training dataflow so ranking/contrastive and synthetic augmentation operate on fixed dataset metadata and epoch-state indices instead of per-batch token tuple mining.
- Requirements:
  - add fixed metadata fields to samples/batches/dataset for:
    - `dataset_index`
    - `peptide_id`
    - `allele_id`
    - `bind_target_log10`
    - `bind_qual`
  - preserve current training semantics for active losses
  - generate epoch-specific synthetic append rows and pair/ranking indices from the fixed dataset
  - pair/ranking loss code should no longer call `pep_tok.detach().cpu().tolist()` or rebuild peptide groups from token tuples
  - this work must stay on the canonical shared model, not a separate affinity-only architecture fork

## Execution order
- [x] Phase 1: inspect focused dataset / sample / batch construction path
  - identify where to assign stable dataset indices and peptide/allele IDs
  - identify how current synthetic refresh hooks into epoch loop
- [x] Phase 2: extend sample/batch metadata
  - add fields to `PrestoSample` / `PrestoBatch` / collator outputs
  - ensure dataset construction populates stable `dataset_index`, `peptide_id`, `allele_id`, `bind_target_log10`, `bind_qual`
- [x] Phase 3: add epoch-state object
  - represent current epoch training population as fixed real rows + append-only synthetic rows
  - store pair/ranking candidate indices against that epoch population
- [x] Phase 4: rewrite pair mining/loss input path
  - replace token-tuple grouping with index-based tensors/metadata
  - preserve current loss semantics and metrics
- [x] Phase 5: verification
  - targeted tests for metadata fields, pair-index generation, and synthetic append indexing
  - focused training smoke / existing tests

## Review
- pending

- Indexed pair-mining refactor landed on the canonical shared model path: fixed sample metadata (`dataset_index`, `peptide_id`, `allele_id`, `bind_target_log10`) now flows through `PrestoBatch`, pair candidates are collated as index tensors, and focused ranking losses no longer rebuild peptide groups from `pep_tok.detach().cpu().tolist()`.
- Epoch train state now preserves a fixed real dataset and appends per-epoch synthetic rows with dataset-index offsets via `CombinedSampleDataset`; synthetic IDs begin after the real dataset size.
- Stale receptor-sequence task code (`TCRPairingTask`, `TCRpMHCMatchingTask`) was removed; pMHC-only `tcr_evidence` remains.
- Verification: `pytest -q tests/test_focused_probe.py tests/test_tasks.py tests/test_training_e2e.py tests/test_presto.py` -> `111 passed`


# Multi-Allele Runtime/Training Sweep (2026-03-10)

## Spec

- Goal: benchmark 16 runtime/performance variants on the ~44k multi-allele binding dataset on Modal, using the canonical shared model path.
- Fixed semantic contract:
  - multi-allele class-I binding dataset around the known ~44k-row contract
  - ranking terms enabled in every run
  - censor-aware quantitative losses on KD/IC50/EC50 assay heads
  - 3 epochs per run
  - collect startup time, per-epoch wall-clock, GPU timing breakdown, and accuracy/probe metrics
- Deliverables:
  - manifest of all runs
  - one results table with runtime + accuracy metrics
  - recommendation for fastest effective configuration

## Execution order
- [ ] Phase 1: confirm exact dataset contract and runner support for the 44k binding benchmark
- [ ] Phase 2: define 16 runtime variants and launcher
- [ ] Phase 3: launch all Modal runs in parallel
- [ ] Phase 4: collect summaries / logs / timing metrics
- [ ] Phase 5: rank variants by wall-clock and training effectiveness

## Review
- pending
