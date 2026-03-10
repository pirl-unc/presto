# Runtime Speedup Plan

## Goal

Reduce end-to-end wall clock for Presto training and benchmarking substantially without changing the canonical model contract or baking in a benchmark-only fast path.

This plan targets both:
- startup/setup latency before epoch 1
- steady-state epoch time under the dynamic training regime we actually intend to keep

## Non-Negotiable Constraints

1. One canonical model.
- No separate “fast affinity-only architecture” as the main direction.
- Runtime work should improve the same model that will be used downstream.

2. Preserve option generality.
- The code must continue to support different combinations of:
  - assay filters
  - qualifier policies
  - ranking losses
  - synthetic augmentation modes
  - curricula
  - class-I only vs mixed class-I/class-II
- We are not allowed to hard-code a fast path that only works for one benchmark contract.

3. Dynamic epoch content remains supported.
- New synthetic samples per epoch
- New pair/ranking assignments per epoch
- Optional curricula that change targets over time

4. Benchmarking stays comparable.
- Every optimization must be evaluated on a fixed semantic contract, not just on raw throughput.

## Current Bottlenecks

### Startup / setup

1. Repeated full scans of `data/merged_deduped.tsv`
- `_load_binding_records_from_merged_tsv(...)`
- `_audit_probe_support(...)` once per probe peptide
- update after Phase 1 cache + setup timing:
  - cached runs reduced `prepare_real_binding_state_s` to `~0.003s`
  - the remaining startup bottleneck is now `probe_setup_s` (`~4.33s`), which includes:
    - `_resolve_allele_sequences(...)` loading the full MHC index
    - `_audit_probe_support(...)` rescans for non-cached probe peptides
- follow-up fix:
  - probe evaluation now reuses `PreparedBindingState.mhc_sequences`
  - cached setup after this change:
    - `prepare_real_binding_state_s ~0.001s`
    - `probe_setup_s ~0.0006s`
    - total `setup_wall_s ~1.27s`
  - larger 7-allele exact-`IC50` cold miss on the same code path still measured:
    - `setup_wall_s ~53.8s`
    - `prepare_real_binding_state_s ~29.8s`
    - `probe_setup_s ~23.1s`
  - implication:
    - the cache-hit path is now fast enough that contract preparation dominates only on cold starts
    - keeping the prepared-state cache warm matters on the real multi-allele contracts

2. Eager per-record sample materialization
- `PrestoDataset(...)` builds every `PrestoSample` eagerly
- MHC resolution / groove preparation is repeated per record, not per unique allele/sequence
- current measured cost on a cached focused run: `dataset_build_s ~0.47s`
- this is now the largest measured startup component on the cached focused path

3. Repeated startup diagnostics on the critical path
- probe audits
- groove audits
- sequence-resolution summaries

4. Artifact/plot initialization on the critical path
- `scripts/focused_binding_probe.py` imports Matplotlib during `_write_summary_artifacts(...)`
- local smoke after the prepared-state cache still spent `~3.4s` to `4.1s` in summary writing
- logs show Matplotlib/font-cache initialization, so plotting needs to be moved off the hot path or made optional/final-only
- status:
  - `probe_plot_frequency` added with `epoch|final|off`
  - default switched to `final`
  - cached 2-epoch smoke moved summary-write cost from epoch 1 (`0.0017s`) to final epoch only (`3.86s`)

5. Modal startup overhead
- image/build/app bootstrap is significant and should be amortized where possible

### Per-epoch

1. Forward/loss dominates runtime
- already measured in the 44k sweep

2. Epoch state rebuild
- current focused runner rebuilds epoch train state, including dynamic dataset objects and audits

3. Balanced batching inflates effective epoch size
- strict balancing turns a nominal dataset into a much larger sampled epoch

4. Dynamic augmentation and pair generation are not yet organized around fixed indices and reusable metadata

## High-Level Design

The main idea is:

- build one fixed, canonical **base dataset state**
- represent all dynamic per-epoch variation as **epoch plans** over that base
- keep options configurable by composing planners, not by branching into specialized code paths

### Core abstractions

1. `DatasetContract`
- Immutable description of the selected training data contract
- Includes:
  - source selection
  - allele set / class filter
  - measurement profile / type filter
  - qualifier policy
  - split seed
  - probe set

2. `PreparedDatasetState`
- Cached result of applying a `DatasetContract`
- Contains:
  - normalized rows
  - fixed train/val split
  - stable integer IDs
  - cached sequence/groove preparation
  - cached tokenized representations or tokenization-ready compact arrays

3. `EpochPlan`
- Dynamic additions on top of the fixed dataset for one epoch
- Contains:
  - sampled real row indices
  - synthetic row specs and their appended indices
  - pair/ranking index tensors
  - optional curriculum state for that epoch

4. `EpochMaterializer`
- Converts `PreparedDatasetState + EpochPlan` into runtime tensors/loaders/batches
- Must support:
  - no augmentation
  - synthetic augmentation
  - ranking-only
  - mixed augmentation + ranking

This preserves generality:
- the contract chooses the data universe
- the epoch plan chooses the epoch contents
- no option is hard-coded away

## Phase Plan

### Phase 0: Instrumentation and acceptance criteria

Goal:
- make every future speedup measurable

Work:
- Add explicit timing sections for:
  - raw TSV scan/filter
  - probe-support audit
  - split generation
  - MHC resolution/groove prep
  - tokenization
  - dataset object build
  - epoch-plan generation
  - dataloader creation
  - batch transfer
- Add one canonical benchmark schema row per run:
  - contract hash
  - option hash
  - startup time
  - mean epoch time
  - accuracy/probe metrics

Acceptance:
- every benchmark writes one compact machine-readable summary row

### Phase 1: Contract-hash caching for startup

Goal:
- eliminate repeated full-file preprocessing when the semantic contract is unchanged

Work:
- Add `DatasetContract` hashing
- Cache:
  - filtered normalized rows
  - split assignments
  - probe-support summaries
- Cache location:
  - `artifacts/cache/` or `modal_runs/cache/`
- Keep cache invalidation explicit:
  - code version stamp
  - source file mtimes / sizes
  - contract hash

Files likely touched:
- `scripts/focused_binding_probe.py`
- new helper module, e.g. `scripts/focused_dataset_cache.py`

Acceptance:
- repeated runs on the same contract do not rescan the merged TSV or rebuild split/probe-support state
- status:
  - implemented in `scripts/focused_binding_probe.py`
  - local miss/hit smoke reduced setup from `35.43s` to `5.53s` on the same focused contract

### Phase 2: Prepared dataset state with reusable IDs and sequence caches

Goal:
- stop recomputing expensive per-record transforms

Work:
- Precompute once per fixed dataset:
  - `dataset_index`
  - `peptide_id`
  - `allele_id`
  - target values / qualifiers
- Cache groove-prepared MHC segments by:
  - allele key and/or direct sequence key
- Cache tokenization inputs by:
  - peptide string
  - groove-half strings
- Optionally cache token tensors if memory is acceptable

Important:
- keep this representation generic across training options
- do not tie it to one loss or one benchmark

Files likely touched:
- `data/loaders.py`
- `data/collate.py`
- new helper module for prepared-state caching

Acceptance:
- sample construction cost scales with number of unique peptides / MHCs, not raw row count

### Phase 3: EpochPlan abstraction for dynamic augmentation and ranking

Goal:
- preserve fully dynamic training while avoiding per-batch Python regrouping and full dataset rebuilds

Work:
- Implement `EpochPlan` containing:
  - real sampled indices
  - synthetic sample specs for this epoch
  - pair tensors for ranking/contrastive losses
- Ranking/contrastive pairs should be:
  - derived from fixed integer metadata
  - represented as index tensors
  - regenerated per epoch if desired
- Synthetic rows should be:
  - append-only relative to the fixed base dataset
  - assigned indices `>= real_dataset_size`

Important:
- dynamic choices remain configurable:
  - which synthetic modes
  - synthetic ratio
  - ranking strategies
  - curricula

Files likely touched:
- `scripts/focused_binding_probe.py`
- `data/collate.py`
- new `epoch_plan` helper module

Acceptance:
- no token-to-CPU regrouping for pair mining
- no need to rebuild the full real dataset every epoch

### Phase 4: Async next-epoch preparation

Goal:
- overlap CPU-side epoch preparation with GPU training

Work:
- background worker/thread/process prepares epoch `t+1` while GPU trains on epoch `t`
- next-epoch work includes:
  - synthetic sample specs
  - ranking pair indices
  - optional curriculum state
- runtime consumes completed epoch plans from a queue

Important:
- this is only valid once epoch plans are disentangled from mutable dataset rebuilding

Acceptance:
- dynamic augmentation cost is largely hidden behind GPU work

### Phase 5: Make diagnostics optional and off critical path

Goal:
- preserve visibility without taxing every run

Work:
- mark diagnostics as one of:
  - required correctness gate
  - startup-only optional audit
  - deferred post-run analysis
- move the following off the hot path by default:
  - probe-support rescans
  - groove audits on every epoch state

Acceptance:
- production benchmark path does not perform expensive optional audits before epoch 1 unless explicitly requested

### Phase 6: Runtime semantics of balanced batching

Goal:
- make epoch size explicit and controllable

Work:
- separate:
  - semantic balancing policy
  - epoch budget policy
- Add explicit controls for:
  - examples per epoch
  - batches per epoch
  - oversampling policy
- Keep strict per-batch balancing available, but do not silently let it expand the epoch by 2-3x without being visible in the contract

Acceptance:
- effective epoch size is an explicit benchmark parameter, not a side effect

### Phase 7: Modal orchestration / benchmark harness

Goal:
- reduce repeated remote setup and improve comparability

Work:
- run multiple variants inside one Modal app when possible
- reuse:
  - prepared dataset cache
  - checkpoint payloads
  - benchmark result schema
- keep per-variant summaries compact and uniform

Acceptance:
- reduced Modal orchestration overhead per compared variant

## What Not To Do

1. Do not add a benchmark-only “fast path” that bypasses the real dynamic training regime.
2. Do not special-case a single assay contract in core runtime abstractions.
3. Do not optimize for static no-augmentation runs and assume that carries over.
4. Do not optimize only dataloader knobs when measured runtime says model/setup dominate.

## Suggested Order of Implementation

1. Phase 0: instrumentation cleanup
2. Phase 1: contract-hash startup cache
3. Phase 2: prepared dataset state + sequence caches
4. Phase 3: epoch-plan abstraction
5. Phase 5: move optional diagnostics off hot path
6. Phase 4: async next-epoch preparation
7. Phase 6: explicit epoch-budget semantics
8. Phase 7: Modal benchmark consolidation

This order attacks the largest current waste first while preserving option generality.

## Success Criteria

### Startup
- repeated runs on the same contract cut startup/setup time by at least 50%

### Steady-state epoch time
- dynamic-augmentation epoch time improves materially without worse probe behavior

### Generality
- the same runtime path still supports:
  - exact vs censored quantitative training
  - peptide ranking / contrastive losses
  - synthetic augmentation on/off and mode subsets
  - future curricula
  - class-I only and mixed class-I/class-II runs

### Developer ergonomics
- every run emits one compact benchmark summary row
- caches are inspectable and invalidation is explicit

## First Concrete Deliverable

Implement the minimum generic infrastructure needed for the next runtime pass:

1. `DatasetContract` + hash
2. cached filtered rows + split + probe-support summary
3. cached groove-prepared unique MHC segments
4. fixed-metadata `PreparedDatasetState`
5. `EpochPlan` with:
  - real row indices
  - synthetic row specs
  - ranking pair tensors

Only after that should we re-run a realistic augmentation-on runtime bake-off.
