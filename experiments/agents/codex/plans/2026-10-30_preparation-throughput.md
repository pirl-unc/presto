# Common cohort preparation throughput

Date: 2026-10-30. Owner: Codex (GPT-6). Baseline: ed9b1b9b.

## Problem and scope

The 30,252,937,352-byte, eleven-file cohort is decoded repeatedly by
ObservationIndex admission, sampling preflight, target-scale fitting, capability
preflight, trainable-row filtering and epoch ordering. Each metadata-only random
access reopens and normalizes a full provenance-heavy JSON record. Current frozen
direct-factorized fits remain untouched; seed 43 has started optimization.

## Implementation contract

1. Retain a compact interned immutable metadata projection during normal verified
   admission: endpoint, split, effective class set and supervised target names.
   Store integer codes per row rather than full observations/sequences/provenance.
   Metadata access must retain the existing per-row source mutation check.
2. Use the projection for common sampling preflight, balanced epoch construction
   and trainable-row filtering. Keep fallback support for ordinary row sequences.
   Preserve grouping/insertion order, quotas, all RNG calls and exact output order.
3. Accumulate target-scale moments inside the existing capability-preflight scan
   using the same canonical fitting implementation. Preserve Welford update order,
   TRAIN-only exact outcomes, quantity contract checks, inherited scalers, floor
   and errors. Fold validation availability checks into that same scan.
4. Keep source hashes, row admission, lineage, split isolation, all capability
   checks, actual batch observations and checkpoint resume fingerprint enforcement.
   No persistent unverified cache, new data contract, model or loss change.

## Verification and runtime study

Register one CPU-only experiment with frozen old/new source, invocation and data
hashes. Benchmark the actual provenance-heavy cohort or a deterministic bounded
slice of its source files. Record hardware, stage timings, RSS, normalization
calls and exact parity of strata, scales, preflight reports, sample order and RNG.
Report measured stage/combined speedups without projecting them onto unmeasured
full-cohort startup. No predictive training or test metrics are produced by this
runtime study; existing scientific fits retain their validation/test contracts.

Tests cover multiple sources, source mutation, weighted/unweighted/empty/single
strata, class overrides/mixed/unknown class, subset order, inherited scalers,
exact/bounded targets, trainable/frozen targets, full fit and resume trajectory.
Run focused tests, lint and full suite after freezing edits. Add runtime results,
limitations and rollout status to the registry and task review.

## Rollout

The improvement belongs to the common manifest trainer. Do not hotpatch or restart
the existing frozen fits: their source fingerprint is part of their reproducible
contract and they have already paid most preparation costs. Future launches use
the optimized implementation. Any later continuation must explicitly declare its
source revision; never bypass resume-contract checks.

## Verification environment correction

The first full suite completed with 2,923 passes, four skips and six Hitlist
fixture-setup errors: scratch free space transiently fell to 0.341 GiB, below
Hitlist's 1 GiB reserve. The whole affected file passed separately (23 tests).
Preserve the original error log, inspect available capacity, clean only disposable
temporary output owned by this verification run, and repeat the unchanged full
suite with an explicit temporary directory. Do not lower the storage reserve or
change source code to suppress this environmental failure.
