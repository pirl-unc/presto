# MPS Training Stabilization

## Goal

Make Apple Silicon `mps` usable for local focused PF07 training instead of diverging immediately with `non_finite_train_loss`.

## Starting Point

- The registered smoke comparison in `experiments/2026-03-19_1007_codex_cpu-vs-mps-focused-pf07-smoke` is the current truth.
- Matched `cpu` completes.
- Matched `mps` diverges at epoch `1` with `non_finite_train_loss`.
- The MPS launch log also reports a fallback warning for `nonzero`, so the path is not a clean fully-native run.

## Hypotheses To Check

1. Forward-path instability on MPS:
   - one or more affinity outputs are already non-finite before loss construction
2. Loss-path instability on MPS:
   - forward outputs are finite, but target encoding / censor-aware loss math becomes non-finite
3. Optimizer-step instability on MPS:
   - first forward/loss/backward are finite, but AdamW update creates non-finite parameters
4. Precision-path issue:
   - same batch is stable on CPU and unstable on MPS because of backend-specific math / reduction behavior

## Debugging Plan

1. Reproduce the exact tiny smoke contract in a local debug script.
2. Compare the same first batch on `cpu` and `mps`.
3. Record:
   - finite/non-finite status of batch tensors after `.to(device)`
   - finite/non-finite status of each affinity output head
   - finite/non-finite status of each supervised loss component
   - gradient finiteness before optimizer step
   - parameter finiteness after optimizer step
4. If needed, bisect by:
   - `probe_only`
   - `assay_heads_only`
   - `full`
   - optimizer step skipped

## Candidate Fixes

- Force MPS-safe runtime defaults only on `device == "mps"`:
  - explicit float32 matmul behavior
  - optimizer epsilon or implementation changes if AdamW is unstable
  - selective CPU loss computation for unstable reductions if forward is otherwise stable
- Reject hidden partial fallback approaches unless they are explicit and well-documented.

## Verification

- Add targeted tests for any new runtime helper / config logic.
- Run the matched registered smoke experiment again.
- Compare:
  - completion vs divergence
  - CPU vs MPS held-out metrics
  - run summaries and manifest status

## Exit Criteria

- If MPS completes with finite metrics close to CPU: mark it usable for local continuation, with any caveats.
- If MPS still diverges after the smallest defensible fix: document the blocker precisely and keep CPU as the default.
