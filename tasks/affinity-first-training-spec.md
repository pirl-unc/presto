# Affinity-first training controls (spec)

Goal: opt-in manifest `training` options that let affinity rows steer the common modular
trainer without elution dominating by row count. Every option defaults to today's behavior;
unset options are NOT written into the resolved `training` dict, so default resolved configs
and contracts are unchanged.

## Options (all validated in `preflight`, all part of the resume contract via `training`)

1. `loss_normalization`: `"observation_mean"` (default) | `"per_endpoint"`; optional
   `endpoint_loss_weights` (`{endpoint: w>0}`, only with `per_endpoint`). Mean within each
   endpoint present in the batch, then weighted mean over those endpoints (default weight 1).
2. `schedule`: `[{until_step, stratum_weights}]`, strictly increasing `until_step`; a segment
   is active while `steps < until_step`; afterwards the top-level `stratum_weights` apply.
   `state["segment"]` is the segment whose epoch order is in use; crossing a boundary
   discards the unconsumed order and rebuilds it from the sampler RNG.
3. `lr_schedule` `{warmup_steps, decay: "none"|"cosine", min_lr_fraction, decay_steps}`,
   a pure function of the completed update count (so resume is exact);
   `grad_clip_norm`; `ema_decay` (EMA of trainable parameters, persisted in resume state;
   validation, selection, `selected.pt`, `last.pt` use EMA weights; resume keeps raw weights).
4. `within_group_ranking_loss` `{weight, endpoints}` and `cell_batching` `{rows_per_cell}`.
   Cell = (endpoint, source, assay method preparation/mechanism/detector, study). Pairwise
   logistic loss `softplus(pred_i - pred_j)` for exact (`=`) rows with `target_i < target_j`
   in the same cell within a batch. Censored rows and rows with no study never pair.
   `cell_batching` re-blocks the epoch order into blocks of up to k same-cell rows, shuffled
   as blocks (same multiset of rows, so stratum frequencies are unchanged).
5. `selection_metric`: `"validation_loss"` (default) | `"within_study_spearman"` with
   `selection_spearman_endpoints` and `selection_min_cell_rows` (default 20). Score is
   `-mean(Spearman)` over validation (endpoint, study) cells with >= min exact rows; a cell
   whose predictions are constant counts 0. No eligible cell is a preflight error (no
   silent fallback to loss). History reports the cells used and skipped.

## Verification

Unit tests per option; golden default trace (`tests/golden_default_training_trace.json`,
recorded before the change); resume equivalence inside and across a schedule boundary and with
EMA/LR/clip; full suite; ruff.
