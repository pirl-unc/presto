# Flank context wired end to end, on corrected mapping data

> **Validity erratum (2026-09-04):** Retain these files as the exact historical
> record, but do not use this rerun to choose a mapping policy. It still
> converted nullable unmapped flanks into the valid-looking sequence `NAN` in
> the legacy arm and cleared them in the masked arm, so the purportedly paired
> intervention included 6,212 accidental unmapped-row changes. Best-checkpoint
> rescoring fixed the checkpoint defect but cannot fix this data-contract
> confound. Sparse held-out targets, seed-dependent caps, and all-positive
> elution are also documented by the 2026-09-04 audit.

**Experiment id:** `2026-09-03_1746_claude_flank-context-fixes`
**Run dates:** 2026-09-03
**Agent/model:** Claude / Opus 5
**Git state:** `6f32d46126a73665e8927d85fc7a7983b2f05c09`, branch `claude/flank-context-and-data-fixes`, dirty: yes
**Status:** complete

## Question and decision

Two questions, one paired design:

1. **Does the legacy-vs-masked conclusion survive on corrected code?** The
   2026-09-02 family answered "use `mask_unresolved`, defer candidate
   marginalization" on a codebase carrying the defects listed below.
2. **What happens once a protein terminus actually reaches the model?** It
   never had. `flank_n_is_terminus` was declared on three record dataclasses,
   moved to the device by `PrestoBatch.to`, and branched on by
   `Presto._pad_for_side` -- but no `PrestoSample(...)` construction in
   `data/loaders.py` passed it, so every row arrived `False` and the `X`
   boundary pad was unreachable. Measured on this corpus, **3.2% of binding
   rows carry an N-terminus and 2.9% a C-terminus**; all of them previously
   padded with `?` ("unknown") instead of `X` ("nothing is here"). On the C
   side that is the sharper error: a peptide ending at its protein's own
   C-terminus required no proteasomal cut, and the excision head was scoring a
   cleavage that never had to happen from a token asserting ignorance.

**Decision: keep `mask_unresolved` in production and defer candidate-junction
marginalization.** The 2026-09-02 verdict reproduces on corrected code, and
survives the terminus wiring, the corrected mapping data and a new split seed.

Two secondary findings matter more than the verdict:

1. Re-scoring the same runs from their selected checkpoints flips the sign of
   `unresolved_exact_rmse_mean_reduction` and nearly triples
   `unresolved_exact_spearman_mean_delta`. The evaluation-checkpoint artifact
   was comparable in size to the effect being measured.
2. The unresolved test stratum is 37 rows (28 exact) and its per-seed deltas
   flip sign. The negative result is a statement about power on
   HLA-A*02:01, not evidence that junction context does not matter.

## What changed since 2026-09-02_1720

Corrected in `claude/flank-context-and-data-fixes`:

- Terminus flags wired through binding, processing and elution records.
  `bulk_ms` carries no flanks, so its `False` default was already right.
- Flank context has one representation per layer: `""` in the frame, `None`
  on the sample, a pad token in the tensor. `?` no longer appears in any
  sequence string, so `is_encodable_sequence` rejects it plainly instead of
  `flank_context` special-casing one exact marker past the validator.
- The candidate collapse could let a left-join miss outrank the real mapping
  (empty identifiers sort first) and still stamp the row `single`/resolved.
- `_canonical_mask` read a float-encoded flag as False, disabling the
  canonical tie-break and masking the bug above.
- `_flank_coverage` gated on `flank_context_resolved`, so it reported
  resolution, not coverage. See the table below for why that mattered.
- Three drifted copies of "which categories are resolved" collapsed into one
  table, with the partition asserted.
- The three-way split reused the two-way helper's `seed + 53` peptide order
  and filled *test* from the front where two-way fills *validation* from the
  front, so at a given seed every held-out test peptide had already been used
  for model selection. Separate stream now.

**Confounding, stated explicitly.** Within this experiment, legacy vs masked
remains a clean paired comparison: both arms get the same terminus wiring, the
same corrected mapping data, and the same split. Comparing these numbers
against the 2026-09-02 table is *not* clean -- the terminus wiring and the
split seed both moved. Mapping categories are byte-identical between the two
families seed for seed, so the data-correctness fixes are defensive on this
corpus rather than corpus-changing; the differences that remain are the
terminus wiring and the new test split.

## The coverage stat now separates the arms

The previous family printed `flank coverage [binding]: 60.0%` for **both**
arms, because the stat gated on resolution -- which masking does not change.
Split into two stats, the arms are distinguishable:

| Arm | coverage [binding] | resolution [binding] | coverage [elution] | resolution [elution] |
|---|---|---|---|---|
| `legacy_global_canonical` | 100.0% | 60.0% | 100.0% | 86.8--87.3% |
| `mask_unresolved` | 60.0% | 60.0% | 86.8--87.3% | 86.8--87.3% |

Legacy keeps every flank while only 60% of binding junctions are genuinely
resolved. Masking clears exactly that 40% gap, so its coverage collapses onto
its resolution. That 60.0% is precisely the number the old single stat
reported for both arms.

So the masked arm trains **40% of binding rows and ~13% of elution rows** on an
explicitly unknown junction rather than an arbitrarily chosen one.

## Dataset and curation contract

- Source: frozen hitlist 1.55.8 curated indexes on the Modal `presto-data`
  volume, hash-verified before the GPU is touched. Note the local
  `~/code/hitlist` checkout has since moved to 1.56.0; the remote contract is
  unaffected and remains 1.55.8.
- Allele `HLA-A*02:01`; 16,721 binding, 20,000 elution, 1,000 stability and
  2 kinetics records per seed (37,723 total). Binding uncapped; capped
  modalities use reservoir sampling with sampling seed `training_seed + 17`.
- Split: peptide-disjoint 80/10/10, roughly 30,178 / 3,772 / 3,772 rows and
  ~23,200 / ~2,870 / ~2,900 peptides.
- Mapping strata: `single`, `flanks_agree`, `within_gene_canonical`,
  `cross_gene_unresolved`, `within_gene_unresolved`, `unmapped`.
- Supervision parity between the two policies verified for all four evidence
  families in `results/data_audit.json`.

Frozen artifact hashes are recorded in `reproduce/` and re-verified per run in
`results/runs/*/data_contract.json`.

## Training contract

Presto expanded topology, d_model 128 / 2 layers / 4 heads, AdamW
lr 2.8e-4 weight_decay 0.01, batch 256, 10 epochs, bf16 autocast.
Synthetic negatives, MHC augmentation and probe tracking all disabled.
Requested GPU `H100!`; observed hardware recorded per run in
`results/runs/*/hardware.json`.

## Conditions

| Condition | Policy | Seeds |
|---|---|---|
| legacy | `legacy_global_canonical` | 42, 43, 44 |
| masked | `mask_unresolved` | 42, 43, 44 |

## Results

**Status: complete.** Six conditions, all `status: complete`, observed hardware
`NVIDIA H100 80GB HBM3, 81559 MiB` matching the requested `H100!`. Runtime
4,104--5,556 s per run; 28,168 GPU-seconds (7.8 GPU-hours) for the family,
~1h21m wall clock in parallel.

### Two sets of numbers, and why

These runs launched before `train_iedb` reloaded the best-validation
checkpoint for the held-out pass, so their original artifacts score whatever
epoch training stopped on. `model.pt` is the selected epoch, so both sets are
reported:

- `results/` -- **as run** (final epoch), what the pipeline produced.
- `results/best_checkpoint/` -- **re-scored** from each run's own `model.pt`
  via `analysis/recompute_holdout_from_best.py`. Splits were re-derived and
  verified identical to each original run before any metric was written.

Four of six runs selected an epoch before the last:

| Run | best epoch | best val loss | final val loss |
|---|---|---|---|
| legacy s42 | 7 | 0.6417 | 0.6652 |
| legacy s43 | 4 | 0.5917 | 0.5971 |
| legacy s44 | 10 | 0.5696 | 0.5696 |
| masked s42 | 7 | 0.6538 | 0.6703 |
| masked s43 | 10 | 0.5798 | 0.5798 |
| masked s44 | 9 | 0.5004 | 0.5164 |

The gap is not constant across runs, so it does not cancel in a paired
difference. **Prefer `results/best_checkpoint/`.**

### Held-out test, overall (best checkpoint, mean +- sd over seeds 42--44)

| Metric | `legacy_global_canonical` | `mask_unresolved` |
|---|---|---|
| exact Spearman | 0.7536 +- 0.0062 | **0.7604 +- 0.0057** |
| exact Pearson | 0.7462 +- 0.0064 | **0.7522 +- 0.0055** |
| exact RMSE log10(nM) | 1.0039 +- 0.0177 | **0.9948 +- 0.0135** |
| all Spearman | 0.7844 +- 0.0052 | **0.7899 +- 0.0120** |
| <=500 nM AUROC | 0.9164 +- 0.0034 | **0.9190 +- 0.0025** |
| <=500 nM AUPRC | 0.8677 +- 0.0048 | **0.8711 +- 0.0080** |
| <=500 nM F1 | **0.8274 +- 0.0067** | 0.8191 +- 0.0068 |
| <=500 nM balanced acc. | **0.8484 +- 0.0089** | 0.8419 +- 0.0062 |
| <=500 nM accuracy | **0.8461 +- 0.0144** | 0.8418 +- 0.0057 |
| <=500 nM precision | 0.7936 +- 0.0385 | **0.7963 +- 0.0070** |
| <=500 nM recall | **0.8671 +- 0.0443** | 0.8434 +- 0.0194 |

Masking wins the regression metrics and AUROC/AUPRC; legacy wins the
thresholded decision metrics. Every gap is within roughly one seed-level
standard deviation, so none of it is a result on its own.

### Decision gate, against the 2026-09-02 family

| Quantity | 20260902b | this, as-run | this, best ckpt |
|---|---|---|---|
| `overall_exact_spearman_mean_delta` | +0.00300 | +0.00216 | +0.00683 |
| `overall_exact_rmse_mean_delta` | -0.00594 | +0.00367 | -0.00909 |
| `unresolved_exact_spearman_mean_delta` | -0.01377 | -0.00611 | -0.01779 |
| `unresolved_exact_rmse_mean_reduction` | -0.01290 | -0.01118 | +0.03204 |
| `invest_in_candidate_marginalization` | false | false | false |
| `no_material_overall_regression` | true | true | true |

**The decision is unchanged: keep `mask_unresolved` in production and defer
candidate-junction marginalization.** That verdict is stable across all three
result sets and across the terminus wiring, the corrected mapping data and a
new split seed.

The individual numbers are not stable. Re-scoring the same six runs from their
selected checkpoints moves `unresolved_exact_rmse_mean_reduction` from -0.0112
to +0.0320 -- a sign flip -- and nearly triples
`unresolved_exact_spearman_mean_delta`. The checkpoint artifact alone is
comparable in magnitude to the effect the experiment is trying to measure,
which is the real reason the fix mattered.

### Why the unresolved stratum stays inconclusive

Small, exactly as in the previous family. Seed 42 test: `unresolved_union` is
**37 rows, 28 exact**, against 1,712 overall. Per-seed exact-Spearman deltas
(masked minus legacy) are -0.0094 / +0.0459 / -0.0899 -- the sign flips across
seeds. Three seeds on ~30 rows cannot separate a real effect from noise, so
"masking does not help on unresolved junctions" remains a statement about
statistical power on HLA-A*02:01, not a claim that junction context is
unimportant.

### Artifacts

| Path | Contents |
|---|---|
| `results/runs/<run_id>/` | per-run config, hardware, data contract, training curve, as-run summaries and per-example dumps |
| `results/runs/<run_id>/best_checkpoint_eval/` | re-scored summaries, metrics and per-example dumps |
| `results/condition_metrics.csv`, `condition_metric_summary.csv` | as-run, per condition and aggregated |
| `results/paired_differences{,_summary}.csv`, `.json` | as-run paired deltas and the decision gate |
| `results/best_checkpoint/` | the same four products, re-scored |
| `results/best_checkpoint_eval_status.json` | per-run split verification for the replay |
| `results/contract_checks.json` | hash, hitlist version, GPU and completeness checks |
| `results/data_parity.json` | supervision parity between the two policies |
| `results/data_audit.json` | pre-launch capped-corpus and split audit |

`model.pt` is gitignored; the Modal volume paths are in `manifest.json`.

## Reproduce

```bash
modal run <exp>/code/launch.py --preflight-only      # CPU: hashes + pipeline
modal run --detach <exp>/code/launch.py              # 6 x H100!
python scripts/fetch_experiment_modal_runs.py --experiment-dir <exp> --wait
python <exp>/code/aggregate.py
```

Bundle: [`reproduce/`](./reproduce/)
