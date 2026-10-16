# Prospective bounded IC50 fit with explicit effective targets

## Problem and decision

The original corrected-human fit was stopped before optimization because its
raw-target plan conflicted with historical clipping. Preserve that family and
all inputs. After #112 ships, run a new family using released Presto 0.1.16 and
Hitlist 1.78.0 (the frozen source producer). The proposed fit explicitly accepts
`log10(clip(nM, 0.001, 50000))`, float32 and unchanged censor direction, as the
historical training objective. This is not a new scientific loss or a claim
that a clipped exact target equals its original measurement.

## Frozen contract before launch

- Reuse all 9,053 corrected human model observations and their exact 7,241/906/906
  membership. Same frozen human source manifest b180eba8…; same MHC catalog and
  raw reference pins; data/model seed 42, reservoir caps, relative_deficit_v2,
  strict resolved MHC, mask_unresolved/no flanks, no generated/auxiliary data.
- Publish a NEW prospective supported-output manifest. Its original dataset,
  supervision, evidence and membership fingerprints and support minima remain
  identical. Add the canonical per-objective affinity_target_policies,
  drawn from the released executable policy, and explicitly update the expected
  frozen-membership policy.json receipt from producer 0.1.13 to 0.1.16. Verify
  that the released partition recipe bytes are identical; preserve the original
  receipt and publish the expected new receipt. Only that producer field may
  change inside it; every membership/assignment/parent hash remains fixed. The
  complete canonical gate must reproduce the expected artifact hash before any
  model is constructed. Preserve and hash the old manifest separately. Name and
  justify both changes explicitly; never overwrite/rebind
  the stopped family's declaration. The canonical gate must independently
  recompute the complete contract before model construction and fail on drift.
- Only direct assays.IC50_nM is declared: all train/val/test minima remain 50
  unique observations, 50 distinct peptides, 50 exact values. Other tasks are
  still optimized if active, but their outputs remain undeclared.
- Freeze producer wheel URL/hash, dependency environment, code revision, full
  launch config, new manifest and all input pins in a clean recipe commit.

## Fit and baselines

Use the previous unexecuted bounded design: one CPU-only d32/l1/h4 expanded
model; two epochs of at most 100 batches; shuffled batches of 64; AdamW base lr .001,
weight decay .01; canonical 5% linear warmup (10 of 200 steps) followed by cosine
decay to .0001; uncertainty weighting; no PCGrad/AMP/compile; all augmentation,
consistency/orthogonality/sparsity/contrastive priors zero; complete validation
and complete MIL bags. Canonical aggregate validation loss selects checkpoint.
No tuning or selection uses test results. Resource limits: eight CPU, 64 GiB RAM,
512 GiB scratch, zero GPU/workers, one Torch/BLAS thread; 90-minute work alarm,
95-minute Modal function limit. Reuse shared verified caches through datacache.

Train-only constant mean and ridge(alpha=1) baselines fit the 1,599 exact IC50
training rows in EFFECTIVE target space. Ridge features stay peptide length,
20 amino-acid fractions and MHC class; normalization uses train only. No held-out
fitting, calibration or hyperparameter search. Evaluate all three predictors
against both effective targets and original raw log10(nM) values/bounds, with
separately named reports and columns. No source value or operator is changed.

## Acceptance and retained evidence

- Before model work, compare all frozen raw/effective labels, policy declarations
  and source directions; canonical census must recover 211 clipped IDs including
  54 exact values. A mismatch stops and preserves the run.
- Check initial versus selected-checkpoint errors on all 2,074 direct IC50 train
  observations, with RNG-preserving diagnostic passes and exact source-ID checks.
- Preserve every validation/test prediction and full lineage/context, raw and
  effective targets, censor directions, checkpoint identities and loss ledgers.
  Report per-split/endpoint coverage, direct gradients, optimizer steps/changes,
  and all zero-support outputs. No output existence substitutes for evidence.
- For effective and raw spaces separately, report exact RMSE/MAE/Pearson/Spearman,
  censor-aware MSE/satisfaction, and conservative <=500 nM classification counts,
  accuracy/balanced accuracy/precision/recall/F1/AUROC/AUPRC. Undefined stays null.
- Prespecified fitting check: 200 finite optimizer steps, no aborted batches or
  nonfinite tracked elements, nonzero direct IC50 gradient and improved exact
  training RMSE in effective space. Quality screen: positive validation Spearman
  and upper bound of 1,000 paired peptide-bootstrap (seed 4242) 95% RMSE difference
  versus train mean below zero. Prefer over ridge only if the analogous interval
  is below zero. Raw-space results are separate limitations, not posthoc criteria.
- Preserve failures, logs, versions, runtime/RSS, initial/selected checkpoints and
  all artifacts as one immutable datacache generation. Publish/independently
  verify download hashes, complete context joins, predictions, reports and recipe.
  Use a separate PR/version 0.1.17, format/lint/full tests, review, merge/release.

Shared studies and unknown donor/specimen/acquisition identities remain explicit.
The held-out set's labels have been inspected for qualification, so this bounded
result is exploratory and peptide-disjoint, not blinded external validation or
biological independence. Positive-only elution and missing processing/T-cell/TCR
support do not become claims. Hitlist #678 and Presto #48/#53 remain open wherever
these broader requirements are unsupported. MHCflurry remains out of scope.

## Optimizer wording clarification — #114

The immutable published plan used the short name “Adam.” The pinned executable
trainer has always used AdamW plus the canonical warmup/cosine scheduler described
above. `results/protocol-notes.json` records the released script hashes, precise
settings and a 10:48:59 UTC observation showing no model gate or checkpoint. The
executed code, launch parameters, data, target policy and evaluation decisions
have not changed. Preserve the original public plan alongside this clarification.
