# Remove the observed nuisance-optimization gap before expanding biology

## Motivation and prospective scope

The200-update joint-offset analytical control in2026-10-23_2108 has independently
verified development shapeRMSE1.28894, worse than constant1.28060 and the older
2,000-full-batch exact fit1.26510. Its joint training dataMSE1.59302 exceeds its
profiled shapeMSE1.39693 by0.19609, exactly the weighted squared distance between
learned and optimal condition offsets. Its joint training objective is also above
the common initialization. This is an observed optimization limitation, not proof
that it causes the poor development prediction or that fragment-length biology
is wrong. Finish and publish both current arms before executing this follow-up.

This prospective experiment changes nuisance optimization only. It adds no
biological feature, dataset filter, sampler variant, seed search or predictor head.
The full APC availability/concentration goal and original test exclusion persist.

## Same physical models and observation objective

Reuse the exact pinned90,630training/14,822development observations, original
40-coefficient initializer,38 response coordinates, source/instrument features,
weights and all28condition/time blocks. Preserve the fixed200draw seed42 schedule
from the joint-offset pilot. Two new fits: analytical eta0 and fragment eta=tanh(rho).
Same common initialization,200Adamupdates,lr0.03,betas0.9/0.999,eps1e-8, existing
nonintercept/response penalty0.01, physical-eta penalty0.01, final checkpoint only.
Retain the two joint-offset arms as frozen references; do not rerun them merely to
obtain different outcomes. Report prior2,000-full-batch controls separately.

Allocate only40theta+38gamma (+one rho for the fragment arm). Each condition
retains its same one constant nuisance offset, but compute its weighted optimum
analytically at the current parameter state instead of allocating an Adam variable:

    beta_c* = sum_{i in c} w_i*(y_i - log10(count_i) - response_i) / sum_{i in c} w_i.

For the sampled condition/time block, first compute all required times/spans of
that condition under no_grad to obtain beta_c*. Use every admitted observation of
that condition. Then compute the selected block's differentiable amounts and
hold beta_c* detached in its probability-weighted loss. Do not reuse stale offsets
across parameter updates, fit separate offsets by time/peptide, or drop any row.

The estimator is unbiased for the profiled objective gradient: the missing beta
chain term cancels only after weighting all blocks of a condition, because its
weighted residual sum at beta_c* is zero. It is not the derivative of a separately
profiled time block. Prove the expectation by enumerating blocks on both the
nonlinear toy and actual analytical reference before launching. Verify the
profiled full objective and envelope gradients agree with the old source trainer.

## Execution gates and cost

Keep all fragment lengths1..L. Check2/4-step positive finite required log amounts
at every queried condition/time before each update and the selected block after
it; preserve the1e-5natural-log threshold. Reuse same-state detached numerical
references within an update if implemented and explicitly qualified; never reuse
across updates. Final complete train/development outputs must pass. No floors,
row exclusions, eta resets, failed-checkpoint substitution or outcome-driven
schedule extension. If an existing resource cap is inadequate, measure a
quantity-closed forward/backward cost and freeze a new explicit bounded receipt
before fitting; do not extend a failed fit after seeing its outcomes.

Record all allocated-coordinate gradient/update counts, full200Adam states,
source-linked predictions, learned eta/rates/response ranges, per-block numerical
checks, runtime and peak RSS. Condition offsets are analytically solved nuisance
quantities, not untrained exposed model parameters. Development offsets remain
label-derived scale-invariant diagnostics, not absolute out-of-sample predictions.

## Decisions

Primary evidence is actual biological development shapeRMSE, Pearson/Spearman,
spread and original temporal-ratio metrics versus matched eta0, constant and
published longer-fit controls. Training profiled objective must be reported
relative to initialization; profile gaps must be numerical zero by construction.
The three already-inspected development parents are not a new blind test, and
one seed cannot establish robustness. An improvement qualifies this bounded
predictive contract only; it does not calibrate concentration or identify the
length effect causally. If neither model improves after coherent optimization,
reconsider the biological/observation hypothesis or source support before more
sampler engineering. Independent known-load MS calibration, APC machinery,
transport/clearance and cellular validation remain required.
