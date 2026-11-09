# Local full-cohort IC50 convergence and replication

Prospective continuation of `full-ic50-fitting.md`, Codex / GPT-6, 2026-10-26.
The user requires scientific milestones before merging PR 131. This family
addresses the exact-IC50 gate only; all remaining architecture/data gates persist.

## Unchanged scientific contract

Original manifest SHA-256:
`914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be`.
Use all 21,636 exact training measurements and original 2,925 validation / 2,803
test measurements, including evaluation bounds. Historical peptide-disjoint
partitions have prior outcome exposure, shared studies and unknown donor/specimen
independence. No cap, clipping, averaging or replacement of observations.

Conditions: training mean; length/20-composition/class ridge with alpha 1 and
unpenalized intercept; fresh modular exact-IC50 fits with seeds 42/43/44; seed-42
within-class training-label permutation with NumPy seed 424242. Preserve original
labels and permutation assignments in provenance. Test labels and splits remain
unchanged. No auxiliary pretraining/replay, MS/APC/TCR training or generated
biological negatives. Permuted values are explicitly a negative control.

Model: shared d128/l2/h4 sequence encoder, supplied species/class, whole-peptide
class I and explicit class-II registers/flanks, direct IC50 scalar. AdamW 3e-4,
weight decay 1e-4, dropout .1, batch 64, complete epochs, maximum 100, patience 15
after minimum 10, constant LR; no arbitrary optimizer-step cap. Per-condition
exact training-only target scaling, standardized squared error weight 1.
Checkpoint selection uses validation exact RMSE; test is evaluated once after
selection. All four conditions are declared before fitting; no best-seed choice.

## Local execution amendment

The later October 22 feasibility plan records the user's choice of local compute.
Use CPU, four Torch/BLAS threads, one interop thread, float32, no autocast/compile,
no package changes and no cloud GPU. Execute one condition at a time. Freeze all
manifests and runtime source before starting. Save resume every 50 updates.

For this full-cohort family, require 10 GiB available RAM and 15 GiB free disk at
startup; retain 5 GiB available RAM and 10 GiB free disk during execution. Stop
and retain failure/checkpoints at process RSS 6 GiB or six hours per condition.
The prior 2.59 GiB one-epoch peak motivates explicit headroom for richer saved
diagnostics and evaluation caches. These limits belong to this experiment; no
earlier source-audit resource contract is altered. No background child training
processes or parallel source materialization. A guard failure is not convergence.

## Gates and independent verification

Retain the original screens: every real seed must improve training RMSE over
mean, pass tensor/feature coverage, have positive exact validation Spearman and
paired peptide-bootstrap upper 95% RMSE-difference bounds below zero versus
both mean and ridge, overall and separately for I/II. Use 2,000 peptide-component
resamples with seed 20261021. The permutation must not reproduce those advantages.
Report contraction when prediction/target SD < .1; never inflate variance.

Save every split's raw/transformed values, qualifiers, original observations,
resolved MHC/context/register diagnostics, predictions and losses. Report exact
RMSE/Pearson/Spearman, spread, 500 nM metrics, bound loss/satisfaction, and supported
allele/study strata. Include bound-established binding classes only under their
strict/inclusive inequalities; retain ambiguous counts. Keep all conditions.
Recompute metrics and baseline coefficients independently from saved outputs,
verify source joins and hashes, replay a fixed training batch from each selected
checkpoint in a fresh process, and never rescore test data to repair reporting.

Copy compact results and compressed predictions into the timestamped family;
keep checkpoints/evaluation caches in `artifacts/<family>/` with hashes and paths.
Freeze `reproduce/launch.sh`, `launch.json`, manifests and runtime/launcher source.
Update README and the canonical experiment log for each terminal result and final
family closure. A failed or incomplete gate keeps PR 131 draft. No APC quantity,
MS transfer, TCR or independent-study qualification follows from this family.

## Reporting and sequential continuation, before neural completion

Preserve the already frozen launcher, runtime and independent verifier. Add a
companion report of all catalog-allele-pair and reported-study strata, without
minimum-count selection: each observation contributes once per axis; keep
unknown labels explicit and retain counts when correlations are undefined.
Retain the original support rule (at least 20 exact measurements and nonzero
target spread); label smaller/constant strata unsupported even when displaying
descriptive estimates. Include class-specific bound/threshold metrics as well.
Study keys use original reference IRI, otherwise PMID, otherwise unknown.
These are descriptive strata, not newly independent holdouts or promotion gates.
Also retain prediction range/quantiles and threshold confusion counts.

A sequential continuation may wait for the currently running seed-42 action's
terminal receipt, then verify it and run the remaining declared conditions in
their original order. Each foreground trainer retains its own resource monitor;
never overlap trainers, change the recipe, automatically retry failed actions,
or rescore a completed test partition. Stop on execution, source-pin or numeric
verification failure. A scientific gate failure is a result: preserve it and
finish the other declared arms. Record each transition durably and refresh the
family README and canonical log from terminal saved outputs. No automatic merge
or checkpoint qualification follows from the continuation.

The user confirmed on October 26 that the authorized TCRvdb release is not yet
available. That is an external blocker for TCR qualification; it does not change
this independent IC50 contract or authorize substituting a different TCR cohort.

Continuation attempt 1 stopped before any training/verification action because
the sandbox rejects signal-zero process probes. Preserve its source, contract,
state and log. Attempt 2 waits on the atomic terminal receipt and original
six-hour deadline plus 120 seconds for finalization; it does not infer liveness
from a PID or restart training. A stale starting receipt eventually stops this
continuation rather than authorizing another fit. Keep a distinct attempt state
and log. No runtime/model/data/fit-recipe changes.
