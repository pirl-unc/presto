# Fixed-budget affinity refinement with existing censor bounds

Codex / GPT-6, 2026-10-26. Parent objective: `tasks/mhc2_improvement_spec.md`.
Prospective family: `2026-10-26_1929_codex_mhc2-censor-refinement`.

## Question and scope

The verified first convergence run learns class-II ranking but has weak bound
satisfaction. Test whether the already admitted censored affinity measurements
improve a fixed refinement of that same checkpoint. This is a real full-cohort
training comparison, not a fixture or new source-label inference. It is an initial
bounded development comparison; replication and independent-source qualification
remain necessary even if it succeeds. Do not change the older active family.

## Frozen data and parent

- Parent: selected epoch26 of old seed42, SHA256
  `db2585e63489cb5aac8c7a32fd7e522bcbc51035846c05f8cd2794c4449e3dc5`, at
  `/Users/iskander/code/presto/artifacts/2026-10-26_1610_codex_ic50-convergence/seed42/selected.pt`.
- Exact source manifest: SHA256
  `914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be`.
- Exact-plus-censor source manifest: SHA256
  `7fd143e22082dc18c1b7332954f66608bb7b0840813e5e12d74b3c85806dfc7c`.
  Both remain in the original `data/modular_cohorts/2026-10-21-full-ic50/` directory.
- Exact train21,636; bounded train1,919; validation2,925; test2,803. Class-II exact
  train7,261 and bounded train1,028. Preserve every original source row, target,
  qualifier, candidate pair, grouping and partition. Check all file hashes.
- Human, direct IEDB/CEDAR purified-MHC IC50, canonical unmodified peptides,
  explicit complete groove pairs. Existing physical nM targets are unclipped.
  Exact squared error and one-sided bound loss use the *inherited parent scaler*;
  do not recompute a different scale when censor rows enter the training set.
- Historical peptide-disjoint partitions share studies and have already been
  inspected. All validation/test results here are development evidence, never a
  newly independent test. No MS, receptor, identity or proxy-affinity loss enters.

## Conditions, compute and checkpoint policy

Two conditions, both initialized from exactly the same parent:

1. `exact_continuation`: existing21,636exact training rows.
2. `censor_refinement`: the same exact rows plus all1,919admitted bounds.

Unchanged native ModularPresto architecture and canonical `fit_index` trainer.
All inherited parameters remain trainable. Seed101, batch64, AdamW1e-4,
weight decay1e-4, dropout.1, identity weight0. Each condition executes exactly
**1,476optimizer updates**, at most5epochs, min_epochs5, patience100. This equals
four complete passes of the23555-row censor cohort; the exact condition executes
four complete passes plus part of the fifth. All rows are used, no cap/subsample.
The changed exact/bound exposure mixture is the intervention; report actual
counts. There is no claim of identical exact-label exposures or causal isolation
of optimization effects merely because the update budget matches.

The **primary endpoint is the final `last.pt` checkpoint at update1476**. It is
chosen before fitting and is not changed using either holdout. Evaluate that
checkpoint once on all train/validation/test rows, preserving per-example dumps.
The canonical trainer also saves and evaluates its validation-selected checkpoint;
retain those outputs as secondary artifacts. Its5versus4validation opportunities
are not used for primary condition selection. No early stopping or further updates
are authorized by a favorable/unfavorable result in this family.

There is no new pretraining. The parent was trained on all exact rows from scratch
for41epochs, selecting26. No synthetic labels, permuted targets, extra sampling
or new auxiliary losses are introduced. Existing mean/ridge/parent outputs remain
context, while the matched continuation is the primary comparison.

## Local resource amendment

Both new conditions run serially within this family, using two numerical CPU
threads and one interop thread. A separate CPU stream may coexist with the older
four-thread family on the observed10-core machine. This supersedes any inferred
cross-family serial-only preference in the initial working plan; it does not
change or restart old jobs. Their measured peak was3.09GB and available RAM at
planning was13.8GiB. No cloud, GPU, package install or shared-environment change.

Start each child only with>=13GiB available RAM and>=15GiB free disk. Enforce
4GiB child RSS,>=10GiB available RAM,>=10GiB free disk and a two-hour wall limit
per fit/evaluation action. If a guard fails, stop the new family with durable
receipts/checkpoints; never kill or restart the old job. The higher available-RAM
floor protects the older runner's10GiB startup requirement. Runtime includes
concurrent activity and cannot be compared as an isolated hardware benchmark.

## Metrics, gate and independent verification

Report validation and test overall/within-class exact Spearman, Pearson,
log10(nM) RMSE, prediction spread, and500nM accuracy/balanced accuracy/precision/
recall/F1/AUROC/AUPRC. Report one-sided censor loss/satisfaction overall and by
class, including unresolved500nM classification bounds. Report train metrics,
all peptide exposures, steps, runtime/RSS, convergence history and coverage.

Development screen, separately on validation and test:

- Class-II exact RMSE improves at least2% against exact continuation.
- Class-II AUROC and balanced accuracy drop no more than.01; recall drops no
  more than.03. Report all directions, including Spearman and AUPRC.
- Class-II censored mean loss improves at least10% and bound satisfaction rises
  by at least.05 absolute. These are source-bound comparisons, not exact errors.
- Class-I exact RMSE worsens by at most1%; AUROC falls by at most.01.

These thresholds govern whether to replicate this fixed recipe from later parent
seeds; they do not qualify the full model. Independently compute paired exact-
peptide bootstrap intervals (2,000draws,seed20261021) for class-II RMSE differences.
An interval crossing zero means statistical evidence remains inconclusive even
if the point-estimate screen passes. Do not select a best seed or retune on test.

Independent checker must reconstruct source IDs/values/qualifiers, original
physical target transforms, inherited scaler, every saved prediction/loss and
all reported metrics. Reuse the previously frozen SciPy/sklearn metric reference
as independent arithmetic; do not call canonical `summarize` as verification.
Numerical loss tolerance remains the old1e-4relative/2e-6absolute. Audit equal
initial tensors/scalers, exactly1476updates, full cohort coverage, final-checkpoint
identity and unchanged source/runtime hashes. Fresh-process prediction probes
must reproduce saved outputs. Preserve failed attempts without changing criteria.
Fresh-process prediction probes allow2e-5absolute error in log10(nM), covering
float32 batch-shape arithmetic; this tolerance is frozen before fitting.

## Artifacts and closure

Freeze launcher/controller/verifier code and complete source/runtime hashes before
fits. Raw checkpoints and full predictions live in the declared local artifact
directory; experiment summaries, compressed predictions or exact raw paths/hashes,
all three split metrics and comparison tables belong in the registered family.
Record exact commands/environment, source contract and selected/final distinction
in README, launch.json and the canonical experiment log. Do not call the family
complete until both conditions and independent verification are terminal.

- [x] Prepare immutable source manifests and verify the common parent/scalers.
- [ ] Launch both canonical refinements under the resource guard.
- [ ] Evaluate and independently verify both fixed-update checkpoints.
- [ ] Close all outcomes and decide replication without changing the full goal.
