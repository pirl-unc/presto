# Local full-cohort IC50 convergence and replication

Status: active; all three true-label seeds independently verified; permutation pending.
Agent/model: Codex / GPT-6; date 2026-10-26; evidence updated 2026-10-27.

Prospective contract: [local convergence plan](../agents/codex/plans/2026-10-26_ic50-convergence.md).
Frozen runtime commit: `4d4c029777af357c35eea47762ce07a9f66f488a`.

All 21,636 exact training IC50 measurements, 2,925 validation and 2,803 test
measurements from the previously qualified source; original peptide-disjoint,
historically inspected splits and shared studies retained. Exact training,
bounded/exact held-out evaluation; direct log10(nM) IC50 with training-only
standardization. No pretraining or auxiliary/APC/MS/TCR losses. A within-class
label-permutation control is explicitly synthetic supervision, never biology.

Conditions: training mean, fixed length/composition/class ridge, neural seeds
42/43/44 and permuted seed 42. Local CPU/four numerical threads/one interop,
AdamW 3e-4, weight decay 1e-4, dropout .1, batch 64, maximum 100 complete epochs,
validation patience 15 after minimum 10. Test is evaluated once after validation
selection; preserve every condition without test-based recipe changes.

Raw artifacts: `artifacts/2026-10-26_1610_codex_ic50-convergence/` in the execution
checkout. Reproduction, manifests, frozen runtime sources and per-action receipts
will be under `reproduce/`; portable summaries and compressed predictions under
`results/`. No GPU/Modal requested. No independent-study or broader model
qualification follows from this within-contract comparison.

The family remains incomplete until all fixed conditions, baseline comparisons,
coverage, original-source replay, fresh-process training checkpoint checks and
independent metric verification close. PR 131 remains draft.

## Preparation and software verification

The integrated runtime passes 2,624 software tests (four explicit skips), pinned
Ruff and strict docs. Canonical prediction dumps preserve original observation
provenance and resolved register diagnostics; numerical predictions/losses are
unchanged. All four manifests and 21,636 within-class permutation assignments
were frozen before fitting. Source commit: `4d4c029777af357c35eea47762ce07a9f66f488a`.

Fixed mean/ridge baselines are complete. Independent verification checks all
54,728 train/validation/test predictions and source joins, rederives all reported
metrics, and refits ridge using augmented least squares (maximum coefficient
difference 3.35e-10). This is verification, not a learned-model quality result.
See `results/baselines-independent-verification.json`. All three true-label seeds are now verified; the permutation is running.
Each completed seed has train/validation/test predictions. Raw checkpoints persist at
`/Users/iskander/code/presto/artifacts/2026-10-26_1610_codex_ic50-convergence/`.

## Reporting and continuation audit

`code/verify.py` independently recomputes scientific metrics and the fixed
bootstrap screens; its frozen hash and tolerances are in
`reproduce/verification-contract.json`. `code/report.py` reads saved predictions
only, records every allele-pair/study stratum, flags insufficient exact support,
and updates the status below plus the canonical registry. The first baseline
descriptive report omitted explicit support flags; its compressed output and
source are retained under `results/reporting-attempt01/` and
`reproduce/source/report-v1.py`. The corrected view preserves all predictions and
the prospective 20-exact/nonconstant support rule. This is a reporting correction,
not new model evaluation or a changed scientific gate.

The sequential continuation waits for seed42's terminal receipt, independently
verifies it, then fits/verifies seeds43/44 and the permuted control in order.
Each trainer enforces the frozen resource guard. Three companion checks verify
row conservation/joint-pair grouping, undefined small-stratum metrics, failure
stopping and refusal to overwrite prior attempt logs. Explicit experiment lint
and format pass. The first continuation failed on a sandbox-blocked signal-zero
process probe before launching anything; its state, contract, log and source are
preserved. `reproduce/continue.sh` records attempt02, which uses the atomic receipt
and original six-hour deadline plus120seconds for finalization. It does not retry
or restart the active fit. The process PID is provenance only.

The user confirmed TCRvdb is not yet available. No author outreach or substitute
TCR cohort is authorized. This blocker does not alter the independent IC50 work.

<!-- ic50-convergence-status:begin -->
## Current saved-output status

As of 2026-10-27T01:41:50.502022+00:00.

| Condition | State | Selected / completed epochs | Validation RMSE | Test RMSE |
| --- | --- | --- | --- | --- |
| mean | verified | n/a | 1.360611 | 1.342001 |
| ridge | verified | n/a | 1.223458 | 1.229134 |
| seed42 | verified | 26 / 41 | 0.981908 | 0.990536 |
| seed43 | verified | 23 / 38 | 0.970887 | 1.025608 |
| seed44 | verified | 29 / 44 | 0.977051 | 1.021521 |
| permuted42 | not_started | pending | pending | pending |

All fixed conditions verified: **False**. IC50 comparison screen passes: **False**.
No best-seed selection, test-based retuning, or full-model promotion. All class-specific, 500 nM, censoring, allele/study and paired-bootstrap details remain in `results/`.
TCRvdb is not yet available (confirmed by the user October 26). Auxiliary, censor-training, APC/MS transfer and serving gates remain open. PR 131 stays draft.

Status: fixed comparison active. Next Step: finish and verify the remaining declared arms. Open Questions: authorized TCRvdb input and the separately scoped biological milestones.
<!-- ic50-convergence-status:end -->


## 2026-10-27 immutable replication evidence

All three true-label seeds passed the original independent scientific screen.
Each verifier checks 27,364 source-linked predictions; original source/allele/study
strata, coverage, runtime, selected checkpoint and train/validation/test metrics
are retained. No best-seed selection or test-driven training change is made.

| Seed | Selected / completed epochs | II validation RMSE | II test RMSE | II test Spearman | II test AUROC | II test recall |
|---|---:|---:|---:|---:|---:|---:|
| seed42 | 26 / 41 | 0.797476 | 0.731948 | 0.619303 | 0.824459 | 0.460784 |
| seed43 | 23 / 38 | 0.762530 | 0.734982 | 0.621880 | 0.823476 | 0.495098 |
| seed44 | 29 / 44 | 0.771376 | 0.740212 | 0.620564 | 0.836483 | 0.470588 |

RMSE is log10(nM); binding classification uses the original 500 nM threshold.
Exact class-II support is 758 validation / 842 test measurements. The replicated
rank/RMSE signal does not remove the sensitivity gap: test binder recall remains
46.1–49.5%. Class-I exact test support is 1,758; seed42/43/44 test RMSE is
1.092928/1.138833/1.131756, AUROC .807057/.785056/.783803 and recall
.808274/.774975/.748739. Full metrics, including Pearson, precision, F1, AUPRC,
all-label loss and censored-bound satisfaction, remain in each summary and
independent verification. These are historically inspected same-study holdouts.

Fit times for seeds42/43/44 are 7,746.178 / 6,062.245 / 8,327.909 seconds on local
CPU, four numerical threads/one interop; no Modal/GPU. Exact-only training, fixed
early-stopping policy, no pretraining/auxiliary loss and all resource limits above
remain unchanged. Censored training and independent study transfer are separate
unfinished requirements. The permuted42 arm is still running; the overall family
and PR131 scientific qualification remain incomplete.

`code/import_verified_seeds_20261027.py` copied 33 immutable result files,
28,448,826 bytes, into this publication checkout. Source hashes were checked before
and after copying and against each destination; the original running checkout
was not changed. `reproduce/verified-seeds-import-20261027.json` preserves the
complete checksum list and source path.
