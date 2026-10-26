# Local full-cohort IC50 convergence and replication

Status: active; fixed controls verified, first neural condition running.
Agent/model: Codex / GPT-6; date 2026-10-26. No completed new neural result yet.

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
See `results/baselines-independent-verification.json`. Seed 42 is running; other
conditions have not started. No test prediction has been produced by the new
neural condition yet. Raw checkpoints persist at
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

As of 2026-10-26T16:30:23.559768+00:00.

| Condition | State | Selected / completed epochs | Validation RMSE | Test RMSE |
| --- | --- | --- | --- | --- |
| mean | verified | n/a | 1.360611 | 1.342001 |
| ridge | verified | n/a | 1.223458 | 1.229134 |
| seed42 | starting | pending | pending | pending |
| seed43 | not_started | pending | pending | pending |
| seed44 | not_started | pending | pending | pending |
| permuted42 | not_started | pending | pending | pending |

All fixed conditions verified: **False**. IC50 comparison screen passes: **False**.
No best-seed selection, test-based retuning, or full-model promotion. All class-specific, 500 nM, censoring, allele/study and paired-bootstrap details remain in `results/`.
TCRvdb is not yet available (confirmed by the user October 26). Auxiliary, censor-training, APC/MS transfer and serving gates remain open. PR 131 stays draft.

Status: fixed comparison active. Next Step: finish and verify the remaining declared arms. Open Questions: authorized TCRvdb input and the separately scoped biological milestones.
<!-- ic50-convergence-status:end -->
