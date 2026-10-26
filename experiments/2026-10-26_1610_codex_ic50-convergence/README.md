# Local full-cohort IC50 convergence and replication

Status: registered, software verification and launch preparation in progress.
Agent/model: Codex / GPT-6; date 2026-10-26. No new scientific result yet.

Prospective contract: [local convergence plan](../agents/codex/plans/2026-10-26_ic50-convergence.md).
Runtime source before registration: 9e9e0d7 (full hash is frozen at launch).

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
