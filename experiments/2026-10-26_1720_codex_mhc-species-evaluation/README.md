# MHC reference species evaluation

Prospectively registered by Codex / GPT-6, 2026-10-26. **Active: fixed controls independently verified; neural evaluation pending.**
See [frozen plan](reproduce/plan.md) for the source, fixed conditions, selection,
metrics, limitations, resource guards and verification contract. This evaluates
single-domain reference species labels only; full scientific qualification of
PR131 remains incomplete. Neural work follows the active IC50 family serially.

Status: active. Next step: neural evaluation after IC50 finishes and verifies.
Open questions: classification quality and subsequent matched affinity transfer.

<!-- mhc-species-evaluation-20261026:start -->
## MHC reference species evaluation — 2026-10-26_1720_codex

Date: 2026-10-26. Agent/model: Codex / GPT-6.

Incomplete; fixed controls independently verified; neural conditions unfinished.

Experiment directory: `experiments/2026-10-26_1720_codex_mhc-species-evaluation/`.
Exact contract and reproducibility bundle: `reproduce/plan.md`, `reproduce/launch.json`,
`reproduce/launch.sh`, `reproduce/verify-v2.sh` and frozen `reproduce/source/`.
Initial executed commit: `e4d2969ce86f7097652e551653dd91f1cb8bd29e`; exact dirty state is in the receipt.
Original data/runtime remain unchanged; generated audit files were untracked.

Data: 13,218 four-species single-domain reference annotations (6,092/694/6,432
train/validation/test), original aliases and fixed exact-domain/reference-chain
components. Human/mouse/rat/rhesus macaque map to the shared mhc_species head
through their actually observed mhc1 or mhc2 input. No pairing/class labels,
synthetic data, assays, bounds or censoring. Regression/500 nM metrics do not apply.
No affinity pretraining/transfer is evaluated. Prior and length/composition/slot
softmax controls fit only training data; softmax uses summed CE plus L2 alpha1,
unpenalized intercept and training-only standardization. Neural seeds42/43/44
use d128/two layers/four heads, dropout0.1, AdamW3e-4, decay1e-4, CE weight1,
batch64, max50/min10 epochs, patience10, minimum validation CE selection and
one selected-checkpoint test evaluation. Neural conditions run after IC50 serially.

Local CPU only; no GPU/Modal requested or observed. Existing Python3.12.6;
read-only pinned dependency overlay, one control/verifier thread or four neural
threads, interop one. Exact environment and hardware are in each receipt.
Fixed-control elapsed time: 5.523s; peak process RSS: 368852992 bytes.

| Condition | Train CE | Validation CE | Validation balanced accuracy | Validation component CE | Test CE | Test balanced accuracy | Test component CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| composition | 0.308779 | 0.557641 | 0.607901 | 0.486352 | 1.024366 | 0.452151 | 0.448776 |
| prior | 0.644654 | 1.060291 | 0.250000 | 0.736348 | 0.320825 | 0.250000 | 0.698657 |

Full per-species/slot discrimination, confusion, Brier, calibration, probability
variation and component metrics are in `results/comparison.json`; the concise
table is `results/comparison.csv`. Every split's predictions are preserved in
`results/baselines/*-predictions.jsonl.gz` and, as completed, each neural result
directory. Raw checkpoints and native dumps: `/Users/iskander/code/presto/artifacts/2026-10-26_1720_codex_mhc-species-evaluation`.

Independent verification checks all original source fields plus precisely the
manifest evidence-policy normalization, all metrics and control probabilities,
training-only feature scaling and the analytic convex-objective gradient.
The original raw-row equality checker rejected that known loader-added field;
its source, failed receipt and log are retained. The v2 correction and exact
source/prediction hashes are in `reproduce/verification-amendment.md` and
`reproduce/verification-v2-contract.json`. No control was refit and no test rescored.

Interpretation: the composition control improves validation discrimination over
the prior but remains below the prespecified neural screen. Test observation CE
favors the prior while equal-component CE favors composition: the existing
5,664-row human component dominates the observation-weighted result. Neither
statistic establishes homology-independent or population-level generalization.
There is no qualified neural winner until all three fixed seeds pass. Even a
passing species screen would leave affinity transfer, class/pairing inference,
APC/MS/TCR science and the PR merge criteria open.

Status: active; not scientifically complete.
Next step: finish serial neural fits, independent replay and closure.
Open questions: reference generalization and useful affinity transfer.
<!-- mhc-species-evaluation-20261026:end -->
