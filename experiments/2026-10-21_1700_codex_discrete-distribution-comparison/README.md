# Exact versus discrete learning of synthetic span distributions

Registered October 21, 2026, before fitting. Agent: Codex / GPT-6.
Status: implementation verification and source freezing pending.

The [prospective specification](../agents/codex/plans/discrete-distribution-comparison.md)
defines the entire comparison. This is a generic algorithm experiment on integer
strings with a known probability oracle, not a biological simulator or assay fit.

Inputs are 144 unique length-16 strings over eight integer symbols, split by parent
96/24/24. Four generic contexts per parent give 384/96/96 observations. Each has
70 actual spans of lengths 1–5. No biological sequence, machinery intervention,
pretraining, assay unit, censoring or synthetic binding-affinity target is used.
Short substrings intentionally overlap across otherwise disjoint parents.

Both methods use the same two-stage policy and matched starting weights. Exact
reverse-KL optimization is compared with REINFORCE using 8 or 64 discrete samples,
seeds 42/43/44, and a separate seed-42 reward-only control. A uniform distribution
is the unfitted baseline. Both methods have access to the same fixed oracle;
sampled training consumes only sampled target terms. All policy logits and the
teacher dataset are computed explicitly, so logical reward-query reductions do
not imply lower total runtime or fewer initial teacher evaluations.

Training uses Adam 0.003, no weight decay, batch 16, 60 complete epochs/1,440
updates, no initial weights or other loss terms. Validation reverse KL every
five epochs selects the checkpoint; test follows selection once. Sampling uses
a leave-one-out baseline and preserves substring multiplicities.

Execution is local CPU with one Torch/BLAS thread. No Modal GPU is requested or
used. The execution receipt will record the actual source commit/dirty state,
versions, environment and source hash. The source is frozen under `reproduce/`.
Outputs live in this family's `results/run/`; logs in `launch_logs/`.

Required metrics, prospective screens, per-example predictions, gradient probes,
fresh-process replay and independent verification are specified before launch.
Affinity regression in log10(nM) and 500 nM classification metrics do not apply:
there are no biological assay targets. The compatibility flag is an arbitrary
known arithmetic tag, not a learned binding endpoint. A fixed selected/biased
distribution is an evaluation-only check, not learned pMHC or MS supervision.

## Conditions

| Condition | Seeds | Status |
| --- | --- | --- |
| Uniform valid spans | None | Pending |
| Exact reverse KL | 42, 43, 44 | Pending |
| Discrete REINFORCE, 8 samples | 42, 43, 44 | Pending |
| Discrete REINFORCE, 64 samples | 42, 43, 44 | Pending |
| Reward only, 64 samples | 42 | Pending diagnostic, different objective |

No winner or result is asserted before execution. This family does not replace
the real-data pMHC/APC/MS qualification work or establish a biological target
distribution. It tests whether the two generic learning routes work on a known
finite distribution and how their estimators differ.
