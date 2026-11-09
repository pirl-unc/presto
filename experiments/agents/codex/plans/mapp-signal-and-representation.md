# Next availability step: observation signal and identifiable representation

October 23, 2026, Codex / GPT-6. Prospective plan, not launched. Previous family
`2026-10-22_2330_codex_mapp-biological-comparison` fitted nine biological models.
Training KL fell to 0.023–0.030 while held-out KL/RMSE lost to uniform. Exact and
REINFORCE were similar. A new optimizer ablation is not the next informative test.

## Hypotheses and order

1. Quantify whether the admitted UT/TNF+IFN within-parent redistribution exceeds
   variation among the original acquisition replicates. The shared source already
   contains per-run quantities and contributor links: use them, do not request
   data that already exists. Add any reusable observation recipe in Hitlist's
   existing registry. Preserve three filtered runs, zeros, direct evidence and
   all acquisition identities. Publication-level biological-triplicate claims
   do not resolve exact source run/specimen pairing. Use unpaired comparisons
   unless a source establishes a pairing; no run-number matching.
2. Register the exact variability estimand and handling of detection/filtered
   zeros before computation. Avoid log-zero pseudocount choices after seeing
   effects. Keep separate source-positive, complete-observation and filter-state
   audits; report how eligibility changes the universe. Establish what is
   identifiable before fitting another condition-response model. Do not call
   three acquisitions a precise biological noise ceiling.
3. Test a small observed-set scoring baseline with only features supported by
   training data, alongside the existing span-policy exact/REINFORCE reference.
   Use explicit treatment-by-sequence feature interactions so a shared context
   offset that cancels in normalization cannot masquerade as a useful signal.
   Preserve source separation, parent/flank/position meaning and all missingness.
   Do not include IDs or per-example target-derived features as sequence inputs.
4. Separately consider response prediction conditioned on a measured untreated
   baseline. This is a different task from de novo sequence prediction; stable
   peptide-response cancellation is an assumption. Register it explicitly and
   never compare its errors to the earlier sequence-only task as if inputs were
   identical. Protein-wide scale and outside-observed-set mass stay unidentified.

## Gates and evaluation

Freeze a timestamped family with code/data/hash and exact conditions before any
new analysis or fitting. Do not reselect the old source split. The original test
is now observed and any further use is development, not a new blind test. Reserve
untouched source/specimen evidence for actual confirmation; track known inhibitor
and reporter panels in Hitlist759, precursor/run pairing in757, broader APC data
and observation-response calibration in723/739. No author outreach or cloud runs.

First report replicate-support and effect variation. Then specify a small fixed
model matrix and one selection metric, training-only capacity/gradient checks,
source uncertainty and a bounded local runtime. Retain uniform and no-context
controls; exact and sampled losses must share data/objective. Do not introduce
GRPO clipping/reuse until the scores are costly or there is evidence it addresses
an observed bottleneck. Avoid another memorization exercise: saved final weights
already demonstrate substantial fit to the current training target.

The full goal remains distributions of processing products, including short
nonbinders, modulated by machinery and combined with MS response and downstream
pMHC selection. An observed-set scorer is an identifiable diagnostic component,
not a replacement for that generative objective or proof of concentrations.
