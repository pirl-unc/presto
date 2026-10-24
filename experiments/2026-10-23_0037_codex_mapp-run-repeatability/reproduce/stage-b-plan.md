# Stage B: parsimonious sequence scores — fixed before new fitting

October 23, 2026; Codex / GPT-6. Stage A finds reproducible baseline composition (pooled Pearson0.914), weaker run-held-out treatment differences (0.172), and higher change prediction RMSE from other-run differences than zero change (0.220 vs0.192). These are selected acquisition-level observations, not a biological noise ceiling. No model in this stage has been fitted yet.

## Data and endpoint

Use the previous shared **min2 median cohort** unchanged:616peptides/101parents,72/15/14train/validation/test. The complete-run subset is only a source-repeatability audit; its smaller55/7/9partitions are not a new fitted cohort. Pin the previous manifest and every source label. Repeat original mean protein-condition reverse KL against captured-intensity composition, with no missing negatives or concentration claim. No measured UT baseline supplied as an input. Prior test is already exposed; all test results here are development evidence, requiring later untouched biological confirmation.

## Fixed representations

- Length-only exact control: one normalized length feature, no context.
- Sequence-only linear score: peptide length/40, canonical20 amino-acid composition, first/last residue one-hots (20 each), preceding/following residue including explicit boundary (21 each), relative start/(parent_length-1) and end/parent_length.105 raw columns. One candidate score, softmax over the provided observed parent set; no intercept.
- Sequence plus treatment interactions: concatenate the same features and treatment0/1 times those features. No context-only offset that cancels under normalization.

All admitted candidates in the frozen cohort have one exact position; assert that before fitting, rather than silently assigning one when ambiguous. Full proteins remain available for flanks/positions. This is an identifiable conditional scorer, not a replacement generative trajectory policy; it has no predictions for mass outside the supplied observed set.

Center features within each parent set. Determine active columns using only TRAIN covariates: scale nonzero columns by pooled training RMS, then pivoted QR with rank threshold1e-10 times largest diagonal. Keep original named columns selected by pivots, in sorted index order. Save all names/scales/pivots/diagonals/rank, omitted columns and validation/test residual outside the TRAIN column span as an exposure audit. No labels enter feature selection. No IDs, source quantity, run metadata or condition-derived target statistic becomes a sequence feature. Every allocated parameter should receive a nonzero gradient and update during training; report selected/final counts and zero/untrained parameters without promoting unsupported effects.

## Matrix and optimizer

Five prespecified L2 weights:0,0.001,0.01,0.1,1 multiplying sum(theta^2), in training-RMS coordinates. Score theta starts at zero, so every run starts uniform. Adam lr0.03, no weight decay/dropout/pretraining, full batch of144 training distributions,1000updates; CPU one intra-op/interop thread. Exact gradients: all three representations, one deterministic run per L2 weight (15fits). REINFORCE64: treatment-interaction representation only, seeds42/43/44 for independent replacement draws, each L2 weight (15fits). Total30fits. Full objective is mean reverse KL plus the identical analytic L2 term. Leave-one-out advantage, no standardization/clipping/reuse/reference KL. As before, all candidates are scored in both methods; sampling has no score-evaluation saving here.

Validate unpenalized mean reverse KL at update0 and every25updates. Preserve earliest minimum per fit, including uniform at update0. Within each architecture/method/seed, choose the lowest validation KL over the five regularization settings (ties prefer stronger regularization, then earliest step); evaluate test only for those six selected models (length exact, sequence exact, interaction exact and three interaction REINFORCE seeds). Do not inspect other candidates' test metrics. Report best exact architecture by validation separately; do not pick a winner from development-test rankings. No extra seeds, regularizers, features or epochs after outcomes. Stop/replan if total fitting exceeds15minutes, preserving partial results.

## Verification and closure

Before fitting, verify autograd exact KL+L2 against finite differences on TRAIN; verify seeded sampled estimates against the same exact gradient. Existing exhaustive score-function derivation applies. Save initial/best/final weights, optimizer states, histories, gradient coverage and validation predictions for every fit; unselected fits intentionally have no test dump. Save train/validation/development-test predictions for the six selected models, raw source targets and source identities. Report KL, TV, log10 RMSE/Pearson/Spearman, dispersion, condition change metrics, selected regularizer/step, support/coverage and runtime/work counts. Uniform and prior nonlinear-policy results are contextual baselines on the same median cohort.

Independently rebuild feature selection and NumPy scores from source sequences, verify selected checkpoints and all held-out metrics. Explicitly report that min2-source selection uses observed candidate sets, test is historical development data, features can capture observation response, and no free-pool concentration or untouched biological generalization is established. Freeze invocation, environment, git and source hashes before fitting; preserve all outcomes and update the canonical registry/report. This stage advances a potentially learnable observation component of the broader machinery/processing/MS/pMHC architecture.
