# Native response: parent-boundary context, training-only development

Prepared after closed acquisition-sensitivity family0711: median complementary
acquisition Pearson0.8382, all nine positive. This selects the positive branch of
the prior plan. The failed screen0630 outer groups remain closed. This is a new
development comparison, not an independent biological test.

## Question and unchanged scope

Can immediate parent context outside a peptide explain the reproducible part of
its PSMD2 response better than the frozen peptide-only feature map? The current
61 features cannot distinguish the same peptide appearing between different
flanking residues. Test that specific missing information before adding a
neural encoder, physical simulator or sampled optimization to this cohort.

Use only the same158training contrasts,56parents/55dependency components from
recipe0612. Manifest SHA256:
`5f1f388876e2b5b3f65d254ff4939d2559077c2e6af1dffe603b05a1283b836a`.
No new target definition, acquisition selection, exclusion, reference, censoring
policy or held-out access. Retain the six-acquisition mean log10 ratio-change
target and equal component/parent/contrast weights. Do not weight examples by
the observed reliability diagnostic or remove influential components.

The two arms encode this one study's control/knockdown contrast. The model does
not generalize machinery states, identify molecular mechanisms or calibrate APC
amounts. Longer-term exact/discrete availability models must retain species,
cell/machinery context, precursor scale, clearance and MS observation terms.

## Four fixed conditions

1. Zero response.
2. Natural-log peptide-length contrast, the original one-feature reference.
3. Frozen61feature peptide-only contrast: ln(length),20AA fractions,20first-AA
   and20last-AA indicators, alphabetACDEFGHIKLMNPQRSTVWY.
4. The same61features plus88external-boundary indicators: four positions,
   two immediately before peptide start and two immediately after peptide end;
   each position has20canonicalAA+OTHER+BOUNDARY states. OTHER covers any literal
   noncanonical residue; BOUNDARY means outside the complete parent sequence.

For each endpoint, enumerate all overlapping exact occurrences in its admitted
complete parent. Average each boundary indicator over those occurrences; this
preserves unresolved origin uncertainty without choosing a favorable span.
Independently reconcile occurrence sets to canonical source origin start/end
positions for that peptide and parent hash. Require a nonempty match set, correct
parent SHA256 and literal slice equality; fail rather than drop a bad row.
Repeated reference records with the same parent sequence do not multiply weight.
Retain per-endpoint occurrence lists and mapped source origin IDs for audit.

Every nonzero model predicts g(peptide,context)-g(reference,context). No intercept,
parent identity, peptide identity, observed intensity, acquisition label or
reliability-derived feature is supplied. All surviving coefficients are solved
by weighted ridge; there is no untrained encoder or auxiliary head. External
boundary information can encode processing associations, not proven causal rates.

## Nested training-component cross-validation

All membership is deterministic and uses only training dependency IDs.

- Five development folds: integer SHA256 of
  `pxd018722-boundary-dev-v1|{dependency_group_id}` modulo5.
- Within each development-training partition, three selection folds: integer
  SHA256 of `pxd018722-boundary-inner-v1|{dependency_group_id}` modulo3.
- Confirm every outer fold has at least five components and every inner fold
  at least five. If support fails, stop without changing hash keys or fold count.
- For each nonzero condition, use alpha[0.01,0.1,1,10,100]. Minimize mean
  three-fold weighted MSE; exact ties choose larger alpha. No grid extension.
- Each fit uses only its fit rows to compute weighted uncentered RMS scales,
  drops exactly zero-RMS columns and normalizes fit weights to one. No centering.
- Objective is sum_i w_i(y_i-x_i beta)^2+alpha||beta||^2, with no division by
  feature count/rank. Freeze numerical solver and independent augmented-SVD
  verification before biological fitting.
- Refit each selected condition on that development-training partition, predict
  its omitted training components once, and combine all158out-of-fold predictions.
  Report all inner selections, coefficient vectors, scales/dropped features,
  design ranks, stationarity, prediction spread and per-component errors.
- No final all-training deployment fit is required for this hypothesis screen.

Use original normalized weights for pooled out-of-fold RMSE/MSE/Pearson/Spearman
(ordinary average-tie ranks followed by weighted Pearson), predictionSD/targetSD,
and every per-fold metric. Also retain ordinary unweighted descriptive metrics.
Mathematically constant predictions have undefined correlations. NoIC50/500nM
metrics apply. These are development OOF predictions, not new blind test results.

## Decision fixed before fitting

The boundary condition earns priority for independent biological confirmation
only if pooled weighted RMSE improves at least5%over all three controls,
predictionSD>=25%targetSD, at least4of5folds improve versus61features, and the
paired55component bootstrap95%lower bound for RMSE gain versus61features is>0.
Bootstrap2,000draws, NumPyPCG64seed202610250711, sorted component IDs, whole-component
resampling with replacement, retaining within-component weights and linear
percentile intervals. The bootstrap is descriptive: development models have
overlapping training sets and this is not a prospective transfer confidence bound.

If the gate fails, preserve the negative result and assess whether the training
support can distinguish sequence-dependent effects from parent/observation
structure before extending capacity. No successful RL comparison is claimed
under either outcome. A later exact/discrete test must share inputs, objective,
parameterization and evaluation support; sampling cannot rescue absent signal.

## Execution and closure

Before execution register a new timestamped family, freeze source/invocation,
verify data-only covariate mapping and qualify the actual fitting path on synthetic
known-effect examples. Then fit training components only. Independently replay
context tokens/occurrence aggregation, all inner fits/selections, OOF predictions,
weighted metrics, bootstrap draws and the fixed decision. Retain failures.

Local CPU/oneBLASthread, no environment installs. Unchanged20minute/stage,
3GiB RSS,1GiB disk reserve+32MiB startup margin;50ms process-tree monitoring and
OSpeakRSS. Record versions at stage start. Close README, registry, prediction
Parquets, model records, plots, code bindings and runtime. Distinguish preflight
permission failures from numerical training failures.

No outer validation/test rows or prior prediction files are used. No authoremail.
Missing specimen/replication metadata remain tracked on Hitlist837. Full APC
precursor-relative abundance, undetected/sub-ligand support, MS calibration and
prospective biological transfer remain required for the overall goal.
