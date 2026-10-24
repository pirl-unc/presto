# First native-cell treatment response prediction

Prospective plan, before fitting or viewing response-prediction metrics. Continues
native-response-recipe.md and the full APC availability objective; this family
establishes biological controls, not a replacement concentration model.

## Admission closure

Use the frozen 0259 canonical Hitlist v3 recipe manifest c72980c0… unchanged.
Independently verify its source identity grammar, complete reference occurrences,
all-source connected components and split assignment, positive exact/bounded
target admission, possible reuse, response-group contributors, and hierarchical
weights. Use a whole-string sequence grammar, Aho–Corasick reference matching
checked against naive overlapping searches, and bipartite graph traversal rather
than the producer's token scanner, prefix matcher and union-find. Verify reference
and source file hashes and source-table equality. Keep shared denominator proxies
and unresolved specimen identity visible. Preserve any failed attempt; do not
reselect splits or repair quantities locally. No predictive metrics in this step.

## Question and data

Does peptide sequence improve prediction of native extracted peptide treatment /
control ratios over a condition-only predictor on held-out parent components?
Use all 1,861 training response groups / 6,735 effective observations and 245
validation groups / 932 observations. Exact values and source-declared upper/lower
bounds retain the canonical weights. Repeated source copies carry their original
contributor identities and at most one effective observation. Ratios are neither
absolute concentrations nor MHC binding or presentation labels. No synthetic
targets, generated negatives, clipping, target averaging or length restriction.

The fixed 631-group test split is reserved for the predeclared final comparison
with the processing models. This first baseline family must not calculate test
prediction metrics, choose parameters using validation, or change that split.
All recipes contain the full source data for provenance; source verification is
not model selection. No specimen, homology or external-study independence claim.

## Fixed models

1. No change: predict zero log10 ratio.
2. Exact-label train mean, respecting canonical observation weights renormalized
   over exact training observations.
3. One constant minimizing the same exact/censored training objective as the fits.
4. Condition-only: global intercept plus the training condition indicators.
5. Shared sequence: condition indicators plus log peptide length, 20 amino-acid
   fractions, first/last residue indicators, N-terminal acetylation indicator and
   oxidation count. No parent identity, source study, measurement value, raw
   peptide identifier, charge or replicate index enters the features.
6. Compound-dependent sequence: same model plus interactions between the sequence
   features and the literal training compound name. No assumed drug mechanism,
   KO activity or effect direction. This is a richer diagnostic, not a default
   promotion based on training error.

Training-only weighted centering/scaling removes constant columns. A weighted
design SVD removes unsupported linear directions (relative singular threshold
1e-10); retain its transformation and rank. Unknown categories map to zero and
are counted. All allocated fit coordinates have training support. This does not
claim support for arbitrary out-of-domain sequences or unseen treatments.

## Objective, optimization and acceptance

For exact observations use squared log10-ratio error. For a source upper bound use
max(prediction - bound, 0)^2; for a lower bound use min(prediction - bound, 0)^2.
Sum with the unmodified within-split condition/component/form/observation weights.
The three linear fits add 0.01 * squared coefficient norm in the orthogonally
projected standardized-feature space; exclude the global intercept. No regularizer
search. Initialize all coefficients at zero. Use float64 deterministic full-data
L-BFGS-B, maxiter=1000, gtol=1e-9, ftol=1e-12; retain convergence/gradient evidence
and final parameters. No pretraining, validation early stopping or restarts.

Local CPU, one numerical thread, no cloud or environment changes. Cap each fit at
300 seconds and the family at 900 seconds / 4 GiB. Keep any failed or nonconverged
fit visible, without changing its settings after viewing outcomes. If the feature
matrix exceeds the cap, stop and register a new design, rather than subsample.

Primary biological comparison: weighted validation exact/censored data loss on
the same groups. Also report weighted exact-only RMSE/MAE/R2, pooled Pearson and
Spearman, prediction/target spread, sign accuracy for nonzero exact targets,
upper/lower bound violation and satisfaction, and condition/component strata.
500 nM binding metrics do not apply. Sequence value requires improvement over
condition-only AND no-change on the primary validation metric; report whether
exact-value error improves separately. No significance or independent replicate
claim from the source slot counts. Do not promote based only on pooled correlation.

## Artifacts and next experiment

Register a timestamped family before executing. Freeze config/source, git/dirty
state, environment overrides, admitted recipe and verification receipt hashes.
Retain final feature transforms, coefficients, convergence/runtime evidence,
train/validation per-observation predictions, aggregate and stratified metrics,
an interpretation plot if helpful, README and the canonical experiment log.
Independently reconstruct features/predictions and metric reductions from source
examples; verify analytic objective gradients on a small upper/lower-bound toy
and a finite-difference direction at real fit parameters. Do not broaden tests
after these pass without a new failure or concern.

The next physical comparison keeps this same endpoint/reductions and fixed split:
full-parent independent cleavage plus precursor input/removal, treated/control
availability ratios, exact versus explicitly described sampled gradients. The
full goal still needs APC machinery effects and precursor-relative concentration
validation; neither this response baseline nor a source admission passes that gate.
