# Prospective grouped nontryptic MS1 observer

Codex / GPT-6. Plan frozen after canonical source qualification1635 and before
selecting/downloading additional pool quantities. This is a prerequisite observation
constraint for the active APC availability objective, not a replacement objective.

## Hypothesis and endpoint

Peptide sequence predicts relative positive MS1 response within intended synthetic
pools beyond length alone. Fit the canonical registered conditional endpoint,
pxd021013_intended_unmodified_ms1_v1, from Hitlist PR814 commitb9b1047 or its verified
release descendant. Preserve crude-synthesis/handling/unknown-molarity limits. No
MHC pull-down, APC abundance, pure detector probability or concentration claim.
Direct differentiation is appropriate here: this observer has no discrete processing
steps. Later exact/sampled processing comparisons must use the same frozen observer.

## Outcome-independent cohort selection

Use the verified canonical full intended-product mapping and original deposited
README, not quantities or model scores. Limit to HLA Class I/II library pools with
an exact indexed `{pool}_01_01_DDA-1h-R1-unspecific.zip` SEARCH file and single RAW
mapping. The two inspected pilot pools remain development-only and are excluded
from this cohort, along with any pool sharing an intended sequence with either.

Within each library sort candidates by SHA256(`presto_ms1_observer_v1:`+pool_name).
Alternate libraries, accepting the next pool only if none of its intended sequences
overlaps any accepted pool. Stop at12new pools per library. Freeze selected/skipped
pools, reasons, intended sets, index IDs and exact filenames before downloading
quantitative assets. No alternate seed or outcome-dependent replacement. If24pools
cannot be selected, report the metadata limit before changing the design.

Within each selected library retain selection order: ranks0–1test,2–3validation,
4–11training. Thus16training/4validation/4test pools, with no intended sequence
shared across any pools. This is same-study pool generalization, not independent
laboratory, protein-homology or instrument transfer. Do not claim biological specimen
independence from pool disjointness.

Acquire exact original archives with URL/time/status/hash/byte receipts; no access
bypass. Core source assets already qualified. Resource limits: local CPU only,
3GiBprocess RSS,1GiBnew transfer,2GiBnew artifacts,128MiBper archive; require4GiBfree
before acquisition and preserve at least2GiBfree. Stream with bounded concurrency,
retain failures, never overwrite existing assets. Stop at a resource gate rather
than silently shrink the cohort. Canonical source/recipe performs all curation.

Use the frozen source eligibility policy. Record exclusions and context for every
run. This first comparable cohort requires the source's fixed Carbamidomethyl(C)
setting and MaxQuant1.5.3.30, with the indexed DDA/unspecific acquisition/search
contract. Context mismatches are explicit exclusions, not silently harmonized.
Require all24selected pools to yield at least50admitted examples; if not, stop and
report before fitting. No held-out response/CV filtering beyond the source recipe.

## Fixed models and inputs

Every model predicts centered log10 signal over the same admitted candidate set.
No raw intensity, target center, pool name, library label or held-out fitted offset
is a model input. Supplied context defines the cohort, not a learnable constant axis.

1. Zero prediction, the conditional uniform-signal control.
2. Length and squared-length ridge regression, with no output intercept.
3. Sequence ridge: length/squared length,20amino-acid fractions,20N-terminal and
   20C-terminal indicators (62raw features), no output intercept.
4. A small nonlinear control on those62features: one32-unit tanh layer and scalar
   linear output without bias; trainable hidden biases. No pretrained encoder.

Training-only, equal-pool-weighted means/scales; mask exactly constant feature
columns before allocating parameters. Retain transforms/masks and test extrapolation
flags. Ridge minimizes equal-pool mean squared error plus alpha times mean squared
coefficient, alpha in[.001,.01,.1,1]. Center feature predictions within each run;
closed-form solutions must reproduce gradients/normal equations. For the nonlinear
model, standardize raw features, predict each peptide and then subtract the run's
mean predicted score. This centers predictions without reading target offsets.

Nonlinear controls: seeds42/43/44, full-batch Adam400steps,lr.003,betas(.9,.999),
eps1e-8, mean-squared-parameter penalty.001, no schedule, clipping, dropout or
restarts. Equal mass per pool and equal mass per example within pool. Keep checkpoints
at0/25/50/.../400and record finite gradients, nonzero changes and unidentified output
shift directions. Never present numerical parameter activity as biological accuracy.
Budget30CPUminutes total fitting; stop and preserve partial work if exceeded.

## Selection, held-outs and reporting

Validation selects ridge alpha and each nonlinear checkpoint from the predeclared
set; seeds are reported separately and as a predeclared mean-score ensemble, not
selected by test. The primary comparison is validation-best sequence ridge versus
validation-best length ridge. Nonlinear results are a prespecified secondary test.
Seal every model and test covariate-only prediction array before loading test
outcomes for evaluation. No test-driven changes, offset estimation or retries.

Report validation/test equal-pool MSE/RMSE, MAE, Pearson/Spearman, R² and prediction
SD versus target SD; per-pool and library strata, coverage and distributions. Preserve
per-example source-linked target/prediction dumps and checkpoints/transforms/weights.
These are log-relative-signal errors, not log10(nM); binder thresholds do not apply.
With four test pools, uncertainty/generalization conclusions must remain modest.

Predeclared useful-signal screen: sequence ridge improves RMSE at least5% over the
length control on both validation and test, has positive within-pool correlations
in at least3/4test pools, and prediction SD at least0.1times target SD. Secondary
nonlinear results cannot rescue a failed primary screen by renaming the winner.
A passed secondary model may motivate a new independent study but does not validate
transfer to biological elution. No seeds or eligibility rules are retuned after test.

Close the experiment in a timestamped registry directory with exact source, code,
environment, launch and resource receipts, failed attempts, all metrics and all
prediction dumps. Compare to the pilot admission without treating it as earlier
prediction. Keep the previously inspected HAP1/MAPP/native/macrophage held-outs
closed. The subsequent availability experiment must combine an independently
qualified observer with processing on a new biological cohort and compare fixed
exact and sampled objectives; intracellular concentrations remain unachieved until
supported by appropriate independent measurements.
