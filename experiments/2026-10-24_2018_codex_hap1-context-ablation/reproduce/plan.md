# HAP1 sequence versus source-context diagnostic

Frozen before implementation/fitting, October 24, 2026. Codex / GPT-6.

## Question and relation to the full goal

Does the successful HAP1 sequence-response regression depend mainly on peptide
position features, and does full-parent context add transferable information within
the training population? This is a bounded diagnostic to choose the next processing
model hypothesis. It does not substitute for the active APC availability/concentration
goal, claim causal cleavage effects, or compare new RL optimizers. HAP1 elution
combines processing, MHC selection and measurement. Positional indicators are not
allele assignments. Prior exact/sampled physical models failed to improve prediction.

## Fixed data and evaluation contract

Reuse Hitlist hap1_machinery_response_v1 at
`data/hap1_response_recipes/2026-10-24_0730_v1`, manifest SHA256
`a1eec392c06485966e131663f85e7599dcd9e5ff621841eb0d44eeaafe779309`,
canonical source `88cad10d7bc2a638305cb0d37ad96ac2826c3badaf7b3e0aad7928820df5c4b8`.
Read responses through Parquet filters split=train and precursor_status=complete.
Expected 87,260 positive KO-replicate log10-response cells / 6,317 peptides /
1,798 dependency components / ten conditions, as in 0730. Preserve all original
WT/B2M/precursor policies, parent alternatives, repeated occurrences and source IDs.
No new source curator, missing-negative labels, imputation, censor bounds, biology
simulation, synthetic training labels or source changes. Other assays and nM/binder
metrics do not apply. Existing outer validation/test sets were inspected in 1503
and remain closed: no new outer predictions, scores or test-selected changes.
Use the original five SHA256(hap1_baseline_inner_v1:+component_id) mod 5 folds.
These are reused development folds, not independent validation. Use original equal
condition/component/peptide/replicate weights, recomputed within fitting subsets;
pooled CV evaluation retains full training-population weights.

## Fixed feature families and models

Every arm includes a condition offset and the original nonnegative common precursor
coefficient, estimated from fitting-only within-condition covariance. Freeze this
coefficient while fitting sequence/context effects. Global weighted standardization
and within-condition centering are fitted on fitting rows only. Separate ridge
coefficients per condition minimize condition-normalized weighted residual MSE plus
0.1*sum(beta**2). No tuning, pretraining, stochastic initialization or optimizer.
Unsupported constant coordinates are masked/recorded; ridge handles collinearity.

Nine predeclared arms:
1. calibrated_precursor: original context/precursor control, no sequence effects.
2. length: peptide length alone.
3. position_p2_c: length + 20 P2 + 20 C-terminal indicators (41 raw features).
4. position_p1_p2_c: length + 20 P1/P2/C-terminal indicators (61).
5. sequence: original length + 20 composition + 20 P1 + 20 C-terminal (61).
6. sequence_geometry: sequence + four parent-geometry values (65).
7. sequence_flanks: sequence + 42 external-flank indicators (103).
8. sequence_context: sequence + geometry + flanks (107).
9. sequence_shuffled_context: sequence + the same context columns from the fixed
   outcome-independent within-length donor permutation (107).

Geometry for a zero-based half-open occurrence [s,e) in length L is
log1p(s), log1p(L-e), I(s=0), I(e=L). Flanks are the residue immediately outside
each boundary, twenty indicators plus a physical-terminus category per side.
Unknown/noncanonical flank symbols cause an admission stop, never invented values.
Average occurrences equally within each distinct parent sequence, then average
parents equally. Preserve every occurrence and parent weight in a feature-lineage
artifact. Reconstruct all exact overlapping occurrences and reconcile against the
canonical source occurrence table through reference IDs/source record IDs.

For the shuffled control, use unique keys=(peptide, sorted distinct parent IDs).
Within each peptide length and each fitting/evaluation partition separately, sort
keys by SHA256('hap1_context_shuffle_v1:'+key) and cyclically rotate donors by one.
Singletons stay unchanged and are counted. Repeated rows share a donor. Full-fit
training prediction uses the same mapping as fitting. No cross-partition donor,
quantity-dependent assignment or claim that one permutation is a calibrated null.
Save donor mappings and fractions of unchanged/same-component assignments.

## Qualification and decisions

Before new biological fits: original sequence/control full and five-fold predictions
must reproduce archived 0730 values within 1e-10; original weights/folds and complete
cohort identities must match. Independently reconstruct occurrence/flank/geometry
features with a separate implementation. On hand-checkable repeated/terminal/ambiguous
parent examples, verify two-stage equal weighting. Failure stops fitting, preserves
attempt and requires a written correction without changing biology/gates.

Primary development gate: sequence_context reduces pooled CV RMSE by >=2% versus
sequence and >=1% versus shuffled-context, and improves versus sequence in >=4/5 folds.
Secondary descriptive comparisons: position-only versus original sequence, isolated
geometry/flank gains, per-condition errors/spread and training-CV gaps. No result
changes a held-out baseline. Failure stops this context family; no extra feature or
penalty sweep on these folds. Success motivates a separately specified physical
exact/sampled comparison and an independent biological population, not a concentration
claim. No coefficients are interpreted as measured enzyme rates or causal KO effects.

## Resources, artifacts and verification

Local float64 CPU, one BLAS thread, no Modal/GPU/shared-environment change. Nine arms
x (full+5fold) =54 fits; fitting cap 300s/3GiB. Abort on caps/numeric nonfinite values;
no timeout-driven restart. Preserve all training and inner-CV prediction rows, full
and fold parameters/transforms, folds, source/recipe hashes, features/lineage/donors,
normal-equation residuals and supported-coordinate counts, runtime/peak memory,
weighted loss/RMSE/MAE/R2/SD plus unweighted Pearson/Spearman, condition/fold summaries,
comparison plot, exact launch/source/env/dirt receipts and canonical registry entry.
Independent post-fit verification reconstructs features, training weights, affine
predictions and metrics; alternative least-squares/SVD checks every fitted system.
No external biological validation or absolute-concentration success is inferred.
