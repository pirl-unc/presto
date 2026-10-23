# Bounded raw-S4 nominal-response admission and local diagnostic

Prospective plan, October23,2026. Depends on Hitlist#767 / #760 and verified Presto0316 source conservation. Tracks data limits in Hitlist#762. The active objective remains APC peptide availability modulo precursor abundance. This stage tests whether an independent observation-response branch can learn anything useful; it cannot identify intracellular concentration or transfer to MAPP by itself.

## Shared recipe before fitting

Add a registered Hitlist `ms_calibration_raw_s4_nominal_response_v1` recipe through the existing materialization/verifier path, with original source cell/nominal-load/reporter/slot identities in every example and a disposition for every raw S4 slot. No Presto-only workbook parser. Use the currently verified combined generation and pin its manifest, source contract and exact software commit.

Eligibility is conservative and explicit:

- Only Data4 raw S4 observations. Never mix normalized values or curve points with them.
- Preserve all31 raw reporter IDs and all3,069 planned raw slots in dispositions. Positive finite reported response is eligible for a conditional quantitative response target; blank/absent/nonnumeric remains unobserved, not a nondetection/zero.
- Exclude the two CNNVLYIR CAM/PCm reporter forms from nominal-load regression: one sequence design load does not assign fractions to each observed form.
- Exclude the two explicitly aggregated raw reporters (`.sum3`, `.sum4`) pending source-backed transition/integration comparability. Preserve the label and exclusion reason, not a renamed sequence-only row.
- Exclude the four raw labels lacking the heavy-isotope annotations shown in the nominal panel (SFPQ_HUMAN.FAQHGTFEYEYSQR; ANRA2_HUMAN.SVQQVIESHLLK; SFSWA_HUMAN.EAQISSAIVSSVQSK; SNRPA_HUMAN.EVSSATNALR). Absence of a suffix does not prove light peptide or form equivalence. Record exact source cells and the mismatch in the shared disposition. This leaves at most23 distinct panel sequences before response availability checks; independently verify actual counts.
- SMD2 raw retains its own identity. The Data8 `.sum4` version is not joined; all curve data are out of this recipe.
- Any other molecular annotations are explicit retained context; even a unique reporter per nominal standard does not establish certified form molarity. The target is conditional raw instrument response under nominal loading, never response per certified molecule.
- No inferred run/day/preparation IDs. Site and source-column slot are structural grouping/provenance. Shared reference protocol is contextual; no per-run injection volume or APC machinery features.
- No imputation of low-end censoring or high-end saturation. Diagnose observed support; do not label excluded curve positions from S4. All synthetic standards are added after digestion, so the response model does not train processing/digestion effects.

Compile per-sequence×site summaries only after preserving every source observation and contribution. Use median log10(raw peak area) and count/range of supplied positive slots, with no row-level split across repeated measurements of the same peptide. Nominal log10(fmol_on_column) is a covariate, not a denominator asserted to linearize detector response. Retain response-informed group assignment only for audit; it must not become a feature.

## Independent admission and experimental design

Independent XML/Parquet replay checks source IDs, exclusions, nominal amounts, exact summary arithmetic and full slot dispositions. Check unique-row/grain conservation and safe query joins; freeze accepted export before assigning a model. The normal full software runner keeps its memory guard.

Register a new experiment before fitting. First question: do simple sequence features improve held-out-peptide conditional response beyond nominal load and site context in this selected panel? Use fixed peptide-grouped cross-validation (predeclare stable folds and seeds before outcomes; report all folds, not a selected best fold). With at most23 sequences, prefer nested grouped folds or a prespecified ridge penalty over pretending many repeated injections are independent. Report a final validation/test metric only if a separate scientifically defensible split exists; otherwise explicitly document why grouped cross-validation is the diagnostic and no untouched test claim is made.

Start with two small models under the same target and weighting:

1. Intercept + log nominal load + site effects, trained only on the training peptides/sites.
2. Same baseline + sequence length and amino-acid fractions, regularized by a prespecified ridge grid selected only within training folds. Heavy/modification context is retained in input metadata; allocate no coefficient for a constant or unexposed feature. No neural network or RL is needed for this observation-only baseline.

Use equal peptide or peptide×site weighting, not the number of positive source slots. Freeze the feature basis/scaler using training folds only. Independently recompute all predictions/metrics and dump each fold's held-out predictions and coefficient/exposure reports. Report RMSE/MAE/Pearson/Spearman in log10 peak area, per-peptide and per-site errors, predictive spread, training/validation losses, actual wall time and peak process memory. Constant predictors have undefined correlation. IC50/500nM binder metrics are not applicable. Retain source-selection/load confounding and saturation sensitivity as limitations, with no causal coefficient claims.

A separate held-out-site diagnostic can test acquisition transfer with explicit unknown-site handling, but do not mix that question into peptide holdout or tune toward success. Neither result licenses transfer from SWATH to MAPP/immunopeptidomics. If sequence features fail, keep the context baseline and request broader known-load/non-MHC observation panels rather than making the architecture more complex.

## Relationship to differentiable versus sampled availability

Do not compare RL and exact enumeration on a dataset that supplies no processing trajectories. This source can constrain only the observation-response term. Retain the previously registered availability references and biological failures. Any later combined availability experiment must separately declare processing/precursor labels, target observation pool, detector/capture transfer assumptions, held-out biological grouping and which modules actually receive supervision. Raw-S4 source admission is not completion of the concentration goal.

## Frozen diagnostic choices before admission and fitting

Use the canonical five peptide folds defined by sorting all30 nominal panel sequences by SHA256(recipe_id + ':' + sequence) and rank modulo5; excluded forms retain their fold but do not enter fits. Outer fold is the held-out diagnostic/test fold. Select ridge alpha from `[0.0001,0.001,0.01,0.1,1,10,100]` using only the remaining four folds as inner grouped validation. Choose minimum aggregated inner out-of-fold equal-peptide MSE (ties favor the first grid entry). Do not change folds/features/grid after looking at outer outcomes.

Three conditions: training weighted-mean control; ridge with log10 nominal load plus site indicators; same ridge plus length and20 amino-acid fractions. No nominal group, source protein label, fold, peptide identity, raw slot count or response-derived metadata as features. All source sites define a fixed category universe; coefficients for constant/unexposed columns are dropped within each training fold. Features center/scale and intercept use training-only equal-peptide weights. Objective is weighted MSE plus alpha times squared standardized coefficients, with unpenalized intercept. A peptide contributes total weight1 regardless of its number of sites; slots are never separate fit examples. No pretraining, synthetic labels, detector-slope constraint, MHC or processing losses.

Preserve each of280 inner fits,10 selected outer ridge fits and5 outer mean controls with their training row IDs, feature exposure, coefficients/scalers, validation/test predictions and metrics. Report all outer folds and pooled held-out predictions; this is nested grouped diagnostic CV, not an untouched external biological test. Primary metric: equal-peptide RMSE in log10 peak area; also MAE, ordinary pooled Pearson/Spearman, predictive spread and per-peptide/site errors. Use1000 paired peptide bootstrap draws (seed20261023) for sequence-minus-context RMSE uncertainty. Thresholded binding metrics are inapplicable. Freeze exact runner/source/config before execution; independent verification uses a second least-squares implementation and replays all source arithmetic, partitions, alpha choices, predictions and metrics. No held-out-site study or full-data production fit in this initial family.
