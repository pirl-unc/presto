# Numerical details fixed before new cohort admission or fitting

Use float64 on local CPU, one Torch/BLAS thread. Feature means/scales use equal-pool
training weights. Exactly constant raw columns are masked by zero raw range. Ridge
also masks columns with zero range within every training run: their contribution is
annihilated by the prescribed prediction centering and is not identifiable. Nonlinear
models retain nonconstant raw features, which can modify within-run responses through
interactions. Record masks rather than allocate a coordinate and claim it trained.

Ridge solves Xc'WXc + alpha/d I, matching weighted MSE + alpha*mean(beta²), with
per-run centered standardized features and no intercept. Check normal-equation
residuals. Nonlinear loss uses per-run-centered outputs and .001 times the mean square
of all trainable parameters. Validate gradients on numerical fixtures before fitting;
fixtures are not biological labels. Validation ties select lower alpha/earlier step.
Each seed's best prescribed checkpoint and the mean of their centered predictions
are frozen before test outcome loading. Full trajectories are retained; no test
checkpoint selection or additional search. Resource limits and scientific gates
remain those of the original prospective plan.

Error metrics, R² and SD use equal-pool weights. Pooled Pearson/Spearman are
unweighted and labelled as such; also report each individual pool. Source-admission
verification necessarily parses original quantities, including reserved rows, but
reports only equality/coverage gates. Model fitting opens train/validation labels
and test covariates only. Reserved prediction metrics are computed only after the
fit process seals its checkpoints and test covariate-only predictions.

Before any fit, resolve the screen's correlation wording conservatively: require
both Pearson and Spearman to be positive in at least3/4test pools. Require the
prediction-to-target SD ratio of at least0.1 on both validation and test. All gates
are reported separately; the preselected sequence ridge remains the primary model.
Centering is conditional on the source-recipe's admitted positive candidate set.
This does not infer an unconditional detection probability or use unobserved
planned products as negative examples.
