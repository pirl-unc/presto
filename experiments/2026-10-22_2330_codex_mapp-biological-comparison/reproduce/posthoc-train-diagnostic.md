# Post-fit training-only diagnostic — October 23, 2026

The registered fits are complete and test results are already exposed. Before
changing any model, evaluate the saved initial and final-epoch checkpoints on
the original training examples only, using the independent NumPy forward path.
Compare their train KL/RMSE/spread with the selected checkpoint. This distinguishes
failure to fit the training objective from validation deterioration; it does not
identify label noise or the cause of historical IC50 collapse.

No optimizer, new fit, checkpoint selection, hyperparameter tuning or additional
test-checkpoint evaluation. Preserve all training predictions and record this as
posthoc analysis, not a prospectively specified selection criterion. Saved raw
training artifacts remain unchanged. Freeze analysis source and invocation.
