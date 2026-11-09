# Verification and closure

This implements the already frozen analysis scope; no model, data, selection,
training or evaluation settings change. First source-only results were available
before this implementation. Subgroup and paired-comparison analyses were planned
before those outcomes and are secondary development diagnostics.

- Recompute saved validation/test predictions with SciPy/scikit-learn, including
  censored per-example/overall loss and every reported exact/binding metric.
- Verify checkpoint/input hashes, source row IDs, original targets and classes.
  Verify the source-training-mean controls independently, using train-only scales.
- Report seen/no-seen training nine-mer substrings, per-groove-pair and length
  strata. A nine-mer is not an experimentally assigned core.
- Replay each selected checkpoint in a fresh CPU process using its frozen runtime
  and only the 64 saved training probes. Cross-device tolerance is prospectively
  2e-5 log10(nM) absolute plus 2e-5 relative, not bitwise equality. Failures remain
  visible; no holdout inference or checkpoint selection occurs in replay.
- Compare the matched GPU seeds; use 2,000 fixed-seed peptide-cluster bootstrap
  draws for paired Spearman/RMSE differences. Seed-averaged scores retain the same
  resampled peptide groups per model; intervals describe dataset sampling only.
  Do not treat two seeds as population-level optimization uncertainty.
- Produce validation curves against updates and observed elapsed time, final
  class-by-class tables, and archive all raw outputs. CPU reference stays separate.
- Close the canonical registry and both family READMEs after all terminal states.

The user's same-protein MS question is a distinct endpoint. Record a prospective
follow-up design; these IC50 checkpoints have no presentation qualification.
