# Note: Ablation Backlog

This is a practical list of experiments that can reduce uncertainty in design decisions.

1. Compare additive-logit presentation fusion against explicit interaction terms.
2. Evaluate flank encoding variants: FFN windows vs transformer concatenation.
3. Measure benefit of class-specific vs fully shared chaperone latent heads.
4. Test bag-level TCR aggregation (Noisy-OR) against single-TCR assumptions.
5. Compare alternate unified ramp schedules (linear, cosine, delayed) on real data.
6. Evaluate sequence-similarity hard negatives for recognition and elution tasks.
7. Quantify gains from explicit `class1_presentation` / `class2_presentation` outputs.
