# Post-fit verification amendment

Independent NumPy replay matched all nine five-rate models and the constant-rate
control checkpoints/predictions. It failed the constant-rate control test Spearman
comparison: prefix-sum arithmetic and enumerated state sums break theoretical ties
differently (absolute correlation difference0.00183684). Preserve all original
results and this failed verifier. No fit, optimizer budget, seed, prediction,
checkpoint selection, RMSE or scientific gate changes.

Reproduce original stored metrics directly from preserved prediction arrays.
Independently check expected quantities numerically. For the single-rate control,
report an additional canonical Spearman using analytically equal classes defined
by time, span length and number of internal boundaries. Compute each class once:
(1-exp(-rate*t))**boundaries * exp(-rate*t*(span_length-1)). No arbitrary decimal
rounding or expanded tolerance. Keep all three rank calculations in a separate
receipt, because score ties are mathematically specified by this control.
