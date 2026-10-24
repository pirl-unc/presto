# Independent closure and exact feature ties

All six frozen fits completed; no model or setting is retried. The compound model
produced a NearConstantInputWarning in some small strata. Inspection shows repeated
predictions for the same model input differ at roughly 1e-16 because the batched
matrix multiplication accumulates roundoff differently for some rows. Fourteen
strata were reported with finite correlation despite effectively constant inputs.
The earlier rule about analytic prediction ties applies here as well.

Preserve the original launcher, predictions, summary and stratified metrics.
Independently rebuild every raw feature from its named definition and the canonical
source groups. For each distinct raw feature vector, evaluate the frozen model
once using scalar dot products, and assign that value to all identical inputs.
This uses exact feature identity, not target values or a numerical tie tolerance.
Verify every prediction differs from the original by less than 1e-12 and every
loss/RMSE stays within 1e-12. Save qualified per-observation predictions and recompute
all global and stratified metrics, representing mathematically constant spread as
zero and its correlations as undefined. Record every changed correlation. No
checkpoint, split, target, observation weight or optimization result changes.
Likewise, where every literal exact target in a stratum is equal, report R2 as
undefined rather than using roundoff in a weighted mean as a target variance.

Verify source identities/values/qualifiers/weights, training-only centering/scaling,
feature support and projections, final objective/gradient and analytic constant
solutions independently. Primary loss and exact/bound metrics use scalar fsum
reductions; Spearman uses ranks followed by a separate Pearson calculation. Do not
use the producer's feature builder or metric functions. Fixed closure budget:
180 seconds / 4 GiB, local CPU one thread. Report development comparisons with
test still reserved, all limitations intact. This is closure, not another fit.
