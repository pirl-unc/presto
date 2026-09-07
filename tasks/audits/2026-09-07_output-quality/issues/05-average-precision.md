The held-out `auprc` implementation advertises average precision but treats tied scores as distinct decision thresholds. It can report a perfect score for a constant predictor, which invalidates comparisons involving tied or saturated predictions.

Audited at `1535b5a208b2ab59b70b83d850b1f0a7e1959537`; this is an existing metrics defect surfaced by the prediction-quality audit.

### Minimal reproduction

```python
import numpy as np
from presto.training.holdout_eval import auprc

scores = np.array([0.5, 0.5])
print(auprc(np.array([1., 0.]), scores))  # 1.0
print(auprc(np.array([0., 1.]), scores))  # 0.5
```

Both inputs describe the same uninformative predictor with positive prevalence 0.5. Threshold-based average precision must be **0.5** for both.

### Cause and reach

[`auprc`](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/training/holdout_eval.py#L112) uses stable sorting, then accumulates precision at every row instead of at each distinct score threshold. Stable sorting preserves arbitrary input ordering within ties.

This estimator feeds binary tasks, binding-threshold metrics, real/decoy breakdowns and mapping strata. AUROC already averages tied ranks and is not affected by this particular bug. Existing archived metrics are not all known to be wrong; the effect depends on their tied-score distribution and ordering.

### Acceptance criteria

- [ ] Compute non-interpolated average precision using complete tie groups at distinct score thresholds.
- [ ] Verify invariance to permutations within ties and complete dataset reordering.
- [ ] Test all-equal scores, mixed tie groups, perfect/inverted rankings and absent-class behavior.
- [ ] Cross-check a trusted reference implementation on representative tied and untied arrays; retain the documented distinction between AP and trapezoidal PR area.
- [ ] Audit archived prediction dumps for ties and reissue affected metrics from preserved predictions, recording estimator/code version and metric deltas.
- [ ] Use the corrected estimator for the fresh post-#47 real-data quality baseline.

No new training run is necessary to reproduce or repair this calculation.
