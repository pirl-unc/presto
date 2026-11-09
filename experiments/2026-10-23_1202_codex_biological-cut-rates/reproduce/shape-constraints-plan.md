# Training-only independent-cut shape check

Post-fit hypothesis diagnostic; no refit, model selection or held-out evaluation.
For a fragment requiring b boundary cuts, m(t)/m(ref) <= (t/ref)^b for t>ref:
each boundary probability grows at most linearly with time, while internal
survival decreases. Hence log10-ratio prediction has a known upper envelope,
independent of the learned sequence representation. Log m(t) is concave in
physical time because log(1-exp(-kt)) is concave and internal survival is linear
in log space. These are restrictions of this proposed process, not assumptions
about the correctness of the measured data.

Using training source curves only, count ratio-envelope and consecutive-three-
time concavity violations at0,0.3,0.5 log10 units. Report parent-level support and
cases repeated in both source biological labels. Compute a lower bound on the
weighted training MSE imposed solely by the ratio envelope. Preserve every
violation observation/ID rather than excluding it. Do not interpret a violation
as proof of a biological mechanism: background subtraction, noise, selection,
time-dependent response and precursor/processing heterogeneity can contribute.
This diagnostic can motivate the next observation/process specification but
must not alter the present fits or test outcomes.
