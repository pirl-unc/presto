# Verification amendment — 2026-10-26, before any neural condition

The first independent control verification stopped at the original-source equality
check. The canonical source adapter adds exactly `provenance.evidence_policy`
from the frozen manifest's `label_policy`. The raw immutable observation lacks
that field. This is documented normalization, not a differing source label.

Preserve code/verify.py, its frozen copy, the failed receipt and verify-baselines.log.
A new code/verify_v2.py must require the entire raw observation plus precisely this
manifest-derived field; if a source already has a policy, require it to match the manifest.
Verify the policy against the frozen manifest rather than dropping provenance.
The new checker replaces only the verification entry point; no dataset, fitted
control, model, selection, metric or test-use policy changes. All launch.json
source hashes remain intact. Keep a separately frozen v2 source/hash contract and
explicit replay script. Rerun saved-output verification only, with fresh logs;
do not refit controls or rescore test. Neural launch remains blocked by IC50.
