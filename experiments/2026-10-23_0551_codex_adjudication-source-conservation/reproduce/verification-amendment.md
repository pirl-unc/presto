# Baseline envelope correction

The first independent verifier stopped at equal file-inventory assertion.
All three frozen MAPP baseline paths selected the pre-canonical standalone
recipes; these have no coverage.json and no output_kind. The only added data
file in current exports is coverage.json. No source build, query or target
verification failure was observed.

Use the last canonical 0316 exports, whose source is the same legacy calibration
generation, as the equal-inventory/coverage baseline. Also compare every original
standalone payload except its source manifest, retaining the original hashes.
This strengthens the intended conservation check without changing a target,
split, source package, producer code or scientific admission decision.

Original launch.json is launch.initial.json; original verifier is
source/verify.initial.py; launch_logs/launch.log retains the failure.
Only the independent verifier is rerun; no producer or training is rerun.
