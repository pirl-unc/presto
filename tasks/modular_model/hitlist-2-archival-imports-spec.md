# Preserve frozen archival verification after Hitlist 2.0 API retirement

CI c06fcb7 failed 5 tests and26 fixtures after installing Hitlist2.0. Root cause:
`verify_training_bundle` and fixture `write_training_bundle` were intentionally
removed from Hitlist's top-level public API. The original `training_bundle` source
module remains for historical source export/audit; do not restore retired aliases
or pin/downgrade Hitlist to hide the break.

- Update only the frozen historical-bundle verifier and its archived-source fixture
  imports to the existing explicit `hitlist.training_bundle` module. Keep source
  manifests, hashes, lineage, admission and observation weighting unchanged.
- Document that this path verifies pre-v2 frozen receipts; it does not materialize
  current v2 evidence through old APIs or claim that a schema8 package is a v1
  training bundle. New MAPP materialization uses canonical EvidenceStore.
- Exercise frozen bundle, partition, coverage-preflight and version-floor regressions
  against actual Hitlist2.0 code with removed top-level exports. No monkeypatch
  restores those names; no skips. Run lint and current-head CI. Record remaining
  consumer migration separately from this bounded archival verification repair.

The biological goal and current data contract do not change. This is a concrete
CI failure discovered during canonical data integration.

Verification amendment: the first actual2.0 run passed86 tests and exposed one remaining retired call in the existing v4-index parity adapter (`hitlist.generate_training_table`). Update that explicit legacy adapter to `hitlist.export.generate_training_table`, including its test stub and the mutation guard. This retains existing v4 audit behavior; it does not add a top-level fallback or convert a v2 generation into a legacy table. Document the adapter’s scope accurately. Broader new trainer admission remains the canonical EvidenceStore path.

Verified:159 affected tests pass against actual Hitlist2.0 (17.32s), with no restored top-level aliases or skipped tests. Code lint and strict documentation build pass. The initial86-pass/1-failure run and final regression log are retained in the canonical integration experiment. Current-head hosted CI remains required.
