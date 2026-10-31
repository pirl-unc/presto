# Staged elution spec

Goal: opt-in `elution_staging="affinity_downstream"` so the elution logit is
`a*(-pooled affinity) + b*pooled half_life + processing + ms_response + bias`,
never reading `cellular_score` or the context-fused trunk.

- [x] Config: `elution_staging`, `elution_affinity_endpoint`, `elution_detach_affinity`;
      `to_dict()` omits them when legacy (digests unchanged).
- [x] Model: `elution_processing`, `elution_affinity_weight`, `elution_stability_weight`;
      no `cellular_score`; `context_fusion` only for other contextual endpoints.
- [x] Dependencies/feature layout/freezing/predict gate/continuation updated.
- [x] Tests: tests/test_staged_elution.py (flag-off identity vs base verified bitwise
      against origin branch; gradients; detach; coverage gate; checkpoint + hash).
- [x] Docs: docs/modular_model.md "Staged elution".
- [x] ruff 0.16.10, full suite.

Review: no training or Modal run launched. Detach covers affinity only; the
half-life term still trains the shared trunk.
