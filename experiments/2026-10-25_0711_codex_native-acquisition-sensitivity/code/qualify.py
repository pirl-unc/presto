"""Data-free checks before biological diagnostics."""

import json
from pathlib import Path

import numpy as np

from core import agreement, correlation, diagnostics

r = np.array([[0, 0, 0, v, v, v] for v in (-2, -1, 1, 3)], float)
w = np.array([0.1, 0.2, 0.3, 0.4])
s, y, residual, omitted, pairs, stable = diagnostics(r, w)
assert np.count_nonzero(residual) == 0 and stable.all()
np.testing.assert_allclose(omitted, np.repeat(y[:, None], 6, axis=1))
assert s["stable_nonzero_sign_fraction"] == 1 and s["within_arm_rms"] == 0
assert all(abs(p["pearson"] - 1) < 1e-14 for p in s["partitions"])
varied = r + np.array([0.3, -0.1, -0.2, 0.5, 0.2, -0.7])
s, y, residual, omitted, pairs, stable = diagnostics(varied, w)
np.testing.assert_allclose(omitted[:, :3] - y[:, None], residual[:, :3] / 2, atol=1e-14)
np.testing.assert_allclose(
    omitted[:, 3:] - y[:, None], -residual[:, 3:] / 2, atol=1e-14
)
np.testing.assert_allclose(
    (pairs[:, :, 0] + 2 * pairs[:, :, 1]) / 3,
    np.repeat(y[:, None], 9, axis=1),
    atol=1e-14,
)
dup = diagnostics(np.repeat(varied, 2, axis=0), np.repeat(w / 2, 2))[0]
for key in (
    "within_arm_rms",
    "median_partition_pearson",
    "stable_nonzero_sign_fraction",
):
    np.testing.assert_allclose(s[key], dup[key], atol=1e-14)
rev = diagnostics(-varied, w)[0]
for key in (
    "within_arm_rms",
    "median_partition_pearson",
    "stable_nonzero_sign_fraction",
):
    np.testing.assert_allclose(s[key], rev[key], atol=1e-14)
assert correlation(np.ones(4), y, w) is None
assert agreement(np.zeros(4), y, w)["same_sign_fraction"] is None
constant = diagnostics(np.zeros((4, 6)), w)[0]
assert constant["median_partition_pearson"] is None
assert constant["next_step"] == "observation_stability_and_replication_metadata"
assert s["next_step"] == "training_only_boundary_context_comparison"
(Path(__file__).resolve().parents[1] / "results/qualification.json").write_text(
    json.dumps(
        dict(
            passed=True,
            checks=[
                "identical acquisitions",
                "omission residual identity",
                "complementary response identity",
                "duplicate weight invariance",
                "sign reversal invariance",
                "constant and zero conventions",
                "decision branches",
            ],
        ),
        indent=2,
    )
    + "\n"
)
print("All data-free diagnostic invariants passed")
