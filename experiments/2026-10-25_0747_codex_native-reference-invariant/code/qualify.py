"""Exercise the new pair identities and actual unchanged ridge fitter."""

from itertools import combinations

import numpy as np

from algebra import verify
from context import endpoint
from core import dump, fit, metrics, predict, select
from stats import PRIMARY, REFERENCES, bootstrap, decision

rng = np.random.default_rng(202610250747)
identity_checks = []
for n in (2, 3, 4, 9):
    x, y = rng.normal(size=(n, 7)), rng.normal(size=n)
    identity_checks.append(verify(x, y))
    identity_checks.append(verify(x + 5, y - 3))
    identity_checks.append(verify(x[::-1], y[::-1]))
rows = []
for g in range(30):
    endpoints = [
        endpoint(parent, peptide)[0]
        for parent, peptide in (
            ("KAVDEGGF", "AV"),
            ("RAVDEGGF", "AV"),
            ("KAVDEGGF", "GG"),
        )
    ]
    signals = [1.0, -1.0, 0.0]
    verify(np.array(endpoints), np.array(signals))
    for a, b in combinations(range(3), 2):
        x = endpoints[a] - endpoints[b]
        rows.append(
            dict(
                example_id=f"{g}:{a}:{b}",
                dependency_group_id="qualify-" + str(g),
                split="train",
                weight=1 / 90,
                target_log10_ratio_change=signals[a] - signals[b],
                features=dict(
                    length=x[:1].tolist(), sequence=x[:61].tolist(), boundary=x.tolist()
                ),
            )
        )
model, inner, _, _ = select(rows, "boundary")
p = predict(rows, model)
assert metrics(rows, p)["rmse"] < 0.03
assert all(r["fitted"]["stationarity_max_abs"] < 1e-12 for r in inner)
duplicate = [dict(row, weight=row["weight"] / 2) for row in rows for _ in range(2)]
np.testing.assert_allclose(
    predict(rows, fit(duplicate, "boundary", model["alpha"])), p, rtol=2e-12, atol=2e-12
)
reverse = [
    dict(
        row,
        target_log10_ratio_change=-row["target_log10_ratio_change"],
        features={k: (-np.array(v)).tolist() for k, v in row["features"].items()},
    )
    for row in rows
]
np.testing.assert_allclose(
    predict(rows, fit(reverse, "boundary", model["alpha"])), p, rtol=2e-12, atol=2e-12
)
try:
    fit([dict(rows[0], split="test")], "boundary", 0.01)
except AssertionError:
    pass
else:
    raise AssertionError("Accepted nontraining rows")
predictions = {key: np.zeros(len(rows)) for key in REFERENCES}
predictions[PRIMARY] = p
*_, intervals = bootstrap(rows, predictions)
pooled = {key: metrics(rows, value) for key, value in predictions.items()}
assert decision(pooled, {str(i): pooled for i in range(5)}, intervals)[
    "prioritize_independent_confirmation"
]
constant = {key: metrics(rows, np.zeros(len(rows))) for key in predictions}
assert decision(constant, {str(i): constant for i in range(5)}, intervals)[
    "stop_response_only_feature_optimizer_expansion"
]
dump(
    "results/qualification.json",
    dict(
        passed=True,
        synthetic_pairs=len(rows),
        synthetic_groups=30,
        selected_alpha=model["alpha"],
        known_context_rmse=metrics(rows, p)["rmse"],
        identity_checks=identity_checks,
        checks=[
            "all-pair/anchor quadratic and gradient identities",
            "common shift and permutation invariance",
            "actual fitting and inner selection",
            "pair reversal and duplicate-weight invariance",
            "nontraining rejection",
            "component bootstrap and both decisions",
        ],
        synthetic_biological_training=False,
    ),
)
print("Pair algebra and actual fitting/statistics qualification passed")
