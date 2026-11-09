"""Data-free qualification of actual contexts, ridge selection and statistics."""

import numpy as np

from context import endpoint, names, occurrences, vectors
from core import bootstrap, correlation, decision, dump, fit, metrics, predict, select

assert len(names()) == 149
assert occurrences("AAAA", "AAA") == [0, 1]
f, matches = endpoint("AAAA", "AAA")
assert len(matches) == 2
np.testing.assert_allclose(f[61:].reshape(4, 22).sum(1), 1)
f, matches = endpoint("UAV", "AV")
assert matches[0]["tokens"] == ["BOUNDARY", "OTHER", "BOUNDARY", "BOUNDARY"]
rows = []
for i in range(60):
    positive = i % 2 == 0
    parent = ("K" if positive else "R") + "AVDEGGF"
    features = vectors(parent, "AV", "GG")
    reverse = vectors(parent, "GG", "AV")
    for kind in features:
        np.testing.assert_allclose(features[kind], -np.asarray(reverse[kind]))
    rows.append(
        dict(
            example_id=str(i),
            dependency_group_id="qualification-" + str(i),
            split="train",
            weight=1 / 60,
            target_log10_ratio_change=1.0 if positive else -1.0,
            features=features,
        )
    )
model, inner, predictions, means = select(rows, "boundary")
values = predict(rows, model)
score = metrics(rows, values)
assert score["rmse"] < 0.02 and score["prediction_target_sd_ratio"] > 0.98
assert all(r["fitted"]["stationarity_max_abs"] < 1e-12 for r in inner)
assert score["pearson"] > 0.999
assert metrics(rows, predict(rows, fit(rows, "sequence", 0.01)))["rmse"] > 0.99
assert model["n_raw_features"] == 149 and model["n_solved_coordinates"] < 149
dup = [dict(r, weight=r["weight"] / 2) for r in rows for _ in range(2)]
np.testing.assert_allclose(
    predict(rows, fit(dup, "boundary", model["alpha"])), values, rtol=2e-12, atol=2e-12
)
try:
    fit([dict(rows[0], split="test")], "boundary", 0.01)
except AssertionError:
    pass
else:
    raise AssertionError("Fitter accepted nontraining rows")
assert correlation(np.ones(60), values, np.ones(60) / 60) is None
base = np.zeros(60)
pred = dict(zero=base, length=base, sequence=base, boundary=values)
groups, errors, draws, rmses, gains, intervals = bootstrap(rows, pred)
assert draws.shape == (2000, 60)
np.testing.assert_allclose(rmses["zero"], 1)
assert intervals["sequence"][0] > 0.98
pooled = {k: metrics(rows, v) for k, v in pred.items()}
folds = {str(i): pooled for i in range(5)}
assert decision(pooled, folds, intervals)["prioritize_independent_confirmation"]
negative = {k: metrics(rows, base) for k in pred}
assert not decision(negative, {str(i): negative for i in range(5)}, intervals)[
    "prioritize_independent_confirmation"
]
dump(
    "results/qualification.json",
    dict(
        passed=True,
        n_synthetic_rows=60,
        selected_alpha=model["alpha"],
        identifiable_context_rmse=score["rmse"],
        checks=[
            "149 features and no extra biochemical grouping",
            "overlapping occurrence averaging",
            "unknown and terminal tokens",
            "reference reversal",
            "actual ridge fitting and inner selection learn known context signal",
            "zero-RMS removal",
            "duplicated weight invariance",
            "held-out-row rejection",
            "constant correlation",
            "component bootstrap",
            "both decision branches",
        ],
        synthetic_rows_used_for_biological_training=False,
    ),
)
print("Data-free actual context/fitting/statistics qualification passed")
