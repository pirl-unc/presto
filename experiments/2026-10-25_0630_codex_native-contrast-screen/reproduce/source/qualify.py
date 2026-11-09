"""Exercise the real fitting path with data-free invariants before biological use."""

import hashlib
from pathlib import Path

import numpy as np

from core import (
    CFG,
    bootstrap,
    differences,
    dump,
    fit,
    inner_fold,
    metrics,
    predict,
    select,
    weights,
)
from features import raw_features

assert (
    hashlib.sha256(Path(CFG["feature_source"]).read_bytes()).hexdigest()
    == CFG["feature_source_sha256"]
)
assert len(raw_features("ACDEFGHIK", "sequence")) == 61
np.testing.assert_allclose(raw_features("ACDEFGHIK", "length"), [np.log(9)], atol=1e-14)
feature = raw_features("AACDEFGHA", "sequence")
assert feature[1] == 3 / 9 and feature[21] == 1 and feature[41] == 1
rows = []
for number in range(24):
    peptide = "ACDEFGHIKLMNPQRSTVWY"[: 8 + number % 12]
    reference = "WYVTSRQPNMLKIHGFEDCA"[: 9 + number % 9]
    row = dict(
        example_id=f"fixture-{number}",
        dependency_group_id=f"fixture-group-{number // 2}",
        parent_sequence_sha256=f"fixture-parent-{number // 2}",
        split="train",
        peptide_sequence=peptide,
        reference_peptide_sequence=reference,
        weight=1 / 24,
    )
    delta = np.subtract(
        raw_features(peptide, "sequence"), raw_features(reference, "sequence")
    )
    row["target_log10_ratio_change"] = float(
        2 * delta[0] + 0.4 * delta[1] - 0.3 * delta[21]
    )
    rows.append(row)
assert {inner_fold(r["dependency_group_id"]) for r in rows} == {0, 1, 2}
model, selection, inner_predictions, means = select(rows, "sequence")
values = predict(rows, model)
assert np.ptp(values) > 0.01 and model["stationarity_max_abs"] < 1e-10
assert model["n_solved_coordinates"] > 0 and model["n_nonzero_coefficients"] > 0
x, w = differences(rows, "sequence"), weights(rows)
np.testing.assert_allclose(model["scales"], np.sqrt(w @ x**2), rtol=1e-13, atol=1e-13)
assert np.any(np.abs(w @ x) > 0.01), "Fixture must distinguish RMS from centered SD"
reverse = [
    {
        **r,
        "peptide_sequence": r["reference_peptide_sequence"],
        "reference_peptide_sequence": r["peptide_sequence"],
    }
    for r in rows
]
np.testing.assert_allclose(predict(reverse, model), -values, atol=1e-13)
duplicated = [{**r, "weight": r["weight"] / 2} for r in rows for _ in range(2)]
dup_model = fit(duplicated, "sequence", model["alpha"])
np.testing.assert_allclose(predict(rows, dup_model), values, rtol=1e-11, atol=1e-12)
zero_features = [
    {
        **r,
        "reference_peptide_sequence": r["peptide_sequence"],
        "target_log10_ratio_change": 7.0,
    }
    for r in rows
]
empty = fit(zero_features, "sequence", 0.1)
assert empty["n_solved_coordinates"] == 0 and empty["design_rank"] == 0
assert np.all(predict(rows, empty) == 0), "An intercept must not fit constant effects"
for function in (
    lambda bad: fit(bad, "sequence", 0.1),
    lambda bad: select(bad, "sequence"),
):
    try:
        function([{**rows[0], "split": "val"}])
    except AssertionError:
        pass
    else:
        raise AssertionError("Fit path accepted a held-out row")
for record in selection:
    assert all(
        inner_fold(g) != record["fold"] for g in record["fitted"]["training_components"]
    )
assert model["alpha"] == min(means, key=lambda a: (means[a], -a))
zero_metric = metrics(rows, np.zeros(len(rows)))
assert zero_metric["pearson"] is None and zero_metric["spearman"] is None
assert zero_metric["signed_change_accuracy"] == 0
manual = [
    dict(
        example_id="m0",
        dependency_group_id="g0",
        weight=0.25,
        target_log10_ratio_change=1,
    ),
    dict(
        example_id="m1",
        dependency_group_id="g1",
        weight=0.75,
        target_log10_ratio_change=3,
    ),
]
score = metrics(manual, np.array([2.0, 1.0]))
assert score["mse"] == 3.25 and score["mae"] == 1.75
y = np.array([r["target_log10_ratio_change"] for r in rows])
toy_predictions = {
    "zero": np.zeros(len(rows)),
    "length": 0.7 * y + 0.1,
    "sequence": 0.9 * y,
}
boot = bootstrap(rows, toy_predictions)
duplicate_boot = bootstrap(
    duplicated, {k: np.repeat(v, 2) for k, v in toy_predictions.items()}
)
assert boot[0] == duplicate_boot[0] and np.array_equal(boot[2], duplicate_boot[2])
for kind in toy_predictions:
    np.testing.assert_allclose(boot[3][kind], duplicate_boot[3][kind], atol=1e-13)
independent_rng = np.random.Generator(np.random.PCG64(CFG["bootstrap_seed"]))
independent_indices = np.stack(
    [
        independent_rng.choice(len(boot[0]), size=len(boot[0]), replace=True)
        for _ in range(CFG["bootstrap_draws"])
    ]
)
assert np.array_equal(boot[2], independent_indices)
dump(
    "results/qualification.json",
    dict(
        passed=True,
        synthetic_software_rows=len(rows),
        biological_rows_read=0,
        tests=[
            "original feature definition",
            "un-centered training RMS",
            "reference reversal",
            "weighted duplicate invariance",
            "zero-coordinate removal and no intercept",
            "held-out row rejection",
            "component-only inner selection",
            "constant metrics",
            "nonconstant predictions on an identifiable synthetic signal",
            "stationarity",
            "component bootstrap duplicate invariance and independent RNG replay",
        ],
        max_stationarity_residual=model["stationarity_max_abs"],
        n_inner_predictions=len(inner_predictions),
    ),
)
print("Data-free production-path qualification passed", flush=True)
