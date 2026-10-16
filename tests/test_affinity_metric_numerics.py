"""Physical labels survive finite-precision target transforms and exports (#116)."""

import io

import numpy as np
import pytest

from presto.training.holdout_eval import (
    TaskPredictionAccumulator,
    binding_threshold_metrics,
    pearson,
    regression_metrics,
    write_holdout_artifacts,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("size", [3, 210, 1599])
def test_constant_vectors_are_undefined_before_mean_rounding(dtype, size):
    constant = np.full(size, 0.1, dtype=dtype)
    variable = np.arange(size, dtype=dtype)
    for order in (np.arange(size), np.arange(size)[::-1]):
        assert pearson(variable[order], constant[order]) is None
        assert pearson(constant[order], variable[order]) is None
        assert "pearson" not in regression_metrics(variable[order], constant[order])


def test_small_real_variation_is_not_erased_by_a_tolerance():
    values = 0.1 + np.arange(10) * 1e-12
    assert pearson(values, values) == pytest.approx(1)
    assert pearson(values, -values) == pytest.approx(-1)


def test_physical_boundary_survives_float32_promotion_and_csv():
    raw = np.array([499.99999, 500, 500.00001, 1000])
    transformed = np.log10(raw.astype(np.float32)).astype(float)
    stream = io.StringIO()
    np.savetxt(stream, transformed, delimiter=",")
    restored = np.loadtxt(io.StringIO(stream.getvalue()), delimiter=",")
    predicted = np.log10([400, 400, 800, 800])
    result = binding_threshold_metrics(restored, predicted, np.zeros(4), raw_target_nm=raw)
    assert result["accuracy"] == result["balanced_accuracy"] == 1
    assert result["n_physical_target_observations"] == 4
    assert result["n_transformed_target_observations"] == 0
    # Historical transformed-only arithmetic remains distinguishable.
    old_basis = binding_threshold_metrics(restored, predicted, np.zeros(4))
    assert old_basis["accuracy"] == 0.5
    assert old_basis["n_transformed_target_observations"] == 4


def test_physical_censor_bounds_keep_conservative_direction_policy():
    raw = np.array([500, 500, 499, 501, 499, 501], dtype=float)
    qualifier = np.array([-1, 1, 1, -1, -1, 1])
    result = binding_threshold_metrics(
        np.log10(raw.astype(np.float32)).astype(float),
        np.log10([400, 800, 800, 400, 400, 800]),
        qualifier,
        raw_target_nm=raw,
    )
    # <=500, <=499 and >501 are usable; equal right bounds stay ambiguous.
    assert result["n"] == result["n_physical_target_observations"] == 3
    assert result["accuracy"] == 1


def test_missing_physical_targets_remain_explicit_transformed_diagnostics():
    result = binding_threshold_metrics(
        np.log10([500, 1000]),
        np.log10([400, 800]),
        np.zeros(2),
        raw_target_nm=[500, np.nan],
    )
    assert result["n_physical_target_observations"] == 1
    assert result["n_transformed_target_observations"] == 1
    assert result["accuracy"] == 1


@pytest.mark.parametrize("raw", [[1], [500, np.inf], [500, -1], [0, 500]])
def test_invalid_physical_arrays_fail(raw):
    with pytest.raises(ValueError, match="Physical nM targets"):
        binding_threshold_metrics(
            np.array([2.0, 3.0]), np.array([2.0, 3.0]), np.zeros(2), raw_target_nm=raw
        )


def test_accumulator_preserves_source_precision_and_mapping_basis(tmp_path):
    accumulator = TaskPredictionAccumulator("binding_ic50", "censor")
    raw = np.array([500, 500.00001, 1000, 1000])
    accumulator.add(
        np.log10(raw.astype(np.float32)).astype(float),
        np.log10([400, 800, 800, 800]),
        [1] * 4,
        qualifiers=[0] * 4,
        mapping_categories=["unique"] * 4,
        observations=[
            {"raw_target": 500, "raw_unit": "nM"},
            {"raw_target": 500, "raw_unit": "nM"},
            {"raw_target": 1000, "raw_unit": "nM"},
            {"raw_target": 500, "raw_unit": "minutes"},
        ],
        lineage={
            "source_value": ["500", "500.00001", "1", "500"],
            "source_units": ["nM"] * 4,
        },
    )
    result = accumulator.metrics()
    for prefix in ("threshold_500nm_", "mapping_unique_threshold_500nm_"):
        assert result[prefix + "accuracy"] == 1
        assert result[prefix + "n_physical_target_observations"] == 3
        assert result[prefix + "n_transformed_target_observations"] == 1
    written = write_holdout_artifacts(tmp_path, {"binding_ic50": accumulator})
    assert written["metric_estimators"]["pearson"] == "constant_input_undefined_v2"
    assert written["metric_estimators"]["binding_threshold"] == (
        "physical_nm_when_available_conservative_censor_v2"
    )
