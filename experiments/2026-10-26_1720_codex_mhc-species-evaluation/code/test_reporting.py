"""Regression checks for concentration and unsupported classes in this evaluation."""

from copy import deepcopy

import numpy as np
import pytest

from report import bootstrap, metrics
from verify import audit_metrics


def test_component_weighting_and_missing_classes():
    # Nine easy annotations in one component must not outweigh one hard component.
    rows = [
        dict(
            observation_id=str(i),
            group_id="large" if i < 9 else "small",
            target=0,
            probabilities=[0.9, 0.05, 0.025, 0.025] if i < 9 else [0.1, 0.8, 0.05, 0.05],
        )
        for i in range(10)
    ]
    for row in rows:
        row.update(
            prediction=int(np.argmax(row["probabilities"])),
            loss=float(-np.log(row["probabilities"][0])),
        )
    result = metrics(rows)
    assert result["accuracy"] == 0.9
    assert result["component_mean_accuracy"] == 0.5
    assert result["per_species"]["mouse"]["recall"] is None
    audit_metrics(rows, result)
    damaged = deepcopy(result)
    damaged["component_mean_accuracy"] = 0.9
    with pytest.raises(AssertionError):
        audit_metrics(rows, damaged)


def test_pairing_rejects_wrong_group_and_preserves_permutation():
    rows = [
        dict(observation_id=str(i), group_id=str(i), target=i % 4, probabilities=[0.25] * 4)
        for i in range(8)
    ]
    assert bootstrap(rows, list(reversed(rows)))["interval_95"] == [0, 0]
    damaged = deepcopy(rows)
    damaged[0]["group_id"] = "wrong"
    with pytest.raises(ValueError, match="paired control mismatch"):
        bootstrap(rows, damaged)
