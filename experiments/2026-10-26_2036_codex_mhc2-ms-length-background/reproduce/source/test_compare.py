"""Artificial boundary tests; no biological observations or outcomes."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from compare import WeightedAUC, compare_metrics, gates, reduction, sequence


@pytest.mark.parametrize("seed", [1, 2, 7, 19])
def test_bootstrap_auc_ties_and_multiplicity(seed):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], 60)
    scores = rng.integers(-4, 5, size=len(y)) / 3
    groups = np.repeat(np.arange(30), 4)
    reducer = WeightedAUC(y, scores)
    for _ in range(10):
        counts = np.bincount(rng.integers(30, size=30), minlength=30)
        weights = counts[groups]
        assert reducer(weights) == pytest.approx(
            roc_auc_score(y, scores, sample_weight=weights), abs=1e-14
        )


def test_perfect_tied_and_reversed_auc():
    y = np.array([0, 0, 1, 1])
    assert WeightedAUC(y, y)(np.ones(4)) == 1
    assert WeightedAUC(y, np.zeros(4))(np.ones(4)) == 0.5
    assert WeightedAUC(y, -y)(np.ones(4)) == 0


def test_metrics_tie_cutoff_and_hand_computed_bce():
    result = reduction([0, 1], [0, 0])
    assert result["loss"] == pytest.approx(np.log(2))
    assert result["brier"] == 0.25
    assert result["tp"] == result["fp"] == 1
    assert result["auroc"] == result["auprc"] == 0.5
    assert result["accuracy"] == result["balanced_accuracy"] == 0.5


def test_metric_discrepancy_rejected():
    with pytest.raises(AssertionError):
        compare_metrics({"auroc": 0.6}, {"auroc": 0.61})


def test_gates_zero_uncertainty_and_gain_boundaries():
    controls = {"combined": {"loss": 0.5, "auroc": 0.8}}
    null = {"auroc": 0.77}
    ensemble = {"loss": 0.49, "auroc": 0.79}
    assert all(gates(ensemble, controls, null, [0.001, 0.1]).values())
    assert not gates(ensemble, controls, null, [0, 0.1])["bce_gain_bootstrap_lower_above_zero"]
    unchanged = {"loss": 0.5, "auroc": 0.8}
    assert not gates(unchanged, controls, null, [0.001, 0.1])["combined_bce_gain_at_least_2pct"]


def test_composition_and_termini_are_distinct():
    first, second = sequence("ACCA"), sequence("CAAC")
    assert first[:21] == second[:21]
    assert first[21:] != second[21:]
    assert len(first) == 61
    assert sum(first[1:21]) == 1
    assert sum(first[21:41]) == sum(first[41:]) == 1
