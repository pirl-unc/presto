"""Artificial edge cases for the independent saved-prediction verifier."""

import math
import pytest
from verify import metrics


def row(y, p, qualifier="=", loss=None):
    residual = p - y
    if qualifier in ("<", "<="):
        residual = max(0, residual)
    elif qualifier in (">", ">="):
        residual = min(0, residual)
    return dict(
        target=y, prediction=p, qualifier=qualifier, loss=residual**2 if loss is None else loss
    )


def test_censor_direction_and_loss():
    m = metrics([row(2, 1, "<"), row(2, 3, ">"), row(2, 3, "<="), row(2, 1, ">=")], 1)
    assert m["n"] == 0 and m["loss"] == 0.5 and m["censor_satisfaction"] == 0.5


def test_constant_predictions_have_no_correlation():
    m = metrics([row(1, 2), row(3, 2)], 1)
    assert m["spearman"] is None and m["pearson"] is None
    assert m["binding_at_500_nM"]["auroc"] == 0.5


def test_binding_sign_and_boundary():
    cutoff = math.log10(500)
    m = metrics([row(1, 1), row(cutoff, cutoff), row(4, 4)], 1)
    assert m["spearman"] == 1
    assert m["binding_at_500_nM"]["auprc"] == 1
    assert m["binding_at_500_nM"]["recall"] == 1


def test_incorrect_dump_loss_is_rejected():
    with pytest.raises(AssertionError):
        metrics([row(1, 3, loss=0)], 1)


def test_paired_score_sign_and_ties():
    import numpy as np
    from compare import scores
    from scipy.stats import spearmanr

    y = np.array([1.0, 1.0, 2.0, 4.0])
    p = np.array([y, -y])
    rho, rmse = scores(y, p)
    np.testing.assert_allclose(rho, [spearmanr(y, row).statistic for row in p])
    np.testing.assert_allclose(rho, [1, -1])
    assert rmse[0] == 0


def test_selection_gate_rejects_class_regression_despite_macro_gain():
    import numpy as np
    from selection import gate

    reference = np.array([[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]])
    candidate = reference.copy()
    candidate[:, :, 0] += 0.04
    assert gate(candidate, reference)["passed"]
    candidate[:, 0, 0] = 0.48
    assert not gate(candidate, reference)["passed"]
