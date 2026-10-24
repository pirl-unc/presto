import numpy as np
from actual_metrics import probability_errors, probability_gate


def test_mean_absolute_gate_rejects_cancelling_log_errors():
    relative, bias, mae = probability_errors(
        np.array([[0.08], [-0.08]]), np.array([0.0])
    )
    assert bias[0] == 0 and mae[0] == 0.08
    assert not probability_gate(
        relative,
        mae,
        {
            "maximum_mean_relative_probability_error": 0.1,
            "maximum_mean_abs_log_error": 0.05,
        },
    )
    relative, bias, mae = probability_errors(
        np.array([[2.0, -3.0], [2.0, -3.0]]), np.array([2.0, -3.0])
    )
    np.testing.assert_allclose(relative, 0, atol=1e-15)
    assert probability_gate(
        relative,
        mae,
        {
            "maximum_mean_relative_probability_error": 0.1,
            "maximum_mean_abs_log_error": 0.05,
        },
    )
