import numpy as np
from verify import log_gate


def test_all_refinements_must_match_the_analytical_reference():
    # Mutual agreement and an accurate eight-step reference alone are insufficient.
    logs=np.array([[1.2e-5],[.8e-5],[.4e-5]])
    assert np.ptp(logs)<1e-5 and abs(logs[-1,0])<1e-5
    assert not log_gate(logs,np.zeros(1))
    assert log_gate(logs)  # No analytical oracle supplied in a nonzero-eta case.
    assert not log_gate(np.array([[0.],[np.nan],[0.]]))
    assert log_gate(np.full((3,2),-400.),np.full(2,-400.))
