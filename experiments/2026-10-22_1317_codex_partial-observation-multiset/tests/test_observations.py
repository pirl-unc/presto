"""Estimator expectation, observation support, and training-boundary contracts."""

import dataclasses
import importlib.util
import inspect
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "code/launch.py"
SPEC = importlib.util.spec_from_file_location("partial_observation_launch", PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


@pytest.mark.parametrize("corrected", [True, False])
def test_exhaustive_cost_and_policy_gradient(corrected):
    # Two independent groups of two ordered draws; 3**4 complete outcomes.
    logits = torch.tensor([0.3, -0.8, 0.1], dtype=torch.float64, requires_grad=True)
    logq = logits.log_softmax(-1)
    q = logq.exp()
    f = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    exact = (q - f).square().sum()
    if not corrected:
        exact = exact + (1 - q.square().sum()) / 2
    expected_cost = torch.zeros((), dtype=torch.float64)
    expected_surrogate = torch.zeros((), dtype=torch.float64)
    for ordered in itertools.product(range(3), repeat=4):
        draws = torch.tensor(ordered).reshape(1, 2, 2)
        probability = q[torch.tensor(ordered)].prod().detach()
        counts = torch.stack([torch.bincount(group, minlength=3) for group in draws[0]]).double()
        expected_cost += probability * module.multiset_cost(counts, f, corrected).mean()
        expected_surrogate += probability * module.multiset_surrogate(
            logq[None], f[None], draws, corrected
        )
    torch.testing.assert_close(expected_cost, exact.detach(), atol=1e-13, rtol=1e-13)
    actual_gradient = torch.autograd.grad(expected_surrogate, logits, retain_graph=True)[0]
    exact_gradient = torch.autograd.grad(exact, logits)[0]
    torch.testing.assert_close(actual_gradient, exact_gradient, atol=1e-13, rtol=1e-13)


def test_corrected_sample_cost_may_be_negative():
    cost = module.multiset_cost(torch.tensor([1.0, 1.0]), torch.tensor([0.5, 0.5]), True)
    assert cost == -0.5


def test_zero_weight_support_and_gradients():
    logits = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    logq = module.observed_logp(logits.log_softmax(-1), torch.tensor([[1.0, 0.0, 0.4]]))
    assert logq.exp()[0, 1] == 0
    f = torch.tensor([[0.5, 0.0, 0.5]])
    loss = module.exact_cost(logq, f, "ce").mean()
    loss += module.sampled_cost(logq, f, 16, torch.Generator().manual_seed(1))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert abs(float(logits.grad[0, 1])) < 1e-7
    with pytest.raises(ValueError, match="observable support"):
        module.observed_logp(logq, torch.zeros_like(logq))


def test_blind_observations_do_not_identify_invisible_mass():
    w = torch.tensor([[1.0, 0.0, 0.5]])
    p = torch.tensor([[0.2, 0.6, 0.2]])
    q = torch.tensor([[0.4, 0.2, 0.4]])
    observed_p = module.observed_logp(p.log(), w).exp()
    observed_q = module.observed_logp(q.log(), w).exp()
    torch.testing.assert_close(observed_p, observed_q)
    assert float((p - q).abs().sum() / 2) > 0.39


def test_nested_counts_source_identity_and_support():
    rows, _ = module.base.make_dataset()
    observations = module.make_observations(rows)
    for records in observations.values():
        assert [r["id"] for r in records] == [r["id"] for r in rows]
        for row in records:
            low, high = np.asarray(row["counts32"]), np.asarray(row["counts256"])
            assert low.sum() == 32 and high.sum() == 256 and np.all(low <= high)
            np.testing.assert_array_equal(low, np.bincount(row["draws"][:32], minlength=70))
            np.testing.assert_array_equal(high, np.bincount(row["draws"], minlength=70))
            assert np.all(high[np.asarray(row["weights"]) == 0] == 0)
    assert len(module.conditions()) == 20


def test_fitting_interface_excludes_oracle():
    assert {field.name for field in dataclasses.fields(module.Inputs)} == {
        "features",
        "starts",
        "weights",
        "counts",
    }
    assert set(inspect.signature(module.fit).parameters) == {
        "config",
        "seed",
        "train_inputs",
        "validation_inputs",
        "path",
    }
    for function in (module.fit, module.validation_ce):
        source = inspect.getsource(function)
        for forbidden in ("[\"target\"]", "make_dataset(", "full_metrics(", "make_observations("):
            assert forbidden not in source


def test_fitting_inputs_ignore_corrupted_oracle():
    rows, _ = module.base.make_dataset()
    rows = rows[:2]
    observations = module.make_observations(rows)["positive"]
    original = module.make_inputs(rows, observations, 32)
    for row in rows:
        row["target"] = [float("nan")] * 70
        row["log_target"] = None
    corrupt = module.make_inputs(rows, observations, 32)
    for a, b in zip(original.values(), corrupt.values()):
        torch.testing.assert_close(a, b)
