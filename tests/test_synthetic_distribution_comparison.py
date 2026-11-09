"""Estimator correctness for an isolated, nonbiological algorithm experiment."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def experiment():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/2026-10-21_1700_codex_discrete-distribution-comparison/code/launch.py"
    )
    spec = importlib.util.spec_from_file_location("synthetic_distribution_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dataset(experiment):
    return experiment.make_dataset()


def test_parent_splits_and_full_product_support(experiment, dataset):
    rows, coefficients = dataset
    assert experiment.MASK.sum() == 70
    assert set(coefficients) == {"start", "length"}
    parents = {}
    for split, count in (("train", 384), ("validation", 96), ("test", 96)):
        subset = [row for row in rows if row["split"] == split]
        assert len(subset) == count
        parents[split] = {tuple(row["tokens"]) for row in subset}
        assert len(parents[split]) == count // 4
        for parent in parents[split]:
            assert {row["context_id"] for row in subset if tuple(row["tokens"]) == parent} == set(
                range(4)
            )
    assert not parents["train"] & parents["validation"]
    assert not parents["train"] & parents["test"]
    assert not parents["validation"] & parents["test"]
    for row in rows:
        target = np.asarray(row["target"])
        assert np.isclose(target.sum(), 1, atol=1e-12)
        assert np.all(target > 0)
        assert target[~np.asarray(row["compatible"])].sum() > 0
        assert np.isneginf(row["log_target"][~experiment.MASK]).all()


def test_real_discrete_multisets_obey_coordinates(experiment, dataset):
    rows, _ = dataset
    model = experiment.Policy()
    x, starts, _ = experiment.tensors(rows[:2])
    joint, start_logp, length_logp = model(x, starts)
    assert torch.allclose(joint.exp().sum((1, 2)), torch.ones(2))
    assert torch.all(joint.exp()[:, ~experiment.MASK] == 0)
    choices, offsets, logp = experiment.sample_spans(
        joint, start_logp, length_logp, 256, torch.Generator().manual_seed(42)
    )
    assert choices.shape == (2, 256)
    assert torch.all(choices + offsets + 1 <= 16)
    assert torch.isfinite(logp).all()
    assert len(set(zip(choices[0].tolist(), offsets[0].tolist()))) < 256
    for batch in range(2):
        for draw in range(256):
            start, size = int(choices[batch, draw]), int(offsets[batch, draw]) + 1
            actual = rows[batch]["tokens"][start : start + size]
            assert len(actual) == size
            assert logp[batch, draw] == joint[batch, start, size - 1]


def test_leave_one_out_score_estimator_exact_expectation():
    logits = torch.tensor([-0.4, 0.3, 0.1], dtype=torch.float64, requires_grad=True)
    logp = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64).log()
    logq = logits.log_softmax(0)
    q = logq.exp()
    cost = logq - logp
    reference = torch.autograd.grad((q * cost).sum(), logits, retain_graph=True)[0]
    expected_surrogate = 0
    for first in range(3):
        for second in range(3):
            probability = (q[first] * q[second]).detach()
            advantage = (cost[first] - cost[second]).detach()
            expected_surrogate += probability * advantage * (logq[first] - logq[second]) / 2
    actual = torch.autograd.grad(expected_surrogate, logits)[0]
    assert torch.allclose(actual, reference, atol=1e-12, rtol=1e-12)


def test_exact_network_gradient_against_finite_difference(experiment, dataset):
    rows, _ = dataset
    model = experiment.Policy().double()
    x, starts, target = (value.double() for value in experiment.tensors(rows[:2]))
    loss = experiment.exact_loss(model(x, starts)[0], target)
    loss.backward()
    for name in ("start.0.weight", "length.0.weight", "start.2.weight", "length.2.weight"):
        parameter = dict(model.named_parameters())[name]
        reference = float(parameter.grad.flatten()[0])
        with torch.no_grad():
            original = float(parameter.flatten()[0])
            parameter.flatten()[0] = original + 1e-5
            plus = float(experiment.exact_loss(model(x, starts)[0], target))
            parameter.flatten()[0] = original - 1e-5
            minus = float(experiment.exact_loss(model(x, starts)[0], target))
            parameter.flatten()[0] = original
        assert np.isclose((plus - minus) / 2e-5, reference, atol=1e-8, rtol=1e-5)


def test_metrics_identical_distribution_and_uniform_nonconstant_target(experiment, dataset):
    rows, _ = dataset
    exact = [{**experiment.public_row(row), "prediction": row["target"]} for row in rows[:4]]
    actual = experiment.metrics(exact)
    for key in (
        "reverse_kl",
        "forward_kl",
        "tv",
        "js",
        "rmse_probability",
        "noncompatible_mass_mae",
        "selected_tv",
    ):
        assert abs(actual[key]) < 1e-12
    uniform = experiment.metrics(experiment.predict(None, rows[:4]))
    assert uniform["reverse_kl"] > 0
    assert uniform["pearson_probability"] is None
    assert uniform["spearman_probability"] is None
    assert actual["target_probability_sd"] > 0


def test_single_sample_rejects_invalid_leave_one_out(experiment, dataset):
    rows, _ = dataset
    x, starts, target = experiment.tensors(rows[:1])
    with pytest.raises(ValueError, match="at least two"):
        experiment.sampled_loss(experiment.Policy()(x, starts), target, 1, torch.Generator())
