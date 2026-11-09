"""Exact estimator and accounting checks for the generic GRPO ingredient study."""

import copy
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def experiment():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/2026-10-22_1120_codex_grpo-ingredient-factorial/code/launch.py"
    )
    spec = importlib.util.spec_from_file_location("grpo_ingredient_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("beta", [0, 0.1])
def test_off_policy_estimator_exact_expectation(experiment, beta):
    logits = torch.tensor([-0.6, 0.5, 0.2], dtype=torch.float64, requires_grad=True)
    logq = logits.log_softmax(0)
    old = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    target = torch.tensor([0.2, 0.1, 0.7], dtype=torch.float64).log()
    reference = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64).log()
    exact = experiment.objective(logq[None], target[None], reference[None], beta)
    expected = 0
    for first, second in itertools.product(range(3), repeat=2):
        indices = torch.tensor([[first, second]])
        loss, _ = experiment.importance_surrogate(
            logq[indices],
            old.log()[indices],
            target[indices],
            reference[indices],
            beta,
            False,
            False,
        )
        expected += old[first] * old[second] * loss
    actual_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    exact_grad = torch.autograd.grad(exact, logits)[0]
    torch.testing.assert_close(actual_grad, exact_grad, atol=1e-12, rtol=1e-12)


def test_hybrid_estimator_exact_expectation(experiment):
    start = torch.tensor([0.3, -0.4, 0.1], dtype=torch.float64, requires_grad=True)
    lengths = torch.tensor(
        [[0.1, 0.2], [-0.2, 0.4], [0.6, -0.3]], dtype=torch.float64, requires_grad=True
    )
    mask = torch.tensor([[True, True], [True, False], [True, True]])
    start_logq = start.log_softmax(0)
    conditional = lengths.masked_fill(~mask, -torch.inf).log_softmax(-1)
    joint = start_logq[:, None] + conditional
    target = torch.full_like(joint, -torch.inf)
    target[mask] = torch.tensor([0.1, 0.3, 0.2, 0.1, 0.3], dtype=torch.float64).log()
    exact = (joint[mask].exp() * (joint[mask] - target[mask])).sum()
    expected = 0
    for first, second in itertools.product(range(3), repeat=2):
        sampled = torch.tensor([[first, second]])
        loss = experiment.hybrid_surrogate(
            joint[None], conditional[None], target[None], sampled, mask
        )
        expected += (start_logq[first] + start_logq[second]).exp().detach() * loss
    actual_grad = torch.autograd.grad(expected, (start, lengths), retain_graph=True)
    exact_grad = torch.autograd.grad(exact, (start, lengths))
    for actual, reference in zip(actual_grad, exact_grad):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("standardize", [False, True])
def test_clipping_inactive_on_first_update(experiment, standardize):
    current = torch.tensor([[-0.2, -0.7, -1.2]], dtype=torch.float64, requires_grad=True)
    old = current.detach().clone()
    target = torch.tensor([[-0.8, -0.4, -2.0]], dtype=torch.float64)
    plain, _ = experiment.importance_surrogate(current, old, target, old, 0.1, False, standardize)
    clipped, diagnostics = experiment.importance_surrogate(
        current, old, target, old, 0.1, True, standardize
    )
    plain_grad = torch.autograd.grad(plain, current, retain_graph=True)[0]
    clipped_grad = torch.autograd.grad(clipped, current)[0]
    torch.testing.assert_close(clipped_grad, plain_grad, rtol=0, atol=0)
    assert diagnostics["active_clip_fraction"] == 0


def test_regularization_has_declared_optimum(experiment):
    target = torch.tensor([[0.8, 0.15, 0.05]], dtype=torch.float64).log()
    reference = torch.tensor([[0.05, 0.25, 0.7]], dtype=torch.float64).log()
    optimum = experiment.optimum(target, reference, 0.1).detach().requires_grad_(True)
    loss = experiment.objective(optimum.log_softmax(-1), target, reference, 0.1)
    gradient = torch.autograd.grad(loss, optimum)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)
    assert not torch.allclose(optimum, target)


def test_clipping_stops_updates_beyond_active_bounds(experiment):
    current = torch.tensor([[-2.0, -1.0]], dtype=torch.float64, requires_grad=True)
    old = current.detach() - torch.tensor([[0.5, 2.0]], dtype=torch.float64).log()
    target = current.detach() - torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    loss, diagnostic = experiment.importance_surrogate(current, old, target, old, 0, True, False)
    gradient = torch.autograd.grad(loss, current)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
    assert diagnostic["active_clip_fraction"] == 1


def test_frozen_reference_and_original_gradient_parity(experiment):
    rows, _ = experiment.base.make_dataset()
    x, starts, target = experiment.base.tensors(rows[:4])
    torch.manual_seed(42)
    first = experiment.base.Policy()
    second = copy.deepcopy(first)
    reference = experiment.flat(first(x, starts)).detach().requires_grad_(True)
    output = first(x, starts)
    cache = experiment.collect(
        output, target[:, experiment.MASK], reference, 8, torch.Generator().manual_seed(7)
    )
    loss, _ = experiment.reused_loss(
        output, cache, {"beta": 0, "clip": False, "standardize": False}
    )
    loss.backward()
    other = experiment.base.sampled_loss(
        second(x, starts), target, 8, torch.Generator().manual_seed(7)
    )
    other.backward()
    assert reference.grad is None
    torch.testing.assert_close(
        experiment.base.gradient_vector(first),
        experiment.base.gradient_vector(second),
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize("reuse", [1, 4])
def test_reuse_spends_fewer_collected_target_terms(experiment, tmp_path, reuse):
    rows, _ = experiment.base.make_dataset()
    groups = {
        split: [row for row in rows if row["split"] == split][:16]
        for split in ("train", "validation", "test")
    }
    torch.manual_seed(42)
    model = experiment.base.Policy()
    references = {
        split: experiment.probabilities(model, subset) for split, subset in groups.items()
    }
    original_refs = {split: values.copy() for split, values in references.items()}
    config = {
        "name": "test",
        "method": "factorial",
        "clip": True,
        "standardize": False,
        "reuse": reuse,
        "beta": 0.1,
        "samples": 8,
        "steps": 8,
        "budget": "unit_test",
    }
    report = experiment.train(config, 42, groups, references, tmp_path)
    assert report["updates"] == 8
    assert report["collected_batches"] == 8 // reuse
    assert report["fit_target_terms"] == 8 // reuse * 16 * 8
    assert report["fit_dense_span_scores"] == 8 * 16 * 70
    assert report["diagnostics_by_reuse_step"]["0"]["active_clip_fraction"] == 0
    for split in references:
        np.testing.assert_array_equal(references[split], original_refs[split])


def test_factorial_is_complete(experiment):
    configs = experiment.conditions()
    factorial = [config for config in configs if config["name"].startswith("factorial")]
    assert len(configs) * 3 == 69
    assert {(c["clip"], c["standardize"], c["reuse"], c["beta"]) for c in factorial} == set(
        itertools.product((False, True), (False, True), (1, 4), (0, 0.1))
    )
