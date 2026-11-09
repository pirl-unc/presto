"""Pre-fit independent state sums and real-training finite differences."""

import itertools
import math
import time

import torch

from common import FAMILY, designs, dump, load_data, objective
from presto.models.independent_cut import log_fragment_expectation, partition_contains


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    cfg, groups = load_data()
    rates = torch.tensor([0.08, 0.18, 0.45, 0.9, 1.4], dtype=torch.float64)
    logs = rates.log().requires_grad_()
    cuts = torch.tensor(list(itertools.product((False, True), repeat=5)))
    starts, last = torch.triu_indices(6, 6)
    ends = last + 1
    # Independent per-state explicit fragment lists, without partition_contains.
    explicit = torch.zeros((32, 21), dtype=torch.float64)
    for k, state in enumerate(cuts.tolist()):
        boundaries = [0] + [i + 1 for i, c in enumerate(state) if c] + [6]
        fragments = set(zip(boundaries[:-1], boundaries[1:]))
        for i, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
            explicit[k, i] = float((a, b) in fragments)
    assert torch.equal(explicit, partition_contains(cuts, starts, ends).double())
    errors = []
    for t in (0.1, 0.7, 4.0, 20.0):
        h = logs.exp() * t
        logp = torch.where(cuts, torch.log(-torch.expm1(-h)), -h).sum(1)
        amount = log_fragment_expectation(
            logs, starts, ends, torch.full((21,), t, dtype=torch.float64)
        ).exp()
        state_amount = logp.exp() @ explicit
        a = torch.autograd.grad(amount.sum(), logs, retain_graph=True)[0]
        b = torch.autograd.grad(state_amount.sum(), logs, retain_graph=True)[0]
        errors.append(
            dict(
                time=t,
                amount_error=float((amount - state_amount).abs().max().detach()),
                mass_error=abs(float((amount @ (ends - starts).double()).detach()) - 6),
                gradient_error=float((a - b).abs().max()),
            )
        )
    assert (
        max(
            max(r[k] for k in ("amount_error", "mass_error", "gradient_error"))
            for r in errors
        )
        < 1e-12
    )
    finite = []
    for variant in cfg["variants"]:
        design, basis, audit = designs(groups, variant)
        original = torch.zeros(basis.shape[0], dtype=torch.float64)
        original[0] = math.log(0.01)
        theta = (basis.T @ original).requires_grad_()
        loss, _ = objective(theta, groups, design, basis, cfg["l2_nonintercept"])
        grad = torch.autograd.grad(loss, theta)[0]
        numerical = []
        for j in range(len(theta)):
            plus, minus = theta.detach().clone(), theta.detach().clone()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            numerical.append(
                float(
                    (
                        objective(plus, groups, design, basis, cfg["l2_nonintercept"])[
                            0
                        ]
                        - objective(
                            minus, groups, design, basis, cfg["l2_nonintercept"]
                        )[0]
                    )
                    / (2e-5)
                )
            )
        error = float((grad - torch.tensor(numerical, dtype=torch.float64)).abs().max())
        assert error < 1e-7, (variant, error)
        finite.append(
            dict(
                variant=variant,
                max_absolute_gradient_error=error,
                gradient=grad.tolist(),
                numerical=numerical,
                allocated_dimensions=len(theta),
                initial_objective=float(loss.detach()),
                source_design_rank=audit["allocated_dimensions"],
            )
        )
    result = dict(
        passed=True,
        state_enumeration=errors,
        training_finite_differences=finite,
        wall_seconds=time.monotonic() - started,
        no_optimizer_updates=True,
        no_heldout_response_evaluation=True,
    )
    dump(FAMILY / "results/math-checks.json", result)
    print("Pre-fit mathematics passed", finite, flush=True)


if __name__ == "__main__":
    main()
