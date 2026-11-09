"""Prospective actual-cohort exact/finite-difference and stochastic score gates."""

import json

import numpy as np
import torch
from core import ROOT, Data, Distribution, write


def main():
    assert json.loads((ROOT / "results/independent-admission.json").read_text())["passed"]
    torch.set_num_threads(1)
    data = Data()
    design, basis = data.basis(range(61))
    train = data.indices["train"]
    model = Distribution([data.rows[i] for i in train], design[train], data.labels["train"]["p"])
    dimension = design.shape[1]
    points = [
        np.zeros(dimension),
        np.linspace(-0.15, 0.15, dimension),
        np.random.default_rng(53).normal(0, 0.1, dimension),
    ]
    generator = torch.Generator().manual_seed(9017)
    reports = []
    all_estimates = []
    for point in points:
        beta = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        gradient = torch.autograd.grad(model.exact(beta), beta)[0].detach().numpy()
        analytic = model.analytic(point)
        fd = np.zeros(dimension)
        for j in range(dimension):
            step = np.zeros(dimension)
            step[j] = 1e-5
            fd[j] = float(
                (model.exact(torch.tensor(point + step)) - model.exact(torch.tensor(point - step)))
                / (2e-5)
            )
        estimates = []
        for _ in range(256):
            estimates.append(
                torch.autograd.grad(model.sampled(beta, generator), beta)[0].detach().numpy()
            )
        estimates = np.asarray(estimates)
        mean = estimates.mean(0)
        se = estimates.std(0, ddof=1) / 16
        report = {
            "autograd_analytic_maxabs": float(abs(gradient - analytic).max()),
            "finite_difference_maxabs": float(abs(gradient - fd).max()),
            "sample_relative_mean_l2": float(
                np.linalg.norm(mean - gradient) / max(np.linalg.norm(gradient), 1e-8)
            ),
            "sample_max_standard_error_units": float(
                (abs(mean - gradient) / np.maximum(se, 1e-8)).max()
            ),
            "exact_gradient_l2": float(np.linalg.norm(gradient)),
        }
        report["passed"] = (
            report["autograd_analytic_maxabs"] <= 1e-10
            and report["finite_difference_maxabs"] <= 1e-6
            and report["sample_relative_mean_l2"] <= 0.10
            and report["sample_max_standard_error_units"] <= 5
        )
        reports.append(report)
        all_estimates.append(estimates)
    np.savez_compressed(
        ROOT / "results/qualification-gradients.npz",
        estimates=np.asarray(all_estimates),
        points=np.array(points),
    )
    result = {
        "passed": all(r["passed"] for r in reports),
        "basis": basis,
        "probes": reports,
        "n_training_observations": len(train),
        "n_training_groups": len(model.counts),
        "n_padded_slots": int(model.index.size),
        "n_max_candidates_per_group": int(model.counts.max()),
    }
    write(ROOT / "results/qualification.json", result)
    print(json.dumps(result["probes"], indent=2))
    if not result["passed"]:
        raise RuntimeError("Prespecified numerical gate failed; do not fit")


if __name__ == "__main__":
    main()
