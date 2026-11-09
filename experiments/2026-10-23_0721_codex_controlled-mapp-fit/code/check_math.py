"""Pre-fit numerical checks, without biological inputs or outcomes."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from model import analytic_gradient, exact_loss, log_policy, sampled_loss

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)
started = time.perf_counter()
x = torch.tensor(
    [
        [0.2, -0.3, 0.1, 1.0],
        [0.5, 0.7, -0.8, 0.2],
        [-0.4, 0.1, 1.2, -0.5],
        [0.3, -0.6, -0.2, 0.4],
    ]
)
base, target = (
    torch.tensor([0.1, 0.2, 0.3, 0.4]).log(),
    torch.tensor([0.4, 0.1, 0.35, 0.15]).log(),
)
w = torch.tensor([0.3, -0.2, 0.5, 0.1], requires_grad=True)
g = torch.autograd.grad(exact_loss(w, x, base, target), w)[0].numpy()
analytic = analytic_gradient(
    w.detach().numpy(), x.numpy(), base.numpy(), target.numpy()
)
assert np.allclose(g, analytic, rtol=1e-12, atol=1e-12)
numeric = []
for j in range(4):
    d = torch.zeros(4)
    d[j] = 1e-6
    numeric.append(
        float(
            (
                exact_loss(w + d, x, base, target) - exact_loss(w - d, x, base, target)
            ).detach()
        )
        / 2e-6
    )
assert np.allclose(g, numeric, rtol=1e-7, atol=1e-9)
p = log_policy(w, x, base).detach().exp()
expected = np.zeros(4)
for i in range(4):
    for j in range(4):
        loss = sampled_loss(w, x, base, target, 2, None, indices=torch.tensor([i, j]))
        estimate = torch.autograd.grad(loss, w)[0].numpy()
        expected += float(p[i] * p[j]) * estimate
assert np.allclose(expected, g, rtol=1e-12, atol=1e-12)
generator = torch.Generator().manual_seed(20261023)
estimates = []
for _ in range(4096):
    value = sampled_loss(w, x, base, target, 32, generator)
    estimates.append(torch.autograd.grad(value, w)[0].numpy())
estimates = np.array(estimates)
sem = estimates.std(0, ddof=1) / np.sqrt(len(estimates))
z = np.abs(estimates.mean(0) - g) / sem
assert np.all(z < 6), z
result = {
    "passed": True,
    "biological_inputs_used": False,
    "finite_difference_max_absolute_error": float(np.max(np.abs(g - numeric))),
    "analytic_max_absolute_error": float(np.max(np.abs(g - analytic))),
    "exhaustive_two_draw_LOO_max_absolute_error": float(np.max(np.abs(expected - g))),
    "monte_carlo_groups": len(estimates),
    "monte_carlo_draws_per_group": 32,
    "monte_carlo_standard_errors": sem.tolist(),
    "monte_carlo_mean_error_zscores": z.tolist(),
    "prospective_tolerance": "each coordinate < 6 Monte Carlo standard errors; exhaustive expectation atol1e-12",
    "wall_seconds": time.perf_counter() - started,
}
(ROOT / "results/math-checks.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, sort_keys=True))
