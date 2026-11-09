"""Frozen dense shifted-solve/exponential checks; no biological input access."""

import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm, expm_frechet

from rational import expectations, shifted_solve

FAMILY = Path(__file__).resolve().parents[1]


def dense(logs, eta):
    size = len(logs) + 1
    spans = [(a, b) for a in range(size) for b in range(a + 1, size + 1)]
    lookup = {v: i for i, v in enumerate(spans)}
    # Independent explicit daughter construction, not row/column accumulation.
    columns = []
    for j, (a, b) in enumerate(spans):
        col = logs.new_zeros(len(spans))
        for k in range(a, b - 1):
            stoich = np.zeros(len(spans))
            stoich[j] -= 1
            stoich[lookup[a, k + 1]] += 1
            stoich[lookup[k + 1, b]] += 1
            rate = (logs[k] + eta * math.log((b - a) / 20)).exp()
            col = col + rate * torch.tensor(stoich, dtype=logs.dtype)
        columns.append(col)
    return torch.stack(columns, dim=1), spans


def dense_numpy(logs, eta):
    size = len(logs) + 1
    spans = [(a, b) for a in range(size) for b in range(a + 1, size + 1)]
    lookup = {v: i for i, v in enumerate(spans)}
    a = np.zeros((len(spans), len(spans)))
    da = np.zeros((size, len(spans), len(spans)))
    for j, (start, end) in enumerate(spans):
        length = math.log((end - start) / 20)
        for bond in range(start, end - 1):
            hazard = math.exp(logs[bond] + eta * length)
            for target, sign in [
                (j, -1),
                (lookup[start, bond + 1], 1),
                (lookup[bond + 1, end], 1),
            ]:
                a[target, j] += sign * hazard
                da[bond, target, j] += sign * hazard
                da[-1, target, j] += sign * hazard * length
    return a, da, spans


def max_error(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    freeze = json.loads((FAMILY / "reproduce/qualification.json").read_text())
    for name, digest in freeze["source_hashes"].items():
        assert hashlib.sha256((FAMILY / name).read_bytes()).hexdigest() == digest
    tick = time.perf_counter()
    solutions, grids = [], []
    for size in cfg["sizes"]:
        base = 0.02 + 0.13 * (1 + np.sin(np.arange(size - 1) + 0.3)) / 2
        for power in cfg["eta"]:
            for shift_re, shift_im in cfg["shifted_solve_shifts"]:
                logs = torch.tensor(np.log(base), requires_grad=True)
                eta = torch.tensor(power, requires_grad=True)
                rhs = torch.tensor(
                    np.triu(np.cos(np.arange(size * size).reshape(size, size) + 0.1)),
                    requires_grad=True,
                )
                matrix, spans = dense(logs, eta)
                indices = tuple(
                    torch.tensor(x) for x in zip(*[(a, b - 1) for a, b in spans])
                )
                shift = complex(shift_re, shift_im)
                answer = shifted_solve(logs, eta, rhs, shift)
                expected = torch.linalg.solve(
                    shift * torch.eye(len(spans)) - matrix,
                    rhs[indices].to(torch.complex128),
                )
                weights = torch.tensor(np.sin(np.arange(len(spans))) + 1.1)
                first = torch.autograd.grad(
                    (answer[indices].real * weights).sum(),
                    (logs, eta, rhs),
                    retain_graph=True,
                )
                second = torch.autograd.grad(
                    (expected.real * weights).sum(), (logs, eta, rhs)
                )
                error = max_error(answer[indices].detach(), expected.detach())
                residual = max_error(
                    (
                        (shift * torch.eye(len(spans)) - matrix) @ answer[indices]
                    ).detach(),
                    rhs[indices].detach(),
                )
                gradient_error = max(
                    max_error(x.detach(), y.detach()) for x, y in zip(first, second)
                )
                solutions.append(
                    dict(
                        size=size,
                        eta=power,
                        shift=[shift_re, shift_im],
                        solution_error=error,
                        residual=residual,
                        gradient_error=gradient_error,
                        passed=error <= 1e-10
                        and residual <= 1e-10
                        and gradient_error <= 1e-9,
                    )
                )
            for scale in cfg["rate_scales"]:
                values = np.log(base * scale)
                matrix, derivatives, spans = dense_numpy(values, power)
                indices = tuple(
                    torch.tensor(x) for x in zip(*[(a, b - 1) for a, b in spans])
                )
                initial = np.zeros(len(spans))
                initial[spans.index((0, size))] = 1
                lengths = np.array([b - a for a, b in spans])
                weight = np.sin(np.arange(len(spans))) + 1.1
                for t in cfg["times_hours"]:
                    expected = expm(t * matrix) @ initial
                    exact_gradient = np.array(
                        [
                            weight
                            @ expm_frechet(t * matrix, t * d, compute_expm=False)
                            @ initial
                            for d in derivatives
                        ]
                    )
                    for order in cfg["orders"]:
                        logs = torch.tensor(values, requires_grad=True)
                        eta = torch.tensor(power, requires_grad=True)
                        start = time.perf_counter()
                        counts = expectations(logs, eta, t, order=order)[indices]
                        gradients = torch.autograd.grad(
                            (counts * torch.tensor(weight)).sum(), (logs, eta)
                        )
                        elapsed = time.perf_counter() - start
                        prediction = counts.detach().numpy()
                        gradient = np.r_[
                            gradients[0].detach().numpy(), float(gradients[1])
                        ]
                        error = max_error(prediction, expected)
                        grad_error = max_error(gradient, exact_gradient)
                        mass_error = abs(float(prediction @ lengths) - size)
                        supported = expected > 1e-12
                        relative = float(
                            np.max(
                                np.abs(prediction[supported] / expected[supported] - 1)
                            )
                        )
                        numeric = (
                            error <= 1e-10
                            and grad_error <= 1e-8
                            and mass_error <= 1e-8
                            and np.isfinite(prediction).all()
                        )
                        grids.append(
                            dict(
                                size=size,
                                eta=power,
                                scale=scale,
                                time=t,
                                order=order,
                                count_error=error,
                                gradient_error=grad_error,
                                mass_error=mass_error,
                                minimum_count=float(prediction.min()),
                                negative_count=int((prediction < 0).sum()),
                                maximum_relative_error_above_1e_12=relative,
                                numerical_gate=bool(numeric),
                                log_admission_gate=bool(
                                    numeric and (prediction >= 0).all()
                                ),
                                seconds=elapsed,
                                spans=spans,
                                expected_counts=expected.tolist(),
                                predicted_counts=prediction.tolist(),
                                expected_gradient=exact_gradient.tolist(),
                                predicted_gradient=gradient.tolist(),
                            )
                        )
    (FAMILY / "results/shifted-solves.json").write_text(
        json.dumps(solutions, indent=2) + "\n"
    )
    (FAMILY / "results/exponential-grid.json").write_text(
        json.dumps(grids, indent=2) + "\n"
    )
    summary = dict(
        shifted_solves_pass=all(r["passed"] for r in solutions),
        n_shifted_solves=len(solutions),
        n_exponential_cases=len(grids),
        orders={
            str(o): {
                "numerical_pass": all(
                    r["numerical_gate"] for r in grids if r["order"] == o
                ),
                "log_admission_pass": all(
                    r["log_admission_gate"] for r in grids if r["order"] == o
                ),
                "maximum_count_error": max(
                    r["count_error"] for r in grids if r["order"] == o
                ),
                "maximum_gradient_error": max(
                    r["gradient_error"] for r in grids if r["order"] == o
                ),
                "negative_cases": sum(
                    r["negative_count"] > 0 for r in grids if r["order"] == o
                ),
                "minimum_count": min(
                    r["minimum_count"] for r in grids if r["order"] == o
                ),
            }
            for o in cfg["orders"]
        },
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (FAMILY / "results/qualification-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
