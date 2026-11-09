"""Prospective source/support and gradient qualification before biological fits."""

import json
import resource
import time

import numpy as np
import torch
from core import CONDITIONS, EXP, data, log_response, reference, write


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    z, eta = np.meshgrid([-8.0, -3.0, 0.0, 3.0, 8.0], [-6.0, -3.0, 0.0, 3.0])
    z, eta = z.ravel(), eta.ravel()
    rho, eps = 1 / (1 + np.exp(-z)), 1 / (1 + np.exp(-eta))
    response = 1 - rho + eps * rho
    analytic_z = -(1 - eps) * rho * (1 - rho) / response
    analytic_eta = eps * rho * (1 - eps) / response
    tz = torch.tensor(z, requires_grad=True)
    te = torch.tensor(eta, requires_grad=True)
    exact = log_response(tz, te)
    actual = torch.autograd.grad(exact.sum(), (tz, te))
    forward_error = np.max(abs(exact.detach().numpy() - np.log(response)))
    gradient_error = max(
        np.max(abs(actual[0].numpy() - analytic_z)),
        np.max(abs(actual[1].numpy() - analytic_eta)),
    )
    assert forward_error < 1e-12 and gradient_error < 1e-10
    step = 1e-5
    numerical = []
    for which in range(2):
        plus = log_response(
            torch.tensor(z + (step if which == 0 else 0)),
            torch.tensor(eta + (step if which == 1 else 0)),
        )
        minus = log_response(
            torch.tensor(z - (step if which == 0 else 0)),
            torch.tensor(eta - (step if which == 1 else 0)),
        )
        numerical.append(((plus - minus) / (2 * step)).numpy())
    fd_error = max(
        np.max(abs(a - b)) for a, b in zip(numerical, [analytic_z, analytic_eta], strict=True)
    )
    assert fd_error < 1e-6
    assert np.max(abs(np.log(1 - rho + rho))) == 0
    assert np.all((1 - rho + np.minimum(1, eps + 0.01) * rho) >= response)
    tz = torch.tensor(np.tile(z, (256, 1)), requires_grad=True)
    te = torch.tensor(np.tile(eta, (256, 1)), requires_grad=True)
    sampled = log_response(tz, te, torch.Generator().manual_seed(42))
    grads = torch.autograd.grad(sampled.sum(), (tz, te))
    posterior = eps * rho / response
    se_z = np.sqrt(posterior * (1 - posterior) / (64 * 256))
    se_eta = se_z * (1 - eps)
    standardized_errors = []
    for actual, expected, se in zip(grads, [analytic_z, analytic_eta], [se_z, se_eta], strict=True):
        error = abs(actual.numpy().mean(axis=0) - expected)
        assert np.all(error <= 5 * se + 1e-10), (error, se)
        standardized_errors.append(float(np.max(error / np.maximum(se, 1e-15))))

    # Independent ridge/offset fixture; qualification uses no biological outcomes.
    fixture = []
    phi = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]] * 2)
    for c in range(2):
        for i in range(4):
            fixture.append(
                {
                    "condition_id": CONDITIONS[c],
                    "component_id": str(i),
                    "peptide_sequence": str(i),
                    "precursor_log10_ratio": i / 4,
                    "target_log10_ratio": 0.2 * i + (-1) ** i + c,
                }
            )
    prediction, cp = reference.fit(fixture, fixture, phi, phi)
    w = np.full(8, 1 / 8)
    means, sd = w @ phi, np.sqrt(w @ (phi - w @ phi) ** 2)
    zz = (phi - means) / sd
    independent = np.zeros(8)
    projection_error = 0.0
    for c, name in enumerate(CONDITIONS):
        sl = slice(4 * c, 4 * c + 4)
        a = zz[sl] - zz[sl].mean(0)
        yy = np.array([r["target_log10_ratio"] for r in fixture[sl]])
        xx = np.array([r["precursor_log10_ratio"] for r in fixture[sl]])
        residual = yy - cp["gamma"] * xx - cp["offsets"][name]
        beta = np.linalg.lstsq(
            np.vstack([a / 2, np.sqrt(0.1) * np.eye(2)]), np.r_[residual / 2, [0, 0]], rcond=None
        )[0]
        independent[sl] = cp["gamma"] * xx + cp["offsets"][name] + a @ beta
        projection_error = max(projection_error, abs(np.mean(yy - independent[sl])))
    ridge_error = float(np.max(abs(prediction - independent)))
    assert ridge_error < 1e-12 and projection_error < 1e-12

    d = data()
    counts = {}
    for cond in CONDITIONS:
        rs = [r for r in d["rows"] if r["condition_id"] == cond]
        counts[cond] = {
            "observations": len(rs),
            "peptides": len({r["peptide_sequence"] for r in rs}),
            "components": len({r["component_id"] for r in rs}),
            "fold_components": [],
        }
        assert counts[cond]["peptides"] >= 300 and counts[cond]["components"] >= 100
        for f in range(5):
            groups = {
                r["component_id"]
                for i, r in enumerate(d["rows"])
                if r["condition_id"] == cond and d["fold"][i] == f
            }
            counts[cond]["fold_components"].append(len(groups))
            assert len(groups) >= 20
    result = {
        "status": "passed",
        "counts": counts,
        "features": d["availability"].shape[1],
        "forward_error": float(forward_error),
        "gradient_error": float(gradient_error),
        "finite_difference_error": float(fd_error),
        "sampled_gradient_max_standard_errors": standardized_errors,
        "ridge_fixture_error": ridge_error,
        "offset_projection_error": projection_error,
        "seconds": time.perf_counter() - started,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "outer_validation_test_scored": False,
    }
    assert result["seconds"] < 120 and result["peak_gib"] < 2
    write(EXP / "results/qualification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
