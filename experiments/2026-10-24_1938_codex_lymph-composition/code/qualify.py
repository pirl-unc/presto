"""Freeze execution dependencies and check the exact/score gradient identity."""

import math
import resource
import shutil
import time

import numpy as np
import torch
from core import PHYSICAL, REPO, ROOT, Data, read, sha, write


def main():
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = read(ROOT / "reproduce/launch.json")
    assert not (ROOT / "results/qualification.json").exists()
    files = [
        *sorted((ROOT / "code").glob("*.py")),
        REPO / "models/steady_processing.py",
    ]
    observer = cfg["observer_directory"]
    from pathlib import Path

    files += [
        Path(observer) / "code/model.py",
        Path(observer) / "results/fit/transforms.npz",
        Path(observer) / "results/fit/sequence_0.1.npz",
    ]
    frozen = {
        "files": {str(path): sha(path) for path in files},
        "config_sha256": sha(ROOT / "reproduce/launch.json"),
        "plan_sha256": sha(ROOT / "reproduce/plan.md"),
        "training": cfg["fit"],
    }
    write(ROOT / "reproduce/fitting.json", frozen)
    for path in files:
        if path.parent == ROOT / "code":
            shutil.copy2(path, ROOT / "reproduce/source" / path.name)
    probes = []
    for block, info in read(ROOT / "results/admission.json")["cohort_support"].items():
        if not info["supported"]:
            continue
        data = Data(block)
        y, p = data.labels("train")
        y = torch.tensor(y)
        p = torch.tensor(p)
        theta = torch.zeros(39, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            theta[0] = math.log(0.01)

        def gradient(kind, data=data, theta=theta, y=y, p=p):
            return torch.autograd.grad(data.loss(data.score(theta), y, p, kind), theta)[
                0
            ]

        exact = gradient("exact").detach().numpy()
        expected_gradient = gradient("score_expectation").detach().numpy()
        assert np.max(abs(exact - expected_gradient)) < 1e-8
        numeric = []
        for coord in (0, 1, 20):
            vals = []
            for sign in (-1, 1):
                changed = theta.detach().clone()
                changed[coord] += sign * 1e-5
                vals.append(data.loss(data.score(changed), y, p, "exact").item())
            numeric.append(abs((vals[1] - vals[0]) / 2e-5 - exact[coord]))
        assert max(numeric) < 1e-8
        # Independent direct scalar rational expression, including protein termini.
        for a, l, r in (
            (0.2, 0.0, 0.0),
            (0.3, 0.01, 0.0),
            (0.4, 0.0, 0.2),
            (0.3, 0.1, 0.2),
        ):
            total = 1 + a
            expected = 1 / total
            if l and r:
                expected = (
                    l
                    * r
                    * (2 * total + l + r)
                    / (total * (total + l) * (total + r) * (total + l + r))
                )
            elif l or r:
                expected = (l + r) / (total * (total + l + r))
            actual = (
                PHYSICAL.log_steady_fragment_amount(
                    *[torch.tensor(v, dtype=torch.float64) for v in (a, l, r, 1.0)]
                )
                .exp()
                .item()
            )
            assert abs(actual - expected) < 1e-12
        gen = torch.Generator().manual_seed(42)
        samples = []
        for _ in range(256):
            score = data.score(theta)
            loss = data.loss(score, y, p, "reinforce", gen, 64)
            samples.append(torch.autograd.grad(loss, theta)[0].detach().numpy())
        samples = np.array(samples)
        mean = samples.mean(0)
        error = float(np.linalg.norm(mean - exact))
        relative = error / max(float(np.linalg.norm(exact)), 1e-12)
        admitted = relative <= 0.15 or error <= 0.01
        probes.append(
            {
                "block": block,
                "exact_gradient": exact.tolist(),
                "score_expectation_max_error": float(
                    np.max(abs(exact - expected_gradient))
                ),
                "finite_difference_max_error": max(numeric),
                "sample_mean_gradient": mean.tolist(),
                "sample_relative_mean_error": relative,
                "sample_absolute_mean_error": error,
                "sample_rms_relative_error": float(
                    np.sqrt(np.mean(np.sum((samples - exact) ** 2, axis=1)))
                    / max(np.linalg.norm(exact), 1e-12)
                ),
                "sampled_admitted": admitted,
            }
        )
    result = {
        "status": "passed",
        "probes": probes,
        "runtime_seconds": time.perf_counter() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
    }
    assert result["runtime_seconds"] < 600 and result["peak_rss_gib"] < 3
    write(ROOT / "results/qualification.json", result)
    print(result)


if __name__ == "__main__":
    main()
