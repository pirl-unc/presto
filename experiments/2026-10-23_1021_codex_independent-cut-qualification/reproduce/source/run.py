"""Frozen mathematical gates and matched synthetic rate-recovery experiment."""

import itertools
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from process import expected, sample, sampled_loss, spans
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())
TRUE = np.array(CONFIG["rates"])
TIMES = CONFIG["times"]
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)


def save(name, value):
    (ROOT / "results" / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def enumeration(rates, time_value):
    """Independent Python cut-state construction and analytic log-rate scores."""
    length = len(rates) + 1
    support = [(i, j) for i in range(length) for j in range(i + 1, length + 1)]
    states = np.array(list(itertools.product([False, True], repeat=len(rates))))
    p = -np.expm1(-rates * time_value)
    weights = np.prod(np.where(states, p, 1 - p), axis=1)
    counts = np.zeros((len(states), len(support)))
    for k, cuts in enumerate(states):
        boundaries = [0] + [i + 1 for i, cut in enumerate(cuts) if cut] + [length]
        for interval in itertools.pairwise(boundaries):
            counts[k, support.index(interval)] += 1
    scores = (
        np.zeros_like(states, dtype=float)
        if time_value == 0
        else np.where(states, rates * time_value * (1 - p) / p, -rates * time_value)
    )
    return weights, counts, scores


def initial(seed, constant=False):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(
        torch.log(torch.full((1 if constant else len(TRUE),), 0.3))
        + torch.randn(1 if constant else len(TRUE), generator=generator) * 0.1
    )


def metrics(pred, truth):
    return {
        "mse": float(np.mean((pred - truth) ** 2)),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "pearson": float(pearsonr(pred.ravel(), truth.ravel()).statistic),
        "spearman": float(spearmanr(pred.ravel(), truth.ravel()).statistic),
    }


def checks():
    times = torch.tensor([0.0, *TIMES["train"]])
    rates = torch.tensor(TRUE)
    values = expected(rates, times).numpy()
    start, end = spans(len(TRUE) + 1)
    lengths = (end - start).numpy()
    residuals = []
    for t, value in zip(times.tolist(), values):
        weights, counts, _ = enumeration(TRUE, t)
        assert np.all(counts @ lengths == 6)
        residuals.append(float(np.max(np.abs(weights @ counts - value))))
    mass_error = float(np.max(np.abs(values @ lengths - 6)))
    number_error = float(
        np.max(
            np.abs(
                values.sum(1)
                - (1 + (-np.expm1(-np.array(times)[:, None] * TRUE)).sum(1))
            )
        )
    )
    log_rates = initial(42)
    t = torch.tensor([1.5])
    target = expected(rates, t).detach()
    loss = ((expected(log_rates.exp(), t) - target) ** 2).mean()
    gradient = torch.autograd.grad(loss, log_rates)[0].numpy()
    w, counts, scores = enumeration(log_rates.detach().exp().numpy(), 1.5)
    mean = w @ counts
    derivative = np.einsum("s,sn,sb->nb", w, counts, scores)
    analytic = 2 * ((mean - target.numpy()[0])[:, None] * derivative).mean(0)
    costs = (
        (counts - target.numpy()[0]) @ (counts - target.numpy()[0]).T / counts.shape[1]
    )
    pair_gradient = np.einsum(
        "i,j,ij,ijb->b", w, w, costs, scores[:, None, :] + scores[None, :, :]
    )
    summary = {
        "max_state_enumeration_error": max(residuals),
        "mass_error": mass_error,
        "fragment_number_error": number_error,
        "analytic_gradient_error": float(np.max(np.abs(analytic - gradient))),
        "two_group_expectation_gradient_error": float(
            np.max(np.abs(pair_gradient - gradient))
        ),
        "exact_gradient": gradient.tolist(),
    }
    assert max(v for k, v in summary.items() if k.endswith("error")) < 1e-11
    probes = {}
    for count in (8, 32, 128):
        gen = torch.Generator().manual_seed(20261023 + count)
        estimates = []
        sampled_mass_error = 0.0
        for _ in range(512):
            value, _, _ = sampled_loss(log_rates.exp(), t, target, count, gen)
            estimates.append(torch.autograd.grad(value, log_rates)[0].numpy())
        estimates = np.array(estimates)
        sampled_counts, _ = sample(log_rates.exp(), t, count, gen)
        sampled_mass_error = float(np.max(np.abs(sampled_counts.numpy() @ lengths - 6)))
        assert sampled_mass_error == 0
        average = estimates.mean(0)
        probes[str(count)] = {
            "gradients": estimates.tolist(),
            "mean_gradient": average.tolist(),
            "standard_error": (
                estimates.std(0, ddof=1) / np.sqrt(len(estimates))
            ).tolist(),
            "relative_rms_error": float(
                np.sqrt(np.mean(np.sum((estimates - gradient) ** 2, axis=1)))
                / np.linalg.norm(gradient)
            ),
            "relative_mean_error": float(
                np.linalg.norm(average - gradient) / np.linalg.norm(gradient)
            ),
            "mean_cosine": float(
                average
                @ gradient
                / (np.linalg.norm(average) * np.linalg.norm(gradient))
            ),
            "sampled_mass_error": sampled_mass_error,
        }
    save("math-checks.json", summary)
    save("gradient-probes.json", probes)
    print("Mathematical gates passed", summary, flush=True)


def fit(method, seed):
    start = time.perf_counter()
    constant = method == "constant"
    log_rates = initial(seed, constant)
    optimizer = torch.optim.Adam([log_rates], lr=0.03)
    gen = torch.Generator().manual_seed(seed + 1000)
    times = torch.tensor(TIMES["train"])
    target = expected(torch.tensor(TRUE), times).detach()
    count = (
        int(method.removeprefix("sampled")) if method.startswith("sampled") else None
    )
    checkpoints = {"0": log_rates.detach().tolist()}
    history = []
    fragments = 0
    for update in range(1, 1001):
        optimizer.zero_grad()
        rates = log_rates.exp().expand(len(TRUE))
        if count:
            loss, estimate, nfrag = sampled_loss(rates, times, target, count, gen)
            fragments += nfrag
        else:
            loss = ((expected(rates, times) - target) ** 2).mean()
            estimate = loss.detach()
        loss.backward()
        optimizer.step()
        if update in (1, 100, 500, 1000):
            history.append(
                {
                    "update": update,
                    "objective_estimate": float(estimate.detach()),
                    "gradient": log_rates.grad.tolist(),
                }
            )
        if update in (100, 500, 1000):
            checkpoints[str(update)] = log_rates.detach().tolist()
    elapsed = time.perf_counter() - start
    rates = log_rates.detach().exp().expand(len(TRUE))
    predictions = []
    scores = {}
    a, b = spans(len(TRUE) + 1)
    for split, time_values in TIMES.items():
        pred = expected(rates, torch.tensor(time_values)).numpy()
        truth = expected(torch.tensor(TRUE), torch.tensor(time_values)).numpy()
        scores[split] = metrics(pred, truth)
        for j, t in enumerate(time_values):
            for k, (i, end) in enumerate(zip(a.tolist(), b.tolist())):
                predictions.append(
                    {
                        "split": split,
                        "time_hours": t,
                        "start": i,
                        "end": end,
                        "target": float(truth[j, k]),
                        "prediction": float(pred[j, k]),
                    }
                )
    result = {
        "method": method,
        "seed": seed,
        "rates": rates.tolist(),
        "median_relative_rate_error": float(
            np.median(np.abs(rates.numpy() - TRUE) / TRUE)
        ),
        "scores": scores,
        "fit_seconds": elapsed,
        "checkpoints": checkpoints,
        "history": history,
        "predictions": predictions,
        "work": {
            "updates": 1000,
            "bond_probability_evaluations": 1000 * len(TIMES["train"]) * len(TRUE),
            "sampled_partitions": 0
            if count is None
            else 1000 * len(TIMES["train"]) * 2 * count,
            "sampled_fragments": fragments,
            "exact_span_expectations": 0 if count else 1000 * len(TIMES["train"]) * 21,
        },
    }
    save(f"{method}_seed{seed}.json", result)
    print(
        method,
        seed,
        scores["test"],
        result["median_relative_rate_error"],
        elapsed,
        flush=True,
    )
    return {
        k: v
        for k, v in result.items()
        if k not in ("predictions", "checkpoints", "history")
    }


def main():
    started = time.perf_counter()
    checks()
    runs = [
        fit(method, seed)
        for method in ("exact", "sampled32", "sampled128")
        for seed in (42, 43, 44)
    ]
    control = fit("constant", 42)
    gates = {
        m: {
            "median_relative_rate_error": float(
                np.median(
                    [r["median_relative_rate_error"] for r in runs if r["method"] == m]
                )
            ),
            "median_test_rmse": float(
                np.median(
                    [r["scores"]["test"]["rmse"] for r in runs if r["method"] == m]
                )
            ),
        }
        for m in ("exact", "sampled32", "sampled128")
    }
    for gate in gates.values():
        gate["passed"] = (
            gate["median_relative_rate_error"] < 0.1
            and gate["median_test_rmse"] < control["scores"]["test"]["rmse"]
        )
    save(
        "summary.json",
        {
            "runs": runs,
            "control": control,
            "gates": gates,
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    )


if __name__ == "__main__":
    main()
