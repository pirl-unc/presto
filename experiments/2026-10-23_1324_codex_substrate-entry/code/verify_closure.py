"""Independent source, predictions, continuous gradients and sampled updates."""

import base64
import gzip
import json
import math
import resource
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import pearsonr

from independent import FAMILY, entry_forward, entry_points, forward, read_groups


def sampled_gradient(theta, groups, count, rng):
    gradient = np.zeros_like(theta)
    for g in groups:
        if g["split"] != "train":
            continue
        x = g["x"]
        rates = np.exp(x @ theta[:-1])
        lam = np.exp(theta[-1])
        n = len(rates)
        keys = sorted(
            set(zip(g["a"], g["b"], g["t"])) | set(zip(g["a"], g["b"], g["ref"]))
        )
        lookup = {key: i for i, key in enumerate(keys)}
        a = np.array([k[0] for k in keys])
        b = np.array([k[1] for k in keys])
        times = np.array([k[2] for k in keys])
        ti = np.array([lookup[k] for k in zip(g["a"], g["b"], g["t"])])
        ri = np.array([lookup[k] for k in zip(g["a"], g["b"], g["ref"])])
        out = entry_points(theta, x, a, b, times, order=32)
        prediction = (out[ti, 0] - out[ri, 0]) / math.log(10)
        residual = 2 * g["w"] * (prediction - g["y"]) / math.log(10)
        combined = np.bincount(ti, weights=residual, minlength=len(keys)) - np.bincount(
            ri, weights=residual, minlength=len(keys)
        )
        hx = np.vstack(
            (np.zeros((1, len(theta) - 1)), np.cumsum(rates[:, None] * x, axis=0))
        )
        for t in np.unique(times):
            mask = times == t
            starts = a[mask]
            ends = b[mask]
            coefficient = combined[mask] / np.exp(out[mask, 0])
            uniform = torch.rand(count, dtype=torch.float64, generator=rng).numpy()
            entered_at = -np.log1p(-uniform) / lam
            entered = entered_at < t
            age = np.where(entered, t - entered_at, 0.0)
            safe_age = np.where(entered, age, 1.0)
            cutprob = -np.expm1(-safe_age[:, None] * rates)
            uniforms = torch.rand(
                (count, n), dtype=torch.float64, generator=rng
            ).numpy()
            cuts = (uniforms < cutprob) & entered[:, None]
            boundary = np.pad(cuts, ((0, 0), (1, 1)), constant_values=True)
            prefix = np.pad(cuts.cumsum(axis=1), ((0, 0), (1, 0)))
            present = (
                boundary[:, starts]
                & boundary[:, ends]
                & (prefix[:, ends - 1] == prefix[:, starts])
            )
            weights = present * coefficient[None, :] / count
            gradient[:-1] -= (age @ weights) @ (hx[ends - 1] - hx[starts])
            for indices, valid in ((starts - 1, starts > 0), (ends - 1, ends <= n)):
                safe = np.clip(indices, 0, n - 1)
                h = safe_age[:, None] * rates[safe]
                d = h * np.exp(-h) / (-np.expm1(-h))
                gradient[:-1] += ((weights * d).sum(0) * valid) @ x[safe]
            entryscore = np.where(entered, 1 - lam * entered_at, -lam * t)
            gradient[-1] += entryscore @ weights.sum(1)
    penalty = 0.01 * theta.copy()
    penalty[0] = 0
    penalty[-1] = 0
    return gradient + penalty


def main():
    tick = time.perf_counter()
    torch.set_num_threads(1)
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    groups = read_groups(cfg)
    assert {g["split"] for g in groups} == {"train", "val"}
    names = [
        "immediate_constant",
        "immediate_sequence",
        "entry_constant",
        "entry_sequence",
        "entry_sampled",
    ]
    fits = {
        v: json.loads((FAMILY / f"results/fit-{v}.json").read_text()) for v in names
    }
    expected = {}
    checks = []
    for variant, fit in fits.items():
        theta = np.array(fit["coefficients"])
        gradient = np.zeros_like(theta)
        data = 0.0
        for g in groups:
            x = g["x"][:, :1] if variant.endswith("constant") else g["x"]
            p, grad = (entry_forward if variant.startswith("entry_") else forward)(
                theta, g, x, True
            )
            if g["split"] == "train":
                gradient += grad
                data += np.sum(g["w"] * (p - g["y"]) ** 2)
            for source, value, w in zip(g["rows"], p, g["w"]):
                expected[variant, source[0]["curve_id"], source[1]] = (
                    source,
                    float(value),
                    float(w),
                    g,
                )
        penalty = 0.01 * theta.copy()
        penalty[0] = 0
        if variant.startswith("entry_"):
            penalty[-1] = 0
        gradient += penalty
        check = dict(
            variant=variant,
            data_mse_error=abs(data - fit["data_mse"]),
            max_gradient_error=float(np.max(np.abs(gradient - fit["final_gradient"]))),
        )
        assert check["data_mse_error"] < 1e-6 and check["max_gradient_error"] < 1e-5, (
            check
        )
        checks.append(check)
        print(check, flush=True)
    seen = set()
    collected = defaultdict(list)
    max_error = 0.0
    with gzip.open(FAMILY / "results/all-predictions.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            key = r["variant"], r["curve_id"], r["time"]
            assert key not in seen
            seen.add(key)
            source, p, w, g = expected[key]
            curve, t, ref, signal, reference = source
            error = abs(p - r["prediction"])
            max_error = max(max_error, error)
            assert error < 1e-6
            assert (
                r["weight"] == w
                and abs(r["target"] - (math.log10(signal) - math.log10(reference)))
                < 1e-12
            )
            assert (
                r["signal"] == signal
                and r["reference_signal"] == reference
                and r["reference_time"] == ref
            )
            for name in (
                "construct_id",
                "condition_id",
                "split",
                "peptide_sequence",
                "biological_replicate_raw",
            ):
                assert r[name] == curve[name]
            assert abs(math.log10(reference) + p - r["predicted_log10_signal"]) < 1e-6
            assert (
                abs(math.log10(r["predicted_signal"]) - r["predicted_log10_signal"])
                < 1e-10
            )
            collected[r["variant"], r["split"]].append((p, r["target"], w))
    assert seen == set(expected)
    scores = json.loads((FAMILY / "results/all-metrics-raw.json").read_text())
    max_metric = 0.0
    for (v, s), rows in collected.items():
        p, y, w = np.array(rows).T
        independent = dict(
            rmse=float(np.sqrt(w @ ((p - y) ** 2))),
            pearson=float(pearsonr(p, y).statistic),
            prediction_sd=float(p.std()),
        )
        for name, value in independent.items():
            max_metric = max(max_metric, abs(value - scores[v][s][name]))
            assert abs(value - scores[v][s][name]) < 1e-6
    # All Adam updates replay from saved gradients. Five gradients are independently rebuilt from the saved RNG states.
    theta = np.array(fits["entry_sampled"]["checkpoints"]["0"])
    m = np.zeros_like(theta)
    v = m.copy()
    replay = []
    max_step = 0.0
    count = 0
    with gzip.open(FAMILY / "results/sampled-trace.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            step = row["step"]
            count += 1
            assert step == count
            assert np.max(np.abs(theta - row["before"])) < 1e-10
            gradient = np.array(row["gradient"])
            if row["rng_before"] is not None:
                raw = np.frombuffer(
                    base64.b64decode(row["rng_before"]), dtype=np.uint8
                ).copy()
                rng = torch.Generator()
                rng.set_state(torch.from_numpy(raw))
                got = sampled_gradient(np.array(row["before"]), groups, 512, rng)
                err = float(np.max(np.abs(got - gradient)))
                assert err < 1e-6, (step, err)
                replay.append(dict(step=step, max_gradient_error=err))
                print(replay[-1], flush=True)
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient**2
            theta -= (
                0.03 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
            )
            error = float(np.max(np.abs(theta - row["after"])))
            max_step = max(max_step, error)
            assert error < 1e-10
    assert (
        count == 2000
        and np.max(np.abs(theta - fits["entry_sampled"]["coefficients"])) < 1e-10
    )
    result = dict(
        predictions_verified=len(seen),
        maximum_prediction_error=max_error,
        maximum_metric_error=max_metric,
        final_objective_gradient_checks=checks,
        sampled_optimizer_updates=count,
        maximum_sampled_update_error=max_step,
        independent_sampled_gradient_replay=replay,
        limitation="All sampled Adam arithmetic replayed from saved gradients; five real sampled gradients independently rebuilt, not every stochastic draw. Exact entry optimizer trajectory not independently rerun.",
        test_not_evaluated=True,
        wall_seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (FAMILY / "results/independent-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
