"""Independent hybrid Adam replay; Torch supplies only the frozen uniform stream."""

import gzip
import json
import math
import resource
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from verify import FAMILY, forward, read_groups
from diagnostics import canonical


def templates(groups):
    result = []
    for g in groups:
        if g["split"] != "train":
            continue
        spans, inverse = np.unique(
            np.column_stack((g["a"], g["b"])), axis=0, return_inverse=True
        )
        result.append((g, spans, inverse, np.unique(np.r_[g["t"], g["ref"]])))
    return result


def gradient(theta, records, count, rng, penalty):
    grad = np.zeros_like(theta)
    for g, spans, inverse, times in records:
        x = g["x"]
        rates = np.exp(x @ theta)
        n = len(rates)
        a, b = spans.T
        coefficient = 2 * g["w"] * (forward(theta, g, x) - g["y"]) / math.log(10)
        rprefix = np.r_[0.0, rates.cumsum()]
        inside = rprefix[b - 1] - rprefix[a]
        for t in times:
            combined = np.zeros(len(spans))
            for observed, sign in ((g["t"], 1), (g["ref"], -1)):
                mask = observed == t
                combined += np.bincount(
                    inverse[mask],
                    weights=sign * coefficient[mask],
                    minlength=len(spans),
                )
            h = rates * t
            cutprob = -np.expm1(-h)
            boundary_score = h * np.exp(-h) / cutprob
            logm = -t * inside
            hx = np.vstack(
                (np.zeros((1, len(theta))), np.cumsum(h[:, None] * x, axis=0))
            )
            conditional = -(hx[b - 1] - hx[a])
            for idx, mask in ((a - 1, a > 0), (b - 1, b <= n)):
                safe = np.clip(idx, 0, n - 1)
                logm += np.where(mask, np.log(cutprob[safe]), 0.0)
                conditional += (
                    np.where(mask, boundary_score[safe], 0.0)[:, None] * x[safe]
                )
            # Same RNG bytes, independent probabilities, cut events and gradients.
            uniform = torch.rand((count, n), dtype=torch.float64, generator=rng).numpy()
            cuts = uniform < cutprob
            boundary = np.pad(cuts, ((0, 0), (1, 1)), constant_values=True)
            prefix = np.pad(np.cumsum(cuts, axis=1), ((0, 0), (1, 0)))
            present = (
                boundary[:, a] & boundary[:, b] & (prefix[:, b - 1] == prefix[:, a])
            )
            grad += (present.mean(0) * combined / np.exp(logm)) @ conditional
    reg = penalty * theta.copy()
    reg[0] = 0
    return grad + reg


def main():
    start = time.monotonic()
    torch.set_num_threads(1)
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    stage = json.loads((FAMILY / "reproduce/sampled-fit.json").read_text())
    fit = json.loads((FAMILY / "results/sampled-fit.json").read_text())
    groups = read_groups(cfg)
    records = templates(groups)
    theta = np.array(fit["checkpoints"]["0"])
    m = theta * 0
    v = m.copy()
    rng = torch.Generator().manual_seed(stage["seed"])
    checkpoints = []
    for step in range(1, stage["updates"] + 1):
        g = gradient(theta, records, stage["count"], rng, cfg["l2_nonintercept"])
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        theta -= (
            cfg["learning_rate"]
            * (m / (1 - 0.9**step))
            / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        )
        if str(step) in fit["checkpoints"]:
            error = float(
                np.max(np.abs(theta - np.array(fit["checkpoints"][str(step)])))
            )
            assert error < 1e-7, (step, error)
            checkpoints.append(dict(step=step, max_coefficient_error=error))
            print(checkpoints[-1], flush=True)
    assert np.max(np.abs(theta - fit["coefficients"])) < 1e-7
    expected = {}
    for group in groups:
        p = forward(np.array(fit["coefficients"]), group, group["x"])
        for source, value, w in zip(group["rows"], p, group["w"]):
            expected[source[0]["curve_id"], source[1]] = (source, value, w, group)
    seen = set()
    per_split = defaultdict(list)
    canonical_rows = []
    max_error = 0.0
    with gzip.open(FAMILY / "results/sampled-predictions.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            key = r["curve_id"], r["time"]
            assert key not in seen
            seen.add(key)
            source, p, w, group = expected[key]
            curve, t, ref, signal, reference = source
            y = math.log10(signal) - math.log10(reference)
            max_error = max(max_error, abs(p - r["prediction"]))
            assert abs(y - r["target"]) < 1e-12 and w == r["weight"]
            assert r["signal"] == signal and r["reference_signal"] == reference
            assert abs(math.log10(reference) + p - r["predicted_log10_signal"]) < 1e-10
            assert (
                abs(math.log10(r["predicted_signal"]) - r["predicted_log10_signal"])
                < 1e-10
            )
            for name in (
                "construct_id",
                "condition_id",
                "split",
                "peptide_sequence",
                "biological_replicate_raw",
            ):
                assert r[name] == curve[name]
            cp = canonical(
                fit["coefficients"],
                group,
                "sequence_context",
                curve["start_0based"],
                curve["end_0based"],
                t,
                ref,
            )
            assert abs(cp - p) < 1e-10
            per_split[r["split"]].append((p, y, w, cp, curve["construct_id"]))
            canonical_rows.append(
                dict(
                    curve_id=r["curve_id"],
                    time=t,
                    prediction=cp,
                    raw_difference=cp - r["prediction"],
                )
            )
    assert seen == set(expected) and max_error < 1e-10
    reported = json.loads(json.dumps(fit["metrics"]))
    for split, rows in per_split.items():
        p, y, w, cp = map(np.array, zip(*[r[:4] for r in rows]))
        score = fit["metrics"][split]
        calculated = {
            "mse": float(w @ (p - y) ** 2),
            "rmse": math.sqrt(float(w @ (p - y) ** 2)),
            "pooled_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "prediction_sd": float(p.std()),
            "target_sd": float(y.std()),
            "pearson": float(pearsonr(p, y).statistic),
        }
        for key, value in calculated.items():
            assert abs(score[key] - value) < 1e-10, (split, key)
        reported[split]["spearman"] = float(spearmanr(cp, y).statistic)
        for parent, result in score["by_parent"].items():
            mask = np.array([r[4] == parent for r in rows])
            error = math.sqrt(
                float(np.sum(w[mask] * (p[mask] - y[mask]) ** 2) / w[mask].sum())
            )
            assert abs(error - result["rmse"]) < 1e-10
    with gzip.open(
        FAMILY / "results/sampled-canonical-rank-predictions.jsonl.gz", "wt"
    ) as stream:
        for row in canonical_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (FAMILY / "results/sampled-reported-metrics.json").write_text(
        json.dumps(reported, indent=2, sort_keys=True) + "\n"
    )
    result = dict(
        passed=True,
        updates_replayed=stage["updates"],
        checkpoints=checkpoints,
        predictions_verified=len(seen),
        max_prediction_error=max_error,
        torch_used_only_for_uniform_rng=True,
        independent_probability_event_gradient_adam=True,
        wall_seconds=time.monotonic() - start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (FAMILY / "results/sampled-independent-verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
