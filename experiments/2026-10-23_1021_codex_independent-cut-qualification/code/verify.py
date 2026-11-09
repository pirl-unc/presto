"""Replay every optimizer update independently using enumerated NumPy states.

Torch is used only for the frozen uniform random-number stream, not for model
features, losses, derivatives, optimizer or prediction evaluation.
"""

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())
SUPPORT = [(i, j) for i in range(6) for j in range(i + 1, 7)]
STATES = np.array(list(itertools.product([False, True], repeat=5)))
COUNTS = np.zeros((32, 21))
for n, state in enumerate(STATES):
    boundaries = [0] + [k + 1 for k, cut in enumerate(state) if cut] + [6]
    for span in itertools.pairwise(boundaries):
        COUNTS[n, SUPPORT.index(span)] = 1


def moments(log_rates, times):
    rates = np.broadcast_to(np.exp(log_rates), 5)
    hazard = np.array(times)[:, None] * rates
    p = -np.expm1(-hazard)
    weights = np.prod(np.where(STATES[None], p[:, None], 1 - p[:, None]), -1)
    score = np.where(
        STATES[None], hazard[:, None] * (1 - p[:, None]) / p[:, None], -hazard[:, None]
    )
    return weights @ COUNTS, np.einsum("ts,sn,tsb->tnb", weights, COUNTS, score)


def main():
    start = time.perf_counter()
    for name, digest in CONFIG["source_hashes"].items():
        assert hashlib.sha256((ROOT / "code" / name).read_bytes()).hexdigest() == digest
        assert (ROOT / "code" / name).read_bytes() == (
            ROOT / "reproduce/source" / name
        ).read_bytes()
    summary = json.loads((ROOT / "results/summary.json").read_text())
    records = summary["runs"] + [summary["control"]]
    errors, metric_error, predictions, checkpoints, updates = [], 0.0, 0, 0, 0
    rank_ties = {}
    target, _ = moments(np.log(CONFIG["rates"]), CONFIG["times"]["train"])
    for record in records:
        name = f"{record['method']}_seed{record['seed']}"
        saved = json.loads((ROOT / "results" / f"{name}.json").read_text())
        parameters = np.array(saved["checkpoints"]["0"])
        first = np.zeros_like(parameters)
        second = np.zeros_like(parameters)
        generator = torch.Generator().manual_seed(saved["seed"] + 1000)
        count = (
            int(saved["method"].removeprefix("sampled"))
            if saved["method"].startswith("sampled")
            else None
        )
        nfragments = 0
        for update in range(1, 1001):
            if count:
                hazard = (
                    np.array(CONFIG["times"]["train"])[:, None]
                    * np.exp(parameters)[None, :]
                )
                p = -np.expm1(-hazard)
                cuts = (
                    torch.rand(
                        (5, 2, count, 5), dtype=torch.float64, generator=generator
                    ).numpy()
                    < p[:, None, None, :]
                )
                # Boolean state -> independently enumerated complete fragment multiset.
                index = (cuts * np.array([16, 8, 4, 2, 1])).sum(-1)
                counts = COUNTS[index]
                nfragments += int(counts.sum())
                residual = counts - target[:, None, None, :]
                averages = residual.mean(2)
                costs = (residual * averages[:, ::-1, None, :]).mean(-1)
                advantage = costs - (costs.sum(-1, keepdims=True) - costs) / (count - 1)
                scores = np.where(
                    cuts,
                    hazard[:, None, None, :]
                    * (1 - p[:, None, None, :])
                    / p[:, None, None, :],
                    -hazard[:, None, None, :],
                )
                gradient = (advantage[:, :, :, None] * scores).mean(2).sum(1).mean(0)
            else:
                amount, derivative = moments(parameters, CONFIG["times"]["train"])
                gradient = 2 * ((amount - target)[:, :, None] * derivative).mean((0, 1))
                if saved["method"] == "constant":
                    gradient = np.array([gradient.sum()])
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            parameters -= (
                0.03
                * (first / (1 - 0.9**update))
                / (np.sqrt(second / (1 - 0.999**update)) + 1e-8)
            )
            updates += 1
            if str(update) in saved["checkpoints"]:
                error = float(
                    np.max(
                        np.abs(parameters - np.array(saved["checkpoints"][str(update)]))
                    )
                )
                errors.append(error)
                assert error < 1e-10, (name, update, error)
                checkpoints += 1
        assert nfragments == saved["work"]["sampled_fragments"]
        for split, times in CONFIG["times"].items():
            pred, _ = moments(parameters, times)
            truth, _ = moments(np.log(CONFIG["rates"]), times)
            actual = [r for r in saved["predictions"] if r["split"] == split]
            assert len(actual) == len(times) * 21
            for row in actual:
                i = times.index(row["time_hours"])
                j = SUPPORT.index((row["start"], row["end"]))
                assert abs(row["prediction"] - pred[i, j]) < 1e-11
                assert abs(row["target"] - truth[i, j]) < 1e-14
                predictions += 1
            scores = {
                "mse": float(np.mean((pred - truth) ** 2)),
                "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
                "pearson": float(pearsonr(pred.ravel(), truth.ravel()).statistic),
                "spearman": float(spearmanr(pred.ravel(), truth.ravel()).statistic),
            }
            if saved["method"] == "constant":
                raw_pred = np.array([r["prediction"] for r in actual])
                raw_target = np.array([r["target"] for r in actual])
                original_rank = float(spearmanr(raw_pred, raw_target).statistic)
                assert original_rank == saved["scores"][split]["spearman"]
                classes = {}
                canonical = []
                rate = saved["rates"][0]
                for row in actual:
                    length = row["end"] - row["start"]
                    boundaries = int(row["start"] > 0) + int(row["end"] < 6)
                    key = (row["time_hours"], length, boundaries)
                    if key not in classes:
                        t = row["time_hours"]
                        classes[key] = (-np.expm1(-rate * t)) ** boundaries * np.exp(
                            -rate * t * (length - 1)
                        )
                    canonical.append(classes[key])
                assert np.max(np.abs(np.array(canonical) - raw_pred)) < 1e-11
                rank_ties[split] = {
                    "raw_spearman": original_rank,
                    "enumerated_spearman": scores["spearman"],
                    "canonical_spearman": float(
                        spearmanr(canonical, raw_target).statistic
                    ),
                    "n_analytic_classes": len(classes),
                }
                scores["spearman"] = original_rank
            for key, val in scores.items():
                delta = abs(val - saved["scores"][split][key])
                metric_error = max(metric_error, delta)
                assert delta < 1e-11, (name, split, key, delta)
        print("Independent replay passed", name, flush=True)
    result = {
        "passed": True,
        "updates": updates,
        "noninitial_checkpoints": checkpoints,
        "prediction_values": predictions,
        "max_log_rate_checkpoint_error": max(errors),
        "max_metric_error": metric_error,
        "wall_seconds": time.perf_counter() - start,
        "implementation": "NumPy cut-state enumeration, analytic score derivatives and independently implemented Adam; Torch only for exact uniform RNG stream",
        "constant_control_analytic_ties": rank_ties,
    }
    (ROOT / "results/independent-replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
