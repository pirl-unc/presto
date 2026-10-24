"""Replay original source, direct-sum predictions and every rate gradient independently."""

import gzip
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

FAMILY = Path(__file__).resolve().parents[1]
LN10 = math.log(10)


def boundary(rate, t, ref):
    if rate == 0:
        return math.log(t / ref), -(t - ref) / 2
    value = math.log(-math.expm1(-rate * t)) - math.log(-math.expm1(-rate * ref))
    if rate * t < 0.01:
        derivative = (
            -(t - ref) / 2
            + rate * (t * t - ref * ref) / 12
            - rate**3 * (t**4 - ref**4) / 720
            + rate**5 * (t**6 - ref**6) / 30240
        )
    else:

        def single(time):
            z = rate * time
            return time * math.exp(-z) / (-math.expm1(-z))

        derivative = single(t) - single(ref)
    return value, derivative


def main():
    now = time.perf_counter()
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    source = Path(cfg["recipe_directory"])
    assert (
        hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
        == cfg["recipe_manifest_sha256"]
    )
    manifest = json.loads((source / "manifest.json").read_text())
    assert (
        hashlib.sha256((source / "examples.json").read_bytes()).hexdigest()
        == manifest["files"]["examples.json"]["sha256"]
    )
    grouped = defaultdict(list)
    for row in json.loads((source / "examples.json").read_text()):
        if row["split"] == "train":
            grouped[row["condition_id"]].append(row)
    parent_conditions = Counter(rows[0]["construct_id"] for rows in grouped.values())
    saved = json.loads((FAMILY / "results/direct-fits.json").read_text())
    allocation = {
        r["condition"]: r
        for r in json.loads(
            (Path(cfg["previous_family"]) / "results/rate-allocation.json").read_text()
        )
    }
    with gzip.open(FAMILY / "results/direct-predictions.jsonl.gz", "rt") as stream:
        dumped = [json.loads(line) for line in stream]
    offset = 0
    all_predictions, all_targets = [], []
    results = []
    total = 0.0
    for number, fit in enumerate(saved["fits"]):
        condition = fit["condition"]
        curves = grouped[condition]
        sequence = curves[0]["parent_sequence"]
        rates = np.array(fit["rates"])
        member = allocation[condition]["members"]
        index = np.full(len(sequence) - 1, -1, dtype=int)
        for i, bonds in enumerate(member):
            index[bonds] = i
        gradient = np.zeros_like(rates)
        mse = 0.0
        max_error = mass_error = weight_error = target_error = 0.0
        small = zero = 0
        for curve in curves:
            values = {r["time_hours"]: r["target_value"] for r in curve["observations"]}
            times = curve["common_positive_times"]
            ref = times[0]
            a, b = curve["start_0based"], curve["end_0based"]
            inside = index[a : b - 1]
            edges = ([index[a - 1]] if a else []) + (
                [index[b - 1]] if b < len(sequence) else []
            )
            assert (inside >= 0).all() and all(i >= 0 for i in edges)
            weight = 1 / (
                len(parent_conditions)
                * parent_conditions[fit["parent"]]
                * len(curves)
                * (len(times) - 1)
            )
            for t in times[1:]:
                prediction = -sum(float(rates[i]) for i in inside) * (t - ref)
                derivative = np.zeros_like(rates)
                np.add.at(derivative, inside, -(t - ref))
                for i in edges:
                    v, d = boundary(float(rates[i]), t, ref)
                    prediction += v
                    derivative[i] += d
                prediction /= LN10
                target = math.log10(values[t]) - math.log10(values[ref])
                mse += weight * (prediction - target) ** 2
                gradient += 2 * weight * (prediction - target) * derivative / LN10
                log_mass = -sum(float(rates[i]) for i in inside) * ref
                for i in edges:
                    log_mass += (
                        math.log(-math.expm1(-float(rates[i]) * ref))
                        if rates[i]
                        else -math.inf
                    )
                log_mass /= LN10
                actual = dumped[offset]
                offset += 1
                assert (
                    actual["condition_id"],
                    actual["curve_id"],
                    actual["time"],
                    actual["reference_time"],
                    actual["split"],
                ) == (condition, curve["curve_id"], t, ref, "train")
                assert (
                    actual["signal"] == values[t]
                    and actual["reference_signal"] == values[ref]
                )
                max_error = max(max_error, abs(prediction - actual["prediction"]))
                weight_error = max(weight_error, abs(weight - actual["weight"]))
                target_error = max(target_error, abs(target - actual["target"]))
                if math.isfinite(log_mass):
                    mass_error = max(
                        mass_error, abs(log_mass - actual["reference_log10_amount"])
                    )
                else:
                    assert (
                        actual["zero_reference_amount"]
                        and actual["reference_log10_amount"] is None
                    )
                    zero += 1
                small += log_mass < -20
                all_predictions.append(prediction)
                all_targets.append(target)
        grad_error = float(
            np.max(np.abs(gradient - np.array(fit["direct_mse_gradient"])))
        )
        projected = float(np.max(np.abs(rates - np.maximum(0, rates - gradient))))
        assert max_error < 1e-10 and mass_error < 1e-10 and grad_error < 1e-9
        assert (
            abs(mse - fit["final_mse"]) < 1e-11
            and target_error == 0
            and weight_error == 0
        )
        assert (
            small == fit["reference_amount_below_1e_20_rows"]
            and zero == fit["zero_reference_amount_rows"]
        )
        with gzip.open(
            FAMILY / f"results/condition-{number}-evaluations.jsonl.gz", "rt"
        ) as stream:
            evaluations = [json.loads(line) for line in stream]
        matches = [
            r for r in evaluations if np.array_equal(np.array(r["rates"]), rates)
        ]
        assert matches and len(evaluations) == fit["nfev"] == fit["actual_computations"]
        results.append(
            dict(
                condition=condition,
                observations=sum(len(c["common_positive_times"]) - 1 for c in curves),
                prediction_error=max_error,
                reference_log_mass_error=mass_error,
                gradient_error=grad_error,
                mse=mse,
                projected_gradient_mapping_max=projected,
                source_targets_weights_exact=True,
                saved_final_found_in_evaluation_trace=True,
                trace_evaluations=len(evaluations),
            )
        )
        total += mse
    assert offset == len(dumped) == 67576
    assert abs(total - saved["global_mse"]) < 1e-10
    metrics = dict(
        mse=total,
        rmse=math.sqrt(total),
        pearson=float(pearsonr(all_predictions, all_targets).statistic),
        spearman=float(spearmanr(all_predictions, all_targets).statistic),
        prediction_sd=float(np.std(all_predictions)),
    )
    assert abs(metrics["pearson"] - saved["pooled_pearson"]) < 1e-10
    rank_discrepancy = {
        "producer_raw_spearman": saved["pooled_spearman"],
        "independent_direct_sum_spearman": metrics["spearman"],
        "absolute_difference": abs(metrics["spearman"]-saved["pooled_spearman"]),
        "interpretation": "Raw ranks are not asserted equal. All per-point values agree within the independently checked 1e-10 gate; finite-precision near-tie ordering can differ.",
    }
    with gzip.open(FAMILY / "results/independent-direct-sum-predictions.jsonl.gz", "wt") as stream:
        for original, prediction in zip(dumped, all_predictions):
            row = dict(original, prediction=prediction)
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    result = dict(
        conditions=results,
        rank_discrepancy=rank_discrepancy,
        metrics=metrics,
        all_predictions_verified=offset,
        all_parameter_gradients_verified=sum(len(f["rates"]) for f in saved["fits"]),
        verification_scope="Original source membership, targets, weights, direct bond sums, all final gradients, KKT mapping, metrics, reference masses and trace consistency. No independent optimizer trajectory replay or global-optimum proof.",
        seconds=time.perf_counter() - now,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (FAMILY / "results/independent-verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "metrics",
                    "all_predictions_verified",
                    "all_parameter_gradients_verified",
                    "seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
