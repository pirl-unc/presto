"""Independent biological baseline closure; exact input ties share predictions."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

FAMILY = Path(__file__).resolve().parents[1]


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as f:
        return json.load(f)


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature(group, name):
    if name.startswith("condition:"):
        return float(group["condition_id"] == name.split(":", 1)[1])
    if name.startswith("compound:"):
        compound, rest = name.removeprefix("compound:").split("*", 1)
        return feature(group, rest) if group["condition"][2] == compound else 0.0
    seq = group["peptide_sequence"]
    if name == "log_length":
        return math.log(len(seq))
    if name == "n_terminal_acetylation":
        return float(any(m["name"] == "N_terminal_acetylation" for m in group["modifications"]))
    if name == "oxidation_count":
        return float(sum(m["name"] == "oxidation" for m in group["modifications"]))
    role, residue = name.split(":")
    if role == "fraction":
        return sum(a == residue for a in seq) / len(seq)
    assert role in {"first", "last"}
    return float(seq[0 if role == "first" else -1] == residue)


def error(row):
    residual = row["prediction_log10_ratio"] - row["target_log10_ratio"]
    if row["inequality"] == "lt" and residual < 0:
        return 0.0
    if row["inequality"] == "gt" and residual > 0:
        return 0.0
    return residual


def mean(values):
    return math.fsum(values) / len(values)


def sd(values):
    if min(values) == max(values):
        return 0.0
    center = mean(values)
    return math.sqrt(mean([(v - center) ** 2 for v in values]))


def correlation(x, y):
    if len(x) < 2 or min(x) == max(x) or min(y) == max(y):
        return None
    a, b = mean(x), mean(y)
    numerator = math.fsum((u - a) * (v - b) for u, v in zip(x, y))
    denominator = math.sqrt(math.fsum((u - a) ** 2 for u in x) * math.fsum((v - b) ** 2 for v in y))
    return numerator / denominator


def weighted(rows, function):
    return math.fsum(r["weight"] * function(r) for r in rows) / math.fsum(r["weight"] for r in rows)


def metrics(rows):
    exact = [r for r in rows if r["inequality"] is None]
    bounds = [r for r in rows if r["inequality"] is not None]
    predictions = [r["prediction_log10_ratio"] for r in rows]
    out = {
        "n_observations": len(rows),
        "n_response_groups": len({r["response_group_id"] for r in rows}),
        "n_components": len({r["component_id"] for r in rows}),
        "weight_sum": math.fsum(r["weight"] for r in rows),
        "data_loss": weighted(rows, lambda r: error(r) ** 2),
        "n_exact_observations": len(exact),
        "n_bounded_observations": len(bounds),
        "prediction_sd_pooled": sd(predictions),
        "prediction_min": min(predictions),
        "prediction_max": max(predictions),
    }
    if exact:
        p, y = (
            [r["prediction_log10_ratio"] for r in exact],
            [r["target_log10_ratio"] for r in exact],
        )
        mse = weighted(exact, lambda r: error(r) ** 2)
        center = weighted(exact, lambda r: r["target_log10_ratio"])
        variance = (
            0.0
            if min(y) == max(y)
            else weighted(exact, lambda r: (r["target_log10_ratio"] - center) ** 2)
        )
        directed = [r for r in exact if r["target_log10_ratio"] != 0]

        def sign(x):
            return int(x > 0) - int(x < 0)

        out.update(
            exact_rmse=math.sqrt(mse),
            exact_mae=weighted(exact, lambda r: abs(error(r))),
            exact_r2=None if variance == 0 else 1 - mse / variance,
            exact_pearson_pooled=correlation(p, y),
            exact_spearman_pooled=correlation(list(rankdata(p)), list(rankdata(y))),
            exact_prediction_sd_pooled=sd(p),
            exact_target_sd_pooled=sd(y),
            exact_direction_accuracy=None
            if not directed
            else weighted(
                directed,
                lambda r: sign(r["prediction_log10_ratio"]) == sign(r["target_log10_ratio"]),
            ),
        )

    def good(r):
        return (
            r["prediction_log10_ratio"] < r["target_log10_ratio"]
            if r["inequality"] == "lt"
            else r["prediction_log10_ratio"] > r["target_log10_ratio"]
        )

    if bounds:
        out.update(
            bound_satisfaction_weighted=weighted(bounds, good),
            bound_satisfaction_pooled=mean([good(r) for r in bounds]),
            bound_violation_rmse=math.sqrt(weighted(bounds, lambda r: error(r) ** 2)),
        )
        for qualifier, name in (("lt", "upper"), ("gt", "lower")):
            selected = [r for r in bounds if r["inequality"] == qualifier]
            if selected:
                out[name + "_bound_satisfaction_weighted"] = weighted(selected, good)
    return out


def main():
    start = time.perf_counter()
    config = read(FAMILY / "reproduce/launch.json")
    source = read(Path(config["recipe_directory"]) / "examples.json.gz")
    source = [g for g in source if g["split"] in {"train", "val"}]
    by_observation = {o["observation_id"]: (g, o) for g in source for o in g["observations"]}
    original_summary = read(FAMILY / "results/summary.json")
    original_strata = read(FAMILY / "results/stratified-metrics.json")
    output = FAMILY / "results/qualified-v2"
    assert not output.exists()
    output.mkdir()
    all_metrics, all_strata, corrections, checks = [], [], [], []
    n_predictions = 0
    for fit in original_summary["fits"]:
        variant = fit["variant"]
        with gzip.open(FAMILY / "results" / (variant + "-predictions.jsonl.gz"), "rt") as f:
            rows = [json.loads(line) for line in f]
        assert {r["observation_id"] for r in rows} == by_observation.keys()
        assert len(rows) == len(by_observation)
        for r in rows:
            g, o = by_observation[r["observation_id"]]
            assert all(r[k] == v for k, v in o.items())
            for field in (
                "response_group_id",
                "condition_id",
                "condition",
                "component_id",
                "split",
                "peptide_sequence",
                "modifications",
                "full_parent_model_supported",
            ):
                assert r[field] == g[field]
            assert r["censor_code"] == {None: 0, "lt": -1, "gt": 1}[o["inequality"]]
        if "constant" in fit:
            independent = {g["response_group_id"]: fit["constant"] for g in source}
            train_rows = [r for r in rows if r["split"] == "train"]
            if variant == "exact_train_mean":
                assert (
                    abs(
                        fit["constant"]
                        - weighted(
                            [r for r in train_rows if r["inequality"] is None],
                            lambda r: r["target_log10_ratio"],
                        )
                    )
                    < 1e-14
                )
            if variant == "censored_constant":
                assert abs(weighted(train_rows, error)) < 1e-12
        else:
            model = np.load(FAMILY / "results" / (variant + "-model.npz"))
            features = fit["features"]
            raw = np.array(
                [[feature(g, name) for name in features["all_feature_names"]] for g in source]
            )
            mask = np.array([g["split"] == "train" for g in source])
            weights = np.array([g["group_weight"] for g in source])[mask]
            weights /= weights.sum()
            means = np.array(
                [math.fsum(w * x for w, x in zip(weights, column)) for column in raw[mask].T]
            )
            variance = np.array(
                [
                    math.fsum(w * (x - m) ** 2 for w, x in zip(weights, column))
                    for column, m in zip(raw[mask].T, means)
                ]
            )
            assert np.allclose(means, model["mean"], atol=1e-12, rtol=1e-12)
            keep = variance > 1e-24
            assert np.array_equal(keep, model["keep"])
            assert np.allclose(np.sqrt(variance[keep]), model["scale"], atol=1e-12, rtol=1e-12)
            projection, theta = model["projection"], model["theta"]
            assert np.allclose(projection.T @ projection, np.eye(projection.shape[1]), atol=1e-12)
            assert (
                len(theta) == fit["n_parameters"] == features["n_supported_feature_directions"] + 1
            )
            assert np.sum(abs(theta) > 1e-12) == fit["n_parameters_changed"]
            standardized = (raw[:, keep] - model["mean"][keep]) / model["scale"]
            transformed = np.column_stack([np.ones(len(source)), standardized @ projection])
            by_input, independent = {}, {}
            for i, g in enumerate(source):
                identity = tuple(raw[i])
                if identity not in by_input:
                    # Evaluate once per exact raw input; no rounding or target-dependent grouping.
                    coordinates = standardized[i] @ projection
                    by_input[identity] = float(
                        theta[0] + math.fsum(a * b for a, b in zip(coordinates, theta[1:]))
                    )
                independent[g["response_group_id"]] = by_input[identity]
            group_index = {g["response_group_id"]: i for i, g in enumerate(source)}
            train_rows = [r for r in rows if r["split"] == "train"]
            xt = transformed[[group_index[r["response_group_id"]] for r in train_rows]]
            grad = 2 * xt.T @ np.array([r["weight"] * error(r) for r in train_rows])
            grad[1:] += 2 * config["penalty"] * theta[1:]
            assert abs(float(abs(grad).max()) - fit["final_gradient_inf"]) < 1e-11
            penalized = weighted(train_rows, lambda r: error(r) ** 2) + config[
                "penalty"
            ] * math.fsum(v * v for v in theta[1:])
            assert abs(penalized - fit["training_objective"]) < 1e-12
        maximum_error = max(
            abs(r["prediction_log10_ratio"] - independent[r["response_group_id"]]) for r in rows
        )
        assert maximum_error < 1e-12, (variant, maximum_error)
        with gzip.open(
            FAMILY / "results/qualified" / (variant + "-predictions.jsonl.gz"), "rt"
        ) as f:
            scalar_reference = [json.loads(line) for line in f]
        assert len(scalar_reference) == len(rows)
        scalar_difference = max(
            abs(r["prediction_log10_ratio"] - independent[r["response_group_id"]])
            for r in scalar_reference
        )
        assert scalar_difference < 1e-12
        for r in rows:
            r["prediction_log10_ratio"] = independent[r["response_group_id"]]
        n_predictions += len(rows)
        with gzip.open(output / (variant + "-predictions.jsonl.gz"), "wt") as f:
            for r in rows:
                f.write(json.dumps(r, allow_nan=False) + "\n")
        comparisons = []
        for split in ("train", "val"):
            selected = [r for r in rows if r["split"] == split]
            result = {"variant": variant, "split": split, **metrics(selected)}
            old = next(
                m
                for m in original_summary["metrics"]
                if m["variant"] == variant and m["split"] == split
            )
            all_metrics.append(result)
            comparisons.append((old, result))
            for field in ("condition_id", "component_id"):
                strata = defaultdict(list)
                for r in selected:
                    strata[r[field]].append(r)
                for value, records in sorted(strata.items()):
                    result = {
                        "variant": variant,
                        "split": split,
                        "stratum": field,
                        "stratum_id": value,
                        **metrics(records),
                    }
                    all_strata.append(result)
                    old = next(
                        m
                        for m in original_strata
                        if m["variant"] == variant
                        and m["split"] == split
                        and m["stratum"] == field
                        and m["stratum_id"] == value
                    )
                    comparisons.append((old, result))
        for old, new in comparisons:
            assert old.keys() == new.keys()
            for name, value in old.items():
                if name in {"exact_pearson_pooled", "exact_spearman_pooled"} or (
                    name == "exact_r2" and (value is None or new[name] is None)
                ):
                    if value != new[name] and (
                        value is None or new[name] is None or abs(value - new[name]) > 1e-12
                    ):
                        corrections.append(
                            {
                                k: new[k]
                                for k in ("variant", "split", "stratum", "stratum_id")
                                if k in new
                            }
                            | {"metric": name, "raw": value, "qualified": new[name]}
                        )
                elif isinstance(value, (float, int)):
                    assert math.isclose(value, new[name], abs_tol=1e-12, rel_tol=2e-12), (
                        variant,
                        name,
                        value,
                        new[name],
                    )
                else:
                    assert value == new[name]
        checks.append(
            {
                "variant": variant,
                "n_predictions": len(rows),
                "maximum_prediction_error": maximum_error,
                "maximum_difference_from_scalar_reference": scalar_difference,
                "n_parameters": fit["n_parameters"],
                "converged": fit["converged"],
            }
        )
    qualified = dict(original_summary)
    qualified["metrics"] = all_metrics
    qualified["canonical_prediction_rule"] = (
        "one frozen-model evaluation per exact raw feature vector"
    )
    validation = {r["variant"]: r for r in all_metrics if r["split"] == "val"}
    for name in ("sequence_shared", "sequence_compound"):
        assert (
            validation[name]["data_loss"]
            < min(validation["condition"]["data_loss"], validation["no_change"]["data_loss"])
        ) == qualified["comparisons"][name]["beats_condition_and_no_change_primary"]
    write(output / "summary.json", qualified)
    write(output / "stratified-metrics.json", all_strata)
    write(output / "correlation-corrections.json", corrections)
    receipt = {
        "status": "passed",
        "verifier_sha256": sha(__file__),
        "n_predictions": n_predictions,
        "n_global_metrics": len(all_metrics),
        "n_stratified_metrics": len(all_strata),
        "n_corrected_metric_records": len(corrections),
        "n_corrected_correlations": sum(r["metric"] != "exact_r2" for r in corrections),
        "checks": checks,
        "seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "no_refit": True,
        "test_evaluated": False,
    }
    assert receipt["seconds"] < 180 and receipt["peak_rss_bytes"] < 4 * 1024**3
    write(FAMILY / "results/independent-verification.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
