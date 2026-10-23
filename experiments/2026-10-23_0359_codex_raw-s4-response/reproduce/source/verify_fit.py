"""Replay all folds with augmented least squares and independent metrics."""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import resource
import time
import traceback

import numpy as np

R = Path(__file__).resolve().parents[1]
C = json.loads((R / "reproduce/launch.json").read_text())


def ranks(values):
    groups = defaultdict(list)
    for index, value in enumerate(values):
        groups[value].append(index)
    result = np.zeros(len(values))
    position = 1
    for value in sorted(groups):
        ids = groups[value]
        result[ids] = (2 * position + len(ids) - 1) / 2
        position += len(ids)
    return result


def weight(rows):
    counts = Counter(r["sequence"] for r in rows)
    values = np.array([1.0 / counts[r["sequence"]] for r in rows])
    return values / sum(values)


def measures(rows, predictions):
    p = np.array(predictions)
    y = np.array([r["target_log10_response"] for r in rows])
    errors = p - y
    mse = np.average(errors**2, weights=weight(rows))
    variable = len(set(p)) > 1 and len(set(y)) > 1
    return {
        "n_examples": len(rows),
        "n_sequences": len({r["sequence"] for r in rows}),
        "equal_peptide_mse": float(mse),
        "equal_peptide_rmse": float(math.sqrt(mse)),
        "equal_peptide_mae": float(np.average(abs(errors), weights=weight(rows))),
        "pooled_pearson": float(np.corrcoef(y, p)[0, 1]) if variable else None,
        "pooled_spearman": float(np.corrcoef(ranks(y), ranks(p))[0, 1])
        if variable
        else None,
        "prediction_std": float(np.std(p)),
        "target_std": float(np.std(y)),
        "prediction_min": float(min(p)),
        "prediction_max": float(max(p)),
    }


def compare(a, b):
    assert set(a) == set(b)
    for key in a:
        if a[key] is None:
            assert b[key] is None, (key, a[key], b[key])
        else:
            assert math.isclose(a[key], b[key], rel_tol=1e-8, abs_tol=1e-9), (
                key,
                a[key],
                b[key],
            )


def matrix(rows, names):
    result = []
    for row in rows:
        values = []
        for name in names:
            if name == "log10_nominal_load":
                value = math.log10(row["nominal_load_fmol"])
            elif name == "length":
                value = len(row["sequence"])
            elif name.startswith("site:"):
                value = int(row["site_label"] == name[5:])
            elif name.startswith("fraction:"):
                value = sum(letter == name[-1] for letter in row["sequence"]) / len(
                    row["sequence"]
                )
            else:
                raise AssertionError("Undeclared feature: " + name)
            values.append(value)
        result.append(values)
    return np.array(result, dtype=float)


def main(report):
    assert json.loads((R / "results/fit-execution.json").read_text())["passed"]
    admission = json.loads((R / "results/independent-admission.json").read_text())
    package = Path(C["recipe_output"])
    assert (
        hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest()
        == admission["recipe_manifest_sha256"]
    )
    examples = json.loads((package / "examples.json").read_text())
    by_id = {r["example_id"]: r for r in examples}
    sites = sorted({r["site_label"] for r in examples}, key=int)
    with gzip.open(R / "results/checkpoints.json.gz", "rt") as stream:
        runs = json.load(stream)
    with gzip.open(R / "results/validation-predictions.jsonl.gz", "rt") as stream:
        validation = [json.loads(line) for line in stream]
    with (R / "results/outer-test-predictions.csv").open() as stream:
        testing = list(csv.DictReader(stream))
    predicted = defaultdict(dict)
    for row in validation + testing:
        assert row["example_id"] not in predicted[row["run_id"]]
        predicted[row["run_id"]][row["example_id"]] = float(row["prediction"])
        example = by_id[row["example_id"]]
        assert (
            row["sequence"] == example["sequence"]
            and row["site_label"] == example["site_label"]
        )
        assert float(row["target"]) == example["target_log10_response"]
    assert len(runs) == len(predicted) == 295
    assert len({run["run_id"] for run in runs}) == 295
    replay, max_coef_error, max_pred_error = {}, 0.0, 0.0
    candidates = defaultdict(list)
    for run in runs:
        model = run["model"]
        outer, inner = run["outer_fold"], run["inner_fold"]
        assert run["condition"] == model["condition"] and model["sites"] == sites
        excluded = {outer, inner} if inner is not None else {outer}
        expected_train = {
            r["example_id"] for r in examples if r["fold_id"] not in excluded
        }
        expected_eval = {
            r["example_id"]
            for r in examples
            if r["fold_id"] == (inner if inner is not None else outer)
        }
        assert set(run["training_ids"]) == expected_train and len(
            run["training_ids"]
        ) == len(expected_train)
        assert (
            set(run["evaluation_ids"])
            == expected_eval
            == predicted[run["run_id"]].keys()
        )
        training = [by_id[x] for x in run["training_ids"]]
        evaluation = [by_id[x] for x in run["evaluation_ids"]]
        assert not {r["sequence"] for r in training} & {
            r["sequence"] for r in evaluation
        }
        w = weight(training)
        y = np.array([r["target_log10_response"] for r in training])
        if model["condition"] == "mean":
            assert (
                inner is None and model["alpha"] is None and not model["feature_names"]
            )
            coefficients = np.array([np.average(y, weights=w)])
            a, b = np.ones((len(training), 1)), np.ones((len(evaluation), 1))
        else:
            names = ["log10_nominal_load"] + ["site:" + site for site in sites]
            if model["condition"] == "sequence":
                names += ["length"] + [
                    "fraction:" + aa for aa in "ACDEFGHIKLMNPQRSTVWY"
                ]
            assert model["feature_names"] == names
            x, xx = matrix(training, names), matrix(evaluation, names)
            center = np.average(x, axis=0, weights=w)
            deviation = np.sqrt(np.average((x - center) ** 2, axis=0, weights=w))
            keep = [i for i, d in enumerate(deviation) if d > 1e-12]
            assert keep == model["kept_indices"]
            assert model["dropped_features"] == [
                n for i, n in enumerate(names) if i not in keep
            ]
            assert np.allclose(center[keep], model["mean"], rtol=1e-12, atol=1e-12)
            assert np.allclose(deviation[keep], model["scale"], rtol=1e-12, atol=1e-12)
            a = np.column_stack(
                [np.ones(len(training)), (x[:, keep] - center[keep]) / deviation[keep]]
            )
            b = np.column_stack(
                [
                    np.ones(len(evaluation)),
                    (xx[:, keep] - center[keep]) / deviation[keep],
                ]
            )
            penalty = np.sqrt(model["alpha"]) * np.eye(a.shape[1])[1:]
            augmented = np.concatenate([a * np.sqrt(w[:, None]), penalty])
            target = np.concatenate([y * np.sqrt(w), np.zeros(len(keep))])
            coefficients = np.linalg.lstsq(augmented, target, rcond=None)[0]
        actual = np.array([model["intercept"]] + model["coefficients"])
        max_coef_error = max(max_coef_error, float(np.max(abs(coefficients - actual))))
        assert np.allclose(coefficients, actual, rtol=1e-7, atol=1e-8)
        estimate, train_estimate = b @ coefficients, a @ coefficients
        output = np.array(
            [predicted[run["run_id"]][row["example_id"]] for row in evaluation]
        )
        max_pred_error = max(max_pred_error, float(np.max(abs(estimate - output))))
        assert np.allclose(estimate, output, rtol=1e-9, atol=1e-8)
        compare(measures(training, train_estimate), run["train_metrics"])
        compare(measures(evaluation, output), run["evaluation_metrics"])
        replay[run["run_id"]] = (evaluation, output)
        if inner is not None:
            assert run["role"] == "inner_validation" and model["alpha"] in C["alphas"]
            candidates[outer, run["condition"], model["alpha"]].append(
                (evaluation, output)
            )
    selections = json.loads((R / "results/selections.json").read_text())
    assert len(selections) == 10
    for choice in selections:
        losses = {}
        for candidate in choice["candidates"]:
            key = choice["outer_fold"], choice["condition"], candidate["alpha"]
            groups = candidates[key]
            assert len(groups) == 4
            rows = [r for group, _ in groups for r in group]
            estimates = [float(p) for _, group in groups for p in group]
            result = measures(rows, estimates)
            compare(result, candidate["validation_metrics"])
            losses[candidate["alpha"]] = result["equal_peptide_mse"]
        best = min(C["alphas"], key=lambda alpha: losses[alpha])
        assert choice["selected_alpha"] == best
        selected_run = [
            run
            for run in runs
            if run["role"] == "outer_test"
            and run["condition"] == choice["condition"]
            and run["outer_fold"] == choice["outer_fold"]
        ]
        assert len(selected_run) == 1 and selected_run[0]["model"]["alpha"] == best
    pooled = json.loads((R / "results/pooled-metrics.json").read_text())
    per_peptide, per_site = [
        json.loads((R / f"results/{name}-metrics.json").read_text())
        for name in ("per-peptide", "per-site")
    ]
    errors = {}
    for condition in C["conditions"]:
        records = [
            run
            for run in runs
            if run["role"] == "outer_test" and run["condition"] == condition
        ]
        rows = [row for run in records for row in replay[run["run_id"]][0]]
        estimates = [float(p) for run in records for p in replay[run["run_id"]][1]]
        assert (
            len(rows) == len(examples)
            and {r["example_id"] for r in rows} == by_id.keys()
        )
        compare(measures(rows, estimates), pooled[condition])
        errors[condition] = {}
        for field, expected in (("sequence", per_peptide), ("site_label", per_site)):
            for group in sorted({r[field] for r in rows}):
                selected = [
                    (r, p) for r, p in zip(rows, estimates) if r[field] == group
                ]
                result = measures([r for r, _ in selected], [p for _, p in selected])
                compare(result, expected[condition][group])
                if field == "sequence":
                    errors[condition][group] = result["equal_peptide_mse"]
    sequences = sorted(errors["context"])
    a = np.array([errors["sequence"][s] for s in sequences])
    b = np.array([errors["context"][s] for s in sequences])
    draws = np.random.default_rng(C["bootstrap_seed"]).integers(
        0, len(sequences), (C["bootstrap_draws"], len(sequences))
    )
    delta = [math.sqrt(np.mean(a[d])) - math.sqrt(np.mean(b[d])) for d in draws]
    comparison = json.loads((R / "results/paired-comparison.json").read_text())
    assert math.isclose(
        comparison["delta"], math.sqrt(a.mean()) - math.sqrt(b.mean()), abs_tol=1e-12
    )
    assert np.allclose(
        comparison["paired_peptide_bootstrap_95percent_interval"],
        np.percentile(delta, [2.5, 97.5]),
        atol=1e-12,
    )
    report.update(
        passed=True,
        n_replayed_fits=len(runs),
        n_validation_predictions=len(validation),
        n_outer_test_predictions=len(testing),
        maximum_coefficient_difference=max_coef_error,
        maximum_prediction_difference=max_pred_error,
        pooled_metrics=pooled,
        selection_replay=True,
        source_group_isolation=True,
    )


if __name__ == "__main__":
    start = time.perf_counter()
    report = {"passed": False}
    try:
        main(report)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - start
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (R / "results/independent-fit-verification.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, sort_keys=True), flush=True)
