"""Prespecified nested peptide-grouped response diagnostic on admitted examples."""

from pathlib import Path
from collections import Counter
import csv
import gzip
import hashlib
import json
import resource
import time
import traceback

import numpy as np
from scipy.stats import pearsonr, spearmanr

R = Path(__file__).resolve().parents[1]
C = json.loads((R / "reproduce/launch.json").read_text())
AA = "ACDEFGHIKLMNPQRSTVWY"


def weights(rows):
    counts = Counter(row["sequence"] for row in rows)
    w = np.array([1 / counts[row["sequence"]] for row in rows])
    return w / w.sum()


def design(rows, condition, sites):
    names = ["log10_nominal_load"] + ["site:" + site for site in sites]
    x = [
        [row["nominal_log10_load"]]
        + [float(row["site_label"] == site) for site in sites]
        for row in rows
    ]
    if condition == "sequence":
        names += ["length"] + ["fraction:" + aa for aa in AA]
        for values, row in zip(x, rows):
            seq = row["sequence"]
            values.extend([len(seq)] + [seq.count(aa) / len(seq) for aa in AA])
    return np.asarray(x, dtype=float), names


def metrics(rows, prediction):
    y = np.array([row["target_log10_response"] for row in rows])
    p = np.asarray(prediction)
    w = weights(rows)
    error = p - y
    nonconstant = np.ptp(p) > 0 and np.ptp(y) > 0
    return {
        "n_examples": len(rows),
        "n_sequences": len({row["sequence"] for row in rows}),
        "equal_peptide_mse": float(w @ error**2),
        "equal_peptide_rmse": float(np.sqrt(w @ error**2)),
        "equal_peptide_mae": float(w @ abs(error)),
        "pooled_pearson": float(pearsonr(y, p).statistic) if nonconstant else None,
        "pooled_spearman": float(spearmanr(y, p).statistic) if nonconstant else None,
        "prediction_std": float(p.std()),
        "target_std": float(y.std()),
        "prediction_min": float(p.min()),
        "prediction_max": float(p.max()),
    }


def fit_model(rows, condition, alpha, sites):
    w = weights(rows)
    y = np.array([row["target_log10_response"] for row in rows])
    intercept = float(w @ y)
    if condition == "mean":
        return {
            "condition": condition,
            "alpha": None,
            "sites": sites,
            "intercept": intercept,
            "feature_names": [],
            "kept_indices": [],
            "mean": [],
            "scale": [],
            "coefficients": [],
            "dropped_features": [],
        }
    x, names = design(rows, condition, sites)
    center = w @ x
    scale = np.sqrt(w @ (x - center) ** 2)
    keep = np.flatnonzero(scale > 1e-12)
    z = (x[:, keep] - center[keep]) / scale[keep]
    beta = np.linalg.solve(
        z.T @ (w[:, None] * z) + alpha * np.eye(len(keep)), z.T @ (w * (y - intercept))
    )
    return {
        "condition": condition,
        "alpha": alpha,
        "sites": sites,
        "intercept": intercept,
        "feature_names": names,
        "kept_indices": keep.tolist(),
        "mean": center[keep].tolist(),
        "scale": scale[keep].tolist(),
        "coefficients": beta.tolist(),
        "dropped_features": [name for i, name in enumerate(names) if i not in keep],
    }


def predict(model, rows):
    if model["condition"] == "mean":
        return np.full(len(rows), model["intercept"])
    x, names = design(rows, model["condition"], model["sites"])
    assert names == model["feature_names"]
    return model["intercept"] + (
        (x[:, model["kept_indices"]] - np.array(model["mean"]))
        / np.array(model["scale"])
    ) @ np.array(model["coefficients"])


def main(receipt):
    admission = json.loads((R / "results/independent-admission.json").read_text())
    assert admission["passed"]
    package = Path(C["recipe_output"])
    assert (
        hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest()
        == admission["recipe_manifest_sha256"]
    )
    examples = json.loads((package / "examples.json").read_text())
    by_id = {row["example_id"]: row for row in examples}
    assert len(by_id) == len(examples)
    sites = sorted({row["site_label"] for row in examples}, key=int)
    assert set(row["fold_id"] for row in examples) == set(C["folds"])
    runs, validation, outer_predictions, selections = [], [], [], []

    def run(condition, train, evaluation, outer, inner, alpha, role):
        start = time.perf_counter()
        model = fit_model(train, condition, alpha, sites)
        pred, train_pred = predict(model, evaluation), predict(model, train)
        run_id = f"{condition}:outer{outer}:inner{inner}:alpha{alpha}:{role}"
        runs.append(
            {
                "run_id": run_id,
                "condition": condition,
                "outer_fold": outer,
                "inner_fold": inner,
                "role": role,
                "model": model,
                "training_ids": [r["example_id"] for r in train],
                "evaluation_ids": [r["example_id"] for r in evaluation],
                "train_metrics": metrics(train, train_pred),
                "evaluation_metrics": metrics(evaluation, pred),
                "wall_seconds": time.perf_counter() - start,
            }
        )
        target = validation if role == "inner_validation" else outer_predictions
        for row, value in zip(evaluation, pred):
            target.append(
                {
                    "run_id": run_id,
                    "condition": condition,
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "alpha": alpha,
                    "example_id": row["example_id"],
                    "sequence": row["sequence"],
                    "site_label": row["site_label"],
                    "target": row["target_log10_response"],
                    "prediction": float(value),
                }
            )
        return pred

    for outer in C["folds"]:
        training = [row for row in examples if row["fold_id"] != outer]
        testing = [row for row in examples if row["fold_id"] == outer]
        assert not {r["sequence"] for r in training} & {r["sequence"] for r in testing}
        run("mean", training, testing, outer, None, None, "outer_test")
        for condition in ("context", "sequence"):
            scores = []
            for alpha in C["alphas"]:
                ids, pred = [], []
                for inner in C["folds"]:
                    if inner == outer:
                        continue
                    inner_train = [row for row in training if row["fold_id"] != inner]
                    inner_val = [row for row in training if row["fold_id"] == inner]
                    predictions = run(
                        condition,
                        inner_train,
                        inner_val,
                        outer,
                        inner,
                        alpha,
                        "inner_validation",
                    )
                    ids.extend(inner_val)
                    pred.extend(predictions)
                scores.append(
                    {"alpha": alpha, "validation_metrics": metrics(ids, pred)}
                )
            selected = min(
                scores, key=lambda r: r["validation_metrics"]["equal_peptide_mse"]
            )
            selections.append(
                {
                    "outer_fold": outer,
                    "condition": condition,
                    "selected_alpha": selected["alpha"],
                    "candidates": scores,
                }
            )
            run(
                condition,
                training,
                testing,
                outer,
                None,
                selected["alpha"],
                "outer_test",
            )
        print(f"Finished outer fold {outer}", flush=True)

    def save(name, value):
        (R / "results" / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    with gzip.open(R / "results/checkpoints.json.gz", "wt") as stream:
        json.dump(runs, stream, sort_keys=True, allow_nan=False)
    with gzip.open(R / "results/validation-predictions.jsonl.gz", "wt") as stream:
        for row in validation:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with (R / "results/outer-test-predictions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outer_predictions[0]))
        writer.writeheader()
        writer.writerows(outer_predictions)
    save("selections.json", selections)
    pooled, by_peptide, by_site = {}, {}, {}
    for condition in C["conditions"]:
        predictions = [
            row for row in outer_predictions if row["condition"] == condition
        ]
        assert len(predictions) == len(examples) and len(
            {p["example_id"] for p in predictions}
        ) == len(examples)
        rows = [by_id[p["example_id"]] for p in predictions]
        pooled[condition] = metrics(rows, [p["prediction"] for p in predictions])
        for key, container in (("sequence", by_peptide), ("site_label", by_site)):
            container[condition] = {}
            for group in sorted({r[key] for r in rows}):
                selected = [p for p in predictions if p[key] == group]
                container[condition][group] = metrics(
                    [by_id[p["example_id"]] for p in selected],
                    [p["prediction"] for p in selected],
                )
    sequences = sorted(by_peptide["context"])
    a = np.array(
        [by_peptide["sequence"][seq]["equal_peptide_mse"] for seq in sequences]
    )
    b = np.array([by_peptide["context"][seq]["equal_peptide_mse"] for seq in sequences])
    rng = np.random.default_rng(C["bootstrap_seed"])
    draws = rng.integers(0, len(sequences), size=(C["bootstrap_draws"], len(sequences)))
    delta = np.sqrt(a[draws].mean(axis=1)) - np.sqrt(b[draws].mean(axis=1))
    comparison = {
        "quantity": "sequence_minus_context_equal_peptide_RMSE",
        "delta": float(np.sqrt(a.mean()) - np.sqrt(b.mean())),
        "paired_peptide_bootstrap_95percent_interval": np.quantile(
            delta, [0.025, 0.975]
        ).tolist(),
        "n_draws": C["bootstrap_draws"],
        "seed": C["bootstrap_seed"],
        "uncertainty_scope": "selected_panel_peptides_not_external_biological_population",
    }
    save("pooled-metrics.json", pooled)
    save("per-peptide-metrics.json", by_peptide)
    save("per-site-metrics.json", by_site)
    save("paired-comparison.json", comparison)
    save(
        "fold-metrics.json",
        [
            {
                k: run[k]
                for k in (
                    "run_id",
                    "condition",
                    "outer_fold",
                    "train_metrics",
                    "evaluation_metrics",
                )
            }
            for run in runs
            if run["role"] == "outer_test"
        ],
    )
    receipt.update(
        passed=True,
        n_fits=len(runs),
        n_validation_predictions=len(validation),
        n_outer_test_predictions=len(outer_predictions),
        n_sequences=len(sequences),
        n_examples=len(examples),
        metrics=pooled,
        paired_comparison=comparison,
    )


if __name__ == "__main__":
    start = time.perf_counter()
    receipt = {
        "passed": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        main(receipt)
    except Exception:
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter() - start
        receipt["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (R / "results/fit-execution.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(json.dumps(receipt, sort_keys=True), flush=True)
