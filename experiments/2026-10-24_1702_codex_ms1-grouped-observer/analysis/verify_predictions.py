"""Independent NumPy reconstruction of observer features, fits and all metrics."""

import gzip
import hashlib
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "data/measured_ms1_grouped_observer/2026-10-24_1702_canonical"
FIT = ROOT / "results/fit"
EVAL = ROOT / "results/evaluation"


def read(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines()]


def design(rows):
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    matrix = np.empty((len(rows), 62), dtype=np.float64)
    for i, row in enumerate(rows):
        sequence = row["sequence"]
        length = len(sequence)
        matrix[i, :2] = length, length * length
        counts = {a: 0 for a in alphabet}
        for residue in sequence:
            counts[residue] += 1
        matrix[i, 2:22] = [counts[a] / length for a in alphabet]
        matrix[i, 22:42] = [sequence.startswith(a) for a in alphabet]
        matrix[i, 42:62] = [sequence.endswith(a) for a in alphabet]
    return matrix


def center(value, rows):
    result = value.copy()
    for pool in sorted({r["pool"] for r in rows}):
        mask = np.array([r["pool"] == pool for r in rows])
        result[mask] -= np.mean(value[mask], axis=0)
    return result


def weights(rows):
    groups = sorted({r["pool"] for r in rows})
    counts = {g: sum(r["pool"] == g for r in rows) for g in groups}
    return np.array([1.0 / (len(groups) * counts[r["pool"]]) for r in rows])


def metric(y, p, w):
    w = w / np.sum(w)
    mse = np.sum(w * np.square(p - y))
    ym, pm = np.sum(w * y), np.sum(w * p)
    ys, ps = (
        np.sqrt(np.sum(w * np.square(y - ym))),
        np.sqrt(np.sum(w * np.square(p - pm))),
    )
    varying = max(p) - min(p) > 1e-12 and max(y) - min(y) > 1e-12
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.sum(w * np.abs(p - y))),
        "r2": float(1 - mse / ys**2) if ys else None,
        "pearson": float(np.corrcoef(y, p)[0, 1]) if varying else None,
        "spearman": float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])
        if varying
        else None,
        "prediction_sd": float(ps),
        "target_sd": float(ys),
        "prediction_mean": float(pm),
        "target_mean": float(ym),
        "n_examples": len(y),
    }


def equal_metrics(expected, actual):
    assert set(expected) == set(actual)
    for name, value in expected.items():
        if value is None:
            assert actual[name] is None
        else:
            assert np.isclose(value, actual[name], atol=1e-10, rtol=1e-10), (
                name,
                value,
                actual[name],
            )


def main():
    started = time.perf_counter()
    destination = ROOT / "results/prediction-verification.json"
    assert not destination.exists()
    shutil.copy2(__file__, ROOT / "reproduce/source/verify_predictions.py")
    seal = json.loads((ROOT / "results/pre-test-seal.json").read_text())
    for name, digest in seal["files"].items():
        assert hashlib.sha256((FIT / name).read_bytes()).hexdigest() == digest
    rows = {
        "train": read("train.jsonl"),
        "validation": read("validation.jsonl"),
        "test": read("test-covariates.jsonl"),
    }
    test_targets = read("test-targets.jsonl")
    assert [r["measurement_id"] for r in rows["test"]] == [
        r["measurement_id"] for r in test_targets
    ]
    for row, target in zip(rows["test"], test_targets, strict=True):
        row["target"] = target["target"]
    raw = {s: design(r) for s, r in rows.items()}
    ws = {s: weights(r) for s, r in rows.items()}
    ys = {s: np.array([r["target"] for r in values]) for s, values in rows.items()}
    transform = np.load(FIT / "transforms.npz", allow_pickle=False)
    mean = np.average(raw["train"], axis=0, weights=ws["train"])
    scale = np.sqrt(
        np.average(np.square(raw["train"] - mean), axis=0, weights=ws["train"])
    )
    mask = raw["train"].max(axis=0) != raw["train"].min(axis=0)
    scale[~mask] = 1
    assert np.allclose(mean, transform["means"], atol=1e-10, rtol=1e-12)
    assert np.allclose(scale, transform["scales"], atol=1e-10, rtol=1e-12)
    within = np.zeros(62, dtype=bool)
    for pool in {r["pool"] for r in rows["train"]}:
        subset = raw["train"][[r["pool"] == pool for r in rows["train"]]]
        within |= subset.max(axis=0) != subset.min(axis=0)
    for name, expected in (
        ("nonlinear", mask),
        ("sequence", mask & within),
        ("length", mask & within & (np.arange(62) < 2)),
    ):
        assert np.array_equal(expected, transform[name])
    xs = {s: (x - mean) / scale for s, x in raw.items()}
    prediction = {s: {"zero": np.zeros(len(r))} for s, r in rows.items()}
    for family in ("length", "sequence"):
        active = transform[family]
        x = center(xs["train"][:, active], rows["train"])
        for alpha in (0.001, 0.01, 0.1, 1):
            name = f"{family}_{alpha:g}"
            saved = np.load(FIT / f"{name}.npz", allow_pickle=False)
            assert float(saved["alpha"]) == alpha and np.array_equal(
                saved["mask"], active
            )
            # Augmented weighted least squares is independent of the trainer's normal-equation solve.
            lhs = np.vstack(
                (
                    np.sqrt(ws["train"])[:, None] * x,
                    np.sqrt(alpha / x.shape[1]) * np.eye(x.shape[1]),
                )
            )
            rhs = np.concatenate(
                (np.sqrt(ws["train"]) * ys["train"], np.zeros(x.shape[1]))
            )
            beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            assert np.allclose(beta, saved["beta"], atol=1e-9, rtol=1e-9)
            for s, split_rows in rows.items():
                prediction[s][name] = center(xs[s][:, active] @ beta, split_rows)
    chosen = json.loads((FIT / "validation-selection.json").read_text())
    for info in chosen["nonlinear"]:
        name = info["model"]
        state = np.load(FIT / f"{name}.npz", allow_pickle=False)
        selected = np.load(
            FIT / name / f"step-{info['selected_step']:03}.npz", allow_pickle=False
        )
        for key in state.files:
            assert np.array_equal(state[key], selected[key])
        history = json.loads((FIT / name / "history.json").read_text())
        scores = [r for r in history if "evaluation" in r]
        assert (
            min(
                scores, key=lambda r: (r["evaluation"]["validation"]["mse"], r["step"])
            )["step"]
            == info["selected_step"]
        )
        for s, split_rows in rows.items():
            hidden = np.tanh(
                np.einsum(
                    "ij,kj->ik",
                    xs[s][:, transform["nonlinear"]],
                    state["hidden.weight"],
                )
                + state["hidden.bias"]
            )
            output = np.einsum("ij,j->i", hidden, state["output.weight"][0])
            prediction[s][name] = center(output, split_rows)
    largest, count = 0.0, 0
    for s, split_rows in rows.items():
        prediction[s]["nonlinear_ensemble"] = (
            sum(prediction[s][f"nonlinear_{seed}"] for seed in (42, 43, 44)) / 3
        )
        saved = np.load(FIT / f"{s}-predictions.npz", allow_pickle=False)
        assert list(saved["measurement_ids"]) == [
            r["measurement_id"] for r in split_rows
        ]
        for name, p in prediction[s].items():
            difference = float(np.max(np.abs(p - saved[name])))
            assert difference < 1e-9, (s, name, difference)
            largest = max(largest, difference)
            count += len(p)
    summary = json.loads((EVAL / "summary.json").read_text())
    for family in ("length", "sequence"):
        names = [f"{family}_{a:g}" for a in (0.001, 0.01, 0.1, 1)]
        best = min(
            names,
            key=lambda n: (
                metric(ys["validation"], prediction["validation"][n], ws["validation"])[
                    "mse"
                ],
                float(n.split("_")[1]),
            ),
        )
        assert chosen["primary_" + family] == best
    metrics_checked = 0
    for s in ("validation", "test"):
        for name, expected in summary["overall"][s].items():
            equal_metrics(expected, metric(ys[s], prediction[s][name], ws[s]))
            metrics_checked += 1
        ids = {r["measurement_id"]: i for i, r in enumerate(rows[s])}
        seen = set()
        with gzip.open(EVAL / f"{s}-predictions.jsonl.gz", "rt") as stream:
            for line in stream:
                row = json.loads(line)
                i, name = ids[row["measurement_id"]], row["model"]
                for field, value in rows[s][i].items():
                    assert row[field] == value
                assert row["split"] == s and abs(row["weight"] - ws[s][i]) < 1e-15
                assert abs(row["prediction"] - prediction[s][name][i]) < 1e-9
                assert (i, name) not in seen
                seen.add((i, name))
        assert len(seen) == len(rows[s]) * len(prediction[s])
    strata = json.loads((EVAL / "strata.json").read_text())
    for row in strata:
        s, name = row["split"], row["model"]
        subset = np.array([r[row["stratum_type"]] == row["stratum"] for r in rows[s]])
        expected = {
            k: v
            for k, v in row.items()
            if k not in ("split", "model", "stratum_type", "stratum")
        }
        equal_metrics(
            expected, metric(ys[s][subset], prediction[s][name][subset], ws[s][subset])
        )
        metrics_checked += 1
    for screen in [summary["primary"], *summary["secondary"]]:
        name, base = screen["model"], screen["length_comparator"]
        scores = {
            s: metric(ys[s], prediction[s][name], ws[s]) for s in ("validation", "test")
        }
        gains = {
            s: 1 - scores[s]["rmse"] / metric(ys[s], prediction[s][base], ws[s])["rmse"]
            for s in scores
        }
        positives = sum(
            r["pearson"] is not None
            and r["spearman"] is not None
            and r["pearson"] > 0
            and r["spearman"] > 0
            for r in strata
            if r["split"] == "test"
            and r["model"] == name
            and r["stratum_type"] == "pool"
        )
        assert positives == screen["positive_test_pools"]
        gates = [
            gains["validation"] >= 0.05,
            gains["test"] >= 0.05,
            positives >= 3,
            all(v["prediction_sd"] >= 0.1 * v["target_sd"] for v in scores.values()),
        ]
        assert (
            sorted(gates) == sorted(screen["gates"].values())
            and all(gates) == screen["passed"]
        )
    report = {
        "status": "passed",
        "prediction_values_verified": count,
        "metric_records_verified": metrics_checked,
        "maximum_prediction_error": largest,
        "ridge_refit": "augmented weighted least squares",
        "nonlinear_replay": "NumPy checkpoint inference; optimizer trajectory not replayed",
        "seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    assert report["peak_rss_gib"] <= 3
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
