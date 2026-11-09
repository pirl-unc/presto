"""Independent NumPy full-parent inference, regression and metric reconstruction."""

import csv
import gzip
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    if a is None or b is None:
        assert a is b
    else:
        assert np.isclose(a, b, atol=2e-9, rtol=2e-9), (a, b)


def main():
    started = time.perf_counter()
    output = ROOT / "results/independent-predictions.json"
    assert not output.exists()
    cfg = read(ROOT / "reproduce/launch.json")
    receipt = read(ROOT / "results/fit-complete.json")
    admission = read(ROOT / "results/independent-admission.json")
    metrics = read(ROOT / "results/evaluation/metrics.json")
    n_values = n_metric = 0
    max_error = 0.0
    normal_errors = []
    for block, files in receipt["files"].items():
        directory = ROOT / "results/fit" / block
        for name, digest in files.items():
            assert sha(directory / name) == digest
        prepared = Path(admission["prepared_directory"]) / block
        rows = read(prepared / "covariates.json")
        parents = read(prepared / "parents.json")
        states = read(directory / "states.json")
        trans = np.load(directory / "transforms.npz")
        stored = np.load(directory / "sealed-predictions.npz")
        ids = [r["measurement_id"] for r in rows]
        lookup = {key: i for i, key in enumerate(ids)}
        assert len(ids) == len(lookup)
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[r["group_id"]].append(i)

        def center(values, groups=groups):
            result = values.copy()
            for indices in groups.values():
                result[indices] -= values[indices].mean(axis=0)
            return result

        raw = []
        for row in rows:
            seq = row["sequence"]
            freq = Counter(seq)
            raw.append(
                [len(seq), len(seq) ** 2]
                + [freq[a] / len(seq) for a in AA]
                + [int(seq[0] == a) for a in AA]
                + [int(seq[-1] == a) for a in AA]
            )
        raw = np.asarray(raw, dtype=float)
        obsdir = Path(cfg["observer_directory"]) / "results/fit"
        ot = np.load(obsdir / "transforms.npz")
        os = np.load(obsdir / "sequence_0.1.npz")
        observed = center(
            (((raw - ot["means"]) / ot["scales"])[:, os["mask"]]) @ os["beta"]
        )
        design = center((raw - trans["means"]) / trans["scales"])
        labels = {
            s: read(prepared / (s + "-labels.json")) for s in ("train", "val", "test")
        }
        y = np.array(
            [
                labels[r["split"]][r["measurement_id"]]["target_centered_log10_area"]
                for r in rows
            ]
        )
        p = np.array(
            [
                labels[r["split"]][r["measurement_id"]]["target_area_fraction"]
                for r in rows
            ]
        )
        weights = {}
        for split in labels:
            components = {r["component_id"] for r in rows if r["split"] == split}
            counts = Counter(
                rows[index[0]]["component_id"]
                for index in groups.values()
                if rows[index[0]]["split"] == split
            )
            w = np.zeros(len(rows))
            for index in groups.values():
                r = rows[index[0]]
                if r["split"] == split:
                    w[index] = 1 / (
                        len(components) * counts[r["component_id"]] * len(index)
                    )
            assert np.isclose(w.sum(), 1)
            weights[split] = w
        w = weights["train"]
        mean = w @ raw
        scales = np.sqrt(w @ (raw - mean) ** 2)
        scales[scales == 0] = 1
        assert np.allclose(mean, trans["means"], atol=1e-12) and np.allclose(
            scales, trans["scales"], atol=1e-12
        )
        gamma = float(np.sum(w * y * observed) / np.sum(w * observed**2))
        close(gamma, states["observer_calibrated"]["gamma"])
        computed = {}
        for name in stored.files:
            if name == "zero":
                prediction = np.zeros(len(rows))
            elif name == "observer_fixed":
                prediction = observed
            elif name == "observer_calibrated":
                prediction = observed * gamma
            elif "theta" not in states[name]:
                state = states[name]
                family = state["family"]
                width = 2 if family == "length" else 62
                x = design[:, :width] @ trans[family + "_projection"]
                beta = np.asarray(state["beta"])
                matrix = x.T @ (w[:, None] * x) + state["alpha"] / len(beta) * np.eye(
                    len(beta)
                )
                rhs = x.T @ (w * y)
                beta_check = np.linalg.solve(matrix, rhs)
                normal_errors.append(float(np.max(abs(beta_check - beta))))
                assert np.allclose(beta_check, beta, atol=1e-8)
                prediction = x @ beta
            else:
                state = states[name]
                theta = np.asarray(state["theta"])
                rate_by_parent = {}
                left = (
                    np.r_[theta[1:20], -sum(theta[1:20])]
                    if state["sequence"]
                    else np.zeros(20)
                )
                right = (
                    np.r_[theta[20:], -sum(theta[20:])]
                    if state["sequence"]
                    else np.zeros(20)
                )
                for key, seq in parents.items():
                    rate_by_parent[key] = np.array(
                        [
                            math.exp(
                                theta[0]
                                + left[AA.index(seq[j])]
                                + right[AA.index(seq[j + 1])]
                            )
                            for j in range(len(seq) - 1)
                        ]
                    )
                values = []
                for row in rows:
                    rates = rate_by_parent[row["parent_id"]]
                    amounts = []
                    for start, end in row["occurrences_0based_halfopen"]:
                        a = 1 + math.fsum(rates[start : end - 1])
                        l = rates[start - 1] if start else 0.0
                        r = rates[end - 1] if end <= len(rates) else 0.0
                        if l and r:
                            value = (
                                l
                                * r
                                * (2 * a + l + r)
                                / (a * (a + l) * (a + r) * (a + l + r))
                            )
                        elif l or r:
                            value = (l + r) / (a * (a + l + r))
                        else:
                            value = 1 / a
                        amounts.append(value)
                    values.append(math.log10(math.fsum(amounts)))
                prediction = center(
                    np.asarray(values) + (observed if state["observer"] else 0.0)
                )
                assert (
                    state["n_allocated_parameters"]
                    == len(theta)
                    == state["n_changed_parameters"]
                    == state["n_gradient_supported_parameters"]
                )
            error = float(np.max(abs(prediction - stored[name])))
            max_error = max(max_error, error)
            assert error < 1e-8, (name, error)
            computed[name] = prediction
            n_values += len(prediction)
            for split, w in weights.items():
                mask = w > 0
                yt = y[mask]
                pr = prediction[mask]
                ws = w[mask]
                difference = pr - yt
                mse = float(ws @ difference**2)
                ytmean = float(ws @ yt)
                prmean = float(ws @ pr)
                ytvar = float(ws @ (yt - ytmean) ** 2)
                prvar = float(ws @ (pr - prmean) ** 2)
                variable = np.ptp(pr) > 1e-12 and np.ptp(yt) > 1e-12
                pearson = float(np.corrcoef(yt, pr)[0, 1]) if variable else None
                spearman = (
                    float(np.corrcoef(rankdata(yt), rankdata(pr))[0, 1])
                    if variable
                    else None
                )
                expected = {
                    "mse": mse,
                    "rmse": mse**0.5,
                    "mae": float(ws @ abs(difference)),
                    "r2": 1 - mse / ytvar,
                    "pearson": pearson,
                    "spearman": spearman,
                    "prediction_sd": prvar**0.5,
                    "target_sd": ytvar**0.5,
                    "prediction_mean": prmean,
                    "target_mean": ytmean,
                    "n_examples": int(mask.sum()),
                    "reverse_kl": 0.0,
                    "forward_kl": 0.0,
                    "total_variation": 0.0,
                }
                for index in groups.values():
                    if rows[index[0]]["split"] != split:
                        continue
                    gw = w[index].sum()
                    logq = prediction[index] * math.log(10)
                    logq -= logsumexp(logq)
                    q = np.exp(logq)
                    logp = np.log(p[index])
                    expected["reverse_kl"] += gw * float(q @ (logq - logp))
                    expected["forward_kl"] += gw * float(p[index] @ (logp - logq))
                    expected["total_variation"] += gw * float(
                        abs(p[index] - q).sum() / 2
                    )
                for key, value in expected.items():
                    close(value, metrics[block][name][split][key])
                n_metric += 1
        n_dump = 0
        with gzip.open(
            ROOT / "results/evaluation" / (block + "-predictions.csv.gz"), "rt"
        ) as handle:
            for row in csv.DictReader(handle):
                index = lookup[row["measurement_id"]]
                close(
                    float(row["prediction_centered_log10_area"]),
                    computed[row["model"]][index],
                )
                close(float(row["target_centered_log10_area"]), y[index])
                close(float(row["target_area_fraction"]), p[index])
                close(float(row["weight"]), weights[row["split"]][index])
                n_dump += 1
        assert n_dump == len(rows) * len(stored.files)
        selected = read(directory / "validation-selection.json")
        for family in ("sequence", "length"):
            assert selected[family] == min(
                (
                    name
                    for name in stored.files
                    if name.startswith(family + "_") and "theta" not in states[name]
                ),
                key=lambda name: metrics[block][name]["val"]["mse"],
            )
    result = {
        "status": "passed",
        "n_predictions_recomputed": n_values,
        "n_split_model_metrics_recomputed": n_metric,
        "max_prediction_absolute_error": max_error,
        "max_ridge_coefficient_replay_error": max(normal_errors),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
        "optimizer_trajectory_replayed": False,
        "scope": "Independent original-area/dependence qualification plus NumPy feature/ridge/observer/full-parent/checkpoint/dump/metric reconstruction; no independent rerun of500Adamsteps",
    }
    assert result["runtime_seconds"] < 600 and result["peak_rss_gib"] < 3
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
