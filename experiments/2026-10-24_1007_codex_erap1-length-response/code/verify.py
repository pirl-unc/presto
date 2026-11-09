"""Independent source, scalar-recurrence and metric closure; no producer imports."""

import csv
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p, w):
    w = w / sum(w)
    error = p - y
    mse = float(sum(w * error * error))
    variance = float(sum(w * (y - sum(w * y)) ** 2))
    varying = np.ptp(p) > 1e-14 and np.ptp(y) > 0
    return {
        "n_observations": len(y),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(sum(w * abs(error))),
        "r2": 1 - mse / variance if variance else None,
        "pearson": float(pearsonr(y, p).statistic) if varying else None,
        "spearman": float(spearmanr(y, p).statistic) if varying else None,
        "prediction_sd": math.sqrt(float(sum(w * (p - sum(w * p)) ** 2))),
    }


def main():
    started = time.perf_counter()
    cfg = read(ROOT / "reproduce/launch.json")
    recipe, baseline = Path(cfg["recipe_directory"]), Path(cfg["baseline_directory"])
    summary = read(ROOT / "results/fits/summary.json")
    qualification = read(ROOT / "results/qualification.json")
    gradient_checks = []
    for attempt in qualification["attempts"]:
        for probe in attempt["probes"]:
            exact, draws = np.array(probe["exact"]), np.array(probe["samples"])
            denominator = max(float(np.linalg.norm(exact)), 1e-8)
            mean = float(np.linalg.norm(draws.mean(0) - exact) / denominator)
            rms = float(np.sqrt(np.mean(np.sum((draws - exact) ** 2, axis=1))) / denominator)
            np.testing.assert_allclose(
                [mean, rms],
                [probe["mean_relative_error"], probe["rms_relative_error"]],
                atol=1e-14,
                rtol=1e-14,
            )
            assert bool(mean <= 0.10 and rms <= 0.25) == probe["passed"]
            gradient_checks.append(
                {
                    "k": attempt["k"],
                    "probe": probe["probe"],
                    "mean_relative_error": mean,
                    "rms_relative_error": rms,
                    "exact_norm": float(np.linalg.norm(exact)),
                }
            )
    reference_experiment = Path(cfg["reference_experiment"])
    for name, expected in cfg["reference_sha256"].items():
        assert digest(reference_experiment / name) == expected
    for name, expected in read(reference_experiment / "results/closure.json")["artifact_sha256"].items():
        if name.startswith("results/fits/"):
            assert digest(reference_experiment / name) == expected
    assert summary["reference_arms"] == read(reference_experiment / "results/fits/summary.json")["arms"]
    assert summary["reference_results_sha256"] == digest(reference_experiment / "results/fits/summary.json")
    assert digest(recipe / "manifest.json") == cfg["recipe_manifest_sha256"]
    manifest = read(recipe / "manifest.json")
    for name in [
        "derived/parents.parquet",
        "derived/precursor_inputs.parquet",
        "source/references.parquet",
        "source/occurrences.parquet",
    ]:
        assert digest(recipe / name) == manifest["files"][name]["sha256"]
    for name, expected in read(baseline / "results/closure.json")["artifact_sha256"].items():
        if name.startswith(("results/fits/", "results/independent-")):
            assert digest(baseline / name) == expected
    coverage = read(ROOT / "results/input-coverage.json")
    for name, expected in coverage["files"].items():
        assert digest(ROOT / "results" / name) == expected
    rows = pq.read_table(ROOT / "results/cohort.parquet").to_pylist()
    original = pq.read_table(
        baseline / "results/fits/cohort.parquet",
        filters=[("condition_id", "=", "hap1_erap1_ko")],
    ).to_pylist()
    assert rows == original and all(r["split"] == "train" for r in rows)
    assert len(rows) == 8281
    groups = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        groups[row["component_id"]][row["peptide_sequence"]].append(i)
    weight = np.zeros(len(rows))
    for peptides_in_group in groups.values():
        for indices in peptides_in_group.values():
            weight[indices] = 1 / (len(groups) * len(peptides_in_group) * len(indices))
    fold = np.array(
        [
            int(
                hashlib.sha256(
                    ("hap1_baseline_inner_v1:" + r["component_id"]).encode()
                ).hexdigest(),
                16,
            )
            % 5
            for r in rows
        ]
    )
    peptides = sorted({r["peptide_sequence"] for r in rows})
    query = {p: i for i, p in enumerate(peptides)}
    owner = np.array([query[r["peptide_sequence"]] for r in rows])
    y = np.array([r["target_log10_ratio"] for r in rows])
    x = np.array([r["precursor_log10_ratio"] for r in rows])
    origins = read(ROOT / "results/origins.json")
    parents = read(ROOT / "results/parents.json")
    source_parents = {
        r["parent_id"]: r["reference_sequence"]
        for r in pq.read_table(recipe / "derived/parents.parquet").to_pylist()
    }
    assert all(source_parents[k] == v for k, v in parents.items())
    reference = {
        r["reference_record_id"]: r["sequence_sha256"]
        for r in pq.read_table(recipe / "source/references.parquet").to_pylist()
    }
    source_origins = set()
    for row in pq.read_table(recipe / "source/occurrences.parquet").to_pylist():
        if row["peptide_sequence"] in query and row["reference_record_id"] is not None:
            source_origins.add(
                (
                    row["peptide_sequence"],
                    reference[row["reference_record_id"]],
                    row["start_1based"] - 1,
                    row["end_1based"],
                )
            )
    assert source_origins == {
        (o["peptide_sequence"], o["parent_id"], o["start0"], o["end0"]) for o in origins
    }
    inputs = {
        (r["parent_id"], r["condition_id"]): r
        for r in pq.read_table(recipe / "derived/precursor_inputs.parquet").to_pylist()
    }
    with np.load(ROOT / "results/path-inputs.npz") as archive:
        path = {name: archive[name] for name in archive.files}
    birth_cache = []
    cursor = 0
    query_bounds = defaultdict(list)
    for o in origins:
        seq = parents[o["parent_id"]]
        a, b = o["start0"], o["end0"]
        assert seq[a:b] == o["peptide_sequence"] and o["query_id"] == query[seq[a:b]]
        precursor = inputs[o["parent_id"], "hap1_erap1_ko"]
        assert precursor["status"] == "complete"
        assert o["parent_log10_response"] == precursor["log10_response"]
        assert o["path_start"] == cursor and o["path_end"] == cursor + a + 1
        birth, residues = [], [AA.index(r) for r in seq[: a + 1]]
        for j in range(a + 1):
            length = b - j
            boundaries = int(j > 0) + int(b < len(seq))
            value = (
                1 / length
                if boundaries == 0
                else 1 / (length * (length + 1))
                if boundaries == 1
                else 2 / (length * (length + 1) * (length + 2))
            )
            birth.append(value)
        removed = np.zeros((a + 1, 20), dtype=int)
        for j in range(a - 1, -1, -1):
            removed[j] = removed[j + 1]
            removed[j, residues[j]] += 1
        sl = slice(cursor, cursor + a + 1)
        np.testing.assert_array_equal(path["counts"][sl], removed)
        assert (path["owner"][sl] == o["query_id"]).all()
        np.testing.assert_allclose(path["log_birth"][sl], np.log(birth), atol=1e-13, rtol=0)
        assert (path["input_log10"][sl] == o["parent_log10_response"]).all()
        lengths = b - np.arange(a + 1)
        np.testing.assert_array_equal(path["state_length"][sl], lengths)
        np.testing.assert_array_equal(path["state_residue"][sl], residues)
        np.testing.assert_array_equal(
            path["state_band"][sl], np.where(lengths <= 8, 0, np.where(lengths <= 16, 1, 2))
        )
        np.testing.assert_array_equal(path["state_monomer"][sl], lengths == 1)
        assert (path["terminal_index"][sl] == cursor + a).all()
        birth_cache.append((o, birth, residues))
        query_bounds[o["query_id"]].extend([cursor, cursor + a + 1])
        cursor += a + 1
    assert cursor == coverage["n_paths"]
    for q, p in enumerate(peptides):
        assert path["starts"][q] == min(query_bounds[q])
        assert path["ends"][q] == max(query_bounds[q])
        assert path["target_residue"][q] == AA.index(p[0])
        assert path["target_monomer"][q] == (len(p) == 1)
    print("Source, weights, folds and every birth path verified", flush=True)

    def recurrence(parameter, gamma):
        rates = [
            math.exp(v) for v in (parameter[:1] * 20 if len(parameter) == 4 else parameter[:20])
        ]
        modifiers = [math.exp(parameter[-3]), 1.0, math.exp(parameter[-2])]
        kappa = 1 / (1 + math.exp(-parameter[-1]))
        sums = np.zeros((2, len(peptides)))
        for o, birth, residues in birth_cache:
            wt = ko = previous_wt = previous_ko = 0.0
            for j, source in enumerate(birth):
                length = o["end0"] - j
                band = 0 if length <= 8 else 1 if length <= 16 else 2
                trim = rates[residues[j]] * modifiers[band] if length > 1 else 0.0
                wt = (source + previous_wt * wt) / (1 + trim)
                ko = (source + previous_ko * ko) / (1 + kappa * trim)
                previous_wt, previous_ko = trim, kappa * trim
            sums[0, o["query_id"]] += wt
            sums[1, o["query_id"]] += ko * 10 ** (gamma * o["parent_log10_response"])
        assert np.isfinite(sums).all() and (sums > 0).all()
        return np.log10(sums[1] / sums[0])[owner], sums

    checkpoints = read(baseline / "results/fits/checkpoints.json")["sequence_0.1"]
    phi = []
    for p in peptides:
        counts = Counter(p)
        phi.append(
            [
                len(p),
                *[counts[a] / len(p) for a in AA],
                *[int(p[0] == a) for a in AA],
                *[int(p[-1] == a) for a in AA],
            ]
        )
    phi = np.array(phi, dtype=float)
    reconstructed = {}
    all_metrics, component_metrics, fit_checks = {}, [], []
    max_error = 0.0

    def compare_metrics(expected, actual):
        for key, value in expected.items():
            if value is None:
                assert actual[key] is None, (key, actual[key])
            else:
                np.testing.assert_allclose(actual[key], value, atol=1e-10, rtol=1e-10)

    for arm, reported in summary["arms"].items():
        reconstructed[arm] = {"inner_cv": np.full(len(rows), np.nan)}
        correction_cv = np.full(len(rows), np.nan)
        for f in ["full", *range(5)]:
            directory = ROOT / "results/fits" / arm / str(f)
            record = read(directory / "fit.json")
            old = checkpoints["full"] if f == "full" else checkpoints["folds"][str(f)]
            assert record["gamma"] == old["gamma"]
            selected = np.ones(len(rows), dtype=bool) if f == "full" else fold != f
            fit_weights = weight[selected] / sum(weight[selected])
            p, _amounts = recurrence(record["parameter"], old["gamma"])
            correction = p - old["gamma"] * x if arm.startswith("augmented") else p.copy()
            if arm.startswith("augmented"):
                mask = np.array(old["feature_mask"])
                params = old["sequence"]["hap1_erap1_ko"]
                z = (phi[:, mask] - np.array(old["feature_mean"])[mask]) / np.array(
                    old["feature_sd"]
                )[mask]
                direct = (
                    old["condition_offsets"]["hap1_erap1_ko"]
                    + old["gamma"] * x
                    + ((z - params["center"]) @ params["beta"])[owner]
                )
                p = direct + correction
            offset = sum(fit_weights * (y[selected] - p[selected]))
            p += offset
            saved = np.load(directory / "predictions.npz")
            error = float(np.max(abs(p - saved["prediction"])))
            max_error = max(error, max_error)
            np.testing.assert_allclose(
                p, saved["prediction"], rtol=0, atol=cfg["old_model_absolute_tolerance"]
            )
            np.testing.assert_allclose(
                correction, saved["correction"], rtol=0, atol=cfg["old_model_absolute_tolerance"]
            )
            data_loss = float(sum(fit_weights * (p[selected] - y[selected]) ** 2))
            parameter = np.array(record["parameter"])
            penalty = (
                0.01 * np.mean(parameter[:-3] ** 2)
                + 0.01 * np.mean(parameter[-3:-1] ** 2)
                + 0.01 * parameter[-1] ** 2
            )
            np.testing.assert_allclose(
                [offset, data_loss, penalty, data_loss + penalty],
                [record[k] for k in ["offset", "data_loss", "regularizer", "objective"]],
                rtol=1e-10,
                atol=1e-10,
            )
            assert (
                np.count_nonzero(parameter)
                == record["n_changed_parameters"]
                == record["n_allocated_parameters"]
            )
            history = read(directory / "history.json")
            assert [r["step"] for r in history] == list(range(1, 151))
            assert all(math.isfinite(r["objective"]) for r in history)
            # Recompute reported rank metrics from the saved predictions after
            # validating their values; scalar summation can change roundoff ties.
            if f == "full":
                reconstructed[arm]["train"] = saved["prediction"].copy()
            else:
                reconstructed[arm]["inner_cv"][fold == f] = saved["prediction"][fold == f]
                correction_cv[fold == f] = correction[fold == f]
            fit_checks.append(
                {
                    "arm": arm,
                    "fold": f,
                    "max_prediction_error": error,
                    "kappa": 1 / (1 + math.exp(-parameter[-1])),
                    "n_changed_parameters": record["n_changed_parameters"],
                    "fitting_metrics": metrics(
                        y[selected], saved["prediction"][selected], weight[selected]
                    ),
                    "held_out_metrics": None
                    if f == "full"
                    else metrics(y[~selected], saved["prediction"][~selected], weight[~selected]),
                }
            )
            print(f"Verified {arm}/{f}: max error {error:.3g}", flush=True)
        sd = math.sqrt(float(sum(weight * (correction_cv - sum(weight * correction_cv)) ** 2)))
        np.testing.assert_allclose(sd, reported["correction_sd_inner_cv"], rtol=1e-10, atol=1e-10)
        for pop, values in reconstructed[arm].items():
            saved = pq.read_table(ROOT / "results/fits" / arm / f"{pop}.parquet").to_pydict()
            assert saved["measurement_id"] == [r["measurement_id"] for r in rows]
            np.testing.assert_array_equal(saved["target_log10_ratio"], y)
            np.testing.assert_array_equal(saved["fold"], fold)
            np.testing.assert_allclose(saved["weight"], weight, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(
                saved["prediction_log10_ratio"], values, rtol=1e-10, atol=1e-10
            )
            compare_metrics(reported[pop], metrics(y, values, weight))
    for name, reported in summary["controls"].items():
        reconstructed[name] = {}
        for pop in ["train", "inner_cv"]:
            data = pq.read_table(
                baseline / f"results/fits/{name}-{pop}.parquet",
                filters=[("condition_id", "=", "hap1_erap1_ko")],
            ).to_pydict()
            assert data["measurement_id"] == [r["measurement_id"] for r in rows]
            np.testing.assert_array_equal(data["target_log10_ratio"], y)
            w = np.array(data["weight"])
            np.testing.assert_allclose(w / sum(w), weight, rtol=1e-12, atol=1e-15)
            reconstructed[name][pop] = np.array(data["prediction_log10_ratio"])
            compare_metrics(reported[pop], metrics(y, reconstructed[name][pop], weight))
    for name, reported in summary["reference_arms"].items():
        reconstructed["reference_" + name] = {}
        for pop in ["train", "inner_cv"]:
            data = pq.read_table(reference_experiment / "results/fits" / name / f"{pop}.parquet").to_pydict()
            assert data["measurement_id"] == [r["measurement_id"] for r in rows]
            np.testing.assert_array_equal(data["target_log10_ratio"], y)
            pred = np.array(data["prediction_log10_ratio"])
            reconstructed["reference_" + name][pop] = pred
            compare_metrics(reported[pop], metrics(y, pred, weight))
    length_metrics = []
    row_lengths = np.array([len(r["peptide_sequence"]) for r in rows])
    for name, populations in reconstructed.items():
        all_metrics[name] = {}
        for pop, values in populations.items():
            all_metrics[name][pop] = metrics(y, values, weight)
            for length in sorted(set(row_lengths)):
                selected = row_lengths == length
                length_metrics.append(
                    {
                        "model": name,
                        "population": pop,
                        "length": int(length),
                        **metrics(y[selected], values[selected], weight[selected]),
                    }
                )
            for component, sequences in groups.items():
                idx = np.array([i for indices in sequences.values() for i in indices])
                w = weight[idx] / sum(weight[idx])
                error = values[idx] - y[idx]
                mse = float(sum(w * error**2))
                component_metrics.append(
                    {
                        "model": name,
                        "population": pop,
                        "component_id": component,
                        "n_observations": len(idx),
                        "mse": mse,
                        "rmse": math.sqrt(mse),
                        "mae": float(sum(w * abs(error))),
                    }
                )
    improvement = (
        1
        - all_metrics["augmented_exact"]["inner_cv"]["rmse"]
        / all_metrics["sequence_0.1"]["inner_cv"]["rmse"]
    )
    gate = (
        improvement >= 0.02 and summary["arms"]["augmented_exact"]["correction_sd_inner_cv"] > 0.001
    )
    assert gate == summary["development_gate_pass"]
    np.testing.assert_allclose(
        improvement, summary["relative_augmented_exact_cv_improvement"], atol=1e-10
    )
    assert not summary["outer_validation_test_scored"]
    with (ROOT / "results/component-metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(component_metrics)
    with (ROOT / "results/length-metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(length_metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(length_metrics)
    output = {
        "status": "passed",
        "gradient_gate_reduction_checks": gradient_checks,
        "n_observations": len(rows),
        "n_paths_verified": cursor,
        "n_fit_predictions_reconstructed": len(fit_checks) * len(rows),
        "n_component_metrics": len(component_metrics),
        "max_prediction_error": max_error,
        "metrics": all_metrics,
        "fit_checks": fit_checks,
        "development_gate_pass": gate,
        "relative_augmented_exact_cv_improvement": improvement,
        "seconds": time.perf_counter() - started,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "scope": "Independent scalar recurrence/source/weights/folds/final objectives/metrics; optimizer updates and saved gradients not independently replayed",
        "source_sha256": digest(__file__),
    }
    (ROOT / "results/independent-verification.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    assert output["seconds"] <= 900 and output["peak_gib"] <= 4
    print(
        json.dumps({k: v for k, v in output.items() if k not in {"metrics", "fit_checks"}}),
        flush=True,
    )


if __name__ == "__main__":
    main()
