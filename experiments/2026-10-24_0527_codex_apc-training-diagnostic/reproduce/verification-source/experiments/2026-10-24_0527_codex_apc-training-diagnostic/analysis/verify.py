"""Independent source, calibration, profiled-offset, physical and metric closure."""

import gzip
import hashlib
import importlib.util
import json
import math
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    output = FAMILY / "results/independent-verification.json"
    assert not output.exists()
    cfg = read(FAMILY / "reproduce/launch.json")
    for path, digest in read(FAMILY / "reproduce/execution.json")["sha256"].items():
        assert sha(path) == digest
    source = Path(cfg["source_family"])
    receipt = read(source / "results/cohort.json")
    root = Path(receipt["model_input_directory"])
    groups = [g for g in read(root / "groups.json.gz") if g["split"] == "train"]
    parents = read(root / "parents.json.gz")
    parents = {
        p: parents[p] for p in {o["parent_sequence_sha256"] for g in groups for o in g["origins"]}
    }
    by_obs = {
        o["measurement_id"]: (i, g, o) for i, g in enumerate(groups) for o in g["observations"]
    }
    assert len(by_obs) == 2952 and len(groups) == 984
    folds = np.array(
        [
            int(hashlib.sha256((cfg["fold_salt"] + g["component_id"]).encode()).hexdigest(), 16) % 5
            for g in groups
        ]
    )
    conditions = sorted({g["condition_id"] for g in groups})
    c = np.array([conditions.index(g["condition_id"]) for g in groups])
    helper_path = source / "analysis/verify_predictions.py"
    spec = importlib.util.spec_from_file_location("independent_arithmetic", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    summary = read(FAMILY / "results/final-summary.json")
    for path, digest in summary["prediction_files_sha256"].items():
        assert sha(FAMILY / path) == digest
    variants = sorted({r["variant"] for r in summary["metrics"]})
    parameter_checks, calibrations, all_rows = {}, [], {}
    maximum_error = 0.0
    for fold in [-1, *range(5)]:
        directory = FAMILY / "results" / ("full" if fold == -1 else f"fold-{fold}")
        fit_groups = folds != fold
        coverage = read(directory / "coverage.json")
        transforms = np.load(directory / "transforms.npz")
        cx = {
            condition: helper.coordinate(
                np.array([float(condition == other) for other in conditions]),
                transforms,
                "condition_",
            )
            for condition in conditions
        }
        owned = {g["component_id"] for i, g in enumerate(groups) if fit_groups[i]}
        validation = {g["component_id"] for i, g in enumerate(groups) if not fit_groups[i]}
        assert owned.isdisjoint(validation)
        rows = [(i, g, o) for i, g in enumerate(groups) if fit_groups[i] for o in g["observations"]]
        indices = np.array([i for i, _, _ in rows])
        y = np.array([o["target_value"] for _, _, o in rows])
        w = np.array([o["weight"] for _, _, o in rows])
        w /= w.sum()
        z = np.array([g["precursor_only_log10_ratio"] for _, g, _ in rows])
        design = np.column_stack([(c[indices] == j).astype(float) for j in range(2)] + [z])
        coef = np.linalg.lstsq(np.sqrt(w[:, None]) * design, np.sqrt(w) * y, rcond=None)[0]
        cal = read(directory / "calibration.json")
        assert abs(coef[-1] - cal["unconstrained_gamma"]) < 1e-10
        assert abs(max(0, coef[-1]) - cal["gamma"]) < 1e-10
        calibrations.append(
            {
                "fold": fold,
                "independent_gamma": float(max(0, coef[-1])),
                "n_fitting_observations": len(rows),
            }
        )
        # Reconstruct feature scaling from unique reachable fitting bonds.
        bonds = {
            (o["parent_sequence_sha256"], j)
            for i, g in enumerate(groups)
            if fit_groups[i]
            for o in g["origins"]
            for j in range(
                max(0, o["start0"] - 1),
                min(len(parents[o["parent_sequence_sha256"]]) - 1, o["end0_exclusive"]),
            )
        }
        counts = Counter(p for p, _ in bonds)
        raw = []
        bw = []
        for p, j in sorted(bonds):
            v = np.zeros(40)
            v[AA.index(parents[p][j])] = 1
            v[20 + AA.index(parents[p][j + 1])] = 1
            raw.append(v)
            bw.append(1 / len(counts) / counts[p])
        raw, bw = np.array(raw), np.array(bw)
        center = bw @ raw
        variance = bw @ (raw - center) ** 2
        np.testing.assert_allclose(center, transforms["bond_center"], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(variance > 1e-24, transforms["bond_keep"])
        np.testing.assert_allclose(
            np.sqrt(variance[transforms["bond_keep"]]), transforms["bond_scale"], atol=1e-12, rtol=0
        )
        assert coverage["n_training_bonds"] == len(bonds)
        for variant in variants:
            physical = variant.removesuffix("_hybrid") in cfg["arms"]
            if physical:
                local = directory / variant
                record = read(local / "status.json")
                assert record["status"] == "completed" and record["steps"] == 400
                assert sha(local / "fit.npz") == record["fit_sha256"]
                fit = np.load(local / "fit.npz")
                theta = fit["theta"]
                rank = transforms["bond_projection"].shape[1]
                nr = 1 + rank
                nc = 2
                settings = cfg["arms"][variant.removesuffix("_hybrid")]

                def rates_from_weights(weights, rho, transforms=transforms):
                    raw_coef = np.zeros(40)
                    raw_coef[transforms["bond_keep"]] = (
                        transforms["bond_projection"] @ weights
                    ) / transforms["bond_scale"]
                    intercept = rho - transforms["bond_center"] @ raw_coef
                    out = {}
                    for p, sequence in parents.items():
                        aa = np.array([AA.index(a) for a in sequence])
                        out[p] = np.exp(intercept + raw_coef[aa[:-1]] + raw_coef[20 + aa[1:]])
                    return out

                base_rates = rates_from_weights(theta[1:nr], theta[0])
                treatment = {}
                for condition, x in cx.items():
                    delta = (
                        x @ theta[nr + nc :].reshape(nc, rank)
                        if settings["flexible"]
                        else np.zeros(rank)
                    )
                    treatment[condition] = rates_from_weights(
                        theta[1:nr] + delta, theta[0] + x @ theta[nr : nr + nc]
                    )
                raw_prediction = []
                log_control = []
                log_treated = []
                for g in groups:
                    control, treated = [], []
                    for o in g["origins"]:
                        p, a, b = o["parent_sequence_sha256"], o["start0"], o["end0_exclusive"]
                        for dest, rates, input_scale in (
                            (control, base_rates, 1.0),
                            (
                                treated,
                                treatment[g["condition_id"]],
                                10 ** (cal["gamma"] * o["precursor_log10_ratio"]),
                            ),
                        ):
                            r = rates[p]
                            dest.append(
                                input_scale
                                * helper.amount(
                                    math.fsum(r[a : b - 1]),
                                    float(r[a - 1]) if a else 0.0,
                                    float(r[b - 1]) if b < len(parents[p]) else 0.0,
                                )
                            )
                    nparents = len({o["parent_sequence_sha256"] for o in g["origins"]})
                    control, treated = math.fsum(control) / nparents, math.fsum(treated) / nparents
                    raw_prediction.append(math.log10(treated / control))
                    log_control.append(math.log(control))
                    log_treated.append(math.log(treated))
                raw_prediction = np.array(raw_prediction)
                b = np.array(
                    [
                        w[c[indices] == j]
                        @ (y[c[indices] == j] - raw_prediction[indices][c[indices] == j])
                        / w[c[indices] == j].sum()
                        for j in range(2)
                    ]
                )
                prediction = raw_prediction + b[c]
                for key, value in (
                    ("prediction", prediction),
                    ("offsets", b),
                    ("log_control_amount", log_control),
                    ("log_treated_amount", log_treated),
                ):
                    error = float(np.max(abs(fit[key] - value)))
                    assert error < 1e-9, (variant, fold, key, error)
                    maximum_error = max(maximum_error, error)
                mse = float(w @ (prediction[indices] - y) ** 2)
                assert abs(mse - record["train_mse"]) < 1e-10
                assert (
                    abs(mse + settings["penalty"] * sum(theta**2) - record["training_objective"])
                    < 1e-10
                )
                assert fit["trajectory"].shape == (401, len(theta))
                np.testing.assert_array_equal(fit["trajectory"][0], np.zeros(len(theta)))
                np.testing.assert_array_equal(fit["trajectory"][-1], theta)
                parameter_checks[str(local.relative_to(FAMILY))] = {
                    "kinetic_parameters": len(theta),
                    "changed": int(np.sum(abs(theta) > 1e-12)),
                    "train_mse": mse,
                }
                path = local / "predictions.jsonl.gz"
            else:
                records = read(directory / "controls/summary.json")
                record = next(r for r in records if r["variant"] == variant)
                gamma = {
                    "no_change": 0,
                    "condition": 0,
                    "precursor_fixed": 1,
                    "precursor_calibrated": cal["gamma"],
                }[variant]
                raw_prediction = gamma * np.array([g["precursor_only_log10_ratio"] for g in groups])
                b = (
                    np.zeros(2)
                    if variant == "no_change"
                    else np.array(
                        [
                            w[c[indices] == j]
                            @ (y[c[indices] == j] - raw_prediction[indices][c[indices] == j])
                            / w[c[indices] == j].sum()
                            for j in range(2)
                        ]
                    )
                )
                prediction = raw_prediction + b[c]
                np.testing.assert_allclose(b, record["offsets"], atol=1e-12, rtol=0)
                path = directory / "controls" / (variant + "-predictions.jsonl.gz")
            with gzip.open(path, "rt") as stream:
                saved = [json.loads(line) for line in stream]
            expected_ids = {
                o["measurement_id"]
                for i, g in enumerate(groups)
                if fold == -1 or folds[i] == fold
                for o in g["observations"]
            }
            assert {r["observation_id"] for r in saved} == expected_ids and len(saved) == len(
                expected_ids
            )
            for row in saved:
                i, g, obs = by_obs[row["observation_id"]]
                for key, value in obs.items():
                    assert row[key] == value
                for key in (
                    "condition_id",
                    "component_id",
                    "peptide_sequence",
                    "response_group_id",
                    "source_peptide_row_id",
                    "precursor_only_log10_ratio",
                ):
                    assert row[key] == g[key]
                assert (
                    row["original_split"] == "train"
                    and row["inner_fold"] == folds[i]
                    and row["fit_fold"] == fold
                )
                assert (
                    row["origin_inputs"] == g["origins"]
                    and row["target_log10_ratio"] == obs["target_value"]
                )
                error = abs(row["prediction_log10_ratio"] - prediction[i])
                assert error < 1e-9
                maximum_error = max(maximum_error, error)
            phase = "full_train" if fold == -1 else "inner_cv"
            all_rows.setdefault((variant, phase), []).extend(saved)
    metric_count = 0
    for (variant, phase), rows in all_rows.items():
        assert len(rows) == 2952 and len({r["observation_id"] for r in rows}) == 2952
        stored = next(
            r for r in summary["metrics"] if r["variant"] == variant and r["phase"] == phase
        )
        helper.verify_metrics(rows, stored)
        metric_count += 1
        for field in ("condition_id", "component_id"):
            for value in {r[field] for r in rows}:
                members = [r for r in rows if r[field] == value]
                stored = next(
                    r
                    for r in summary["strata"]
                    if r["variant"] == variant
                    and r["phase"] == phase
                    and r["stratum"] == field
                    and r["stratum_id"] == value
                )
                helper.verify_metrics(members, stored)
                metric_count += 1
    # Descriptive observed-channel lower bound, not a prediction model.
    within = 0.0
    means = {}
    for g in groups:
        values = [o["target_value"] for o in g["observations"]]
        weights = [o["weight"] for o in g["observations"]]
        mean = math.fsum(w * y for w, y in zip(weights, values)) / math.fsum(weights)
        means[g["response_group_id"]] = mean
        within += math.fsum(w * (y - mean) ** 2 for w, y in zip(weights, values))
    decomposition = []
    for (variant, phase), rows in all_rows.items():
        between = math.fsum(
            r["weight"] * (r["prediction_log10_ratio"] - means[r["response_group_id"]]) ** 2
            for r in rows
        )
        observed = next(
            r["data_loss"]
            for r in summary["metrics"]
            if r["variant"] == variant and r["phase"] == phase
        )
        assert abs(observed - between - within) < 1e-10
        decomposition.append(
            {
                "variant": variant,
                "phase": phase,
                "observed_mse": observed,
                "within_group_channel_mse": within,
                "between_group_mean_error": between,
            }
        )
    result = {
        "status": "passed",
        "original_held_out_used": False,
        "prediction_rows": sum(len(r) for r in all_rows.values()),
        "metric_records": metric_count,
        "maximum_prediction_error": maximum_error,
        "calibrations": calibrations,
        "parameter_checks": parameter_checks,
        "loss_decomposition": decomposition,
        "seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "independent source, split, train-only scaling/calibration, scalar final physical/profiled predictions and metrics; no optimizer replay or fresh biological test",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"parameter_checks", "loss_decomposition"}}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
