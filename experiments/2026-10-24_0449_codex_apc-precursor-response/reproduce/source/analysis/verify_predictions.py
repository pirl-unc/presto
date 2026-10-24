"""Independent scalar physical reconstruction and source-linked metric closure."""

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
from scipy.stats import pearsonr, spearmanr

FAMILY = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def amount(internal, left, right):
    a = 1 + internal
    if left and right:
        return (
            left
            * right
            * (2 * a + left + right)
            / (a * (a + left) * (a + right) * (a + left + right))
        )
    if left or right:
        b = left or right
        return b / (a * (a + b))
    return 1 / a


def coordinate(raw, transforms, prefix):
    keep = transforms[prefix + "keep"]
    z = (raw[keep] - transforms[prefix + "center"][keep]) / transforms[prefix + "scale"]
    return np.r_[1.0, z @ transforms[prefix + "projection"]]


def verify_metrics(rows, stored):
    p, y, w = (
        np.array([r[key] for r in rows])
        for key in ("prediction_log10_ratio", "target_log10_ratio", "weight")
    )
    wsum = w.sum()
    w = w / wsum
    residual = p - y
    mse = float(w @ residual**2)
    variance = float(w @ (y - w @ y) ** 2)
    variable = len(y) > 1 and p.min() != p.max() and y.min() != y.max()
    directed = y != 0
    expected = {
        "n_observations": len(rows),
        "n_response_groups": len({r["response_group_id"] for r in rows}),
        "n_components": len({r["component_id"] for r in rows}),
        "weight_sum": float(wsum),
        "data_loss": mse,
        "n_exact_observations": len(rows),
        "n_bounded_observations": 0,
        "prediction_sd_pooled": float(p.std()),
        "prediction_min": float(p.min()),
        "prediction_max": float(p.max()),
        "exact_rmse": math.sqrt(mse),
        "exact_mae": float(w @ abs(residual)),
        "exact_r2": 1 - mse / variance if variance else None,
        "exact_pearson_pooled": float(pearsonr(p, y).statistic) if variable else None,
        "exact_spearman_pooled": float(spearmanr(p, y).statistic) if variable else None,
        "exact_prediction_sd_pooled": float(p.std()),
        "exact_target_sd_pooled": float(y.std()),
        "exact_direction_accuracy": float(
            w[directed] @ (np.sign(p[directed]) == np.sign(y[directed])) / w[directed].sum()
        )
        if directed.any()
        else None,
    }
    for key, value in expected.items():
        actual = stored[key]
        if value is None:
            assert actual is None, (key, actual, value)
        else:
            assert math.isclose(actual, value, abs_tol=1e-10, rel_tol=1e-10), (key, actual, value)


def main():
    start = time.perf_counter()
    output = FAMILY / "results/independent-prediction-verification.json"
    assert not output.exists()
    execution = read(FAMILY / "reproduce/model-execution.json")
    for path, digest in execution["source_sha256"].items():
        assert sha(path) == digest
    sealed = read(FAMILY / "reproduce/pre-evaluation-receipt.json")
    assert sha(FAMILY / "reproduce/model-execution.json") == sealed["model_execution_sha256"]
    for path, digest in sealed["checkpoints_and_constants"].items():
        assert sha(FAMILY / path) == digest
    cohort = read(FAMILY / "results/cohort.json")
    assert read(FAMILY / "results/cohort-verification.json")["status"] == "passed"
    root = Path(cohort["model_input_directory"])
    for path, digest in cohort["files"].items():
        assert sha(root / path) == digest
    groups, parents = read(root / "groups.json.gz"), read(root / "parents.json.gz")
    by_id = {o["measurement_id"]: (g, o) for g in groups for o in g["observations"]}
    assert len(by_id) == sum(len(g["observations"]) for g in groups)
    transforms = np.load(FAMILY / "results/feature-transforms.npz")
    coverage = read(FAMILY / "results/feature-coverage.json")
    cx = {
        c: coordinate(
            np.array([float(c == x) for x in coverage["conditions"]]), transforms, "condition_"
        )
        for c in coverage["conditions"]
    }
    nc, rank = coverage["n_condition_coordinates"], coverage["n_bond_coordinates"]
    settings = read(FAMILY / "reproduce/model-config.json")
    summary = read(FAMILY / "results/evaluation/summary.json")
    strata = read(FAMILY / "results/evaluation/stratified-metrics.json")
    count, metric_count, max_error, checks = 0, 0, 0.0, []
    for path in sorted((FAMILY / "results/evaluation").glob("*-predictions.jsonl.gz")):
        variant = path.name.removesuffix("-predictions.jsonl.gz")
        baseline = variant in settings["baselines"]
        directory = FAMILY / "results" / ("baselines" if baseline else "arms")
        theta = np.load(
            directory / (variant + ".npz") if baseline else directory / variant / "fit.npz"
        )["theta"]
        rates = {}
        if not baseline:
            cfg = settings["arms"][variant]
            nr = 0 if cfg["kind"] == "prior" else 1 + (rank if cfg["kind"] == "sequence" else 0)
            coef = np.zeros(40)
            if cfg["kind"] == "prior":
                prior = Path(read(FAMILY / "reproduce/launch.json")["prior_family"])
                tr = np.load(prior / "results/feature-transforms.npz")
                old = np.load(prior / "results/arms/sequence_exact/fit.npz")["theta"]
                n = tr["bond_projection"].shape[1]
                coef[tr["bond_keep"]] = (tr["bond_projection"] @ old[1 : 1 + n]) / tr["bond_scale"]
                intercept = old[0] - tr["bond_center"] @ coef
            else:
                if cfg["kind"] == "sequence":
                    coef[transforms["bond_keep"]] = (
                        transforms["bond_projection"] @ theta[1:nr]
                    ) / transforms["bond_scale"]
                intercept = theta[0] - transforms["bond_center"] @ coef
            for parent, sequence in parents.items():
                aa = np.array([AA.index(a) for a in sequence])
                rates[parent] = np.exp(intercept + coef[aa[:-1]] + coef[20 + aa[1:]])
        expected = {}
        for g in groups:
            x = cx[g["condition_id"]]
            if baseline:
                pp = {o["parent_sequence_sha256"]: o["precursor_log10_ratio"] for o in g["origins"]}
                precursor = (
                    math.log10(math.fsum(10**v for v in pp.values()) / len(pp))
                    if "precursor" in variant
                    else 0.0
                )
                expected[g["response_group_id"]] = (float(x @ theta + precursor),)
                continue
            scale, offset = math.exp(float(x @ theta[nr : nr + nc])), float(x @ theta[nr + nc :])
            control, treated = [], []
            for o in g["origins"]:
                parent, a, b = o["parent_sequence_sha256"], o["start0"], o["end0_exclusive"]
                assert parents[parent][a:b] == g["peptide_sequence"]
                r = rates[parent]
                internal = math.fsum(r[a : b - 1])
                left, right = (
                    float(r[a - 1]) if a else 0.0,
                    float(r[b - 1]) if b < len(parents[parent]) else 0.0,
                )
                control.append(amount(internal, left, right))
                treated.append(
                    amount(internal * scale, left * scale, right * scale)
                    * (10 ** o["precursor_log10_ratio"] if cfg["precursor"] else 1)
                )
            nparents = len({o["parent_sequence_sha256"] for o in g["origins"]})
            c, t = math.fsum(control) / nparents, math.fsum(treated) / nparents
            expected[g["response_group_id"]] = (
                offset + math.log10(t / c),
                math.log(c),
                math.log(t),
            )
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == len(by_id) and {r["observation_id"] for r in rows} == by_id.keys()
        local = 0.0
        for row in rows:
            g, obs = by_id[row["observation_id"]]
            for key, value in obs.items():
                assert row[key] == value, (key, row[key], value)
            for key in (
                "response_group_id",
                "source_peptide_row_id",
                "condition_id",
                "component_id",
                "split",
                "peptide_sequence",
                "precursor_only_log10_ratio",
            ):
                assert row[key] == g[key]
            assert row["origin_inputs"] == g["origins"]
            assert row["target_log10_ratio"] == obs["target_value"] and row["inequality"] is None
            assert row["n_candidate_parent_sequences"] == len(
                {o["parent_sequence_sha256"] for o in g["origins"]}
            )
            assert row["n_occurrences"] == len(g["origins"])
            values = expected[g["response_group_id"]]
            for key, value in zip(
                (
                    "prediction_log10_ratio",
                    "log_control_amount_per_unit_input",
                    "log_treated_amount_per_unit_input",
                ),
                values,
            ):
                err = abs(row[key] - value)
                assert err < 1e-9, (variant, key, err)
                local = max(local, err)
        for split in ("train", "val", "test"):
            chosen = [r for r in rows if r["split"] == split]
            known = next(
                m for m in summary["metrics"] if m["variant"] == variant and m["split"] == split
            )
            verify_metrics(chosen, known)
            metric_count += 1
            for field in ("condition_id", "component_id", "multiple_parent_candidates"):
                bins = defaultdict(list)
                for row in chosen:
                    value = (
                        row["n_candidate_parent_sequences"] > 1
                        if field == "multiple_parent_candidates"
                        else row[field]
                    )
                    bins[str(value)].append(row)
                stored = {
                    r["stratum_id"]: r
                    for r in strata
                    if r["variant"] == variant and r["split"] == split and r["stratum"] == field
                }
                assert stored.keys() == bins.keys()
                for value, members in bins.items():
                    verify_metrics(members, stored[value])
                    metric_count += 1
        count += len(rows)
        max_error = max(max_error, local)
        checks.append(
            {"variant": variant, "n_prediction_rows": len(rows), "maximum_scalar_error": local}
        )
    result = {
        "status": "passed",
        "n_prediction_rows": count,
        "n_metric_records": metric_count,
        "maximum_prediction_error": max_error,
        "checks": checks,
        "seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "source-linked predictions, scalar final physical arithmetic and independent metrics; no optimizer replay or external biological validation",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
