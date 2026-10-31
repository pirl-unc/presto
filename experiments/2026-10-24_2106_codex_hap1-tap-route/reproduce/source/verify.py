"""Independent source/features, SVD regression and analytic route replay."""

import hashlib
import json
import resource
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
RECIPE = ROOT / "data/hap1_response_recipes/2026-10-24_0730_v1"
PREDICTOR = ROOT / "experiments/2026-10-24_2041_codex_hap1-affinity-control"
AA = "ACDEFGHIKLMNPQRSTVWY"
CONDS = ["hap1_tap1_ko", "hap1_tap2_ko"]


def weights(rows):
    f = pd.DataFrame(rows)
    distinct = f[["condition_id", "component_id", "peptide_sequence"]].drop_duplicates()
    ng = distinct.groupby("condition_id").component_id.nunique().to_dict()
    ns = distinct.groupby(["condition_id", "component_id"]).size().to_dict()
    nr = f.groupby(["condition_id", "component_id", "peptide_sequence"]).size().to_dict()
    return np.array(
        [
            1
            / len(ng)
            / ng[r["condition_id"]]
            / ns[r["condition_id"], r["component_id"]]
            / nr[r["condition_id"], r["component_id"], r["peptide_sequence"]]
            for r in rows
        ]
    )


def metrics(y, p, w):
    w = w / w.sum()
    mse = np.sum(w * (y - p) ** 2)
    var = np.sum(w * (y - np.sum(w * y)) ** 2)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.sum(w * abs(y - p))),
        "r2": float(1 - mse / var),
        "pearson": float(pearsonr(y, p).statistic),
        "spearman": float(spearmanr(y, p).statistic),
        "prediction_sd": float(np.sqrt(np.sum(w * (p - np.sum(w * p)) ** 2))),
        "target_sd": float(np.sqrt(var)),
        "n_observations": len(y),
    }


def main():
    started = time.perf_counter()
    receipt = json.loads((EXP / "reproduce/launch.json").read_text())
    for name, digest in receipt["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    original = pq.read_table(
        RECIPE / "derived/responses.parquet",
        filters=[
            ("split", "=", "train"),
            ("precursor_status", "=", "complete"),
            ("condition_id", "in", CONDS),
        ],
    )
    archived = pq.read_table(EXP / "results/fits/cohort.parquet")
    assert original.equals(archived)
    rows = original.to_pylist()
    w = weights(rows)
    y = np.array([r["target_log10_ratio"] for r in rows])
    x = np.array([r["precursor_log10_ratio"] for r in rows])
    cond = np.array([CONDS.index(r["condition_id"]) for r in rows])
    folds = np.array(
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
    parents = {
        r["parent_id"]: r["reference_sequence"]
        for r in pq.read_table(RECIPE / "derived/parents.parquet").to_pylist()
    }
    feature_lookup = {}
    for r in rows:
        seq = r["peptide_sequence"]
        ids = tuple(sorted(set(json.loads(r["parent_ids_json"]))))
        if (seq, ids) in feature_lookup:
            continue
        geometry = np.zeros(4)
        for pid in ids:
            parent = parents[pid]
            positions = [
                i for i in range(len(parent) - len(seq) + 1) if parent[i : i + len(seq)] == seq
            ]
            assert positions
            for i in positions:
                tail = len(parent) - i - len(seq)
                geometry += (
                    np.array([np.log(1 + i), np.log(1 + tail), i == 0, tail == 0])
                    / len(ids)
                    / len(positions)
                )
        one = lambda symbol: [float(symbol == a) for a in AA]
        feature_lookup[seq, ids] = [
            len(seq),
            *[sum(ch == a for ch in seq) / len(seq) for a in AA],
            *one(seq[0]),
            *one(seq[-1]),
            *one(seq[1]),
            *one(seq[2]),
            *geometry,
        ]
    phi = np.array(
        [
            feature_lookup[
                r["peptide_sequence"], tuple(sorted(set(json.loads(r["parent_ids_json"]))))
            ]
            for r in rows
        ]
    )
    scores = pd.read_parquet(PREDICTOR / "results/predictor/scores.parquet")
    pivot = scores.pivot(index="peptide", columns="allele", values="prediction")
    score = np.log10(
        pivot.loc[
            [r["peptide_sequence"] for r in rows], ["HLA-A*02:01", "HLA-B*40:01", "HLA-C*03:04"]
        ].to_numpy()
    )
    a = np.load(EXP / "results/fits/features.npz")
    feature_error = float(np.max(abs(a["availability"] - phi)))
    assert feature_error < 1e-12 and np.max(abs(a["scores"] - score)) < 1e-12
    assert np.array_equal(a["fold"], folds) and np.array_equal(a["condition"], cond)
    assert np.max(abs(a["weights"] - w)) < 1e-15
    predictions = pq.read_table(EXP / "results/fits/predictions.parquet").to_pandas()
    assert predictions.measurement_id.tolist() == original["measurement_id"].to_pylist()
    exposure = pd.read_parquet(PREDICTOR / "results/predictor/exposure.parquet").set_index(
        "peptide"
    )
    assert np.array_equal(
        predictions.frozen_predictor_training_exposure,
        exposure.loc[predictions.peptide_sequence, "any_training_peptide"],
    )
    summary = json.loads((EXP / "results/fits/summary.json").read_text())
    variants = list(summary["metrics"])
    errors = Counter()
    replay = {}
    audits = {}
    for variant in variants:
        full = np.full(len(rows), np.nan)
        cv = full.copy()
        for part in ["full", "0", "1", "2", "3", "4"]:
            it = np.arange(len(rows)) if part == "full" else np.flatnonzero(folds != int(part))
            ie = np.arange(len(rows)) if part == "full" else np.flatnonzero(folds == int(part))
            fit = json.loads((EXP / "results/fits" / variant / part / "fit.json").read_text())
            cp = fit["checkpoint"]
            wt = weights([rows[i] for i in it])
            c = cond[it]
            assert np.array_equal(cp.get("fit_indices", fit.get("fit_indices")), it)
            assert np.array_equal(cp.get("eval_indices", fit.get("eval_indices")), ie)
            ym = np.array([np.sum(wt[c == k] * y[it][c == k]) / wt[c == k].sum() for k in range(2)])
            xm = np.array([np.sum(wt[c == k] * x[it][c == k]) / wt[c == k].sum() for k in range(2)])
            yr = y[it] - ym[c]
            xr = x[it] - xm[c]
            gamma = max(0.0, float(np.sum(wt * xr * yr) / np.sum(wt * xr * xr)))
            assert abs(cp["gamma"] - gamma) < 1e-12
            if variant.endswith("ridge") or variant == "sequence_scores":
                f = np.c_[phi[:, :61] if variant == "sequence_scores" else phi, score]
                mu = wt @ f[it]
                sd = np.sqrt(wt @ (f[it] - mu) ** 2)
                mask = sd > 1e-12
                zz = (f[:, mask] - mu[mask]) / sd[mask]
                p = gamma * x[ie] + (ym - gamma * xm)[cond[ie]]
                for k, name in enumerate(CONDS):
                    m = c == k
                    we = wt[m] / wt[m].sum()
                    center = we @ zz[it][m]
                    design = zz[it][m] - center
                    residual = yr[m] - gamma * xr[m]
                    matrix = np.vstack(
                        [np.sqrt(we)[:, None] * design, np.sqrt(0.1) * np.eye(mask.sum())]
                    )
                    target = np.r_[np.sqrt(we) * residual, np.zeros(mask.sum())]
                    beta = np.linalg.lstsq(matrix, target, rcond=None)[0]
                    errors["coefficient"] = max(
                        errors["coefficient"],
                        float(np.max(abs(beta - np.array(cp["coefficients"][name]["beta"])))),
                    )
                    e = cond[ie] == k
                    p[e] += (zz[ie][e] - center) @ beta
            else:
                mu = wt @ phi[it]
                sd = np.sqrt(wt @ (phi[it] - mu) ** 2)
                mask = sd > 1e-12
                z = (phi[:, mask] - mu[mask]) / sd[mask]
                sm = wt @ score[it]
                ss = np.sqrt(wt @ (score[it] - sm) ** 2)
                sz = (score - sm) / ss
                for field, expected in [("mean", mu), ("sd", sd), ("mask", mask)]:
                    assert (
                        np.max(
                            abs(
                                np.array(cp["availability_transform"][field], float)
                                - expected.astype(float)
                            )
                        )
                        < 1e-12
                    )
                checkpoints = json.loads(
                    (EXP / "results/fits" / variant / part / "checkpoints.json").read_text()
                )
                for step, cpoint in checkpoints.items():
                    theta, eta, beta = (np.array(cpoint[k]) for k in ["theta", "eta", "beta"])
                    rho = expit(theta[0] + z @ theta[1:])
                    eps = expit(eta)[cond]
                    rr = 1 - rho + rho * eps
                    assert np.all((rr > 0) & (rr <= 1))
                    raw = gamma * x + np.log10(rr) + np.sum(sz * beta[cond], axis=1)
                    offsets = np.array(
                        [
                            np.sum(wt[c == k] * (y[it] - raw[it])[c == k]) / wt[c == k].sum()
                            for k in range(2)
                        ]
                    )
                    pred = raw + offsets[cond]
                    err = float(np.max(abs(offsets - np.array(cpoint["offsets"]))))
                    errors["offset"] = max(errors["offset"], err)
                    res = pred[it] - y[it]
                    dz = -(1 - eps[it]) * rho[it] * (1 - rho[it]) / rr[it] / np.log(10)
                    de = eps[it] * rho[it] * (1 - eps[it]) / rr[it] / np.log(10)
                    gt = np.c_[np.ones(len(it)), z[it]].T @ (2 * wt * res * dz) + 0.02 * theta
                    ge = np.array(
                        [np.sum(2 * wt[c == k] * res[c == k] * de[c == k]) for k in range(2)]
                    )
                    ge += 0.01 * (eta - np.log(0.05 / 0.95))
                    parts = [gt, ge]
                    if variant != "route_only_exact":
                        gb = (
                            np.stack(
                                [
                                    sz[it][c == k].T @ (2 * wt[c == k] * res[c == k])
                                    for k in range(2)
                                ]
                            )
                            + 0.1 * beta
                        )
                        parts.append(gb.ravel())
                    gradient = np.concatenate(parts)
                    errors["gradient"] = max(
                        errors["gradient"],
                        float(np.max(abs(gradient - np.array(cpoint["analytic_gradient"])))),
                    )
                    obj = np.sum(wt * res**2) + 0.01 * (
                        np.sum(theta**2) + np.mean((eta - np.log(0.05 / 0.95)) ** 2)
                    )
                    obj += 0.1 * np.mean(np.sum(beta**2, axis=1))
                    errors["objective"] = max(
                        errors["objective"], abs(float(obj) - cpoint["objective"])
                    )
                    assert errors["gradient"] < 1e-9 and errors["objective"] < 1e-10
                    record = next(h for h in fit["history"] if h["step"] == int(step))
                    reconstructed = metrics(y[it], pred[it], wt)
                    for k, v in reconstructed.items():
                        if k != "spearman":
                            errors["history_metric"] = max(
                                errors["history_metric"], abs(v - record["train"][k])
                            )
                    errors["history_prediction_values"] += len(it)
                    if step == "250":
                        p = pred[ie]
                        prefix = f"{variant}_{'train' if part == 'full' else 'inner_cv'}_"
                        for field, values in [
                            ("rho", rho),
                            ("epsilon", eps),
                            ("relative_delivery", rr),
                            ("transport_log10", np.log10(rr)),
                            ("score_term", np.sum(sz * beta[cond], axis=1)),
                        ]:
                            errors["latent"] = max(
                                errors["latent"],
                                float(
                                    np.max(
                                        abs(predictions[prefix + field].to_numpy()[ie] - values[ie])
                                    )
                                ),
                            )
                        if part == "full":
                            audits[variant] = {
                                "epsilon": expit(eta).tolist(),
                                "rho_quantiles": np.quantile(rho, [0, 0.1, 0.5, 0.9, 1]).tolist(),
                                "delivery_quantiles": np.quantile(
                                    rr, [0, 0.1, 0.5, 0.9, 1]
                                ).tolist(),
                                "parameter_count": cpoint["supported_parameter_count"],
                                "gradient_touched_count": cpoint["gradient_touched_count"],
                                "jacobian": cp["profiled_prediction_jacobian"],
                                "last_50_step_relative_objective_change": fit[
                                    "last_50_step_relative_objective_change"
                                ],
                            }
            (full if part == "full" else cv)[ie] = p
            errors["prediction_values"] += len(ie)
            errors["fits"] += 1
        replay[variant] = {"train": full, "inner_cv": cv}
        for population, p in replay[variant].items():
            archived = predictions[f"{variant}_{population}"].to_numpy()
            errors["prediction"] = max(errors["prediction"], float(np.max(abs(archived - p))))
            actual = metrics(y, archived, w)
            for k, v in actual.items():
                errors["archived_metric"] = max(
                    errors["archived_metric"], abs(v - summary["metrics"][variant][population][k])
                )
        print(json.dumps({"variant": variant, "independently_verified": True}), flush=True)
    strata = json.loads((EXP / "results/fits/strata.json").read_text())
    for s in strata:
        kind = s["kind"]
        population = "train" if kind.startswith("train") else "inner_cv"
        if kind == "fold":
            m = folds == int(s["label"])
        elif kind.endswith("condition"):
            m = cond == int(s["label"])
        else:
            m = predictions.frozen_predictor_training_exposure.to_numpy() == (s["label"] == "True")
        actual = metrics(y[m], predictions[f"{s['variant']}_{population}"].to_numpy()[m], w[m])
        for k, v in actual.items():
            errors["stratum_metric"] = max(errors["stratum_metric"], abs(v - s[k]))
    for key in [
        "prediction",
        "latent",
        "coefficient",
        "offset",
        "objective",
        "archived_metric",
        "stratum_metric",
        "history_metric",
    ]:
        assert errors[key] < 1e-9, (key, errors[key])

    def rmse(v):
        return summary["metrics"][v]["inner_cv"]["rmse"]

    gain = 1 - rmse("route_scores_exact") / rmse("sequence_scores")
    rich = 1 - rmse("route_scores_exact") / rmse("expanded_ridge")
    sf = {(s["variant"], s["label"]): s["rmse"] for s in strata if s["kind"] == "fold"}
    improved = sum(
        sf["route_scores_exact", str(f)] < sf["expanded_ridge", str(f)] for f in range(5)
    )
    gate = gain >= 0.02 and rich >= 0.01 and improved >= 4
    assert gate == summary["development_gate_pass"]
    result = {
        "status": "passed",
        "errors_and_counts": dict(errors),
        "feature_error": feature_error,
        "parameter_audits": audits,
        "development_gate_pass": gate,
        "n_strata": len(strata),
        "seconds": time.perf_counter() - started,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "outer_validation_test_scored": False,
        "rank_scope": "Archived-value Spearman replay; continuous history metrics independently replayed. No raw rank tolerance changed.",
    }
    (EXP / "results/independent-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.copyfile(__file__, EXP / "reproduce/source/verify.py")
    print(json.dumps({k: v for k, v in result.items() if k != "parameter_audits"}), flush=True)


if __name__ == "__main__":
    main()
