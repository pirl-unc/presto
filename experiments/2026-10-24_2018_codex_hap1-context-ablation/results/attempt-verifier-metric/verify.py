"""Reconstruct features, SVD fits, predictions and metrics independently."""

import hashlib
import json
import resource
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
from independent_features import reconstruct

AA = "ACDEFGHIKLMNPQRSTVWY"


def weights(rows):
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, r in enumerate(rows):
        tree[r["condition_id"]][r["component_id"]][r["peptide_sequence"]].append(i)
    result = np.zeros(len(rows))
    for groups in tree.values():
        for peptides in groups.values():
            for indices in peptides.values():
                result[indices] = 1 / len(tree) / len(groups) / len(peptides) / len(indices)
    return result


def encode(rows, variant, contexts, donor):
    result = []
    for r in rows:
        s = r["peptide_sequence"]
        if variant == "calibrated_precursor":
            value = []
        elif variant == "length":
            value = [len(s)]
        elif variant.startswith("position"):
            value = [len(s)]
            positions = [0, 1, -1] if variant == "position_p1_p2_c" else [1, -1]
            for pos in positions:
                a = np.zeros(20)
                a[AA.index(s[pos])] = 1
                value.extend(a)
        else:
            composition = np.zeros(20)
            for residue in s:
                composition[AA.index(residue)] += 1 / len(s)
            first, last = np.zeros(20), np.zeros(20)
            first[AA.index(s[0])] = 1
            last[AA.index(s[-1])] = 1
            value = [len(s), *composition, *first, *last]
            k = r["key"]
            if variant == "sequence_geometry":
                value.extend(contexts[k][:4])
            elif variant == "sequence_flanks":
                value.extend(contexts[k][4:])
            elif variant in {"sequence_context", "sequence_shuffled_context"}:
                value.extend(contexts[donor.get(k, k)])
        result.append(value)
    return np.asarray(result)


def metric(y, p, weight):
    w = weight / sum(weight)
    error = np.subtract(p, y)
    mse = np.average(np.square(error), weights=w)
    var = np.average(np.square(y - np.average(y, weights=w)), weights=w)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": np.average(abs(error), weights=w),
        "r2": 1 - mse / var,
        "pearson": pearsonr(y, p).statistic,
        "spearman": spearmanr(y, p).statistic,
        "prediction_sd": np.sqrt(np.average(np.square(p - np.average(p, weights=w)), weights=w)),
        "target_sd": np.sqrt(var),
    }


def main():
    start = time.perf_counter()
    shutil.copyfile(__file__, EXP / "reproduce/source/verify.py")
    recipe = ROOT / "data/hap1_response_recipes/2026-10-24_0730_v1"
    out = EXP / "results/fits"
    rows = pq.read_table(
        recipe / "derived/responses.parquet",
        filters=[("split", "=", "train"), ("precursor_status", "=", "complete")],
    ).to_pylist()
    for r in rows:
        r["key"] = json.dumps(
            [r["peptide_sequence"], sorted(set(json.loads(r["parent_ids_json"])))],
            separators=(",", ":"),
        )
    keys = sorted({r["key"] for r in rows})
    parents = {
        r["parent_id"]: r["reference_sequence"]
        for r in pq.read_table(recipe / "derived/parents.parquet").to_pylist()
    }
    context, _ = reconstruct(keys, parents)
    checkpoint = json.loads((out / "checkpoints.json").read_text())
    donors = json.loads((out / "donor-mappings.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    dump = pq.read_table(out / "predictions.parquet")
    assert dump["measurement_id"].to_pylist() == [r["measurement_id"] for r in rows]
    folds = np.array(
        [
            int.from_bytes(
                hashlib.sha256(("hap1_baseline_inner_v1:" + r["component_id"]).encode()).digest(),
                "big",
            )
            % 5
            for r in rows
        ]
    )
    assert np.array_equal(folds, dump["fold"].to_numpy())
    global_weight = weights(rows)
    assert max(abs(global_weight - dump["weight"].to_numpy())) < 1e-14
    y_all = np.array([r["target_log10_ratio"] for r in rows])
    assert np.array_equal(y_all, dump["target_log10_ratio"].to_numpy())
    worst_pred = worst_coef = worst_metric = 0.0
    n_pred = n_systems = 0
    donor_counts = {}
    for variant, partitions in checkpoint.items():
        replay = {"train": np.full(len(rows), np.nan), "inner_cv": np.full(len(rows), np.nan)}
        for part, cp in partitions.items():
            it = np.arange(len(rows)) if part == "full" else np.where(folds != int(part))[0]
            ie = np.arange(len(rows)) if part == "full" else np.where(folds == int(part))[0]
            tr, ev = [rows[i] for i in it], [rows[i] for i in ie]
            if part != "full":
                assert {r["component_id"] for r in tr}.isdisjoint({r["component_id"] for r in ev})
            mapping = donors[variant].get(part, {"fit": {}, "eval": {}})
            for phase, population in [("fit", tr), ("eval", ev)]:
                d = mapping[phase]
                if not d:
                    continue
                groups = defaultdict(list)
                components = {r["key"]: r["component_id"] for r in population}
                for k in {r["key"] for r in population}:
                    groups[len(json.loads(k)[0])].append(k)
                assert set(d) == set(components) == set(d.values())
                for group in groups.values():
                    ordered = sorted(
                        group,
                        key=lambda k: hashlib.sha256(
                            ("hap1_context_shuffle_v1:" + k).encode()
                        ).digest(),
                    )
                    for i, k in enumerate(ordered):
                        assert d[k] == ordered[(i + 1) % len(ordered)]
                donor_counts[f"{part}:{phase}"] = {
                    "n_keys": len(d),
                    "unchanged": sum(k == v for k, v in d.items()),
                    "same_component": sum(components[k] == components[v] for k, v in d.items()),
                }
            a = encode(tr, variant, context, mapping["fit"])
            b = encode(ev, variant, context, mapping["eval"])
            w = weights(tr)
            mean = np.average(a, axis=0, weights=w)
            sd = np.sqrt(np.average((a - mean) ** 2, axis=0, weights=w))
            mask = sd > 1e-12
            assert np.array_equal(mask, cp["mask"])
            assert np.max(abs(mean - np.array(cp["mean"])), initial=0) < 1e-10
            assert np.max(abs(sd - np.array(cp["sd"])), initial=0) < 1e-10
            z, ze = (a[:, mask] - mean[mask]) / sd[mask], (b[:, mask] - mean[mask]) / sd[mask]
            x = np.array([r["precursor_log10_ratio"] for r in tr])
            y = np.array([r["target_log10_ratio"] for r in tr])
            xe = np.array([r["precursor_log10_ratio"] for r in ev])
            c = np.array([r["condition_id"] for r in tr])
            ce = np.array([r["condition_id"] for r in ev])
            xr, yr = x.copy(), y.copy()
            for cond in cp["offsets"]:
                idx = c == cond
                xr[idx] -= np.average(x[idx], weights=w[idx])
                yr[idx] -= np.average(y[idx], weights=w[idx])
            gamma = max(0.0, np.average(xr * yr, weights=w) / np.average(xr * xr, weights=w))
            assert abs(gamma - cp["gamma"]) < 1e-10
            prediction = np.empty(len(ev))
            for cond in cp["offsets"]:
                idx, ide = c == cond, ce == cond
                wc = w[idx] / sum(w[idx])
                offset = np.average(y[idx] - gamma * x[idx], weights=wc)
                center = np.average(z[idx], axis=0, weights=wc)
                centered = z[idx] - center
                response = y[idx] - gamma * x[idx] - offset
                u, s, vt = np.linalg.svd(np.sqrt(wc[:, None]) * centered, full_matrices=False)
                beta = vt.T @ (s / (s * s + 0.1) * (u.T @ (np.sqrt(wc) * response)))
                coef_error = np.max(
                    abs(beta - np.array(cp["coefficients"][cond]["beta"])), initial=0
                )
                worst_coef = max(worst_coef, float(coef_error))
                assert coef_error < 1e-10
                assert abs(offset - cp["offsets"][cond]) < 1e-10
                assert (
                    np.max(abs(center - np.array(cp["coefficients"][cond]["center"])), initial=0)
                    < 1e-10
                )
                prediction[ide] = offset + gamma * xe[ide] + (ze[ide] - center) @ beta
                n_systems += 1
            replay["train" if part == "full" else "inner_cv"][ie] = prediction
        for population, p in replay.items():
            original = dump[f"{variant}_{population}"].to_numpy()
            err = float(max(abs(p - original)))
            assert err < 1e-10
            worst_pred = max(worst_pred, err)
            n_pred += len(p)
            for name, value in metric(y_all, p, global_weight).items():
                err = abs(value - summary["metrics"][variant][population][name])
                assert err < 1e-10
                worst_metric = max(worst_metric, float(err))
        print(json.dumps({"variant": variant, "verified": True}), flush=True)
    receipt = {
        "status": "passed",
        "n_prediction_values": n_pred,
        "n_condition_systems": n_systems,
        "max_prediction_error": worst_pred,
        "max_coefficient_error": worst_coef,
        "max_metric_error": worst_metric,
        "donor_counts": donor_counts,
        "seconds": time.perf_counter() - start,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "argv": sys.argv,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outer_validation_test_scored": False,
    }
    (EXP / "results/independent-verification.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
