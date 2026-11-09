"""Reconstruct features, SVD fits, predictions and metrics independently."""

import bz2
import csv
import hashlib
import json
import resource
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PREVIOUS = ROOT / "experiments/2026-10-24_2018_codex_hap1-context-ablation"
sys.path.insert(0, str(PREVIOUS / "code"))
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


def encode(rows, variant, contexts, donor, scores):
    result = []
    for r in rows:
        s = r["peptide_sequence"]
        if variant == "calibrated_precursor":
            value = []
        elif variant == "length_affinity":
            value = [len(s), *scores[s]]
        else:
            composition = np.zeros(20)
            for residue in s:
                composition[AA.index(residue)] += 1 / len(s)
            first, last = np.zeros(20), np.zeros(20)
            first[AA.index(s[0])] = 1
            last[AA.index(s[-1])] = 1
            value = [len(s), *composition, *first, *last]
            if variant != "sequence":
                peptide = json.loads(donor[r["key"]])[0] if donor else s
                value.extend(scores[peptide])
            if variant == "sequence_affinity_geometry":
                value.extend(contexts[r["key"]][:4])
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
    score_frame = pd.read_parquet(EXP / "results/predictor/scores.parquet")
    alleles = ["HLA-A*02:01", "HLA-B*40:01", "HLA-C*03:04"]
    scores = {
        p: np.log10(g.set_index("allele").loc[alleles, "prediction"].to_numpy())
        for p, g in score_frame.groupby("peptide")
    }
    member_columns = [c for c in score_frame if c.startswith("model_pan_")]
    assert len(member_columns) == 11
    member_error = float(
        np.max(
            abs(
                np.log10(score_frame.prediction.to_numpy())
                - np.mean(np.log10(score_frame[member_columns].to_numpy()), axis=1)
            )
        )
    )
    assert member_error < 1e-10
    training_path = Path(
        "/Users/iskander/Library/Application Support/mhcflurry/4/2.3.0/models_class1_presentation/models/affinity_predictor_train_data.csv.bz2"
    )
    exposed = set()
    mass_spec = set()
    pairs = set()
    training_count = 0
    peptides = set(scores)
    with bz2.open(training_path, "rt") as stream:
        for r in csv.DictReader(stream):
            training_count += 1
            if r["peptide"] in peptides:
                exposed.add(r["peptide"])
                if r["measurement_kind"] == "mass_spec":
                    mass_spec.add(r["peptide"])
                if r["allele"] in alleles:
                    pairs.add((r["peptide"], r["allele"]))
    exposure = pd.read_parquet(EXP / "results/predictor/exposure.parquet")
    for r in exposure.to_dict("records"):
        assert r["any_training_peptide"] == (r["peptide"] in exposed)
        assert r["mass_spec_training_peptide"] == (r["peptide"] in mass_spec)
        for allele in alleles:
            assert r[allele] == ((r["peptide"], allele) in pairs)
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
    assert np.array_equal(
        dump["frozen_predictor_training_exposure"].to_numpy(),
        np.array([r["peptide_sequence"] in exposed for r in rows]),
    )
    y_all = np.array([r["target_log10_ratio"] for r in rows])
    assert np.array_equal(y_all, dump["target_log10_ratio"].to_numpy())
    worst_pred = worst_coef = worst_metric = 0.0
    n_pred = n_systems = 0
    donor_counts = {}
    rank_diagnostics = {}
    for variant, partitions in checkpoint.items():
        replay = {"train": np.full(len(rows), np.nan), "inner_cv": np.full(len(rows), np.nan)}
        signatures = {name: np.empty(len(rows), dtype="U64") for name in replay}
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
            a = encode(tr, variant, context, mapping["fit"], scores)
            b = encode(ev, variant, context, mapping["eval"], scores)
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
            population = "train" if part == "full" else "inner_cv"
            replay[population][ie] = prediction
            signatures[population][ie] = [
                hashlib.sha256(
                    part.encode() + cond.encode() + float(xi).hex().encode() + feature.tobytes()
                ).hexdigest()
                for cond, xi, feature in zip(ce, xe, ze, strict=True)
            ]
        for population, p in replay.items():
            original = dump[f"{variant}_{population}"].to_numpy()
            err = float(max(abs(p - original)))
            assert err < 1e-10
            worst_pred = max(worst_pred, err)
            n_pred += len(p)
            archived_metrics = metric(y_all, original, global_weight)
            replay_metrics = metric(y_all, p, global_weight)
            for name, value in archived_metrics.items():
                err = abs(value - summary["metrics"][variant][population][name])
                assert err < 1e-10, (variant, population, name, err)
                worst_metric = max(worst_metric, float(err))
                if name != "spearman":
                    assert abs(value - replay_metrics[name]) < 1e-10
            delta = abs(archived_metrics["spearman"] - replay_metrics["spearman"])
            if delta > 1e-10:
                groups = defaultdict(list)
                for i, signature in enumerate(signatures[population]):
                    groups[signature].append(i)
                canonical_original, canonical_replay = original.copy(), p.copy()
                original_spread = replay_spread = 0.0
                for indices in groups.values():
                    original_spread = max(original_spread, float(np.ptp(original[indices])))
                    replay_spread = max(replay_spread, float(np.ptp(p[indices])))
                    canonical_original[indices] = original[indices[0]]
                    canonical_replay[indices] = p[indices[0]]
                assert max(original_spread, replay_spread) < 1e-10
                original_rank = float(spearmanr(y_all, canonical_original).statistic)
                replay_rank = float(spearmanr(y_all, canonical_replay).statistic)
                assert abs(original_rank - replay_rank) < 1e-10, (
                    variant,
                    population,
                    original_rank,
                    replay_rank,
                )
                rank_diagnostics[f"{variant}:{population}"] = {
                    "raw_difference": delta,
                    "raw_archived_spearman": archived_metrics["spearman"],
                    "raw_replayed_spearman": replay_metrics["spearman"],
                    "canonical_archived_spearman": original_rank,
                    "canonical_replayed_spearman": replay_rank,
                    "max_identical_input_original_spread": original_spread,
                    "max_identical_input_replay_spread": replay_spread,
                    "n_input_groups": len(groups),
                }
                np.savez_compressed(
                    EXP / "results" / f"rank-ties-{variant}-{population}.npz",
                    original=original,
                    replay=p,
                    target=y_all,
                    input_group=signatures[population],
                )
        print(json.dumps({"variant": variant, "verified": True}), flush=True)
    receipt = {
        "status": "passed",
        "n_prediction_values": n_pred,
        "n_condition_systems": n_systems,
        "max_prediction_error": worst_pred,
        "max_coefficient_error": worst_coef,
        "max_metric_error": worst_metric,
        "donor_counts": donor_counts,
        "rank_diagnostics": rank_diagnostics,
        "n_predictor_training_rows": training_count,
        "n_exposed_hap1_peptides": len(exposed),
        "member_aggregation_error": member_error,
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
