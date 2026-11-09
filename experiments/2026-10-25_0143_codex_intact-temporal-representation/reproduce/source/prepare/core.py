"""Explicit fold-local representations of measured temporal changes."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

import frozen_model as physical

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def configuration():
    return json.loads((ROOT / "reproduce/launch.json").read_text())


def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def hash_fold(sequence, salt, n_folds):
    return int(hashlib.sha256((salt + sequence).encode()).hexdigest(), 16) % n_folds


def weights(rows):
    times = defaultdict(set)
    counts = Counter((r["peptide_sequence"], r["time_minutes"]) for r in rows)
    for row in rows:
        times[row["peptide_sequence"]].add(row["time_minutes"])
    assert rows and times
    result = np.array([1 / (len(times) * len(times[r["peptide_sequence"]]) *
                           counts[r["peptide_sequence"], r["time_minutes"]]) for r in rows])
    assert abs(result.sum() - 1) < 1e-12
    return result


def source_changes():
    config = configuration()
    prior = Path(config["prior_family"])
    covariates = pq.read_table(prior / "results/data/train/covariates.parquet").to_pylist()
    labels = pq.read_table(prior / "results/data/train/outcomes.parquet").to_pylist()
    assert [r["assignment_id"] for r in covariates] == [r["assignment_id"] for r in labels]
    grouped = defaultdict(list)
    for row, label in zip(covariates, labels):
        assert row["time_minutes"] <= 80
        grouped[row["peptide_sequence"]].append({**row, "source_target": label["target"]})
    changes, baselines = [], []
    for sequence, records in sorted(grouped.items()):
        first = min(r["time_minutes"] for r in records)
        base = [r for r in records if r["time_minutes"] == first]
        baseline = float(np.mean([r["source_target"] for r in base]))
        ids = [r["assignment_id"] for r in base]
        later = [r for r in records if r["time_minutes"] > first]
        assert later
        baselines.append({"peptide_sequence": sequence, "baseline_time": first,
                          "baseline_target": baseline, "baseline_assignment_ids": ids,
                          "baseline_source_targets": [r["source_target"] for r in base]})
        for row in later:
            changes.append({**row, "source_weight": row["weight"], "baseline_time": first,
                            "baseline_target": baseline, "baseline_assignment_ids": ids,
                            "target": row["source_target"] - baseline,
                            "elapsed": (row["time_minutes"] - first) / 80,
                            "outer_fold": hash_fold(sequence, config["outer_salt"], 5),
                            "inner_fold": hash_fold(sequence, config["inner_salt"], 3)})
    for row, weight in zip(changes, weights(changes)):
        row["weight"] = float(weight)
    return changes, baselines


def load_changes():
    return pq.read_table(ROOT / "results/changes.parquet").to_pylist()


def sequences(rows):
    return sorted({r["peptide_sequence"] for r in rows})


def raw_features(sequence, representation):
    assert sequence and set(sequence) <= set(AA)
    features = [math.log(len(sequence))]
    if representation == "sequence":
        features += [sequence.count(a) / len(sequence) for a in AA]
        features += [float(sequence[0] == a) for a in AA]
        features += [float(sequence[-1] == a) for a in AA]
    else:
        assert representation == "length"
    return features


def transform_fit(rows, representation):
    seq = sequences(rows)
    x = np.array([raw_features(s, representation) for s in seq])
    center = x.mean(0)
    scale = np.sqrt(((x - center) ** 2).mean(0))
    constant = np.ptp(x, axis=0) == 0
    scale = np.where((scale > 0) & ~constant, scale, 1.)
    z = (x - center) / scale
    z[:, constant] = 0.
    eigenvalues, eigenvectors = np.linalg.eigh(z.T @ z / len(seq))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    retained = eigenvalues > max(float(eigenvalues[0]), 1.) * 1e-10
    return {"representation": representation, "training_sequences": seq,
            "center": center.tolist(), "scale": scale.tolist(), "constant_columns": constant.tolist(),
            "basis": eigenvectors[:, retained].tolist(), "eigenvalues": eigenvalues.tolist(),
            "retained": retained.tolist(), "rank": int(retained.sum())}


def design(rows, transform):
    x = np.array([raw_features(r["peptide_sequence"], transform["representation"]) for r in rows])
    standardized = (x - np.array(transform["center"])) / np.array(transform["scale"])
    standardized[:, np.array(transform["constant_columns"])] = 0.
    z = standardized @ np.array(transform["basis"])
    return np.array([r["elapsed"] for r in rows])[:, None] * np.column_stack([np.ones(len(rows)), z])


def ridge_fit(rows, representation, alpha):
    transform = transform_fit(rows, representation)
    a = design(rows, transform)
    y = np.array([r["target"] for r in rows])
    w = weights(rows)
    penalty = np.ones(a.shape[1]) * (alpha / transform["rank"] if transform["rank"] else 0.)
    penalty[0] = 0.
    normal = (a.T * w) @ a + np.diag(penalty)
    rhs = a.T @ (w * y)
    beta = np.linalg.solve(normal, rhs)
    residual = float(np.max(np.abs(normal @ beta - rhs)))
    assert residual <= 1e-9
    return {"transform": transform, "coefficients": beta.tolist(), "alpha": alpha,
            "normal_equation_max_error": residual, "all_coefficients_nonzero": bool(np.all(beta != 0)),
            "mse": float(w @ ((a @ beta - y) ** 2)), "penalty": float(penalty @ beta ** 2)}


def ridge_predict(rows, fitted):
    return design(rows, fitted["transform"]) @ np.array(fitted["coefficients"])


def common_fit(rows):
    x = np.array([r["elapsed"] for r in rows])
    y = np.array([r["target"] for r in rows])
    w = weights(rows)
    return float(np.sum(w * x * y) / np.sum(w * x * x))


def physical_batch(rows):
    parent = rows[0]["parent_sequence"]
    assert all(r["parent_sequence"] == parent for r in rows)
    amino_acids = sorted(set(parent[:-1]))
    index = torch.tensor([amino_acids.index(a) for a in parent[:-1]])
    batch = {"start": torch.tensor([r["start_1based"] - 1 for r in rows]),
             "end": torch.tensor([r["end_1based"] for r in rows]),
             "time": torch.tensor([r["time_minutes"] for r in rows], dtype=torch.float64),
             "baseline_time": torch.tensor([r["baseline_time"] for r in rows], dtype=torch.float64),
             "weight": torch.tensor(weights(rows)), "target": torch.tensor([r["target"] for r in rows])}
    return batch, index, amino_acids


def physical_change(theta, kind, batch, aa_index):
    baseline = {**batch, "time": batch["baseline_time"]}
    return physical.component(theta, kind, batch, aa_index) - physical.component(theta, kind, baseline, aa_index)


def physical_objective(theta, kind, batch, aa_index, alpha):
    prediction = physical_change(theta, kind, batch, aa_index)
    mse = torch.sum(batch["weight"] * (prediction - batch["target"]).square())
    _, _, rates = physical.decode(theta, kind, aa_index)
    penalty = alpha * (rates.log() - rates.log().mean()).square().mean() if kind == "residue" else theta.new_zeros(())
    return mse + penalty, mse, penalty


def data_free_initial(kind, n_residues, rate, tau):
    r = (rate - 1e-7) / (1 - 1e-7)
    t = (tau - .001) / (120 - .001)
    return np.array([math.log(r / (1 - r))] * (1 if kind == "uniform" else n_residues) + [math.log(t / (1 - t))])


def predictions(rows, values, **context):
    assert len(rows) == len(values)
    return [{**row, "weight": float(weight), "prediction": float(value),
             "predicted_source_target": float(row["baseline_target"] + value), **context}
            for row, weight, value in zip(rows, weights(rows), values)]
