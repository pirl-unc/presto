"""Source/conditioning conservation, supported ridge, and physical-gradient gates."""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from core import (AA, ROOT, configuration, data_free_initial, design, dump, load_changes,
                  physical_batch, physical_change, physical_objective, raw_features,
                  ridge_fit, ridge_predict, sequences, weights)


def independent_raw(sequence, representation):
    result = [math.log(len(sequence))]
    if representation == "sequence":
        result.extend(sum(1 for residue in sequence if residue == aa) / len(sequence) for aa in AA)
        result.extend(int(sequence.startswith(aa)) for aa in AA)
        result.extend(int(sequence.endswith(aa)) for aa in AA)
    return np.array(result)


assert json.loads((ROOT / "results/prepare-resource.json").read_text())["passed"]
config = configuration()
prior = Path(config["prior_family"])
covariates = pq.read_table(prior / "results/data/train/covariates.parquet").to_pylist()
labels = {r["assignment_id"]: r["target"] for r in pq.read_table(prior / "results/data/train/outcomes.parquet").to_pylist()}
by_sequence = defaultdict(list)
for row in covariates:
    by_sequence[row["peptide_sequence"]].append(row)
rows = load_changes()
max_baseline_error = 0.
for row in rows:
    original = by_sequence[row["peptide_sequence"]]
    earliest = min(r["time_minutes"] for r in original)
    base_ids = [r["assignment_id"] for r in original if r["time_minutes"] == earliest]
    mean = sum(labels[name] for name in base_ids) / len(base_ids)
    error = max(abs(mean - row["baseline_target"]),
                abs(labels[row["assignment_id"]] - mean - row["target"]))
    max_baseline_error = max(max_baseline_error, error)
    assert error <= 1e-12 and set(base_ids) == set(row["baseline_assignment_ids"])
    assert row["time_minutes"] > row["baseline_time"] == earliest
    assert row["assignment_id"] not in base_ids

ridge_checks = []
for outer in range(5):
    training = [r for r in rows if r["outer_fold"] != outer]
    validation = [r for r in rows if r["outer_fold"] == outer]
    assert not set(sequences(training)) & set(sequences(validation))
    for representation in ("length", "sequence"):
        fitted = ridge_fit(training, representation, .1)
        transform = fitted["transform"]
        seq = sequences(training)
        x = np.vstack([independent_raw(s, representation) for s in seq])
        assert np.max(np.abs(x - np.array([raw_features(s, representation) for s in seq]))) <= 1e-12
        center, scale = np.array(transform["center"]), np.array(transform["scale"])
        assert np.max(np.abs(center - x.mean(0))) <= 1e-12
        expected_scale = np.std(x, axis=0)
        expected_scale[(expected_scale == 0) | (np.ptp(x, axis=0) == 0)] = 1
        assert np.max(np.abs(scale - expected_scale)) <= 1e-12
        basis = np.array(transform["basis"])
        assert np.max(np.abs(basis.T @ basis - np.eye(transform["rank"])), initial=0) <= 1e-10
        a = design(training, transform)
        w = weights(training)
        penalty = np.zeros(a.shape[1])
        if transform["rank"]:
            penalty[1:] = .1 / transform["rank"]
        augmented = np.vstack([np.sqrt(w)[:, None] * a, np.diag(np.sqrt(penalty))])
        target = np.r_[np.sqrt(w) * np.array([r["target"] for r in training]), np.zeros(a.shape[1])]
        independent_beta = np.linalg.lstsq(augmented, target, rcond=None)[0]
        error = float(np.max(np.abs(design(validation, transform) @ independent_beta - ridge_predict(validation, fitted))))
        assert error <= 1e-8
        ridge_checks.append({"outer_fold": outer, "representation": representation,
                             "rank": transform["rank"], "prediction_error": error})

gradient_checks = []
for outer in range(5):
    training = [r for r in rows if r["outer_fold"] != outer]
    batch, aa_index, amino_acids = physical_batch(training)
    for kind, alpha in (("uniform", 0.), ("residue", .01), ("residue", 0.)):
        initial = data_free_initial(kind, len(amino_acids), rate=.001, tau=10)
        for state, x in (("initial", initial), ("fixed_perturbation", initial + .1 * np.sin(np.arange(len(initial))))):
            theta = torch.tensor(x, requires_grad=True)
            value = physical_objective(theta, kind, batch, aa_index, alpha)[0]
            gradient = torch.autograd.grad(value, theta)[0].detach().numpy()
            finite = np.zeros(len(x))
            for j in range(len(x)):
                direction = np.zeros(len(x)); direction[j] = 1e-4
                plus = physical_objective(torch.tensor(x + direction), kind, batch, aa_index, alpha)[0].item()
                minus = physical_objective(torch.tensor(x - direction), kind, batch, aa_index, alpha)[0].item()
                finite[j] = (plus - minus) / 2e-4
            error = float(np.max(np.abs(finite - gradient)))
            assert error <= 1e-6
            equal_time = {**batch, "time": batch["baseline_time"]}
            cancellation = physical_change(theta, kind, equal_time, aa_index)
            assert torch.max(torch.abs(cancellation)).item() == 0.
            cancelled_gradient = torch.autograd.grad(cancellation.sum(), theta)[0]
            assert torch.max(torch.abs(cancelled_gradient)).item() == 0.
            gradient_checks.append({"outer_fold": outer, "kind": kind, "alpha": alpha, "state": state,
                                    "max_gradient_error": error, "zero_elapsed_value_and_gradient": True})
result = {"passed": True, "n_verified_source_changes": len(rows), "max_baseline_error": max_baseline_error,
          "ridge": ridge_checks, "physical_gradients": gradient_checks,
          "data_free_initialization": True, "original_outer_outcomes_read": False}
dump("results/qualification.json", result)
print(json.dumps(result), flush=True)
