"""Canonical native response consumer and shared exact/hybrid physical model."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from presto.models.steady_processing import (
    conditional_fragment_age,
    fragment_event_log_score,
    log_steady_fragment_amount,
)

FAMILY = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
LN10 = math.log(10)


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as f:
        return json.load(f)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config():
    return read(FAMILY / "reproduce/launch.json")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def baseline_modules():
    family = Path(config()["baseline_family"])
    return module(family / "code/launch.py", "native_baseline_train"), module(
        family / "analysis/verify.py", "native_baseline_metrics"
    )


def raw_bond(sequence, bond):
    row = np.zeros(40)
    row[AA.index(sequence[bond])] = 1
    row[20 + AA.index(sequence[bond + 1])] = 1
    return row


class Dataset:
    def __init__(self):
        self.cfg = config()
        directory = Path(self.cfg["recipe_directory"])
        assert sha(directory / "manifest.json") == self.cfg["recipe_manifest_sha256"]
        manifest = read(directory / "manifest.json")
        for name in ("examples", "parents", "occurrences"):
            assert (
                sha(directory / (name + ".json.gz"))
                == manifest["files"][name + ".json.gz"]["sha256"]
            )
        admitted = read(self.cfg["admission_receipt"])
        assert (
            admitted["status"] == "passed"
            and admitted["recipe_manifest_sha256"] == self.cfg["recipe_manifest_sha256"]
        )
        all_groups = read(directory / "examples.json.gz")
        self.excluded = [g for g in all_groups if not g["full_parent_model_supported"]]
        self.groups = [g for g in all_groups if g["full_parent_model_supported"]]
        self.split = np.array([g["split"] for g in self.groups])
        original_weights = np.array([g["group_weight"] for g in self.groups])
        self.normalizers = {
            s: float(original_weights[self.split == s].sum()) for s in ("train", "val", "test")
        }
        self.group_weights = np.array(
            [w / self.normalizers[s] for w, s in zip(original_weights, self.split)]
        )
        training, _ = baseline_modules()
        self.cx, self.condition_metadata, self.condition_arrays = training.design(
            self.groups,
            self.split == "train",
            self.group_weights,
            "condition",
            self.cfg["rank_relative_threshold"],
        )
        self.condition_metadata["unseen_by_split"] = {
            s: sum(
                g["split"] == s and g["condition_id"] not in self.condition_metadata["conditions"]
                for g in self.groups
            )
            for s in ("val", "test")
        }
        assert not any(self.condition_metadata["unseen_by_split"].values())
        self.parents = {
            p["sequence_sha256"]: p["sequence"] for p in read(directory / "parents.json.gz")
        }
        eligible_sequences = {g["peptide_sequence"] for g in self.groups}
        origins = defaultdict(set)
        for o in read(directory / "occurrences.json.gz"):
            if o["peptide_sequence"] in eligible_sequences:
                parent, start, end = (
                    o["reference_sequence_sha256"],
                    o["start_1based"] - 1,
                    o["end_1based"],
                )
                sequence = self.parents[parent]
                assert sequence[start:end] == o["peptide_sequence"] and set(sequence) <= set(AA)
                origins[o["peptide_sequence"]].add((parent, start, end))
        self.origins = {s: sorted(v) for s, v in origins.items()}
        assert eligible_sequences == self.origins.keys()
        training_sequences = {g["peptide_sequence"] for g in self.groups if g["split"] == "train"}
        bonds, reachable_training = set(), set()
        for peptide, locations in self.origins.items():
            for parent, a, b in locations:
                involved = {
                    (parent, i) for i in range(max(0, a - 1), min(len(self.parents[parent]) - 1, b))
                }
                bonds.update(involved)
                if peptide in training_sequences:
                    reachable_training.update(involved)
        self.bonds = sorted(bonds)
        bond_index = {key: i for i, key in enumerate(self.bonds)}
        raw = np.array([raw_bond(self.parents[p], i) for p, i in self.bonds])
        supported = np.array([bond in reachable_training for bond in self.bonds])
        parent_bonds = defaultdict(int)
        for p, _ in reachable_training:
            parent_bonds[p] += 1
        weights = np.array(
            [
                1 / len(parent_bonds) / parent_bonds[p]
                for (p, _), selected in zip(self.bonds, supported)
                if selected
            ]
        )
        center = weights @ raw[supported]
        variance = weights @ (raw[supported] - center) ** 2
        keep = variance > 1e-24
        scale = np.sqrt(variance[keep])
        standardized = (raw[:, keep] - center[keep]) / scale
        _, singular, vt = np.linalg.svd(
            np.sqrt(weights[:, None]) * standardized[supported], full_matrices=False
        )
        rank = int(np.sum(singular > singular[0] * self.cfg["rank_relative_threshold"]))
        projection = vt[:rank].T
        self.bond_features = standardized @ projection
        residual = standardized - self.bond_features @ projection.T
        self.bond_arrays = {
            "center": center,
            "keep": keep,
            "scale": scale,
            "projection": projection,
            "singular_values": singular,
        }
        self.bond_metadata = {
            "n_raw_features": 40,
            "n_supported_coordinates": rank,
            "n_bonds_in_queries": len(bonds),
            "n_reachable_training_bonds": int(supported.sum()),
            "n_training_parent_sequences": len(parent_bonds),
            "n_bonds_outside_training_feature_span": int(
                np.sum(np.linalg.norm(residual, axis=1) > 1e-10)
            ),
        }
        assert self.bond_metadata["n_bonds_outside_training_feature_span"] == 0
        self.occurrences = sorted({o for locations in self.origins.values() for o in locations})
        occurrence_index = {o: i for i, o in enumerate(self.occurrences)}
        self.internal_bond, self.internal_owner, left, right, counts = [], [], [], [], []
        for oi, (p, a, b) in enumerate(self.occurrences):
            self.internal_bond.extend(bond_index[p, i] for i in range(a, b - 1))
            self.internal_owner.extend([oi] * (b - a - 1))
            left.append(bond_index[p, a - 1] if a else len(bonds))
            right.append(bond_index[p, b - 1] if b < len(self.parents[p]) else len(bonds))
            counts.append(b - a - 1)
        self.left, self.right, self.internal_counts = (
            np.array(left),
            np.array(right),
            np.array(counts),
        )
        width = max(len(v) for v in self.origins.values())
        self.query = np.zeros((len(self.groups), width), dtype=np.int64)
        self.valid = np.zeros_like(self.query, dtype=bool)
        self.n_parents, self.ties = [], {"uniform": [], "sequence": []}
        for i, g in enumerate(self.groups):
            seq = g["peptide_sequence"]
            locations = self.origins[seq]
            self.query[i, : len(locations)] = [occurrence_index[o] for o in locations]
            self.valid[i, : len(locations)] = True
            self.n_parents.append(len({p for p, _, _ in locations}))
            geometry = [0, 0, 0]
            for p, a, b in locations:
                geometry[int(a > 0) + int(b < len(self.parents[p]))] += 1
            divisor = math.gcd(*geometry)
            geometry = tuple(v // divisor for v in geometry)
            self.ties["uniform"].append((g["condition_id"], len(seq), geometry))
            self.ties["sequence"].append((g["condition_id"], seq))

    def rows(self, split):
        rows, group_indices = [], []
        for i, g in enumerate(self.groups):
            if g["split"] != split:
                continue
            for o in g["observations"]:
                row = {
                    k: g[k]
                    for k in (
                        "response_group_id",
                        "condition_id",
                        "condition",
                        "component_id",
                        "split",
                        "peptide_sequence",
                        "modifications",
                        "full_parent_model_supported",
                    )
                }
                row.update(o)
                row["canonical_weight"] = row["weight"]
                row["weight"] /= self.normalizers[split]
                row["censor_code"] = {None: 0, "lt": -1, "gt": 1}[row["inequality"]]
                row["n_candidate_parent_sequences"] = self.n_parents[i]
                row["n_candidate_occurrences"] = len(self.origins[g["peptide_sequence"]])
                rows.append(row)
                group_indices.append(i)
        return rows, np.array(group_indices)

    def packing(self, split):
        group_ids = np.flatnonzero(self.split == split)
        row_records, row_group_ids = self.rows(split)
        position = {g: i for i, g in enumerate(group_ids)}
        tensor = lambda x, dtype=torch.float64: torch.as_tensor(x, dtype=dtype)
        return {
            "groups": group_ids,
            "rows": row_records,
            "owner": tensor([position[g] for g in row_group_ids], torch.long),
            "target": tensor([r["target_log10_ratio"] for r in row_records]),
            "censor": tensor([r["censor_code"] for r in row_records], torch.long),
            "weight": tensor([r["weight"] for r in row_records]),
        }

    def summary(self):
        # Counts only; never evaluate held-out quantities here.
        return {
            "recipe_manifest_sha256": self.cfg["recipe_manifest_sha256"],
            "splits": {
                s: {
                    "n_response_groups": int(np.sum(self.split == s)),
                    "n_effective_observations": sum(
                        len(g["observations"]) for g in self.groups if g["split"] == s
                    ),
                    "n_exact_observations": sum(
                        o["inequality"] is None
                        for g in self.groups
                        if g["split"] == s
                        for o in g["observations"]
                    ),
                    "weight_normalizer": self.normalizers[s],
                }
                for s in ("train", "val", "test")
            },
            "excluded_response_groups": [
                {
                    "response_group_id": g["response_group_id"],
                    "split": g["split"],
                    "source_record_ids": sorted(
                        {r for o in g["observations"] for r in o["source_record_ids"]}
                    ),
                    "n_effective_observations": len(g["observations"]),
                }
                for g in self.excluded
            ],
            "bond_features": self.bond_metadata,
            "condition_features": self.condition_metadata,
            "n_occurrences": len(self.occurrences),
            "maximum_occurrences_per_group": self.query.shape[1],
        }


class PhysicalModel:
    def __init__(self, dataset, representation):
        self.data = dataset
        self.representation = representation
        self.rank = dataset.bond_features.shape[1] if representation == "sequence" else 0
        self.nc = dataset.cx.shape[1]
        self.n_parameters = 1 + self.rank + 2 * self.nc
        self.a_slice = slice(1 + self.rank, 1 + self.rank + self.nc)
        self.b_slice = slice(1 + self.rank + self.nc, self.n_parameters)
        self.penalty_mask = torch.ones(self.n_parameters, dtype=torch.float64)
        self.penalty_mask[self.b_slice.start] = 0
        self.cx = torch.tensor(dataset.cx, dtype=torch.float64)
        self.features = torch.tensor(dataset.bond_features, dtype=torch.float64)
        self.internal_bond = torch.tensor(dataset.internal_bond)
        self.internal_owner = torch.tensor(dataset.internal_owner)
        self.left, self.right = torch.tensor(dataset.left), torch.tensor(dataset.right)
        self.counts = torch.tensor(dataset.internal_counts, dtype=torch.float64)
        self.query, self.valid = torch.tensor(dataset.query), torch.tensor(dataset.valid)
        self.log_parent_count = torch.tensor(dataset.n_parents, dtype=torch.float64).log()

    def terms(self, theta):
        if self.rank:
            rate = (theta[0] + self.features @ theta[1 : 1 + self.rank]).exp()
            internal = theta.new_zeros(len(self.data.occurrences)).index_add(
                0, self.internal_owner, rate[self.internal_bond]
            )
        else:
            rate = theta[0].exp().expand(len(self.data.bonds))
            internal = self.counts * theta[0].exp()
        extended = torch.cat([rate, theta.new_zeros(1)])
        return internal, extended[self.left], extended[self.right]

    def forward(self, theta, groups, *, uniforms=None, trace=False):
        groups = torch.as_tensor(groups, dtype=torch.long)
        terms = self.terms(theta)
        selected = self.query[groups]
        control = tuple(t[selected] for t in terms)
        scale = (self.cx[groups] @ theta[self.a_slice]).exp()[:, None]
        treated = tuple(t * scale for t in control)
        valid = self.valid[groups]
        delta = theta.new_tensor(1.0)
        log_control = log_steady_fragment_amount(*control, delta, validate=False).masked_fill(
            ~valid, -torch.inf
        )
        log_treated = log_steady_fragment_amount(*treated, delta, validate=False).masked_fill(
            ~valid, -torch.inf
        )
        totals0, totals1 = torch.logsumexp(log_control, 1), torch.logsumexp(log_treated, 1)
        offset = self.cx[groups] @ theta[self.b_slice]
        prediction = offset + (totals1 - totals0) / LN10
        # Proven analytic input equivalence, not floating-point/target-based rounding.
        seen, representatives = {}, []
        for i, index in enumerate(groups.tolist()):
            key = self.data.ties[self.representation][index]
            representatives.append(seen.setdefault(key, i))
        prediction = prediction[torch.tensor(representatives)]
        trace_rows = []
        if uniforms is not None:
            assert uniforms.shape == (len(groups), self.data.cfg["sample_count"], 5)
            scores = []
            for rates, logf in ((control, log_control), (treated, log_treated)):
                with torch.no_grad():
                    probability = logf.softmax(1)
                    cumulative = probability.cumsum(1)
                    cumulative[:, -1] = 1
                    chosen = torch.searchsorted(
                        cumulative.contiguous(), uniforms[..., 0].contiguous()
                    )
                    assert valid.gather(1, chosen).all()
                chosen_terms = tuple(t.gather(1, chosen) for t in rates)
                age = conditional_fragment_age(
                    *chosen_terms, delta, uniforms[..., 1:], validate=False
                )
                score = fragment_event_log_score(*chosen_terms, delta, age, validate=False).mean(1)
                scores.append(score)
                if trace:
                    trace_rows.append(
                        {"chosen": chosen[:4, :8].tolist(), "age": age[:4, :8].tolist()}
                    )
            surrogate = offset + (scores[1] - scores[0]) / LN10
            prediction = prediction.detach() + (surrogate - surrogate.detach())
        assert torch.isfinite(prediction).all()
        return prediction, {
            "log_control_amount": totals0 - self.log_parent_count[groups],
            "log_treated_amount": totals1 - self.log_parent_count[groups],
            "trace": trace_rows,
        }


def data_loss(prediction, pack):
    residual = prediction[pack["owner"]] - pack["target"]
    residual = torch.where(
        pack["censor"] == -1,
        residual.clamp_min(0),
        torch.where(pack["censor"] == 1, residual.clamp_max(0), residual),
    )
    return (pack["weight"] * residual.square()).sum()


def objective(theta, model, pack, *, uniforms=None, penalized=True, trace=False):
    pred, extra = model.forward(theta, pack["groups"], uniforms=uniforms, trace=trace)
    loss = data_loss(pred, pack)
    total = (
        loss + model.data.cfg["penalty"] * (model.penalty_mask * theta.square()).sum()
        if penalized
        else loss
    )
    return total, loss, pred, extra
