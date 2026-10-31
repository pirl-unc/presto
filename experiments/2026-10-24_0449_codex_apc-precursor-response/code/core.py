"""Shared fixed APC cohort, supported features and exact/conditional-score model."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
from collections import Counter
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
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as stream:
        return json.load(stream)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_module():
    return load_module(
        FAMILY.parent / "2026-10-24_0314_codex_native-response-baselines/analysis/verify.py",
        "apc_metrics",
    )


def basis(raw, train, weights):
    w = weights[train] / weights[train].sum()
    center = w @ raw[train]
    variance = w @ (raw[train] - center) ** 2
    keep = variance > 1e-24
    scale = np.sqrt(variance[keep])
    z = (raw[:, keep] - center[keep]) / scale
    _, singular, vt = np.linalg.svd(np.sqrt(w[:, None]) * z[train], full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-10))
    projection = vt[:rank].T
    x = z @ projection
    unsupported = int(np.sum(np.linalg.norm(z - x @ projection.T, axis=1) > 1e-10))
    assert unsupported == 0, "Held-out input direction lacks training support"
    return x, {
        "center": center,
        "keep": keep,
        "scale": scale,
        "projection": projection,
        "singular": singular,
    }


class Dataset:
    def __init__(self):
        self.cfg = read(FAMILY / "reproduce/model-config.json")
        receipt = read(FAMILY / "results/cohort.json")
        assert receipt["status"] == "admitted"
        assert read(FAMILY / "results/cohort-verification.json")["status"] == "passed"
        root = Path(receipt["model_input_directory"])
        for name, digest in receipt["files"].items():
            assert sha(root / name) == digest
        self.groups = read(root / "groups.json.gz")
        self.parents = read(root / "parents.json.gz")
        self.split = np.array([g["split"] for g in self.groups])
        self.weights = np.array([g["group_weight"] for g in self.groups])
        conditions = sorted({g["condition_id"] for g in self.groups if g["split"] == "train"})
        assert set(conditions) == {g["condition_id"] for g in self.groups}
        raw_condition = np.array(
            [[float(g["condition_id"] == c) for c in conditions] for g in self.groups]
        )
        contrasts, self.condition_transform = basis(
            raw_condition, self.split == "train", self.weights
        )
        self.conditions = conditions
        self.cx = np.column_stack([np.ones(len(self.groups)), contrasts])
        self.baseline_precursor = np.array([g["precursor_only_log10_ratio"] for g in self.groups])
        self.occurrences = sorted(
            {
                (o["parent_sequence_sha256"], o["start0"], o["end0_exclusive"])
                for g in self.groups
                for o in g["origins"]
            }
        )
        oi = {o: i for i, o in enumerate(self.occurrences)}
        bonds, trained_bonds = set(), set()
        for g in self.groups:
            for o in g["origins"]:
                p, a, b = o["parent_sequence_sha256"], o["start0"], o["end0_exclusive"]
                touched = {(p, i) for i in range(max(0, a - 1), min(len(self.parents[p]) - 1, b))}
                bonds.update(touched)
                if g["split"] == "train":
                    trained_bonds.update(touched)
        self.bonds = sorted(bonds)
        bi = {b: i for i, b in enumerate(self.bonds)}
        self.raw_bonds = np.zeros((len(self.bonds), 40))
        for j, (p, i) in enumerate(self.bonds):
            self.raw_bonds[j, AA.index(self.parents[p][i])] = 1
            self.raw_bonds[j, 20 + AA.index(self.parents[p][i + 1])] = 1
        train = np.array([b in trained_bonds for b in self.bonds])
        counts = Counter(p for p, _ in trained_bonds)
        weights = np.array(
            [
                1 / len(counts) / counts[p] if selected else 0
                for (p, _), selected in zip(self.bonds, train)
            ]
        )
        self.features, self.bond_transform = basis(self.raw_bonds, train, weights)
        self.internal_bonds, self.internal_owner, self.left, self.right, self.internal_counts = (
            [],
            [],
            [],
            [],
            [],
        )
        for i, (p, a, b) in enumerate(self.occurrences):
            self.internal_bonds.extend(bi[p, j] for j in range(a, b - 1))
            self.internal_owner.extend([i] * (b - a - 1))
            self.left.append(bi[p, a - 1] if a else len(bonds))
            self.right.append(bi[p, b - 1] if b < len(self.parents[p]) else len(bonds))
            self.internal_counts.append(b - a - 1)
        width = max(len(g["origins"]) for g in self.groups)
        self.query = np.zeros((len(self.groups), width), dtype=np.int64)
        self.valid = np.zeros_like(self.query, dtype=bool)
        self.input_log = np.zeros_like(self.query, dtype=np.float64)
        self.nparents = []
        for i, g in enumerate(self.groups):
            for j, o in enumerate(g["origins"]):
                self.query[i, j] = oi[o["parent_sequence_sha256"], o["start0"], o["end0_exclusive"]]
                self.valid[i, j] = True
                self.input_log[i, j] = o["precursor_log10_ratio"] * LN10
            self.nparents.append(len({o["parent_sequence_sha256"] for o in g["origins"]}))
        prior = Path(read(FAMILY / "reproduce/launch.json")["prior_family"])
        transform = np.load(prior / "results/feature-transforms.npz")
        theta = np.load(prior / "results/arms/sequence_exact/fit.npz")["theta"]
        rank = transform["bond_projection"].shape[1]
        coef = np.zeros(40)
        coef[transform["bond_keep"]] = (
            transform["bond_projection"] @ theta[1 : 1 + rank]
        ) / transform["bond_scale"]
        intercept = theta[0] - transform["bond_center"] @ coef
        self.fixed_rates = np.exp(intercept + self.raw_bonds @ coef)
        self.prior_rate_coefficients = {
            "intercept": float(intercept),
            "raw_coefficients": coef.tolist(),
            "checkpoint_sha256": sha(prior / "results/arms/sequence_exact/fit.npz"),
        }
        self.feature_coverage = {
            "n_training_bonds": int(train.sum()),
            "n_query_bonds": len(bonds),
            "n_training_parents": len(counts),
            "n_bond_coordinates": self.features.shape[1],
            "n_condition_coordinates": self.cx.shape[1],
            "conditions": conditions,
            "unsupported_feature_directions": 0,
        }

    def rows(self, split):
        rows, indices = [], []
        for i, g in enumerate(self.groups):
            if g["split"] != split:
                continue
            for obs in g["observations"]:
                r = dict(obs)
                r.update(
                    {
                        "observation_id": obs["measurement_id"],
                        "response_group_id": g["response_group_id"],
                        "source_peptide_row_id": g["source_peptide_row_id"],
                        "condition_id": g["condition_id"],
                        "component_id": g["component_id"],
                        "split": split,
                        "peptide_sequence": g["peptide_sequence"],
                        "target_log10_ratio": obs["target_value"],
                        "inequality": None,
                        "precursor_only_log10_ratio": g["precursor_only_log10_ratio"],
                        "n_candidate_parent_sequences": self.nparents[i],
                        "n_occurrences": len(g["origins"]),
                        "origin_inputs": g["origins"],
                    }
                )
                rows.append(r)
                indices.append(i)
        return rows, np.array(indices)

    def pack(self, split):
        ids = np.flatnonzero(self.split == split)
        positions = {i: j for j, i in enumerate(ids)}
        rows, owner = self.rows(split)
        return {
            "groups": ids,
            "rows": rows,
            "owner": torch.tensor([positions[i] for i in owner]),
            "target": torch.tensor([r["target_log10_ratio"] for r in rows], dtype=torch.float64),
            "weight": torch.tensor([r["weight"] for r in rows], dtype=torch.float64),
        }


class Model:
    def __init__(self, data, kind, precursor=True):
        self.data, self.kind, self.precursor = data, kind, precursor
        self.rank = data.features.shape[1] if kind == "sequence" else 0
        self.nr = 0 if kind == "prior" else 1 + self.rank
        self.nc = data.cx.shape[1]
        self.n_parameters = self.nr + 2 * self.nc
        self.a_slice = slice(self.nr, self.nr + self.nc)
        self.b_slice = slice(self.nr + self.nc, self.n_parameters)
        self.mask = torch.ones(self.n_parameters, dtype=torch.float64)
        self.mask[self.b_slice.start] = 0
        for name, value in (
            ("features", data.features),
            ("cx", data.cx),
            ("fixed_rates", data.fixed_rates),
            ("counts", data.internal_counts),
            ("input_log", data.input_log),
        ):
            setattr(self, name, torch.tensor(value, dtype=torch.float64))
        for name, value in (
            ("internal_bond", data.internal_bonds),
            ("internal_owner", data.internal_owner),
            ("left", data.left),
            ("right", data.right),
            ("query", data.query),
        ):
            setattr(self, name, torch.tensor(value, dtype=torch.long))
        self.valid = torch.tensor(data.valid)
        self.log_nparents = torch.tensor(data.nparents, dtype=torch.float64).log()

    def terms(self, theta):
        if self.kind == "uniform":
            rates = theta[0].exp().expand(len(self.data.bonds))
            internal = theta[0].exp() * self.counts
        else:
            rates = (
                self.fixed_rates
                if self.kind == "prior"
                else (theta[0] + self.features @ theta[1 : self.nr]).exp()
            )
            internal = theta.new_zeros(len(self.data.occurrences)).index_add(
                0, self.internal_owner, rates[self.internal_bond]
            )
        padded = torch.cat([rates, theta.new_zeros(1)])
        return internal, padded[self.left], padded[self.right]

    def forward(self, theta, groups, uniforms=None):
        groups = torch.as_tensor(groups, dtype=torch.long)
        selected = self.query[groups]
        control = tuple(t[selected] for t in self.terms(theta))
        scale = (self.cx[groups] @ theta[self.a_slice]).exp()[:, None]
        treated = tuple(t * scale for t in control)
        valid = self.valid[groups]
        delta = theta.new_tensor(1.0)
        log0 = log_steady_fragment_amount(*control, delta, validate=False).masked_fill(
            ~valid, -torch.inf
        )
        log1 = log_steady_fragment_amount(*treated, delta, validate=False)
        if self.precursor:
            log1 = log1 + self.input_log[groups]
        log1 = log1.masked_fill(~valid, -torch.inf)
        totals0, totals1 = torch.logsumexp(log0, 1), torch.logsumexp(log1, 1)
        offset = self.cx[groups] @ theta[self.b_slice]
        pred = offset + (totals1 - totals0) / LN10
        seen, rep = {}, []
        for local, index in enumerate(groups.tolist()):
            g = self.data.groups[index]
            if self.kind == "uniform":
                geometry = Counter(
                    (
                        int(o["start0"] > 0)
                        + int(
                            o["end0_exclusive"]
                            < len(self.data.parents[o["parent_sequence_sha256"]])
                        ),
                        o["precursor_log10_ratio"] if self.precursor else 0.0,
                    )
                    for o in g["origins"]
                )
                divisor = math.gcd(*geometry.values())
                key = (
                    g["condition_id"],
                    len(g["peptide_sequence"]),
                    tuple(sorted((bound, p, n // divisor) for (bound, p), n in geometry.items())),
                )
            else:
                key = (g["condition_id"], g["peptide_sequence"])
            rep.append(seen.setdefault(key, local))
        pred = pred[torch.tensor(rep)]
        if uniforms is not None:
            assert uniforms.shape == (len(groups), self.data.cfg["sample_count"], 5)
            scores = []
            for terms, logf in ((control, log0), (treated, log1)):
                with torch.no_grad():
                    cdf = logf.softmax(1).cumsum(1)
                    cdf[:, -1] = 1.0
                    chosen = torch.searchsorted(cdf.contiguous(), uniforms[..., 0].contiguous())
                    assert valid.gather(1, chosen).all()
                rates = tuple(t.gather(1, chosen) for t in terms)
                age = conditional_fragment_age(*rates, delta, uniforms[..., 1:], validate=False)
                scores.append(fragment_event_log_score(*rates, delta, age, validate=False).mean(1))
            surrogate = offset + (scores[1] - scores[0]) / LN10
            pred = pred.detach() + (surrogate - surrogate.detach())
        assert torch.isfinite(pred).all()
        return pred, {
            "log_control_amount": totals0 - self.log_nparents[groups],
            "log_treated_amount": totals1 - self.log_nparents[groups],
        }


def objective(theta, model, pack, uniforms=None, penalty=True):
    p, extra = model.forward(theta, pack["groups"], uniforms)
    loss = (pack["weight"] * (p[pack["owner"]] - pack["target"]).square()).sum()
    total = (
        loss + model.data.cfg["penalty"] * (model.mask * theta.square()).sum() if penalty else loss
    )
    return total, loss, p, extra
