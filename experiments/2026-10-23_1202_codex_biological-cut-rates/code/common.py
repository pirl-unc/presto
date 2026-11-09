"""Frozen canonical-recipe consumption and biological temporal objective."""

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parents[1]
sys.path.insert(0, "/private/tmp/presto-modular-imports")
from presto.models.independent_cut import (
    log_fragment_expectation,
    ordered_bond_features,
)


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_data():
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    path = Path(cfg["recipe_directory"])
    assert sha(path / "manifest.json") == cfg["recipe_manifest_sha256"]
    manifest = json.loads((path / "manifest.json").read_text())
    for name in ("examples.json", "components.json", "recipe.json"):
        assert sha(path / name) == manifest["files"][name]["sha256"]
    for name, value in cfg["source_hashes"].items():
        assert sha(REPO / name) == value, name
    examples = json.loads((path / "examples.json").read_text())
    by_condition = defaultdict(list)
    for e in examples:
        by_condition[e["condition_id"]].append(e)
    groups = []
    for condition, curves in sorted(by_condition.items()):
        first = curves[0]
        sequence = first["parent_sequence"]
        context = first["context"]
        enzyme = (
            context["substrate_concentration_um"]
            / context["enzyme_substrate_denominator"]
        )
        rows = []
        for e in curves:
            assert e["parent_sequence"] == sequence and e["split"] == first["split"]
            vals = {o["time_hours"]: o["target_value"] for o in e["observations"]}
            ref = e["common_positive_times"][0]
            times = e["common_positive_times"][1:]
            for t in times:
                rows.append(
                    dict(
                        curve_id=e["curve_id"],
                        construct_id=e["construct_id"],
                        condition_id=condition,
                        split=e["split"],
                        peptide_sequence=e["peptide_sequence"],
                        start=e["start_0based"],
                        end=e["end_0based"],
                        time=t,
                        reference_time=ref,
                        signal=vals[t],
                        reference_signal=vals[ref],
                        target=math.log10(vals[t] / vals[ref]),
                        curve_observations=len(times),
                        biological_replicate_raw=e["biological_replicate_raw"],
                    )
                )
        groups.append(
            dict(
                condition=condition,
                parent=first["construct_id"],
                split=first["split"],
                sequence=sequence,
                context=math.log(enzyme / 0.2),
                enzyme_concentration_proxy_um=enzyme,
                rows=rows,
                n_curves=len(curves),
                starts=torch.tensor([r["start"] for r in rows]),
                ends=torch.tensor([r["end"] for r in rows]),
                times=torch.tensor([r["time"] for r in rows], dtype=torch.float64),
                refs=torch.tensor(
                    [r["reference_time"] for r in rows], dtype=torch.float64
                ),
                target=torch.tensor([r["target"] for r in rows], dtype=torch.float64),
            )
        )
    for split in ("train", "val", "test"):
        subset = [g for g in groups if g["split"] == split]
        parents = Counter(g["parent"] for g in subset)
        for g in subset:
            weights = [
                1
                / (
                    len(parents)
                    * parents[g["parent"]]
                    * g["n_curves"]
                    * r["curve_observations"]
                )
                for r in g["rows"]
            ]
            g["weight"] = torch.tensor(weights, dtype=torch.float64)
        assert abs(sum(g["weight"].sum().item() for g in subset) - 1) < 1e-12
    return cfg, groups


def features(group, variant):
    if variant == "constant":
        return torch.ones((len(group["sequence"]) - 1, 1), dtype=torch.float64)
    return ordered_bond_features(
        group["sequence"],
        context=group["context"] if variant == "sequence_context" else None,
    )


def designs(groups, variant):
    raw = [features(g, variant) for g in groups]
    train = torch.cat([x for x, g in zip(raw, groups) if g["split"] == "train"])
    _, s, vh = torch.linalg.svd(train, full_matrices=False)
    rank = int((s > s[0] * max(train.shape) * torch.finfo(train.dtype).eps).sum())
    basis = torch.eye(train.shape[1], dtype=torch.float64)
    if rank < train.shape[1]:
        basis = vh[:rank].T.contiguous()
        for column in range(rank):
            first = torch.nonzero(basis[:, column].abs() > 1e-12)[0, 0]
            if basis[first, column] < 0:
                basis[:, column] *= -1
    return (
        [x @ basis for x in raw],
        basis,
        dict(
            raw_dimensions=train.shape[1],
            allocated_dimensions=rank,
            singular_values=s.tolist(),
            zero_training_columns=torch.nonzero(train.abs().sum(0) == 0)
            .flatten()
            .tolist(),
            basis=basis.tolist(),
            heldout_projection_residual_max={
                split: max(
                    float((x - x @ basis @ basis.T).abs().max())
                    for x, g in zip(raw, groups)
                    if g["split"] == split
                )
                for split in ("val", "test")
            },
        ),
    )


def predict(log_rates, group):
    a = log_fragment_expectation(
        log_rates, group["starts"], group["ends"], group["times"], validate=False
    )
    b = log_fragment_expectation(
        log_rates, group["starts"], group["ends"], group["refs"], validate=False
    )
    return (a - b) / math.log(10)


def objective(theta, groups, design, basis, penalty):
    loss = theta.new_zeros(())
    for g, x in zip(groups, design):
        if g["split"] == "train":
            loss = (
                loss + ((predict(x @ theta, g) - g["target"]) ** 2 * g["weight"]).sum()
            )
    original = basis @ theta
    reg = penalty * original[1:].square().sum() / 2
    return loss + reg, loss


def metrics(rows):
    y = np.array([r["target"] for r in rows])
    p = np.array([r["prediction"] for r in rows])
    w = np.array([r["weight"] for r in rows])
    assert abs(w.sum() - 1) < 1e-12
    constant = np.ptp(p) < 1e-12
    byparent = {}
    for parent in sorted({r["construct_id"] for r in rows}):
        mask = np.array([r["construct_id"] == parent for r in rows])
        ww = w[mask] / w[mask].sum()
        byparent[parent] = dict(
            n=int(mask.sum()),
            rmse=float(np.sqrt(np.sum(ww * (p[mask] - y[mask]) ** 2))),
        )
    return dict(
        n_predictions=len(rows),
        n_parents=len(byparent),
        mse=float(np.sum(w * (p - y) ** 2)),
        rmse=float(np.sqrt(np.sum(w * (p - y) ** 2))),
        pooled_rmse=float(np.sqrt(np.mean((p - y) ** 2))),
        pearson=None if constant else float(pearsonr(y, p).statistic),
        spearman=None if constant else float(spearmanr(y, p).statistic),
        prediction_sd=0.0 if constant else float(p.std()),
        target_sd=float(y.std()),
        by_parent=byparent,
    )
