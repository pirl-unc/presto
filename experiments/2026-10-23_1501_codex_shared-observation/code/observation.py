"""Fixed source signals, training-supported response features, and profiled scale."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from presto.models.independent_cut import (
    log_fragment_expectation,
    ordered_bond_features,
)

FAMILY = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
LN10 = math.log(10)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def center(value, weight):
    return value - (weight * value).sum() / weight.sum()


def load_data():
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    path = Path(cfg["recipe_directory"])
    assert sha(path / "manifest.json") == cfg["recipe_manifest_sha256"]
    manifest = json.loads((path / "manifest.json").read_text())
    assert sha(path / "examples.json") == manifest["files"]["examples.json"]["sha256"]
    context = json.loads((FAMILY / "results/context-summary.json").read_text())
    grouped = defaultdict(list)
    for e in json.loads((path / "examples.json").read_text()):
        if e["split"] in cfg["allowed_splits"]:
            grouped[e["condition_id"]].append(e)
    groups = []
    for condition, curves in sorted(grouped.items()):
        first = curves[0]
        sequence = first["parent_sequence"]
        ctx = first["context"]
        instrument_counts = {
            k.removeprefix("instrument:"): v
            for k, v in context["per_condition"][condition].items()
            if k.startswith("instrument:")
        }
        assert len(instrument_counts) == 1 and "None" not in instrument_counts
        instrument = next(iter(instrument_counts))
        rows = []
        for e in curves:
            values = {o["time_hours"]: o["target_value"] for o in e["observations"]}
            ref = e["common_positive_times"][0]
            for t in e["common_positive_times"]:
                rows.append(
                    dict(
                        curve_id=e["curve_id"],
                        condition_id=condition,
                        construct_id=e["construct_id"],
                        split=e["split"],
                        peptide_sequence=e["peptide_sequence"],
                        start=e["start_0based"],
                        end=e["end_0based"],
                        time=t,
                        reference_time=ref,
                        signal=values[t],
                        reference_signal=values[ref],
                        target=math.log10(values[t]),
                        n_times=len(e["common_positive_times"]),
                        biological_replicate_raw=e["biological_replicate_raw"],
                        instrument=instrument,
                    )
                )
        enzyme = ctx["substrate_concentration_um"] / ctx["enzyme_substrate_denominator"]
        g = dict(
            condition=condition,
            parent=first["construct_id"],
            split=first["split"],
            sequence=sequence,
            instrument=instrument,
            rows=rows,
            ncurves=len(curves),
            x=ordered_bond_features(sequence, context=math.log(enzyme / 0.2)),
            a=torch.tensor([r["start"] for r in rows]),
            b=torch.tensor([r["end"] for r in rows]),
            t=torch.tensor([r["time"] for r in rows], dtype=torch.float64),
            y=torch.tensor([r["target"] for r in rows], dtype=torch.float64),
        )
        groups.append(g)
    for split in cfg["allowed_splits"]:
        subset = [g for g in groups if g["split"] == split]
        parents = Counter(g["parent"] for g in subset)
        for g in subset:
            g["w"] = torch.tensor(
                [
                    1
                    / (
                        len(parents)
                        * parents[g["parent"]]
                        * g["ncurves"]
                        * r["n_times"]
                    )
                    for r in g["rows"]
                ],
                dtype=torch.float64,
            )
        assert abs(sum(float(g["w"].sum()) for g in subset) - 1) < 1e-12
    assert sum(len(g["rows"]) for g in groups if g["split"] == "train") == 90630
    assert sum(len(g["rows"]) for g in groups if g["split"] == "val") == 14822
    instruments = sorted({g["instrument"] for g in groups if g["split"] == "train"})
    assert {g["instrument"] for g in groups} == set(instruments)
    names = [
        f"{ins}:{name}" for ins in instruments for name in ["length/40", *list(AA[1:])]
    ]
    train_gram = np.zeros((len(names), len(names)))
    for g in groups:
        z = np.zeros((len(g["rows"]), len(names)))
        offset = instruments.index(g["instrument"]) * 20
        for i, r in enumerate(g["rows"]):
            peptide = r["peptide_sequence"]
            z[i, offset : offset + 20] = [
                len(peptide) / 40,
                *[peptide.count(a) / len(peptide) for a in AA[1:]],
            ]
        w = g["w"].numpy()
        zc = z - (w[:, None] * z).sum(0) / w.sum()
        g["raw_response"] = z
        if g["split"] == "train":
            train_gram += (zc * w[:, None]).T @ zc
    rms = np.sqrt(np.maximum(np.diag(train_gram), 0))
    supported = rms > 1e-12
    scale = np.where(supported, rms, 1.0)
    gram = train_gram / scale[:, None] / scale[None, :]
    eigen, vectors = np.linalg.eigh(gram)
    keep = eigen > max(eigen[-1], 1.0) * 1e-10
    basis = vectors[:, keep]
    for i in range(basis.shape[1]):
        first = np.flatnonzero(np.abs(basis[:, i]) > 1e-10)[0]
        if basis[first, i] < 0:
            basis[:, i] *= -1
    for g in groups:
        g["z"] = torch.tensor((g["raw_response"] / scale) @ basis, dtype=torch.float64)
    feature = dict(
        names=names,
        instruments=instruments,
        raw_dimensions=len(names),
        allocated_dimensions=int(keep.sum()),
        scale=scale.tolist(),
        basis=basis.tolist(),
        eigenvalues=eigen.tolist(),
        absent_columns=[n for n, s in zip(names, supported) if not s],
        interpretation="Instrument-specific length/composition slopes; training within-condition contrasts only. Molecular-form mixture remains unresolved.",
    )
    reference_path = Path(cfg["prior_ratio_family"]) / "results/fits.json"
    saved = json.loads(reference_path.read_text())["sequence_context"]
    initial = torch.tensor(saved["coefficients"], dtype=torch.float64)
    assert initial.shape == (40,)
    return cfg, groups, feature, initial


def log_amount(theta, g):
    return (
        log_fragment_expectation(g["x"] @ theta, g["a"], g["b"], g["t"], validate=False)
        / LN10
    )


def objective(theta, gamma, groups, penalty=0.01):
    data = theta.new_zeros(())
    for g in groups:
        if g["split"] != "train":
            continue
        value = log_amount(theta, g)
        if gamma.numel():
            value = value + g["z"] @ gamma
        residual = center(value - g["y"], g["w"])
        data += (g["w"] * residual.square()).sum()
    reg = penalty / 2 * (theta[1:].square().sum() + gamma.square().sum())
    return data + reg, data


def metrics(rows):
    p = np.array([r["centered_prediction"] for r in rows])
    y = np.array([r["centered_target"] for r in rows])
    w = np.array([r["weight"] for r in rows])
    curves = defaultdict(list)
    for row in rows:
        curves[row["curve_id"]].append(row)
    ratio_p, ratio_y, ratio_w = [], [], []
    for curve in curves.values():
        ref = next(r for r in curve if r["time"] == r["reference_time"])
        for row in curve:
            if row is ref:
                continue
            ratio_p.append(row["raw_prediction"] - ref["raw_prediction"])
            ratio_y.append(row["target"] - ref["target"])
            ratio_w.append(row["weight"] * len(curve) / (len(curve) - 1))
    rp, ry, rw = map(np.asarray, (ratio_p, ratio_y, ratio_w))
    assert abs(w.sum() - 1) < 1e-12 and abs(rw.sum() - 1) < 1e-12

    def corr(a, b, fn):
        return (
            None
            if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12
            else float(fn(a, b).statistic)
        )

    return dict(
        n_observations=len(rows),
        shape_mse=float(w @ (p - y) ** 2),
        shape_rmse=float(np.sqrt(w @ (p - y) ** 2)),
        shape_pearson=corr(p, y, pearsonr),
        shape_spearman=corr(p, y, spearmanr),
        prediction_sd=float(p.std()),
        target_sd=float(y.std()),
        ratio_rmse=float(np.sqrt(rw @ (rp - ry) ** 2)),
        ratio_pearson=corr(rp, ry, pearsonr),
        ratio_spearman=corr(rp, ry, spearmanr),
        minimum_log10_amount=min(
            r["log10_amount"] for r in rows if r["log10_amount"] is not None
        )
        if any(r["log10_amount"] is not None for r in rows)
        else None,
        amount_below_1e_20_count=sum(
            r["log10_amount"] is not None and r["log10_amount"] < -20 for r in rows
        ),
    )


def predictions(groups, values, amounts=None, responses=None):
    result = []
    for i, (g, value) in enumerate(zip(groups, values)):
        w = g["w"]
        yc = center(g["y"], w)
        pc = center(value, w)
        offset = float((w * (g["y"] - value)).sum() / w.sum())
        for j, row in enumerate(g["rows"]):
            result.append(
                dict(
                    row,
                    weight=float(w[j]),
                    raw_prediction=float(value[j]),
                    centered_prediction=float(pc[j]),
                    centered_target=float(yc[j]),
                    profiled_offset=offset,
                    log10_amount=None if amounts is None else float(amounts[i][j]),
                    log10_response=None
                    if responses is None
                    else float(responses[i][j]),
                )
            )
    return result
