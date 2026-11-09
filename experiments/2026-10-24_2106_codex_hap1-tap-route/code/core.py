"""Conditional two-route response model; no absolute concentration interpretation."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
CONTEXT = ROOT / "experiments/2026-10-24_2018_codex_hap1-context-ablation"
SCORES = ROOT / "experiments/2026-10-24_2041_codex_hap1-affinity-control"
RECIPE = ROOT / "data/hap1_response_recipes/2026-10-24_0730_v1"
CONDITIONS = ["hap1_tap1_ko", "hap1_tap2_ko"]
ALLELES = ["HLA-A*02:01", "HLA-B*40:01", "HLA-C*03:04"]
VARIANTS = [
    "sequence_scores",
    "expanded_ridge",
    "route_only_exact",
    "route_scores_exact",
    "route_scores_sampled",
]
ETA0 = np.log(0.05 / 0.95)
LN10 = np.log(10.0)

spec = importlib.util.spec_from_file_location("frozen_reference", CONTEXT / "code/core.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data():
    context_manifest = json.loads((CONTEXT / "artifact-manifest.json").read_text())
    score_manifest = json.loads((SCORES / "artifact-manifest.json").read_text())["files"]
    for name in ["code/core.py", "results/fits/context-features.npz"]:
        assert digest(CONTEXT / name) == context_manifest[name]["sha256"]
    for name in ["results/predictor/scores.parquet", "results/predictor/exposure.parquet"]:
        assert digest(SCORES / name) == score_manifest[name]["sha256"]
    assert digest(RECIPE / "manifest.json") == (
        "a1eec392c06485966e131663f85e7599dcd9e5ff621841eb0d44eeaafe779309"
    )
    table = pq.read_table(
        RECIPE / "derived/responses.parquet",
        filters=[
            ("split", "=", "train"),
            ("precursor_status", "=", "complete"),
            ("condition_id", "in", CONDITIONS),
        ],
    )
    rows = table.to_pylist()
    ctx = np.load(CONTEXT / "results/fits/context-features.npz")
    geometry = dict(zip(ctx["keys"], ctx["features"], strict=True))
    scores = pd.read_parquet(SCORES / "results/predictor/scores.parquet")
    lookup = {
        p: np.log10(g.set_index("allele").loc[ALLELES, "prediction"].to_numpy())
        for p, g in scores.groupby("peptide")
    }
    exposure = pd.read_parquet(SCORES / "results/predictor/exposure.parquet")
    exposure = exposure.set_index("peptide")["any_training_peptide"].to_dict()
    availability, observation = [], []
    for r in rows:
        seq = r["peptide_sequence"]
        assert 8 <= len(seq) <= 14
        availability.append(
            [
                *reference.sequence_features(seq),
                *reference.onehot(seq[1]),
                *reference.onehot(seq[2]),
                *geometry[reference.key(r)][:4],
            ]
        )
        observation.append(lookup[seq])
    fold = np.array(
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
    return {
        "table": table,
        "rows": rows,
        "availability": np.array(availability),
        "scores": np.array(observation),
        "fold": fold,
        "condition": np.array([CONDITIONS.index(r["condition_id"]) for r in rows]),
        "precursor": np.array([r["precursor_log10_ratio"] for r in rows]),
        "y": np.array([r["target_log10_ratio"] for r in rows]),
        "weights": reference.weights(rows),
        "exposure": np.array([exposure[r["peptide_sequence"]] for r in rows]),
    }


def log_response(z, eta, generator=None):
    """Natural log of the two-path sum, optionally using conditional-score gradients."""
    bypass = torch.nn.functional.logsigmoid(-z)
    affected = torch.nn.functional.logsigmoid(z) + torch.nn.functional.logsigmoid(eta)
    exact = torch.logaddexp(bypass, affected)
    if generator is None:
        return exact
    posterior = (affected - exact).exp().detach()
    count = torch.binomial(torch.full_like(posterior, 64), posterior, generator=generator)
    fraction = count / 64
    proxy = (1 - fraction) * bypass + fraction * affected
    return exact.detach() + proxy - proxy.detach()


def standardize(a, w):
    mean = w @ a
    sd = np.sqrt(w @ (a - mean) ** 2)
    mask = sd > 1e-12
    return (a[:, mask] - mean[mask]) / sd[mask], {
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "mask": mask.tolist(),
    }


def transform(a, state):
    mean, sd, mask = (np.array(state[k]) for k in ["mean", "sd", "mask"])
    return (a[:, mask] - mean[mask]) / sd[mask]


def objective(theta, eta, beta, z, scores, condition, y, precursor, w, gamma, generator=None):
    response = log_response(theta[0] + z @ theta[1:], eta[condition], generator) / LN10
    score_term = (scores * beta[condition]).sum(-1)
    raw = gamma * precursor + response + score_term
    offsets = torch.stack(
        [
            (w[condition == c] * (y - raw)[condition == c]).sum() / w[condition == c].sum()
            for c in range(2)
        ]
    )
    prediction = raw + offsets[condition]
    mse = (w * (prediction - y).square()).sum()
    penalty = 0.01 * (theta.square().sum() + (eta - ETA0).square().mean())
    penalty = penalty + 0.1 * beta.square().sum(-1).mean()
    return mse + penalty, prediction, offsets, response, score_term


def numpy_prediction(d, indices, cp):
    z = transform(d["availability"][indices], cp["availability_transform"])
    s = transform(d["scores"][indices], cp["score_transform"])
    theta, eta, beta = (np.array(cp[k]) for k in ["theta", "eta", "beta"])
    condition = d["condition"][indices]
    logit = theta[0] + z @ theta[1:]
    lb = -np.logaddexp(0, logit)
    la = -np.logaddexp(0, -logit) - np.logaddexp(0, -eta[condition])
    log_r = np.logaddexp(lb, la)
    score = np.sum(s * beta[condition], axis=1)
    prediction = cp["gamma"] * d["precursor"][indices]
    prediction += np.array(cp["offsets"])[condition] + log_r / LN10 + score
    return prediction, {
        "rho": 1 / (1 + np.exp(-logit)),
        "epsilon": 1 / (1 + np.exp(-eta[condition])),
        "relative_delivery": np.exp(log_r),
        "transport_log10": log_r / LN10,
        "score_term": score,
    }
