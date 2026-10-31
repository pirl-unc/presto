"""Frozen linear observed-composition model; no test labels are loaded here."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
LN10 = math.log(10)
REGULARIZER = 0.01


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def groups(rows):
    values, inverse = np.unique([r["group_id"] for r in rows], return_inverse=True)
    return values, inverse, np.bincount(inverse)


def center(values, inverse, counts):
    if values.ndim == 1:
        return values - (np.bincount(inverse, weights=values) / counts)[inverse]
    means = np.zeros((len(counts), values.shape[1]))
    np.add.at(means, inverse, values)
    return values - (means / counts[:, None])[inverse]


def features(rows):
    result = np.zeros((len(rows), 61), dtype=np.float64)
    for i, row in enumerate(rows):
        sequence = row["peptide_sequence"]
        result[i, 0] = math.log(len(sequence))
        result[i, 1:21] = [sequence.count(a) / len(sequence) for a in AA]
        result[i, 21 + AA.index(sequence[0])] = 1
        result[i, 41 + AA.index(sequence[-1])] = 1
    return result


class Data:
    def __init__(self):
        seal = json.loads((ROOT / "results/input-seal.json").read_text())
        assert sha(ROOT / "results/covariates.json") == seal["covariates.json"]
        self.rows = json.loads((ROOT / "results/covariates.json").read_text())
        self.raw = features(self.rows)
        self.weights = np.array([r["weight"] for r in self.rows])
        self.indices = {
            s: np.array([i for i, r in enumerate(self.rows) if r["split"] == s], dtype=int)
            for s in ("train", "val", "test")
        }
        self.labels = {}
        for split in ("train", "val"):
            path = ROOT / f"results/{split}-labels.json"
            assert sha(path) == seal[path.name]
            source = json.loads(path.read_text())
            self.labels[split] = {
                k: np.array(
                    [source[self.rows[i]["measurement_id"]][k] for i in self.indices[split]]
                )
                for k in ("y", "p")
            }
        self.group_ids, self.inverse, self.counts = groups(self.rows)

    def basis(self, indices):
        train = self.indices["train"]
        raw = self.raw[:, indices]
        weights = self.weights[train]
        mean = weights @ raw[train]
        scale = np.sqrt(weights @ ((raw[train] - mean) ** 2))
        scale[scale <= 1e-14] = 1
        design = center((raw - mean) / scale, self.inverse, self.counts)
        _, singular, vh = np.linalg.svd(
            design[train] * np.sqrt(weights[:, None]), full_matrices=False
        )
        mask = singular > singular[0] * 1e-10
        projection = vh[mask].T
        assert projection.shape[1] > 0
        basis = {
            "indices": list(indices),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "projection": projection.tolist(),
            "singular_values": singular.tolist(),
            "rank": int(mask.sum()),
        }
        return design @ projection, basis


class Distribution:
    def __init__(self, rows, design, p):
        self.design = torch.as_tensor(design, dtype=torch.float64)
        _, inverse, counts = groups(rows)
        self.inverse, self.counts = inverse, counts
        self.index = np.zeros((len(counts), int(counts.max())), dtype=int)
        mask = np.zeros_like(self.index, dtype=bool)
        group_weights = np.zeros(len(counts))
        for g in range(len(counts)):
            ix = np.flatnonzero(inverse == g)
            self.index[g, : len(ix)] = ix
            mask[g, : len(ix)] = True
            group_weights[g] = rows[ix[0]]["group_weight"]
        self.mask = torch.as_tensor(mask)
        self.ix = torch.as_tensor(self.index)
        self.gw = torch.as_tensor(group_weights, dtype=torch.float64)
        self.logp = torch.as_tensor(np.where(mask, np.log(p[self.index]), 0), dtype=torch.float64)
        assert abs(group_weights.sum() - 1) < 1e-12

    def probabilities(self, beta):
        logits = (LN10 * (self.design @ beta))[self.ix].masked_fill(~self.mask, -torch.inf)
        raw = torch.log_softmax(logits, dim=1)
        logq = raw.masked_fill(~self.mask, 0)
        q = raw.exp()
        return q, logq

    def exact(self, beta):
        q, logq = self.probabilities(beta)
        return (
            self.gw * (q * (logq - self.logp)).sum(1)
        ).sum() + REGULARIZER * beta.square().mean()

    def sampled(self, beta, generator, k=64):
        q, logq = self.probabilities(beta)
        index = torch.multinomial(q, k, replacement=True, generator=generator)
        selected = logq.gather(1, index)
        reward = (logq - self.logp).detach().gather(1, index)
        advantage = (reward - reward.mean(1, keepdim=True)) * (k / (k - 1))
        return (self.gw * (advantage * selected).mean(1)).sum() + REGULARIZER * beta.square().mean()

    def analytic(self, beta):
        x = self.design.numpy()[self.index]
        logits = LN10 * (self.design.numpy() @ beta)[self.index]
        logits[~self.mask.numpy()] = -np.inf
        raw = logits - logsumexp(logits, axis=1, keepdims=True)
        q = np.exp(raw)
        logq = np.where(self.mask.numpy(), raw, 0)
        reward = logq - self.logp.numpy()
        expected = np.einsum("gi,gid->gd", q, x)
        return LN10 * np.einsum(
            "g,gi,gid->d", self.gw.numpy(), q * reward, x - expected[:, None, :]
        ) + 2 * REGULARIZER * beta / len(beta)


def metrics(rows, y, p, prediction):
    weight = np.array([r["weight"] for r in rows])
    weight = weight / weight.sum()
    _, inverse, counts = groups(rows)
    prediction = center(np.asarray(prediction), inverse, counts)
    error = float(weight @ ((prediction - y) ** 2))
    mean = float(weight @ y)
    variance = float(weight @ ((y - mean) ** 2))
    reverse_kl = 0.0
    fractions = np.zeros(len(rows))
    for g in range(len(counts)):
        ix = np.flatnonzero(inverse == g)
        lq = LN10 * prediction[ix] - logsumexp(LN10 * prediction[ix])
        q = np.exp(lq)
        fractions[ix] = q
        reverse_kl += weight[ix].sum() * float(q @ (lq - np.log(p[ix])))
    nonconstant = float(np.ptp(prediction)) > 1e-14
    return {
        "mse": error,
        "rmse": math.sqrt(error),
        "mae": float(weight @ abs(prediction - y)),
        "r2": 1 - error / variance,
        "spearman": float(spearmanr(y, prediction).statistic) if nonconstant else None,
        "pearson": float(pearsonr(y, prediction).statistic) if nonconstant else None,
        "prediction_sd": float(np.sqrt(weight @ ((prediction - weight @ prediction) ** 2))),
        "target_sd": math.sqrt(variance),
        "reverse_kl": reverse_kl,
    }, fractions
