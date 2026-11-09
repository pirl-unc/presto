"""Small conditional MS1 observers; raw labels and source IDs are never features."""

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

AA = "ACDEFGHIKLMNPQRSTVWY"
FEATURE_NAMES = (
    ["length", "length_squared"]
    + ["fraction_" + aa for aa in AA]
    + ["first_" + aa for aa in AA]
    + ["last_" + aa for aa in AA]
)


def features(rows):
    output = []
    for row in rows:
        seq = row["sequence"]
        assert seq and set(seq) <= set(AA)
        output.append(
            [len(seq), len(seq) ** 2]
            + [seq.count(aa) / len(seq) for aa in AA]
            + [float(seq[0] == aa) for aa in AA]
            + [float(seq[-1] == aa) for aa in AA]
        )
    return np.asarray(output, dtype=np.float64)


def group_data(rows):
    groups = sorted({row["pool"] for row in rows})
    lookup = {group: i for i, group in enumerate(groups)}
    indices = np.array([lookup[row["pool"]] for row in rows], dtype=np.int64)
    counts = np.bincount(indices).astype(np.float64)
    weights = 1 / (len(groups) * counts[indices])
    return groups, indices, counts, weights


def center_numpy(values, indices, counts):
    if values.ndim == 1:
        means = np.bincount(indices, weights=values, minlength=len(counts)) / counts
        return values - means[indices]
    means = np.zeros((len(counts), values.shape[1]), dtype=np.float64)
    np.add.at(means, indices, values)
    means /= counts[:, None]
    return values - means[indices]


def center_torch(values, indices, counts):
    means = torch.zeros(len(counts), dtype=values.dtype).index_add(0, indices, values) / counts
    return values - means[indices]


def metrics(target, prediction, weights):
    assert np.all(np.isfinite(target)) and np.all(np.isfinite(prediction))
    weights = weights / weights.sum()
    error = prediction - target
    mse = float(np.dot(weights, error**2))
    target_mean, predicted_mean = float(weights @ target), float(weights @ prediction)
    target_sd = float(np.sqrt(weights @ (target - target_mean) ** 2))
    prediction_sd = float(np.sqrt(weights @ (prediction - predicted_mean) ** 2))
    varies = np.ptp(prediction) > 1e-12 and np.ptp(target) > 1e-12
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": float(weights @ abs(error)),
        "r2": 1 - mse / target_sd**2 if target_sd else None,
        "pearson": float(pearsonr(target, prediction).statistic) if varies else None,
        "spearman": float(spearmanr(target, prediction).statistic) if varies else None,
        "prediction_sd": prediction_sd,
        "target_sd": target_sd,
        "prediction_mean": predicted_mean,
        "target_mean": target_mean,
        "n_examples": len(target),
    }


class Observer(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.hidden = torch.nn.Linear(width, 32, bias=True, dtype=torch.float64)
        self.output = torch.nn.Linear(32, 1, bias=False, dtype=torch.float64)

    def forward(self, x):
        return self.output(torch.tanh(self.hidden(x))).squeeze(-1)


def parameter_penalty(network):
    return torch.cat([value.reshape(-1) for value in network.parameters()]).square().mean()


def numpy_observer(x, state):
    return (
        np.tanh(x @ state["hidden.weight"].T + state["hidden.bias"]) @ state["output.weight"].T
    ).ravel()
