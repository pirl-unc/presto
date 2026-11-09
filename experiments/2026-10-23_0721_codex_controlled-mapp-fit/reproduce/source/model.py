"""Four-feature conditional captured-signal policy and matched estimators."""

import numpy as np
import torch

FEATURE_NAMES = ("log_length", "fraction_AVILMFYW", "fraction_KR", "fraction_DE")


def features(sequences):
    return np.array(
        [
            [
                np.log(len(s)),
                *(
                    sum(s.count(a) for a in alphabet) / len(s)
                    for alphabet in ("AVILMFYW", "KR", "DE")
                ),
            ]
            for s in sequences
        ],
        dtype=np.float64,
    )


def log_policy(weights, x, log_baseline):
    return torch.log_softmax(log_baseline + x @ weights, dim=0)


def exact_loss(weights, x, log_baseline, log_target, penalty=0.1):
    logp = log_policy(weights, x, log_baseline)
    kl = (logp.exp() * (logp - log_target)).sum()
    return kl + penalty / 2 * weights.square().sum()


def sampled_loss(
    weights, x, log_baseline, log_target, count, generator, penalty=0.1, indices=None
):
    logp = log_policy(weights, x, log_baseline)
    if indices is None:
        indices = torch.multinomial(
            logp.detach().exp(), count, replacement=True, generator=generator
        )
    assert count >= 2 and len(indices) == count
    cost = (logp[indices] - log_target[indices]).detach()
    baseline = (cost.sum() - cost) / (count - 1)
    return (
        (cost - baseline) * logp[indices]
    ).mean() + penalty / 2 * weights.square().sum()


def analytic_gradient(weights, x, log_baseline, log_target, penalty=0.1):
    logp = log_baseline + x @ weights
    logp -= np.logaddexp.reduce(logp)
    p = np.exp(logp)
    cost = logp - log_target
    return x.T @ (p * (cost - p @ cost)) + penalty * weights
