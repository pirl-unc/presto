"""One irreversible independent-cut process, exact amounts and sampled partitions."""

import torch


def spans(length, device=None):
    starts, last = torch.triu_indices(length, length, device=device)
    return starts, last + 1


def expected(rates, times):
    """Expected fragment copies per initial molecule; no normalization over products."""
    length = rates.numel() + 1
    starts, ends = spans(length, rates.device)
    hazard = times[:, None] * rates[None, :]
    boundary = torch.cat(
        [
            torch.ones_like(times[:, None]),
            -torch.expm1(-hazard),
            torch.ones_like(times[:, None]),
        ],
        dim=1,
    )
    prefix = torch.cat([rates.new_zeros(1), rates.cumsum(0)])
    internal = prefix[ends - 1] - prefix[starts]
    return (
        boundary[:, starts] * boundary[:, ends] * torch.exp(-times[:, None] * internal)
    )


def sample(rates, times, count, generator):
    """Two independent groups of partitions at each time, with exact policy scores.

    This bounded fixture stores dense span counts for comparison; production
    trajectory sampling can retain only the returned nonzero spans.
    """
    if count < 2 or bool((times <= 0).any()):
        raise ValueError("Score estimator requires count>=2 and positive times")
    length = rates.numel() + 1
    hazard = times[:, None] * rates[None, :]
    probability = -torch.expm1(-hazard)
    cuts = (
        torch.rand(
            (len(times), 2, count, length - 1), dtype=rates.dtype, generator=generator
        )
        < probability.detach()[:, None, None, :]
    )
    log_probability = torch.where(
        cuts, probability.log()[:, None, None, :], -hazard[:, None, None, :]
    ).sum(-1)
    flat = cuts.reshape(-1, length - 1)
    terminal = torch.ones((len(flat), 1), dtype=torch.bool)
    boundaries = torch.cat([terminal, flat, terminal], dim=1)
    owners, positions = boundaries.nonzero(as_tuple=True)
    same = owners[:-1] == owners[1:]
    owners = owners[:-1][same]
    starts, ends = positions[:-1][same], positions[1:][same]
    index = starts * length - starts * (starts - 1) // 2 + ends - starts - 1
    counts = rates.new_zeros((len(flat), length * (length + 1) // 2))
    counts[owners, index] = 1
    return counts.reshape(len(times), 2, count, -1), log_probability


def sampled_loss(rates, times, target, count, generator):
    counts, logp = sample(rates, times, count, generator)
    residual = counts - target[:, None, None, :]
    means = residual.mean(2)
    costs = (residual * means.flip(1)[:, :, None, :]).mean(-1)
    baseline = (costs.sum(-1, keepdim=True) - costs) / (count - 1)
    surrogate = ((costs - baseline).detach() * logp).mean(-1).sum(-1).mean()
    estimate = (means[:, 0] * means[:, 1]).mean()
    return surrogate, estimate, int(counts.sum())
