"""Exact N-terminal trimming paths and a matched conditional-score gradient.

Source influx and clearance establish the units. This module supports monomers and
nonbinding products; it does not infer enzyme activity or MHC selection from labels.
"""

import torch
import torch.nn.functional as F


def log_trimming_paths(
    log_birth, removed_counts, owner, target_residue, target_monomer, log_rates, log_scale
):
    """Log contribution of every birth path to its queried terminal span.

    Clearance is one. A path loses each residue counted in ``removed_counts``;
    its end state has clearance plus its own trimming hazard, except monomers,
    which clear without further trimming. Counts are exact sufficient statistics
    when a trimming rate depends only on the residue removed, as assumed here.
    """
    rates = log_rates + log_scale
    log_survival = F.logsigmoid(rates)
    log_hazard = torch.where(target_monomer, 0.0, F.softplus(rates[target_residue]))
    return log_birth + removed_counts @ log_survival - log_hazard[owner]


def grouped_logsumexp(scores, owner, n_queries):
    """Stable finite mixture sums, without normalizing over observed peptides."""
    with torch.no_grad():
        maximum = torch.full((n_queries,), -torch.inf, dtype=scores.dtype, device=scores.device)
        maximum.scatter_reduce_(0, owner, scores, reduce="amax", include_self=True)
    shifted = (scores - maximum[owner]).exp()
    total = torch.zeros_like(maximum).scatter_add(0, owner, shifted)
    return maximum + total.log(), shifted / total[owner]


def conditional_path_indices(probabilities, starts, ends, uniforms):
    """Sample each contiguous ragged query's path with supplied uniform quantiles."""
    with torch.no_grad():
        cumulative = torch.cat((probabilities.new_zeros(1), probabilities.cumsum(0)))
        lower, upper = cumulative[starts], cumulative[ends]
        target = lower[:, None] + (upper - lower)[:, None] * uniforms
        selected = torch.searchsorted(cumulative[1:].contiguous(), target, right=True)
        if ((selected < starts[:, None]) | (selected >= ends[:, None])).any():
            raise ValueError("Conditional trimming sample crossed its query boundary")
        return selected


def trimmed_log_amount(
    log_birth,
    removed_counts,
    owner,
    starts,
    ends,
    target_residue,
    target_monomer,
    log_rates,
    log_scale,
    *,
    uniforms=None,
):
    """Exact forward amounts; optional unbiased conditional-score gradient.

    Sampling uses the exact posterior birth-path distribution. The forward value
    remains exact: this is a hybrid estimator, not a simulator-only policy model.
    """
    n_queries = len(target_residue)
    if uniforms is None:
        scores = log_trimming_paths(
            log_birth, removed_counts, owner, target_residue, target_monomer, log_rates, log_scale
        )
        return grouped_logsumexp(scores, owner, n_queries)[0]
    with torch.no_grad():
        scores = log_trimming_paths(
            log_birth, removed_counts, owner, target_residue, target_monomer, log_rates, log_scale
        )
        exact, probability = grouped_logsumexp(scores, owner, n_queries)
        selected = conditional_path_indices(probability, starts, ends, uniforms)
    rates = log_rates + log_scale
    survival = F.logsigmoid(rates)
    hazard = torch.where(target_monomer, 0.0, F.softplus(rates[target_residue]))
    sampled = (log_birth[selected] + removed_counts[selected] @ survival - hazard[:, None]).mean(1)
    return exact + sampled - sampled.detach()


def log_state_trimming_paths(log_birth, log_state_rates, terminal_index, state_monomer):
    """Birth-to-target contributions for contiguous, ordered origin segments.

    Each segment contains starts 0..a for fixed end b, ending at target start a.
    ``terminal_index[j]`` is the last index of the segment containing j. Rates
    may depend on any state context. Prefix differences sum transitions j..a-1;
    the target contributes its residence factor. Monomers clear without trimming.
    This evaluates every enumerated state/path. Float64 is recommended when many
    segments share the prefix sum, because subtraction can lose precision.
    """
    survival = F.logsigmoid(log_state_rates)
    before = torch.cat((survival.new_zeros(1), survival.cumsum(0)[:-1]))
    hazard = torch.where(state_monomer, 0.0, F.softplus(log_state_rates))
    return log_birth + (before[terminal_index] - before) - hazard[terminal_index]


def state_trimmed_log_amount(
    log_birth,
    log_state_rates,
    terminal_index,
    state_monomer,
    owner,
    starts,
    ends,
    *,
    uniforms=None,
):
    """State-dependent exact amounts or matched conditional-score gradients.

    The optional estimator samples from the exact path posterior. It keeps exact
    forward values and still evaluates the complete state/path representation.
    """
    scores = log_state_trimming_paths(log_birth, log_state_rates, terminal_index, state_monomer)
    if uniforms is None:
        return grouped_logsumexp(scores, owner, len(starts))[0]
    with torch.no_grad():
        exact, probability = grouped_logsumexp(scores.detach(), owner, len(starts))
        selected = conditional_path_indices(probability, starts, ends, uniforms)
    sampled = scores[selected].mean(1)
    return exact + sampled - sampled.detach()
