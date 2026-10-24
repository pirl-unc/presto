"""Independent cleavage with continuous precursor input and fragment removal.

Amounts cover every contiguous product, without normalizing over observed peptides.
The time and input units require external calibration before concentration claims.
"""

import torch


def _terms(internal, left, right, removal):
    if internal.shape != left.shape or internal.shape != right.shape:
        raise ValueError("Internal and boundary rates must have identical shapes")
    if any(not torch.isfinite(x).all() or (x < 0).any() for x in (internal, left, right)):
        raise ValueError("Supply finite nonnegative internal/boundary rates")
    if not torch.isfinite(removal).all() or (removal <= 0).any():
        raise ValueError("Supply a finite positive removal rate")


def log_steady_fragment_amount(internal, left, right, removal, *, validate=True):
    """Log expected amount per unit continuous precursor input.

    `internal` is the sum of rates of bonds inside the fragment. A zero left/right
    rate denotes a physical protein terminus. This stable positive formula is the
    age integral of survival times the fragment's cut/no-cut event probability.
    """
    if validate:
        _terms(internal, left, right, removal)
    total = internal + removal
    value = -total.log()
    for boundary in (left, right):
        present = boundary > 0
        safe = torch.where(present, boundary, torch.ones_like(boundary))
        value = value + torch.where(
            present, safe.log() - (total + safe).log(), torch.zeros_like(total)
        )
    both = (left > 0) & (right > 0)
    return value + torch.where(
        both,
        (2 * total + left + right).log() - (total + left + right).log(),
        torch.zeros_like(total),
    )


def conditional_fragment_age(internal, left, right, removal, uniforms, *, validate=True):
    """Sample age conditional on the requested fragment being observed.

    The final axis of `uniforms` contains boundary-order and three waiting-time
    uniforms. All preceding axes broadcast with the rates. Rates and sampled times
    are detached: use `fragment_event_log_score` for a score-function derivative.
    Outside-fragment bonds have been integrated out, not simulated.
    """
    if validate:
        _terms(internal, left, right, removal)
        if uniforms.shape[-1] != 4 or not torch.isfinite(uniforms).all():
            raise ValueError("Supply four finite uniforms per sample")
        if (uniforms < 0).any() or (uniforms >= 1).any():
            raise ValueError("Uniforms must lie in [0,1)")
    with torch.no_grad():
        total = internal + removal
        waiting = -torch.log1p(-uniforms[..., 1:])
        age = waiting[..., 0] / total
        both = (left > 0) & (right > 0)
        one = (left > 0) ^ (right > 0)
        left_first = uniforms[..., 0] < (total + left) / (2 * total + left + right)
        second_boundary = torch.where(left_first, right, left)
        two_waits = waiting[..., 1] / (total + left + right)
        two_waits = two_waits + waiting[..., 2] / (total + second_boundary)
        one_wait = waiting[..., 1] / (total + left + right)
        age = age + torch.where(both, two_waits, torch.where(one, one_wait, 0.0))
        if validate and (not torch.isfinite(age).all() or (age <= 0).any()):
            raise ValueError("Conditional age must be finite and positive")
    return age


def fragment_event_log_score(internal, left, right, removal, age, *, validate=True):
    """Unnormalized log survival/event likelihood at a detached sampled age.

    Its conditional expectation of the parameter derivative is the derivative of
    log amount. It is not the normalized conditional sampling density, whose mean
    score would be zero, and must not be used as a likelihood for observed MS data.
    """
    if validate:
        _terms(internal, left, right, removal)
        if not torch.isfinite(age).all() or (age <= 0).any():
            raise ValueError("Supply finite positive ages")
    age = age.detach()
    value = -(internal + removal) * age
    for boundary in (left, right):
        present = boundary > 0
        safe = torch.where(present, boundary, torch.ones_like(boundary))
        contribution = torch.log(-torch.expm1(-safe * age))
        value = value + torch.where(present, contribution, torch.zeros_like(value))
    return value
