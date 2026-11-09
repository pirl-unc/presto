"""Prespecified descriptive diagnostics, with no fitted parameters."""

import itertools

import numpy as np
from scipy.stats import rankdata

LABELS = ["Cont1", "Cont2", "Cont3", "KD1", "KD2", "KD3"]


def correlation(a, b, w):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ac, bc = a - np.dot(w, a), b - np.dot(w, b)
    return float(np.dot(w, ac * bc) / np.sqrt(np.dot(w, ac**2) * np.dot(w, bc**2)))


def agreement(a, b, w):
    eligible = (a != 0) & (b != 0)
    mass = float(w[eligible].sum())
    same = eligible & (np.sign(a) == np.sign(b))
    return {
        "eligible_weight": mass,
        "excluded_zero_count": int((~eligible).sum()),
        "same_sign_fraction": float(w[same].sum() / mass) if mass else None,
        "opposite_sign_fraction": float(w[eligible & ~same].sum() / mass)
        if mass
        else None,
    }


def spread(x, w):
    return {
        "mean": float(w @ x),
        "rms": float(np.sqrt(w @ (x**2))),
        "sd": float(np.sqrt(w @ ((x - w @ x) ** 2))),
    }


def diagnostics(ratios, weights):
    r, w = np.asarray(ratios, float), np.asarray(weights, float)
    assert r.ndim == 2 and r.shape[1] == 6
    assert np.isfinite(r).all() and np.isfinite(w).all() and (w > 0).all()
    w = w / w.sum()
    arm_mean = np.stack([r[:, :3].mean(1), r[:, 3:].mean(1)], axis=1)
    y = arm_mean[:, 1] - arm_mean[:, 0]
    residual = r - np.repeat(arm_mean, 3, axis=1)
    omitted = np.stack(
        [
            r[:, 3:].mean(1) - np.delete(r[:, :3], j, axis=1).mean(1)
            if j < 3
            else np.delete(r[:, 3:], j - 3, axis=1).mean(1) - r[:, :3].mean(1)
            for j in range(6)
        ],
        axis=1,
    )
    within_rms = float(np.sqrt(w @ (residual**2).mean(1)))
    response = spread(y, w)
    omissions = []
    for j, label in enumerate(LABELS):
        delta = omitted[:, j] - y
        omissions.append(
            dict(
                omitted_acquisition=label,
                response=spread(omitted[:, j], w),
                change=spread(delta, w),
                mean_absolute_change=float(w @ np.abs(delta)),
                sign=agreement(y, omitted[:, j], w),
            )
        )
    pairs, partitions = [], []
    for c, k in itertools.product(range(3), range(3, 6)):
        a = r[:, k] - r[:, c]
        b = np.delete(r[:, 3:], k - 3, axis=1).mean(1) - np.delete(
            r[:, :3], c, axis=1
        ).mean(1)
        pairs.append(np.stack([a, b], axis=1))
        partitions.append(
            dict(
                control=LABELS[c],
                kd=LABELS[k],
                selected=spread(a, w),
                remaining=spread(b, w),
                pearson=correlation(a, b, w),
                spearman=correlation(rankdata(a), rankdata(b), w),
                sign=agreement(a, b, w),
            )
        )
    eligible = y != 0
    stable = eligible & np.all(np.sign(omitted) == np.sign(y[:, None]), axis=1)
    mass = float(w[eligible].sum())
    values = [row["pearson"] for row in partitions if row["pearson"] is not None]
    median = float(np.median(values)) if values else None
    summary = dict(
        n_contrasts=len(r),
        response=response,
        within_arm_rms=within_rms,
        within_control_rms=float(np.sqrt(w @ (residual[:, :3] ** 2).mean(1))),
        within_kd_rms=float(np.sqrt(w @ (residual[:, 3:] ** 2).mean(1))),
        within_to_response_rms_ratio=within_rms / response["rms"]
        if response["rms"]
        else None,
        acquisitions=[
            dict(label=label, **spread(residual[:, j], w))
            for j, label in enumerate(LABELS)
        ],
        omissions=omissions,
        partitions=partitions,
        stable_nonzero_sign_fraction=float(w[stable].sum() / mass) if mass else None,
        sign_eligible_weight=mass,
        zero_original_response_count=int((~eligible).sum()),
        median_partition_pearson=median,
        defined_partition_correlations=len(values),
        next_step="training_only_boundary_context_comparison"
        if median is not None and median > 0
        else "observation_stability_and_replication_metadata",
    )
    return summary, y, residual, omitted, np.stack(pairs, axis=1), stable
