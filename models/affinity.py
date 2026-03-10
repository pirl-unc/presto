"""Shared affinity calibration utilities.

Centralizes all KD/IC50 scaling decisions so training, model logits,
and inference probabilities stay consistent.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch


Number = Union[float, int]


# Global affinity range for Presto (nM).
DEFAULT_MIN_AFFINITY_NM: float = 1e-3
DEFAULT_MAX_AFFINITY_NM: float = 50000.0

# Logistic calibration used for binding logits/probabilities.
DEFAULT_BINDING_MIDPOINT_NM: float = 500.0
DEFAULT_BINDING_LOG10_SCALE: float = 0.35


def max_log10_nM(max_affinity_nM: float = DEFAULT_MAX_AFFINITY_NM) -> float:
    """Return the log10-space upper bound corresponding to max affinity (nM)."""
    return math.log10(max(float(max_affinity_nM), 1e-12))


def affinity_nm_to_log10(
    value: Union[torch.Tensor, Number],
    *,
    min_affinity_nM: float = DEFAULT_MIN_AFFINITY_NM,
    max_affinity_nM: float = DEFAULT_MAX_AFFINITY_NM,
) -> Union[torch.Tensor, float]:
    """Convert affinity values from nM -> log10(nM) with shared clamping."""
    min_nm = max(float(min_affinity_nM), 1e-12)
    max_nm = max(float(max_affinity_nM), min_nm)

    if isinstance(value, torch.Tensor):
        return torch.log10(value.float().clamp(min=min_nm, max=max_nm))
    return math.log10(min(max(float(value), min_nm), max_nm))


def affinity_log10_to_nm(value: Union[torch.Tensor, Number]) -> Union[torch.Tensor, float]:
    """Convert affinity values from log10(nM) -> nM."""
    if isinstance(value, torch.Tensor):
        return torch.pow(10.0, value.float())
    return float(10.0 ** float(value))


def binding_logit_from_kd_log10(
    kd_log10_nM: Union[torch.Tensor, Number],
    *,
    midpoint_nM: float = DEFAULT_BINDING_MIDPOINT_NM,
    log10_scale: float = DEFAULT_BINDING_LOG10_SCALE,
) -> Union[torch.Tensor, float]:
    """Map KD log10(nM) to binding logit with shared calibration."""
    midpoint_log10 = math.log10(max(float(midpoint_nM), 1e-12))
    scale = max(float(log10_scale), 1e-6)
    if isinstance(kd_log10_nM, torch.Tensor):
        return (kd_log10_nM.float() - midpoint_log10).neg() / scale
    return (midpoint_log10 - float(kd_log10_nM)) / scale


def binding_prob_from_kd_log10(
    kd_log10_nM: Union[torch.Tensor, Number],
    *,
    midpoint_nM: float = DEFAULT_BINDING_MIDPOINT_NM,
    log10_scale: float = DEFAULT_BINDING_LOG10_SCALE,
) -> Union[torch.Tensor, float]:
    """Map KD log10(nM) to binding probability [0,1]."""
    logits = binding_logit_from_kd_log10(
        kd_log10_nM,
        midpoint_nM=midpoint_nM,
        log10_scale=log10_scale,
    )
    if isinstance(logits, torch.Tensor):
        return torch.sigmoid(logits)
    return 1.0 / (1.0 + math.exp(-float(logits)))


def normalize_binding_target_log10(
    value: torch.Tensor,
    *,
    min_affinity_nM: float = DEFAULT_MIN_AFFINITY_NM,
    max_affinity_nM: float = DEFAULT_MAX_AFFINITY_NM,
    assume_log10: Optional[bool] = None,
    detection_margin_log10: float = 1.0,
) -> torch.Tensor:
    """Normalize binding targets to log10(nM).

    By default this auto-detects representation:
    - values mostly <= max_log10 + margin are treated as already-log10
    - larger values are treated as nM and converted.
    """
    vec = value.float()
    max_log10 = max_log10_nM(max_affinity_nM)
    min_log10 = math.log10(max(float(min_affinity_nM), 1e-12))

    if assume_log10 is None:
        finite = vec[torch.isfinite(vec)]
        if finite.numel() == 0:
            assume_log10 = True
        else:
            assume_log10 = float(finite.max().item()) <= (max_log10 + float(detection_margin_log10))

    if assume_log10:
        return vec.clamp(min=min_log10, max=max_log10)
    return affinity_nm_to_log10(
        vec,
        min_affinity_nM=min_affinity_nM,
        max_affinity_nM=max_affinity_nM,
    )
