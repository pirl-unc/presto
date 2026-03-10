"""Tests for shared affinity calibration utilities."""

import math

import torch


def test_affinity_nm_to_log10_clamps_to_global_max():
    from presto.models.affinity import affinity_nm_to_log10

    values = torch.tensor([10.0, 1000000.0], dtype=torch.float32)
    converted = affinity_nm_to_log10(values)
    assert torch.allclose(converted[0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(converted[1], torch.tensor(math.log10(50000.0)), atol=1e-6)


def test_binding_prob_from_kd_log10_matches_reference_formula():
    from presto.models.affinity import binding_prob_from_kd_log10

    kd_log10 = 3.0
    midpoint = 500.0
    scale = 0.35
    expected = 1.0 / (1.0 + math.exp(-((math.log10(midpoint) - kd_log10) / scale)))
    observed = binding_prob_from_kd_log10(
        kd_log10,
        midpoint_nM=midpoint,
        log10_scale=scale,
    )
    assert abs(observed - expected) < 1e-8


def test_normalize_binding_target_log10_auto_detects_nm_and_log10():
    from presto.models.affinity import normalize_binding_target_log10

    nm_values = torch.tensor([[50000.0], [100000.0]], dtype=torch.float32)
    nm_normalized = normalize_binding_target_log10(nm_values)
    assert torch.allclose(
        nm_normalized.squeeze(-1),
        torch.tensor([math.log10(50000.0), math.log10(50000.0)]),
        atol=1e-5,
    )

    log_values = torch.tensor([[4.7], [5.3]], dtype=torch.float32)
    log_normalized = normalize_binding_target_log10(log_values)
    assert torch.allclose(log_normalized[0], torch.tensor([math.log10(50000.0)]), atol=1e-6)
    assert torch.allclose(log_normalized[1], torch.tensor([math.log10(50000.0)]), atol=1e-6)
