import math

from presto.scripts.sweep_20m_models import (
    generate_candidates,
    summarize_loss_series,
)


def test_generate_candidates_prefers_near_target_and_respects_band():
    candidates = generate_candidates(
        target_params=20_000_000,
        min_params=18_000_000,
        max_params=22_000_000,
        d_models=[192, 224, 256],
        layer_min=2,
        layer_max=5,
        heads=[4, 8, 16],
        max_candidates=6,
    )

    assert candidates
    assert len(candidates) <= 6
    for cand in candidates:
        assert 18_000_000 <= cand.n_params <= 22_000_000
        assert cand.d_model % cand.n_heads == 0

    deltas = [abs(c.n_params - 20_000_000) for c in candidates]
    assert deltas == sorted(deltas)


def test_summarize_loss_series_reports_drop_speed():
    series = [(1, 10.0), (2, 8.0), (3, 7.0)]
    summary = summarize_loss_series(series, prefix="val")

    assert summary["val_first_loss"] == 10.0
    assert summary["val_last_loss"] == 7.0
    assert summary["val_best_loss"] == 7.0
    assert summary["val_best_step"] == 3.0
    assert summary["val_drop_abs"] == 3.0
    assert summary["val_drop_per_epoch"] == 1.5
    assert summary["val_slope"] == -1.5
    assert summary["val_speed"] == 1.5
    assert summary["val_num_points"] == 3.0


def test_summarize_loss_series_empty_defaults_to_nan():
    summary = summarize_loss_series([], prefix="train")

    assert summary["train_num_points"] == 0.0
    assert math.isnan(summary["train_first_loss"])
    assert math.isnan(summary["train_last_loss"])
    assert math.isnan(summary["train_drop_per_epoch"])
