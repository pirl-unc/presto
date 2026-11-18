from __future__ import annotations

import pytest

from presto.evaluation.performance_figures import (
    render_affinity_calibration,
    render_affinity_scatter,
    render_discrimination_curves,
    render_study_deltas,
)


def _affinity_rows():
    rows = []
    for mhc_class, offset in (("I", 0.0), ("II", 0.2)):
        for method, error in (("Presto", 0.1), ("Comparator", -0.15)):
            for index, target in enumerate((1.0, 2.0, 3.0, 4.0)):
                rows.append(
                    {
                        "observation_id": f"{mhc_class}-{index}",
                        "method": method,
                        "mhc_class": mhc_class,
                        "target_log10_nM": target + offset,
                        "prediction_log10_nM": target + offset + error * (-1) ** index,
                    }
                )
    return rows


def _binary_rows():
    rows = []
    for method, scores in (
        ("Presto", (0.95, 0.75, 0.30, 0.05)),
        ("Comparator", (0.8, 0.6, 0.4, 0.2)),
    ):
        for index, (target, score) in enumerate(zip((1, 1, 0, 0), scores, strict=True)):
            rows.append(
                {
                    "observation_id": str(index),
                    "method": method,
                    "target": target,
                    "score": score,
                }
            )
    return rows


def test_diagnostic_figures_are_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        render_affinity_scatter(_affinity_rows(), root)
        render_affinity_calibration(_affinity_rows(), root)
        render_discrimination_curves(
            _binary_rows(), root, stem="presentation_curves", title="Presentation"
        )
        render_study_deltas(
            [
                {
                    "study": "study-a",
                    "metric": "auroc",
                    "delta": 0.1,
                    "ci_low": 0.05,
                    "ci_high": 0.15,
                },
                {
                    "study": "pooled_bootstrap",
                    "metric": "auroc",
                    "delta": 0.08,
                    "ci_low": 0.02,
                    "ci_high": 0.13,
                },
            ],
            root,
            stem="presentation_study_deltas",
            title="Presentation deltas",
        )
    first_files = sorted(path.relative_to(first) for path in first.iterdir())
    second_files = sorted(path.relative_to(second) for path in second.iterdir())
    assert first_files == second_files
    for relative in first_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_diagnostic_figures_reject_nonfinite_values(tmp_path):
    rows = _affinity_rows()
    rows[0]["prediction_log10_nM"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        render_affinity_scatter(rows, tmp_path)
