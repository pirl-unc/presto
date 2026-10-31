"""Artificial fixtures for scientific selection and independent metric semantics."""

from pathlib import Path
import importlib.util, json
import pytest


def module(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_promotion_rejects_one_class_regression_and_breaks_capacity_tie(
    tmp_path, monkeypatch
):
    m = module("select_validation")
    monkeypatch.setattr(m, "F", tmp_path)
    (tmp_path / "results").mkdir()
    values = {
        "ic50-exact": (0.5, 0.5, 500),
        "ic50-bounds": (0.49, 0.7, 500),
        "multi-assay": (0.53, 0.53, 600),
        "multi-assay-wide": (0.531, 0.531, 1600),
    }
    for variant, (i, ii, size) in values.items():
        for seed in (42, 43):
            p = tmp_path / "results" / f"{variant}-s{seed}"
            p.mkdir()
            report = {
                "parameters": size,
                "common_ic50": {
                    "validation": {
                        "ic50": {
                            "by_class": {
                                c: {"spearman": r, "rmse": 1.0, "n": 100}
                                for c, r in [("I", i), ("II", ii)]
                            }
                        }
                    },
                    "test": {
                        "this_payload_must_not_be_used_for_selection": "deliberately unusable"
                    },
                },
            }
            (p / "report.json").write_text(json.dumps(report))
    m.main()
    decision = json.loads((tmp_path / "results/validation-decision.json").read_text())
    assert decision["preferred_variant"] == "multi-assay"
    assert not decision["comparisons"]["ic50-bounds"]["eligible"]
    assert decision["test_used_for_decision"] is False
    with pytest.raises(ValueError, match="already sealed"):
        m.main()


def test_independent_metrics_preserve_bounds_and_half_life_units():
    m = module("verify")
    rows = [
        {"target": 1.0, "prediction": 1.5, "qualifier": "=", "loss": 0.25},
        {"target": 3.0, "prediction": 2.0, "qualifier": "=", "loss": 1.0},
        {"target": 3.0, "prediction": 2.0, "qualifier": ">", "loss": 1.0},
        {"target": 3.0, "prediction": 2.0, "qualifier": "<", "loss": 0.0},
    ]
    concentration = m.metrics(rows, 1.0, True)
    stability = m.metrics(rows, 1.0, False)
    assert (
        concentration["censor_satisfaction"] == 0.5
        and concentration["n"] == 2
        and concentration["n_all"] == 4
    )
    assert concentration["rmse"] == pytest.approx((0.625) ** 0.5)
    assert "binding_at_500_nM" in concentration and "binding_at_500_nM" not in stability
    assert m.physical({"value": 0.2, "unit": "uM"}) == pytest.approx(2.3010299956639813)
