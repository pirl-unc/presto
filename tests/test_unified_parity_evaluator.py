from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-11-08_1800_codex_unified-parity-retrain/code/evaluate_pair.py"
)
SPEC = importlib.util.spec_from_file_location("unified_parity_evaluator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _prediction(
    observation_id: str,
    endpoint: str,
    target: float,
    prediction: float,
    mhc_class: str = "I",
) -> dict:
    raw_value = 10**target
    return {
        "observation_id": observation_id,
        "endpoint": endpoint,
        "mhc_classes": [mhc_class],
        "assay_method": {
            "preparation": "purified",
            "mechanism": "competition",
            "detector": "radioactivity",
            "review_status": "reviewed",
        },
        "qualifier": "=",
        "target": target,
        "raw_value": raw_value,
        "peptide": f"PEPTIDE{observation_id[-1]}",
        "source": "fixture",
        "group_id": "study-1",
        "observation": {"provenance": {"study_id": "study-1"}},
        "prediction": prediction,
    }


def _comparator(row: dict) -> dict:
    method = row["assay_method"]
    return {
        "observation_id": row["observation_id"],
        "peptide": row["peptide"],
        "mhc_class": row["mhc_classes"][0],
        "endpoint": "ic50",
        "qualifier": "=",
        "target_log10": row["target"],
        "raw_nM": row["raw_value"],
        "source": "fixture",
        "study": "study-1",
        "preparation": method["preparation"],
        "mechanism": method["mechanism"],
        "detector": method["detector"],
        "assay_review_status": method["review_status"],
        "comparator_log10": row["target"] + 0.1,
    }


def test_affinity_table_preserves_non_ic50_assay_outputs() -> None:
    truth = {
        "ic50-low": _prediction("ic50-low", "ic50", 2.0, 2.1),
        "ic50-high": _prediction("ic50-high", "ic50", 4.0, 3.9),
        "ec50-low": _prediction("ec50-low", "ec50", 2.2, 2.3),
        "ec50-high": _prediction("ec50-high", "ec50", 4.2, 4.0),
        "ii-ic50-low": _prediction("ii-ic50-low", "ic50", 2.1, 2.2, "II"),
        "ii-ic50-high": _prediction("ii-ic50-high", "ic50", 4.1, 4.0, "II"),
        "ii-ec50-low": _prediction("ii-ec50-low", "ec50", 2.3, 2.4, "II"),
        "ii-ec50-high": _prediction("ii-ec50-high", "ec50", 4.3, 4.1, "II"),
    }
    indexed = {
        method: {
            observation_id: {**row, "prediction": row["prediction"] + offset}
            for observation_id, row in truth.items()
        }
        for method, offset in (
            ("Presto reference", 0.0),
            ("Presto parity", 0.05),
            ("Presto Stage B", -0.05),
        )
    }
    comparator = [
        _comparator(truth[key]) for key in ("ic50-low", "ic50-high", "ii-ic50-low", "ii-ic50-high")
    ]

    rows = EVALUATOR.affinity_rows(indexed, comparator)

    assert len(rows) == 28
    assert {row["endpoint"] for row in rows} == {"ic50", "ec50"}
    assert {row["endpoint"] for row in rows if row["method"] == "NetMHCpan 4.2c"} == {"ic50"}
    assert sum(row["endpoint"] == "ec50" for row in rows) == 12

    summaries, _ = EVALUATOR.affinity_summary(rows)
    ec50_key = "ec50/purified/competition/radioactivity/reviewed"
    ec50 = [
        row for row in summaries if row["assay_method"] == ec50_key and row["population"] == "exact"
    ]
    assert {row["method"] for row in ec50} == {
        "Presto reference",
        "Presto parity",
        "Presto Stage B",
    }
    assert {row["metric"] for row in ec50} == {"spearman", "pearson", "rmse"}
