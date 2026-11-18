from __future__ import annotations

import copy
import math

import pytest

from presto.evaluation.paired_benchmark import (
    affinity_threshold_metrics,
    operational_processing_logit,
    paired_study_bootstrap,
    require_aligned_predictions,
    study_bootstrap,
)


def _prediction(observation_id="row-1"):
    return {
        "observation_id": observation_id,
        "endpoint": "ic50",
        "source": "fixture",
        "group_id": "study-a",
        "split": "validation",
        "peptide": "AAAAAAAAA",
        "qualifier": "=",
        "mhc_classes": ["I"],
        "target": 2.0,
        "raw_value": 100.0,
        "raw_unit": "nM",
        "target_space": "log10(nM)",
        "assay_method": {
            "preparation": "purified",
            "mechanism": "competitive",
            "detector": "radioactivity",
            "review_status": "reviewed",
        },
        "observation": {
            "provenance": {
                "study_id": "study-a",
                "original_measurement_id": observation_id,
            }
        },
        "prediction": 2.1,
    }


def test_complete_prediction_alignment_rejects_label_drift():
    first = [_prediction()]
    second = [copy.deepcopy(first[0])]
    assert set(require_aligned_predictions({"first": first, "second": second})) == {
        "first",
        "second",
    }
    second[0]["target"] = 3.0
    with pytest.raises(ValueError, match="truth identity differs"):
        require_aligned_predictions({"first": first, "second": second})


def test_operational_processing_reconstructs_candidate_pooling():
    record = {
        "endpoint": "elution_observation",
        "prediction": 7.0,
        "component_scores": {"elution_presentation": 3.0, "ms_response": 2.0},
        "diagnostics": {
            "candidates": [
                {
                    "registers": [
                        {
                            "class_probability": 1.0,
                            "elution_terms": {"processing": 0.0},
                        }
                    ]
                },
                {
                    "registers": [
                        {
                            "class_probability": 0.25,
                            "elution_terms": {"processing": 2.0},
                        },
                        {
                            "class_probability": 0.75,
                            "elution_terms": {"processing": 4.0},
                        },
                    ]
                },
            ]
        },
    }
    # Observation offset is 7 - 3 - 2 = 2; second candidate processing is 3.5.
    expected = math.log((math.exp(0.0) + math.exp(3.5)) / 2.0) + 2.0 + 2.0
    assert operational_processing_logit(record) == pytest.approx(expected)


def test_affinity_threshold_metrics_use_definite_censored_labels():
    rows = [
        {
            "qualifier": "=",
            "target_log10_nM": math.log10(100),
            "prediction_log10_nM": math.log10(100),
        },
        {
            "qualifier": ">=",
            "target_log10_nM": math.log10(1000),
            "prediction_log10_nM": math.log10(1000),
        },
        # This lower bound does not establish either side of the 500 nM cutoff.
        {
            "qualifier": ">=",
            "target_log10_nM": math.log10(100),
            "prediction_log10_nM": math.log10(1000),
        },
    ]
    result = affinity_threshold_metrics(rows)
    assert result["n"] == 2
    assert result["accuracy"] == 1
    assert result["auroc"] == 1


def _binary_rows():
    rows = []
    targets = {"study-a": (1, 1, 0, 0), "study-b": (1, 0, 1, 0)}
    for study, labels in targets.items():
        for method, scores in (
            ("first", (0.9, 0.8, 0.2, 0.1)),
            ("second", (0.8, 0.3, 0.7, 0.2)),
        ):
            for index, (target, score) in enumerate(zip(labels, scores, strict=True)):
                rows.append(
                    {
                        "observation_id": f"{study}-{index}",
                        "study": study,
                        "method": method,
                        "target": target,
                        "score": score,
                    }
                )
    return rows


def test_study_bootstraps_are_paired_and_deterministic():
    rows = _binary_rows()
    first = paired_study_bootstrap(
        rows, "first", "second", metric="auroc", kind="binary", replicates=100
    )
    second = paired_study_bootstrap(
        rows, "first", "second", metric="auroc", kind="binary", replicates=100
    )
    assert first == second
    assert first["studies"] == 2
    method = study_bootstrap(rows, "first", metric="auroc", kind="binary", replicates=100)
    assert method["estimate"] == 0.875
    assert method["ci_low"] == 0.75
    assert method["ci_high"] == 1
