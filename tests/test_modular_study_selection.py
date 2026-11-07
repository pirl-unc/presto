"""Unequal study sizes must not silently determine validation checkpoint priority."""

from collections import defaultdict
import json

import pytest
import torch

from presto.data.modular import Observation
from presto.data.modular_sources import ObservationIndex, write_observations
from presto.tests.test_modular_affinity_controls import fixture, run_with_interruption
from presto.training import modular_run as runner


def row(endpoint, group, loss, **extra):
    return dict(endpoint=endpoint, group_id=group, loss=loss, **extra)


def test_studies_then_endpoints_receive_equal_weight():
    records = [row("elution_observation", "large", 0.0)] * 9 + [
        row("elution_observation", "small", 1.0),
        row("ic50", "third", 0.8),
    ]
    metrics = {"elution_observation": {"n": 10, "loss": 0.1}, "ic50": {"n": 1, "loss": 0.8}}
    assert runner._selection_score(
        metrics, reduction="study_mean", records=records
    ) == pytest.approx(0.65)
    assert runner._selection_score(
        metrics, primary_endpoint="elution_observation", reduction="study_mean", records=records
    ) == pytest.approx(0.5)
    assert runner._selection_score(metrics, primary_endpoint="elution_observation") == 0.1


def test_study_selection_excludes_frozen_and_identity_records(monkeypatch):
    monkeypatch.setattr(runner, "frozen_output", lambda model, target: target == "ic50")
    records = [
        row("elution_observation", "a", 0.4),
        row("ic50", None, None),
        row("identity", None, None, identity_task="mhc_class"),
    ]
    metrics = {"elution_observation": {"n": 1}, "ic50": {"n": 1}, "mhc_class": {"n": 1}}
    assert (
        runner._selection_score(metrics, model=object(), reduction="study_mean", records=records)
        == 0.4
    )


@pytest.mark.parametrize(
    "records, message",
    [
        (None, "evaluated assay records"),
        ([row("elution_observation", "", 0.1)], "study group IDs"),
        ([row("elution_observation", "a", float("nan"))], "finite observation losses"),
        ([row("elution_observation", "a", float("inf"))], "finite observation losses"),
        ([], "cover selected endpoints"),
        ([row("elution_observation", "a", 0.1)] * 2, "every evaluated assay record"),
    ],
)
def test_incomplete_or_invalid_study_records_fail(records, message):
    with pytest.raises(ValueError, match=message):
        runner._selection_score(
            {"elution_observation": {"n": 1}}, reduction="study_mean", records=records
        )


def test_study_selection_runs_and_resumes_with_the_same_selected_checkpoint(tmp_path, monkeypatch):
    path = fixture(
        tmp_path,
        selection_endpoint="elution_observation",
        selection_reduction="study_mean",
        validation_every_steps=2,
    )
    manifest = json.loads(path.read_text())
    manifest["model"]["endpoints"] = ["elution_observation"]
    source = tmp_path / "rows.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines()]
    for i, record in enumerate(records):
        record.update(endpoint="elution_observation", unit="binary", value=float(i % 2))
        if record["split"] == "validation":
            record["group_id"] = "small-study" if i == 16 else "large-study"
    receipt = write_observations(
        tmp_path / "binary.jsonl", [Observation.from_dict(record) for record in records]
    )
    manifest["sources"][0]["path"] = receipt["path"]
    manifest["sources"][0]["sha256"] = receipt["sha256"]
    path.write_text(json.dumps(manifest))
    scores = []
    evaluate = runner.evaluate

    def observed(*args, **kwargs):
        metrics, records = evaluate(*args, **kwargs)
        if records and all(record["split"] == "validation" for record in records):
            grouped = defaultdict(list)
            for record in records:
                grouped[record["group_id"]].append(record["loss"])
            scores.append(
                sum(sum(losses) / len(losses) for losses in grouped.values()) / len(grouped)
            )
        return metrics, records

    monkeypatch.setattr(runner, "evaluate", observed)
    reference = run_with_interruption(tmp_path, monkeypatch, path, before_step=6)
    selected = torch.load(reference / "selected.pt", weights_only=True)
    assert selected["metadata"]["selection_reduction"] == "study_mean"
    assert selected["metadata"]["selection_metric"] == pytest.approx(min(scores))
    predictions = [
        json.loads(line)
        for line in (reference / "validation-predictions.jsonl").read_text().splitlines()
    ]
    groups = defaultdict(list)
    for record in predictions:
        groups[record["group_id"]].append(record["loss"])
    balanced = sum(sum(losses) / len(losses) for losses in groups.values()) / len(groups)
    assert balanced == pytest.approx(selected["metadata"]["selection_metric"])
    assert balanced != pytest.approx(
        sum(record["loss"] for record in predictions) / len(predictions)
    )
    assert len(ObservationIndex(path).partitions["validation:assay"]) == 12


def test_study_reduction_cannot_override_spearman_selection(tmp_path):
    path = fixture(
        tmp_path,
        selection_reduction="study_mean",
        selection_metric="within_study_spearman",
        selection_spearman_endpoints=["ic50"],
    )
    with pytest.raises(ValueError, match="other selection overrides"):
        runner.preflight(ObservationIndex(path))
