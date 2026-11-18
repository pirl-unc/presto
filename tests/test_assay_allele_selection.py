"""Selection must reward within-assay peptide ranking, not between-group offsets."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from presto.data.assay_measurement import AssayMethod
from presto.data.modular import MHCInput, Observation
from presto.data.modular_sources import ObservationIndex, write_observations
from presto.tests.test_modular_affinity_controls import fixture, run_with_interruption
from presto.training import modular_run as runner
from presto.training.modular_controls import (
    assay_allele_key,
    assay_allele_spearman_score,
    ranking_loss,
    spearman_score,
)


def row(i, *, mhc_class="I", detector="radioactivity", allele=0, study="S", value=10):
    return Observation(
        f"row{i}",
        "fixture",
        "ic50",
        "ACDEFGHIK",
        (MHCInput("ACDEFGHIK" + "A" * allele, "LMNPQRSTV", mhc_class),),
        value=value,
        unit="nM",
        group_id=f"g{i}",
        assay_method=AssayMethod("purified", "competitive", detector, "verified"),
        provenance={"study_id": study},
    )


def record(obs, target, prediction):
    return dict(
        endpoint=obs.endpoint,
        qualifier=obs.qualifier,
        target=target,
        prediction=prediction,
        observation=obs.to_dict(),
    )


def test_assay_offsets_cannot_win_strict_selection():
    records = [
        record(row(i, detector=detector), offset + i, offset)
        for detector, offset in (("radioactivity", 0), ("fluorescence", 100))
        for i in range(4)
    ]
    old, _ = spearman_score(records, ["ic50"], 4)
    strict, report = assay_allele_spearman_score(records, ["ic50"], 4, ("I",))
    assert old < -0.8  # An assay-only model appears strongly predictive when pooled.
    assert strict == 0
    assert len(report["cells"]) == 2


def test_classes_and_alleles_have_equal_weight_despite_many_studies():
    records = []
    for study in range(10):
        records += [record(row(i, study=f"s{study}"), i, i) for i in range(4)]
    records += [record(row(i, allele=1), i, -i) for i in range(4)]
    records += [record(row(i, mhc_class="II"), i, -i) for i in range(4)]
    score, report = assay_allele_spearman_score(records, ["ic50"], 4, ("I", "II"))
    assert report["by_class"] == pytest.approx({"I": 0, "II": -1})
    assert score == pytest.approx(0.5)


def test_strict_ranking_cannot_compare_different_alleles_or_assays():
    observations = [
        row(0, value=10),
        row(1, value=100),
        row(2, allele=1, value=1000),
        row(3, detector="fluorescence", value=10000),
        replace(row(4, value=100000), qualifier=">"),
    ]
    outputs = [SimpleNamespace(value=torch.tensor(float(i), requires_grad=True)) for i in range(5)]
    loss, count = ranking_loss(observations, outputs, ["ic50"], grouping="assay_allele")
    assert count == 1
    loss.backward()
    assert outputs[0].value.grad is not None and outputs[1].value.grad is not None
    assert all(output.value.grad is None for output in outputs[2:])


def test_unknown_methods_and_constant_targets_do_not_qualify_a_class():
    assert assay_allele_key(replace(row(0), assay_method=None)) is None
    records = [record(row(i), 1, i) for i in range(4)]
    with pytest.raises(ValueError, match="no evaluable"):
        assay_allele_spearman_score(records, ["ic50"], 4, ("I",))
    records = [record(row(i), i, i) for i in range(4)]
    with pytest.raises(ValueError, match="II"):
        assay_allele_spearman_score(records, ["ic50"], 4, ("I", "II"))


def strict_fixture(tmp_path):
    path = fixture(
        tmp_path,
        selection_metric="within_assay_allele_spearman",
        selection_spearman_endpoints=["ic50"],
        selection_min_cell_rows=2,
        within_group_ranking_loss={
            "weight": 0.5,
            "endpoints": ["ic50"],
            "grouping": "assay_allele",
        },
        cell_batching={"rows_per_cell": 2, "grouping": "assay_allele"},
    )
    manifest = json.loads(path.read_text())
    rows = list(ObservationIndex(path))
    rows = [
        replace(
            obs,
            mhc=(replace(obs.mhc[0], mhc_class="II" if i % 2 else "I"),),
            assay_method=AssayMethod("purified", "competitive", "radioactivity", "verified"),
        )
        for i, obs in enumerate(rows)
    ]
    written = write_observations(tmp_path / "strict-rows.jsonl", rows)
    manifest["sources"][0].update(path=written["path"], sha256=written["sha256"])
    manifest["model"]["assay_tracks"] = [
        {
            "endpoint": "ic50",
            "preparation": "purified",
            "mechanism": "competitive",
            "detector": "radioactivity",
        }
    ]
    path.write_text(json.dumps(manifest))
    return path


def test_strict_selection_and_ranking_resume_exactly(tmp_path, monkeypatch):
    path = strict_fixture(tmp_path)
    reference = run_with_interruption(tmp_path, monkeypatch, path, before_step=6)
    history = json.loads((reference / "history.json").read_text())
    for epoch in history:
        report = epoch["selection"]
        assert set(report["by_class"]) == {"I", "II"}
        assert report["cells"]
        assert all(c["preparation"] == "purified" and c["mhc_key"] for c in report["cells"])


def test_strict_preflight_requires_each_class(tmp_path):
    path = strict_fixture(tmp_path)
    rows = [
        replace(obs, mhc=(replace(obs.mhc[0], mhc_class="I"),))
        if obs.split == "validation"
        else obs
        for obs in ObservationIndex(path)
    ]
    written = write_observations(tmp_path / "missing-ii.jsonl", rows)
    manifest = json.loads(path.read_text())
    manifest["sources"][0].update(path=written["path"], sha256=written["sha256"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no evaluable.*II"):
        runner.preflight(ObservationIndex(path))
