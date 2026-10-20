"""Admission and end-to-end software verification using artificial observations."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from presto.data.bulk_ms import BulkMSRecord
from presto.data.loaders import BindingRecord, KineticsRecord, StabilityRecord
from presto.data.modular import MHCInput, Observation
from presto.data.modular_sources import (
    MANIFEST_VERSION,
    ObservationIndex,
    affinity_observations,
    file_sha256,
    normalized_observation,
    scoped_bulk_observation,
    write_observations,
)
from presto.training.modular_evaluation import binary_metrics, regression_metrics
from presto.training.modular_run import fit_index, preflight, run_manifest


def fixture_manifest(tmp_path):
    peptides = ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH", "IKLMNPQRS", "TVWYACDEF", "GHIKLMNPQ"]
    observations = [
        Observation(
            f"r{i}",
            "artificial-software-test",
            "ic50",
            peptide,
            (MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "I"),),
            value=10.0 if i % 2 else 1000.0,
            unit="nM",
            split=("train", "validation", "test")[i // 2],
            group_id=f"g{i}",
        )
        for i, peptide in enumerate(peptides)
    ]
    receipt = write_observations(tmp_path / "rows.jsonl", observations)
    manifest = {
        "schema": MANIFEST_VERSION,
        "model": {"endpoints": ["ic50"], "mhc_classes": ["I"], "dropout": 0},
        "training": {"max_epochs": 2, "batch_size": 2, "max_steps": 2},
        "sources": [
            {
                "path": "rows.jsonl",
                "sha256": receipt["sha256"],
                "source": "artificial-software-test",
                "population": "software fixture only",
                "label_policy": "artificial numeric labels; no biological claims",
                "split_policy": "distinct synthetic sequences and groups",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, observations


def test_affinity_adapter_raw_units_proxies_and_stability():
    mhc = MHCInput("ACDEFGHIK", "LMNPQRSTV", "I")
    record = BindingRecord(
        "ACDEFGHIK",
        "example",
        1e6,
        measurement_type="dissociation constant KD (~IC50)",
        evidence_row_id="source1",
    )
    observation = affinity_observations(record, mhc, split="train", group_id="g")[0]
    assert observation.endpoint == "kd_via_ic50" and observation.value == 1e6
    assert observation.provenance["raw_record"]["measurement_type"] == record.measurement_type
    stability = StabilityRecord("ACDEFGHIK", "example", t_half=2, tm=55, evidence_row_id="s")
    outcomes = affinity_observations(stability, mhc, split="train", group_id="g")
    assert [(x.endpoint, x.value, x.unit) for x in outcomes] == [
        ("half_life", 2, "h"),
        ("tm", 55, "C"),
    ]
    kinetic = KineticsRecord("ACDEFGHIK", "example", kon=1, koff=0.1, evidence_row_id="k")
    assert affinity_observations(kinetic, mhc, split="train", group_id="g")[0].endpoint == "koff"
    with pytest.raises(ValueError, match="generated"):
        affinity_observations(
            replace(record, generation_kind="random"), mhc, split="train", group_id="g"
        )


def test_bulk_adapter_requires_scope_and_preserves_absence_semantics():
    raw = BulkMSRecord("ACDEFGHIK", evidence_row_id="bulk1")
    with pytest.raises(ValueError, match="verified search scope"):
        scoped_bulk_observation(raw, split="train", group_id="g")
    scoped = replace(
        raw,
        detectability_label=0,
        source_endpoint="observed_in_selected_search_scope",
        source_search_space_id="search",
        source_reference_sha256="hash",
        source_n_candidate_occurrences="2",
        flank_n="AAA",
    )
    outcome = scoped_bulk_observation(scoped, split="train", group_id="g")
    assert outcome.endpoint == "bulk_observation" and outcome.value == 0
    assert outcome.context.source_n is None
    assert outcome.provenance["interpretation"] == "within_search_observability"
    with pytest.raises(ValueError, match="verified search scope"):
        scoped_bulk_observation(
            replace(scoped, generated_kind="wrong_enzyme"), split="train", group_id="g"
        )


def test_normalized_tcr_adapter_does_not_invent_author_schema():
    observation = Observation(
        "t1",
        "curated",
        "tcr_response",
        "ACDEFGHIK",
        (MHCInput("ACDEFGHIK", "LMNPQRSTV", "I"),),
        tcr_a="ACDEFGHIKLMN",
        tcr_b="PQRSTVWYACDE",
        value=1,
        unit="binary",
    )
    with pytest.raises(ValueError, match="source-attested"):
        normalized_observation(observation.to_dict(), evidence_policy="measured response")
    admitted = replace(
        observation, provenance={"tcr_sequence_scope": "full_paired", "measured_outcome": True}
    )
    assert (
        normalized_observation(admitted.to_dict(), evidence_policy="measured response").tcr_a
        == observation.tcr_a
    )


def test_streaming_manifest_preflight_and_mutation_detection(tmp_path):
    path, observations = fixture_manifest(tmp_path)
    index = ObservationIndex(path)
    config, _, scales, report = preflight(index)
    assert len(index) == 6 and index[3] == observations[3]
    assert config.endpoints == ("ic50",) and scales["ic50"].mean == 2
    assert report["training_labels"] == {"ic50": 2}
    with (tmp_path / "rows.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed after admission"):
        _ = index[0]
    with pytest.raises(ValueError, match="hash mismatch"):
        ObservationIndex(path)


def test_split_leakage_is_rejected_before_fitting(tmp_path):
    path, observations = fixture_manifest(tmp_path)
    observations[2] = replace(observations[2], peptide=observations[0].peptide)
    data_path = tmp_path / "rows.jsonl"
    data_path.unlink()
    write_observations(data_path, observations)
    manifest = json.loads(path.read_text())
    manifest["sources"][0]["sha256"] = file_sha256(data_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="peptide leaks"):
        ObservationIndex(path)


def test_fit_and_selected_checkpoint_prediction_dump(tmp_path):
    path, _ = fixture_manifest(tmp_path)
    output = tmp_path / "software-integration"
    summary = fit_index(ObservationIndex(path), output)
    assert summary["steps"] == 2 and summary["qualification"] == "not_qualified"
    assert not summary["untrained_selected_parameters"]
    for name in (
        "initial.pt",
        "last.pt",
        "selected.pt",
        "coverage.json",
        "admission.json",
        "resolved-config.json",
    ):
        assert (output / name).exists()
    for split in ("train", "validation", "test"):
        predictions = [
            json.loads(line)
            for line in (output / f"{split}-predictions.jsonl").read_text().splitlines()
        ]
        assert len(predictions) == 2
        assert all(row["target_space"] == "log10(nM)" for row in predictions)
        assert summary["metrics"][split]["ic50"]["exact"]["n"] == 2
    with pytest.raises(ValueError, match="overwrite"):
        fit_index(ObservationIndex(path), output)


def test_canonical_cli_dispatch_and_reject_ignored_legacy_settings(tmp_path, capsys):
    path, _ = fixture_manifest(tmp_path)
    from presto.cli.main import main
    from presto.scripts.train_iedb import main as script_main

    args = ["--model-kind", "modular", "--observations-manifest", str(path), "--modular-preflight"]
    assert main(["train", "unified", *args]) == 0
    script_main(args)
    assert "manifest_sha256" in capsys.readouterr().out
    with pytest.raises(ValueError, match="legacy CLI flags"):
        run_manifest(
            SimpleNamespace(
                model_kind="modular",
                observations_manifest=str(path),
                modular_preflight=True,
                _explicit_cli_dests={"epochs"},
            )
        )


def test_metric_ties_and_constant_outputs_are_honest():
    metrics = binary_metrics([0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0])
    assert metrics["auroc"] == 0.5 and metrics["auprc"] == 0.5 and metrics["recall"] == 0
    regression = regression_metrics([1, 2, 3], [2, 2, 2], binding=True)
    assert regression["pearson"] is None and regression["spearman"] is None
    assert regression["sd_ratio"] == 0


def test_registered_launcher_freezes_manifest_and_closes_software_fixture(tmp_path, monkeypatch):
    from presto.scripts import experiment_registry as registry

    manifest, _ = fixture_manifest(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    monkeypatch.setattr(registry, "EXPERIMENTS_ROOT", experiments)
    monkeypatch.setattr(registry, "REGISTRY_MD", experiments / "experiment_log.md")
    output = experiments / "2026-10-20_0000_codex_software-test"
    summary = run_manifest(
        SimpleNamespace(
            observations_manifest=str(manifest),
            modular_preflight=False,
            run_dir=str(output),
            device="cpu",
            _explicit_cli_dests=set(),
        )
    )
    assert summary["qualification"] == "not_qualified"
    frozen = json.loads((output / "reproduce/source-manifest.json").read_text())
    assert frozen["sources"][0]["path"] == str(tmp_path / "rows.jsonl")
    launch = json.loads((output / "reproduce/launch.json").read_text())
    assert str(output / "reproduce/source-manifest.json") in launch["argv"]
    assert "--run-dir" not in launch["argv"]
    assert "| test | ic50 |" in (output / "README.md").read_text()
    assert "Closure:" in registry.REGISTRY_MD.read_text()
    assert (output / "reproduce/source/models/modular.py").exists()
