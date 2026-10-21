"""Admission and end-to-end software verification using artificial observations."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import numpy as np

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
from presto.data.modular_preparation import prepare_observations
from presto.data.modular_auxiliary import weirdo_observations
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
            provenance={"original_measurement_id": f"measured-r{i}"},
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
    assert len(index) == 6 and index[3] == replace(
        observations[3],
        provenance={
            **observations[3].provenance,
            "evidence_policy": "artificial numeric labels; no biological claims",
        },
    )
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


def test_single_core_classii_subset_prunes_unreachable_pooling(tmp_path):
    path, observations = fixture_manifest(tmp_path)
    data = tmp_path / "rows.jsonl"
    data.unlink()
    observations = [
        replace(row, mhc=(replace(row.mhc[0], mhc_class="II"),)) for row in observations
    ]
    write_observations(data, observations)
    manifest = json.loads(path.read_text())
    manifest["sources"][0]["sha256"] = file_sha256(data)
    manifest["model"]["mhc_classes"] = ["II"]
    path.write_text(json.dumps(manifest))
    config, _, _, report = preflight(ObservationIndex(path))
    assert config.register_pooling is False
    assert report["module_pruning"]
    from presto.models.modular import ModularPresto

    network = ModularPresto(config)
    assert not hasattr(network, "register_score")
    with pytest.raises(ValueError, match="multi-register pooling capability"):
        network([replace(observations[0], peptide="ACDEFGHIKLMN")])
    manifest["model"]["mhc_classes"] = ["I", "II"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no admitted assay training support"):
        preflight(ObservationIndex(path))


def test_training_closure_fails_with_an_untouched_allocated_head(tmp_path, monkeypatch):
    path, rows = fixture_manifest(tmp_path)
    rows = [
        replace(row, split="train", endpoint="kd" if i % 2 else "ic50", value=10 ** (i + 1))
        for i, row in enumerate(rows)
    ]
    data_path = tmp_path / "rows.jsonl"
    data_path.unlink()
    receipt = write_observations(data_path, rows)
    manifest = json.loads(path.read_text())
    manifest["model"]["endpoints"] = ["ic50", "kd"]
    manifest["training"] = {
        "max_epochs": 1,
        "max_steps": 1,
        "batch_size": 1,
        "test_policy": "no_test_training_diagnostic",
        "no_test_reason": "software closure regression",
    }
    manifest["sources"][0]["sha256"] = receipt["sha256"]
    path.write_text(json.dumps(manifest))
    output = tmp_path / "failed-coverage"
    with pytest.raises(ValueError, match="training coverage gate failed"):
        fit_index(ObservationIndex(path), output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["coverage_gate"]["status"] == "failed"
    assert any(name.startswith("heads.") for name in summary["untrained_selected_parameters"])
    from presto.training import modular_run

    monkeypatch.setattr(
        modular_run, "evaluate", lambda *a, **k: pytest.fail("unexpected rescoring")
    )
    with pytest.raises(ValueError, match="training coverage gate failed"):
        fit_index(ObservationIndex(path), output, resume=True)


def test_distinct_export_ids_cannot_duplicate_one_measured_outcome(tmp_path):
    path, rows = fixture_manifest(tmp_path)
    first = replace(rows[0], provenance={"original_measurement_id": "original-assay"})
    duplicate = replace(first, observation_id="another-export-id")
    data_path = tmp_path / "rows.jsonl"
    data_path.unlink()
    receipt = write_observations(data_path, [first, duplicate, *rows[1:]])
    manifest = json.loads(path.read_text())
    manifest["sources"][0]["sha256"] = receipt["sha256"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate measured outcome"):
        ObservationIndex(path)


def test_preparation_deduplicates_lineage_and_preserves_independent_replicates(tmp_path):
    path, rows = fixture_manifest(tmp_path)
    provenance = {"measurement_lineage": {"namespace": "original-study", "measurement_id": "m1"}}
    original = replace(rows[0], provenance=provenance)
    duplicate = replace(
        original, observation_id="copy", source="second-export", group_id="alias-group"
    )
    replicate = replace(
        original,
        observation_id="independent",
        value=2000,
        provenance={
            "measurement_lineage": {
                **provenance["measurement_lineage"],
                "replicate_id": "replicate-2",
            },
        },
    )
    report = prepare_observations(
        [original, duplicate, replicate, *rows[1:]],
        tmp_path / "prepared",
        population="software fixture",
        label_policy="artificial measurements",
        split_policy="frozen synthetic groups",
    )
    assert report["dispositions"] == {"admitted": 7, "duplicate": 1}
    assert report["independent_input_contradictions"] == 1
    manifest = json.loads(path.read_text())
    manifest["sources"] = report["sources"]
    path.write_text(json.dumps(manifest))
    index = ObservationIndex(path)
    assert len(index) == 7
    admitted = next(
        row for row in index if row.provenance.get("group_aliases") == ["alias-group", "g0"]
    )
    assert len(admitted.provenance["export_aliases"]) == 2
    assert preflight(index)[3]["training_labels"]["ic50"] == 3


def test_preparation_reports_invalid_missing_and_conflicting_records(tmp_path):
    _, rows = fixture_manifest(tmp_path)
    invalid = replace(rows[0], observation_id="invalid", peptide="INVALID?")
    missing = replace(rows[1], observation_id="missing", value=None)
    conflicting = replace(rows[0], observation_id="conflicting-copy", value=999)
    no_lineage = replace(rows[1], observation_id="no-lineage", provenance={})
    report = prepare_observations(
        [rows[0], conflicting, invalid, missing, no_lineage, rows[2]],
        tmp_path / "prepared",
        population="software fixture",
        label_policy="artificial measurements",
        split_policy="frozen",
    )
    assert report["dispositions"] == {"deferred": 4, "excluded": 1, "admitted": 1}
    dispositions = [
        json.loads(line)
        for line in (tmp_path / "prepared/dispositions.jsonl").read_text().splitlines()
    ]
    assert len(dispositions) == 6
    assert (
        dispositions[0]["reason"]
        == dispositions[1]["reason"]
        == "conflicting exports of one measured outcome"
    )
    assert dispositions[2]["raw"]["peptide"] == "INVALID?"


def test_preflight_rejects_configured_but_unobserved_numeric_features(tmp_path):
    path, rows = fixture_manifest(tmp_path)
    rows = [
        replace(row, endpoint="ms_response", value=i % 2, unit="binary", mhc=())
        for i, row in enumerate(rows)
    ]
    data_path = tmp_path / "rows.jsonl"
    data_path.unlink()
    receipt = write_observations(data_path, rows)
    manifest = json.loads(path.read_text())
    manifest["sources"][0]["sha256"] = receipt["sha256"]
    manifest["model"] = {
        "endpoints": ["ms_response"],
        "acquisition_features": ["energy"],
        "acquisition_units": {"energy": "eV"},
    }
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no observed training values"):
        preflight(ObservationIndex(path))


def test_source_pretraining_requires_each_category_to_have_varied_labels(tmp_path):
    path, rows = fixture_manifest(tmp_path)
    rows = [
        replace(
            item,
            endpoint="identity",
            value=None,
            unit=None,
            mhc=(),
            identity_labels={
                "peptide_source:human": "present" if i % 2 else "absent",
                "peptide_source:rodents": "absent",
            },
        )
        for i, item in enumerate(rows)
    ]
    data_path = tmp_path / "rows.jsonl"
    data_path.unlink()
    receipt = write_observations(data_path, rows)
    manifest = json.loads(path.read_text())
    manifest["sources"][0]["sha256"] = receipt["sha256"]
    manifest["model"] = {
        "endpoints": [],
        "identity_tasks": ["peptide_source"],
        "source_categories": ["human", "rodents"],
        "dropout": 0,
    }
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="positive and negative reference labels: rodents"):
        preflight(ObservationIndex(path))
    rows = [
        replace(
            item,
            identity_labels={
                **item.identity_labels,
                "peptide_source:rodents": "absent" if i % 2 else "present",
            },
        )
        for i, item in enumerate(rows)
    ]
    data_path.unlink()
    receipt = write_observations(data_path, rows)
    manifest["sources"][0]["sha256"] = receipt["sha256"]
    path.write_text(json.dumps(manifest))
    index = ObservationIndex(path)
    assert preflight(index)[3]["source_label_values"]["rodents"] == ["absent", "present"]
    summary = fit_index(index, tmp_path / "source-pretraining")
    assert summary["coverage_gate"]["status"] == "passed"
    assert set(summary["metrics"]["test"]) == {"peptide_source:human", "peptide_source:rodents"}


def test_weirdo_adapter_preserves_multilabels_hashes_and_source_folds(tmp_path):
    directory = tmp_path / "weirdo"
    directory.mkdir()
    np.savez_compressed(
        directory / "splits.npz",
        train_kmers=np.array(["ACDEFGHI", "KLMNPQRS"]),
        train_masks=np.array([3, 2]),
        train_groups=np.array([0, 1]),
        val_kmers=np.array(["TVWYACDE"]),
        val_masks=np.array([1]),
        val_groups=np.array([0]),
    )
    (directory / "splits.json").write_text(
        json.dumps(
            {
                "categories": ["human", "rodents"],
                "salt": "source-family-rule",
                "k": 8,
                "sample_sizes": {"train": 2, "val": 1},
            }
        )
    )
    kwargs = dict(
        archive_sha256=file_sha256(directory / "splits.npz"),
        metadata_sha256=file_sha256(directory / "splits.json"),
        reference_sha256="a" * 64,
        limits={"train": 2, "validation": 1},
        seed=42,
    )
    rows = list(weirdo_observations(directory, **kwargs))
    assert rows == list(weirdo_observations(directory, **kwargs))
    assert rows[0].identity_labels == {
        "peptide_source:human": "present",
        "peptide_source:rodents": "present",
    }
    assert rows[-1].split == "validation"
    assert rows[-1].group_id != rows[0].group_id
    with pytest.raises(ValueError, match="hash mismatch"):
        list(weirdo_observations(directory, **{**kwargs, "archive_sha256": "b" * 64}))
    # The actual older archive lacks these newer group arrays. Keep them unknown
    # until the selected peptides are independently audited against the reference.
    with np.load(directory / "splits.npz") as archive:
        arrays = {name: archive[name] for name in archive.files if not name.endswith("_groups")}
    np.savez_compressed(directory / "splits.npz", **arrays)
    kwargs["archive_sha256"] = file_sha256(directory / "splits.npz")
    with pytest.raises(ValueError, match="explicit reference reconstruction"):
        list(weirdo_observations(directory, **kwargs))
    pending = list(weirdo_observations(directory, **kwargs, allow_missing_groups=True))
    assert [(row.peptide, row.identity_labels) for row in pending] == [
        (row.peptide, row.identity_labels) for row in rows
    ]
    assert all(
        row.group_id == "" and row.provenance["source_family_bucket"] is None for row in pending
    )
    receipt = write_observations(tmp_path / "pending.jsonl", pending)
    manifest = {
        "schema": MANIFEST_VERSION,
        "sources": [
            {
                "path": receipt["path"],
                "sha256": receipt["sha256"],
                "source": "weirdo_swissprot_reference",
                "population": "software",
                "label_policy": "reference membership",
                "split_policy": "not audited",
            }
        ],
    }
    path = tmp_path / "pending-manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing frozen grouping"):
        ObservationIndex(path)


def test_pairing_labels_require_evidence_and_do_not_turn_observed_pairs_into_negatives():
    sample = Observation(
        "pair",
        "reference",
        "identity",
        mhc=(MHCInput("ACDEFGHIK", "LMNPQRSTV", "II"),),
        identity_labels={"mhc_pairing": "mismatched"},
    )
    with pytest.raises(ValueError, match="explicit source pairing evidence"):
        normalized_observation(sample.to_dict(), evidence_policy="reference-pairing")
    evidence = {"basis": "observed_pair", "reference": "source-record"}
    sample = replace(sample, provenance={"pairing_evidence": {"mhc_pairing": evidence}})
    with pytest.raises(ValueError, match="does not supply a measured mismatch"):
        normalized_observation(sample.to_dict(), evidence_policy="reference-pairing")
    sample = replace(sample, identity_labels={"mhc_pairing": "matched"})
    assert (
        normalized_observation(
            sample.to_dict(), evidence_policy="reference-pairing"
        ).identity_labels["mhc_pairing"]
        == "matched"
    )
