"""Real canonical collation/report parity and diagnostic artifact integrity."""

import argparse
import dataclasses
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from presto.data.collate import PrestoCollator, PrestoSample
from presto.data.label_provenance import record_target_provenance
from presto.scripts.train_iedb import _resolve_run_args
from presto.training import coverage_preflight

SPEC = importlib.util.spec_from_file_location(
    "census_under_test", Path(__file__).with_name("census.py")
)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def samples():
    common = dict(
        peptide="ACDEFGHIKLM",
        mhc_a="ACDEFGHIK",
        mhc_b="LMNPQRSTV",
        mhc_class="I",
        species="human",
        sample_source="iedb",
        target_provenance=record_target_provenance("iedb", "binding", "tcell"),
    )
    rows = [
        PrestoSample(
            **common,
            sample_id="binding",
            bind_value=37,
            bind_qual=-1,
            bind_measurement_type="IC50",
            binding_assay_method="purified MHC/direct/fluorescence",
        ),
        PrestoSample(**common, sample_id="row", tcell_label=0, tcell_assay_method="ELISPOT"),
        PrestoSample(
            **common,
            sample_id="bag",
            tcell_label=1,
            tcell_assay_method="ELISPOT",
            use_tcell_pathway_mil=True,
            tcell_mil_mhc_a_list=["ACDEFGHIK", "ACDEFGHIK"],
            tcell_mil_mhc_b_list=["LMNPQRSTV", "LMNPQRSTV"],
            tcell_mil_mhc_class_list=["I", "II"],
        ),
    ]
    return {
        name: [dataclasses.replace(row, sample_id=f"{name}-{row.sample_id}") for row in rows]
        for name in ("train", "val", "test")
    }


@pytest.mark.parametrize("chunk_size", [1, 2, 512])
def test_full_canonical_report_parity_retention_and_training_only_candidates(
    tmp_path, monkeypatch, chunk_size
):
    original_split_audit = coverage_preflight.audit_split_support
    monkeypatch.setattr(
        coverage_preflight,
        "audit_split_support",
        lambda *a, **kw: original_split_audit(*a, chunk_size=chunk_size, **kw),
    )
    kwargs = dict(
        splits=samples(),
        collator=PrestoCollator(),
        args=_resolve_run_args(argparse.Namespace()),
        data_seed=17,
        manifest=None,
    )
    before = coverage_preflight.audit_training_coverage(**kwargs, output_dir=tmp_path / "before")
    original = coverage_preflight.OutputCoverageCensus
    output = tmp_path / "after"
    with WORKER.retained_census(output):
        after = coverage_preflight.audit_training_coverage(**kwargs, output_dir=output)
    assert coverage_preflight.OutputCoverageCensus is original
    assert before == after
    for name in ("output_coverage.json", "split_support.json", "supported_outputs.json"):
        assert json.loads((tmp_path / "before" / name).read_text()) == json.loads(
            (output / name).read_text()
        )
    with sqlite3.connect(output / "output_coverage.sqlite") as db:
        assert db.execute(
            "SELECT split,COUNT(*) FROM samples GROUP BY split ORDER BY split"
        ).fetchall() == [("test", 3), ("train", 3), ("val", 3)]
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    candidates = json.loads((output / "training_candidates.json").read_text())
    identities = {entry["sample"]["sample_id"] for entry in candidates["samples"]}
    assert identities == {"train-binding", "train-row", "train-bag"}
    assert any("ELISPOT" in str(entry["stratum"]) for entry in candidates["strata"])


def test_failed_manifest_restores_wrapper_and_retains_evidence(tmp_path):
    original = coverage_preflight.OutputCoverageCensus
    with pytest.raises(ValueError, match="Manifest requires schema_version=1"):
        with WORKER.retained_census(tmp_path):
            coverage_preflight.audit_training_coverage(
                samples(),
                collator=PrestoCollator(),
                args=_resolve_run_args(argparse.Namespace()),
                data_seed=17,
                manifest={"schema_version": -1},
                output_dir=tmp_path,
            )
    assert coverage_preflight.OutputCoverageCensus is original
    assert (tmp_path / "output_coverage.json").is_file()
    with sqlite3.connect(tmp_path / "output_coverage.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] > 0
    with pytest.raises(FileExistsError):
        with WORKER.retained_census(tmp_path):
            coverage_preflight.audit_training_coverage(
                samples(),
                collator=PrestoCollator(),
                args=_resolve_run_args(argparse.Namespace()),
                data_seed=17,
                manifest=None,
                output_dir=tmp_path,
            )


def test_candidate_order_and_duplicate_identity_ties_are_deterministic(tmp_path):
    rows = [dataclasses.replace(samples()["train"][0], bind_value=value) for value in (10, 20, 30)]
    for name, records in (("forward", rows), ("reverse", list(reversed(rows)))):
        selector = WORKER.CandidateSelection()
        for row in records:
            selector.add(row, ("cell",))
        selector.write(tmp_path / name)
    left = json.loads((tmp_path / "forward/training_candidates.json").read_text())
    right = json.loads((tmp_path / "reverse/training_candidates.json").read_text())
    assert left == right
    assert len(left["samples"]) == 2
    assert {row["sample"]["sample_id"] for row in left["samples"]} == {"train-binding"}


def test_frozen_inputs_reject_drift(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("original")
    manifest = [{"relative_path": path.name, "sha256": WORKER.file_hash(path)}]
    WORKER.verify_files(manifest, root=tmp_path)
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="Frozen input changed"):
        WORKER.verify_files(manifest, root=tmp_path)


def test_candidates_exclude_generated_rows_without_changing_census(tmp_path):
    real = samples()["train"][0]
    generated = dataclasses.replace(real, sample_id="generated", synthetic_kind="peptide_scramble")
    kwargs = dict(
        splits={"train": [real, generated]},
        collator=PrestoCollator(),
        args=_resolve_run_args(argparse.Namespace()),
        data_seed=17,
        manifest=None,
    )
    baseline = coverage_preflight.audit_training_coverage(**kwargs)
    with WORKER.retained_census(tmp_path):
        result = coverage_preflight.audit_training_coverage(**kwargs, output_dir=tmp_path)
    assert result == baseline
    candidates = json.loads((tmp_path / "training_candidates.json").read_text())
    assert {entry["sample"]["sample_id"] for entry in candidates["samples"]} == {real.sample_id}
    assert candidates["excluded_synthetic_observation_hits"] > 0


@pytest.mark.parametrize("identity", [None, ""])
def test_missing_candidate_identity_preserves_census_without_invented_id(tmp_path, identity):
    row = dataclasses.replace(samples()["train"][0], sample_id=identity)
    kwargs = dict(
        splits={"train": [row]},
        collator=PrestoCollator(),
        args=_resolve_run_args(argparse.Namespace()),
        data_seed=17,
        manifest=None,
    )
    baseline = coverage_preflight.audit_training_coverage(**kwargs)
    with WORKER.retained_census(tmp_path):
        result = coverage_preflight.audit_training_coverage(**kwargs, output_dir=tmp_path)
    assert result == baseline
    candidates = json.loads((tmp_path / "training_candidates.json").read_text())
    assert candidates["samples"] == []
    assert candidates["missing_identity_observation_hits"] > 0


def test_registered_source_conditions_are_uncapped_and_do_not_union_sources():
    conditions = json.loads((Path(__file__).parents[1] / "conditions.json").read_text())
    assert len(conditions) == 6
    for name, args in conditions.items():
        assert args["data_source"] == ("merged_tsv" if name.startswith("merged_") else "hitlist")
        assert args["bulk_ms"] == name.startswith("hitlist_bulk_")
        assert all(
            args[f"max_{name}"] == 0
            for name in (
                "binding",
                "kinetics",
                "stability",
                "processing",
                "elution",
                "tcell",
                "vdjdb",
                "bulk_ms",
            )
        )
        assert (args["data_seed"], args["seed"], args["val_frac"], args["test_frac"]) == (
            17,
            42,
            0.1,
            0.1,
        )
        if name.endswith("_measured"):
            assert all(
                value == 0
                for key, value in args.items()
                if key.startswith("synthetic_") and key.endswith("ratio")
            )
            assert args["mhc_augmentation_samples"] == args["bulk_excision_negative_ratio"] == 0


def test_source_archive_excludes_raw_datasets_and_historical_artifacts():
    spec = importlib.util.spec_from_file_location(
        "census_launcher", Path(__file__).with_name("launch.py")
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    prefix = f"experiments/{launcher.FAMILY}/"
    required = [
        "__init__.py",
        "data/loaders.py",
        "models/presto.py",
        "data/b2m_sequences.csv",
        prefix + "code/census.py",
        prefix + "conditions.json",
        prefix + "input_manifest.json",
        prefix + "reproduce/environment.txt",
    ]
    excluded = [
        "data/merged_deduped.tsv",
        "data/mhc_index.csv",
        "data/iedb/tcell_full_v3.zip",
        "data/vdjdb/vdjdb.zip",
        "artifacts/private.json",
        prefix + "results/large.sqlite",
        prefix + "results/reproduce/source/launch.py",
    ]
    assert launcher.archive_paths(required + excluded) == sorted(required)
