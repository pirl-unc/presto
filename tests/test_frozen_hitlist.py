"""Frozen inputs built by Hitlist's real scanner, publisher and bundle verifier."""

import csv
import hashlib
import json
import shutil
from dataclasses import asdict

import pandas as pd
import pytest

from presto.data.frozen_hitlist import load_frozen_hitlist_bundle
from presto.data.hitlist_source import load_records_from_hitlist
from presto.data.source_lineage import SOURCE_LINEAGE_FIELDS


@pytest.fixture
def frozen_bundle(tmp_path, monkeypatch):
    from hitlist import builder, curation, downloads, supplement, write_training_bundle
    from hitlist.parquet_io import atomic_write_parquet

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path / "shared-data")
    # No unrelated curation or network acquisitions in this synthetic fixture.
    original_data_path = curation._data_path
    empty_overrides = tmp_path / "pmid_overrides.yaml"
    empty_overrides.write_text("[]\n")
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda name: (
            str(empty_overrides) if name == "pmid_overrides.yaml" else original_data_path(name)
        ),
    )
    curation._clear_curation_caches()
    try:
        monkeypatch.setattr(supplement, "scan_supplementary", lambda **kwargs: pd.DataFrame())

        def empty(path):
            frame = pd.DataFrame()
            atomic_write_parquet(frame, path)
            return frame

        monkeypatch.setattr(
            builder, "build_bulk_proteomics", lambda **kw: empty(builder._bulk_proteomics_path())
        )
        monkeypatch.setattr(
            builder, "build_line_expression", lambda **kw: empty(builder._line_expression_path())
        )
        fields = [
            "Assay IRI",
            "Reference IRI",
            "PMID",
            "Epitope | Name",
            "Epitope | Source Organism",
            "Host",
            "Host | Process Type",
            "Qualitative Measurement",
            "Effector Cells | Cell Name",
            "MHC Restriction | Name",
            "MHC Allele Class",
            "Assay | Method",
            "Assay | Response measured",
            "Assay | Units",
            "Assay | Measurement Inequality",
            "Assay | Quantitative measurement",
        ]
        common = [
            "ref:1",
            "99999999",
            "SLYNTVATL",
            "Homo sapiens",
            "Homo sapiens",
            "Cellular MHC ligand presentation",
            "Positive",
            "unknown fixture cell",
            "HLA-A*02:01",
            "I",
        ]
        rows = [
            ["http://iedb.org/assay/1", *common, "mass spectrometry", "", "", "", ""],
            [
                "http://iedb.org/assay/2",
                *common,
                "purified MHC/competitive/radioactivity",
                "half maximal inhibitory concentration (IC50)",
                "nM",
                "<",
                "25",
            ],
            ["http://iedb.org/assay/3", *common, "purified MHC", "half life", "min", ">", "120"],
        ]
        missing_fields = [
            "Submission ID",
            "Title",
            "Epitope | Species",
            "Host Age",
            "Host | Disease",
            "Host | Disease Stage",
            "Antigen Processing Comments",
            "Assay Comments",
            "Effector Cells | Source Tissue",
            "Assay | Culture Condition",
            "Host | MHC Types Present",
        ]
        fields.extend(missing_fields)
        rows = [row + [""] * len(missing_fields) for row in rows]
        source = tmp_path / "iedb.csv"
        with source.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([""] * len(fields))
            writer.writerow(fields)
            writer.writerows(rows)
        downloads.register("iedb", source)
        builder.build_observations(build_mappings=False)
        bundle = tmp_path / "frozen"
        write_training_bundle(bundle, include_evidence="both", map_source_proteins=False)
        yield bundle
    finally:
        curation._clear_curation_caches()


def test_verified_bundle_reuses_shared_bytes_and_rejects_damage(frozen_bundle, tmp_path):
    first = load_frozen_hitlist_bundle(frozen_bundle)
    copy = tmp_path / "same-bundle"
    shutil.copytree(frozen_bundle, copy)
    second = load_frozen_hitlist_bundle(copy, expected_sha256=first.manifest_sha256)
    assert first.directory == second.directory
    assert first.directory != frozen_bundle
    assert first.identities().evidence_row_id.nunique() == 3
    assert first.receipt()["coverage"]["n_observations_with_contributors"] == 3
    with pytest.raises(ValueError, match="pinned SHA-256"):
        load_frozen_hitlist_bundle(frozen_bundle, expected_sha256="0" * 64)
    (first.directory / "training.parquet").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="Invalid cached"):
        load_frozen_hitlist_bundle(frozen_bundle)


def test_frozen_labels_match_live_and_never_read_mutable_indexes(frozen_bundle, monkeypatch):
    live = load_records_from_hitlist(include_flanks=False)
    import hitlist

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen input consulted a mutable index")

    monkeypatch.setattr(hitlist, "generate_training_table", forbidden)
    frozen = load_records_from_hitlist(bundle_dir=frozen_bundle, include_flanks=False)
    for old_group, new_group in zip(live[:-1], frozen[:-1], strict=True):
        assert len(old_group) == len(new_group)
        for old, new in zip(old_group, new_group, strict=True):
            ignored = {"source_bundle_id", "provenance_id", "provenance_status"}
            assert {k: v for k, v in asdict(old).items() if k not in ignored} == {
                k: v for k, v in asdict(new).items() if k not in ignored
            }
            assert new.source_bundle_id.startswith("sha256:")
            assert new.provenance_id and new.provenance_status == "indexed"
    binding, _, stability, _, ms, _, _, stats = frozen
    assert len(binding) == len(stability) == len(ms) == 1
    assert (binding[0].value, binding[0].source_units, binding[0].source_inequality) == (
        25,
        "nM",
        "<",
    )
    assert (stability[0].t_half, stability[0].source_value, stability[0].source_units) == (
        2,
        "120.0",
        "min",
    )
    assert stats["input_mode"] == "frozen_bundle"
    assert stats["frozen_bundle"]["input_validation"]["observations"] == "current"


def test_filters_use_mhc_species_and_recorded_restriction(frozen_bundle):
    records = load_records_from_hitlist(
        bundle_dir=frozen_bundle,
        include_flanks=False,
        species="human",
        mhc_allele="A*02:01",
    )
    assert records[-1]["counts"]["binding"] == 1
    assert (
        load_records_from_hitlist(
            bundle_dir=frozen_bundle,
            include_flanks=False,
            length_min=10,
        )[-1]["counts"]["binding"]
        == 0
    )
    assert (
        load_records_from_hitlist(
            bundle_dir=frozen_bundle,
            include_flanks=False,
            mhc_class="II",
        )[-1]["counts"]["elution"]
        == 0
    )


def test_archival_admission_and_missing_columns_fail(frozen_bundle, tmp_path):
    manifest_path = frozen_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_validation"]["require_current_indexes"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="current verified"):
        load_frozen_hitlist_bundle(frozen_bundle)
    manifest["input_validation"]["require_current_indexes"] = True
    path = frozen_bundle / "training.parquet"
    pd.read_parquet(path).drop(columns=["measurement_units"]).to_parquet(path, index=False)
    manifest["artifacts"]["training"].update(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="measurement_units"):
        load_records_from_hitlist(bundle_dir=frozen_bundle, include_flanks=False, mhc_class="II")


def test_bundle_lineage_survives_samples_and_host_batch_metadata(frozen_bundle):
    from presto.data.collate import PrestoCollator
    from presto.data.loaders import PrestoDataset

    binding, _, stability, _, ms, _, _, stats = load_records_from_hitlist(
        bundle_dir=frozen_bundle,
        include_flanks=False,
    )
    samples = list(
        PrestoDataset(
            binding_records=binding,
            stability_records=stability,
            elution_records=ms,
            mhc_sequences={"HLA-A*02:01": "ACDEFGHIKLMNPQRSTVWY" * 10},
            strict_mhc_resolution=False,
        )
    )
    assert len(samples) == 3
    batch = PrestoCollator()(samples).to("cpu")
    assert batch.source_lineage["source_bundle_id"] == [stats["frozen_bundle"]["bundle_id"]] * 3
    for name in SOURCE_LINEAGE_FIELDS:
        assert batch.source_lineage[name] == [getattr(sample, name) for sample in samples]
    assert not set(SOURCE_LINEAGE_FIELDS).intersection(batch.model_inputs())


@pytest.fixture
def mapped_bundle(frozen_bundle, tmp_path, monkeypatch):
    from hitlist import mappings, write_training_bundle
    from hitlist.dataset import update_dataset
    from hitlist.proteome import DEFAULT_FLANK

    mapping = pd.DataFrame(
        [
            dict(
                peptide="SLYNTVATL",
                protein_id=f"P{i}",
                transcript_id=f"T{i}",
                gene_id=f"G{i}",
                gene_name=f"GENE{i}",
                n_flank=flank * 10,
                c_flank=flank * 10,
                position=10,
                is_canonical_transcript=True,
                proteome="Homo sapiens",
                proteome_source="species",
            )
            for i, flank in enumerate(("A", "C"))
        ]
    )

    def write():
        mapping.to_parquet(mappings.mappings_path(), index=False)
        mappings.mappings_meta_path().write_text(
            json.dumps(
                {
                    "observations": mappings._obs_fingerprint(),
                    "contract": mappings._mapping_artifact_contract(
                        release=112,
                        use_uniprot=False,
                        fetch_missing=True,
                        flank=DEFAULT_FLANK,
                    ),
                }
            )
        )
        return mappings.mappings_path()

    path = update_dataset(("peptide_mappings.parquet", "peptide_mappings_meta.json"), write)
    monkeypatch.setattr(
        "hitlist.export._load_training_mappings_for_peptides",
        lambda *a, **kw: pd.read_parquet(path),
    )
    result = tmp_path / "mapped-bundle"
    write_training_bundle(result, include_evidence="both", map_source_proteins=True)
    return result


def test_mapping_alternatives_never_reweight_observations_when_flanks_disabled(mapped_bundle):
    bundle = load_frozen_hitlist_bundle(mapped_bundle)
    assert bundle.manifest["n_rows"] == 6
    for include_flanks in (False, True):
        *groups, stats = load_records_from_hitlist(
            bundle_dir=mapped_bundle, include_flanks=include_flanks
        )
        records = [record for group in groups for record in group]
        assert len(records) == 3
        assert len({record.evidence_row_id for record in records}) == 3
        assert {record.source_bundle_id for record in records} == {bundle.bundle_id}


def test_frozen_measurements_reach_prediction_exports(frozen_bundle, tmp_path):
    from presto.data.collate import PrestoCollator
    from presto.data.loaders import PrestoDataset
    from presto.data.source_lineage import source_lineage_fields
    from presto.models.presto import Presto
    from presto.scripts.train_synthetic import (
        LOSS_TASK_SPECS,
        _get_batch_mask,
        _get_batch_qual,
        _get_batch_target,
        _resolve_task_prediction,
    )
    from presto.training.holdout_eval import collect_holdout_predictions, write_holdout_artifacts

    binding, _, stability, _, ms, _, _, _ = load_records_from_hitlist(
        bundle_dir=frozen_bundle,
        include_flanks=False,
    )
    samples = list(
        PrestoDataset(
            binding_records=binding,
            stability_records=stability,
            elution_records=ms,
            mhc_sequences={"HLA-A*02:01": "ACDEFGHIKLMNPQRSTVWY" * 10},
            strict_mhc_resolution=False,
        )
    )
    expected = {sample.sample_id: source_lineage_fields(sample) for sample in samples}
    accumulators = collect_holdout_predictions(
        Presto(d_model=32, n_layers=1, n_heads=4).eval(),
        [PrestoCollator()(samples)],
        "cpu",
        LOSS_TASK_SPECS,
        forward_fn=lambda model, batch: model(**batch.model_inputs()),
        resolve_pred_fn=_resolve_task_prediction,
        get_target_fn=_get_batch_target,
        get_mask_fn=_get_batch_mask,
        get_qual_fn=_get_batch_qual,
    )
    write_holdout_artifacts(tmp_path, accumulators, split="test")
    with (tmp_path / "test_predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    assert {prediction["sample_id"] for prediction in predictions} == set(expected)
    for prediction in predictions:
        assert {name: prediction[name] for name in SOURCE_LINEAGE_FIELDS} == expected[
            prediction["sample_id"]
        ]


@pytest.mark.parametrize("include_flanks", [False, True])
def test_conflicting_values_for_one_observation_are_rejected(mapped_bundle, include_flanks):
    manifest_path = mapped_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = mapped_bundle / "training.parquet"
    frame = pd.read_parquet(path)
    index = frame.index[frame.response_measured.eq("half maximal inhibitory concentration (IC50)")][
        0
    ]
    frame.loc[index, "quantitative_value"] = 999
    frame.to_parquet(path, index=False)
    manifest["artifacts"]["training"].update(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="conflicting assay"):
        load_records_from_hitlist(bundle_dir=mapped_bundle, include_flanks=include_flanks)


def test_frozen_input_cli_options_and_training_guard(tmp_path, monkeypatch):
    from presto.cli.main import create_parser
    from presto.scripts.train_iedb import run

    parser = create_parser()
    args = parser.parse_args(
        [
            "train",
            "unified",
            "--data-source",
            "hitlist",
            "--data-dir",
            str(tmp_path),
            "--hitlist-bundle",
            str(tmp_path / "bundle"),
            "--hitlist-bundle-sha256",
            "a" * 64,
        ]
    )
    assert args.hitlist_bundle_sha256 == "a" * 64
    monkeypatch.setattr("presto.scripts.train_iedb._assert_mhcseqs_importable", lambda: None)
    args.hitlist_bundle = None
    with pytest.raises(ValueError, match="requires --hitlist-bundle"):
        run(args)
    args.hitlist_bundle = str(tmp_path / "bundle")
    args.bulk_ms = True
    with pytest.raises(ValueError, match="cannot add live --bulk-ms"):
        run(args)
