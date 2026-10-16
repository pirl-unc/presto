"""Frozen scoped inputs through Hitlist verification and real model host batches."""

import hashlib
import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from presto.data.collate import PrestoCollator
from presto.data.loaders import PrestoDataset
from presto.data.scoped_bulk import load_scoped_bulk_records
from presto.training.output_contract import build_output_contract, label_evidence
from presto.training.output_coverage import OutputCoverageCensus

DATASET = "pxd013455-hela-tryp46-v1"


def pin(path):
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def seal(directory):
    candidate_manifest = json.loads((directory / "candidates.manifest.json").read_text())
    candidate_manifest["artifact"] = {
        "name": "candidates.parquet",
        **pin(directory / "candidates.parquet"),
    }
    candidate_manifest["n_candidates"] = pq.ParquetFile(
        directory / "candidates.parquet"
    ).metadata.num_rows
    write_json(directory / "candidates.manifest.json", candidate_manifest)
    write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "dataset_id": DATASET,
            "producer": {"hitlist_version": "1.75.0"},
            "training_readiness": "not_established",
            "artifacts": {
                path.name: pin(path) for path in directory.iterdir() if path.name != "manifest.json"
            },
        },
    )
    return pin(directory / "manifest.json")["sha256"]


@pytest.fixture
def scoped_bundle(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "searched.fasta").write_text(">P1\nACDEFGHIK\n>P2\nLMNPQRSTK\n")
    reference = pin(directory / "searched.fasta")["sha256"]
    scope = {
        "source": "PXD013455",
        "cell_line_name": "HeLa",
        "digestion_enzyme": "Trypsin/P",
        "protocol_id": "PXD013455:fixture",
        "n_fractions_in_run": 46,
        "enrichment": "none",
        "fractionation_ph": 8.0,
    }
    acquisition = {
        "instrument": "fixture",
        "acquisition_mode": "DDA",
        "protein_observation_basis": "same-scope leading razor peptides",
    }
    shared = {**scope, **acquisition, "search_space_id": "fixture-search"}
    search = {"fasta_sha256": reference, "enzyme": "Trypsin/P"}
    registry = {
        "dataset_id": DATASET,
        "scope": scope,
        "acquisition": acquisition,
        "search_space": search,
        "citations": {
            "reanalysis": {
                "pmid": "33893311",
                "doi": "fixture",
                "url": "https://pubmed.ncbi.nlm.nih.gov/33893311/",
            }
        },
        "cell_context": {"donor_id": None, "specimen_id": None},
        "limitations": ["one fixture scope"],
    }
    write_json(directory / "source_registry.json", registry)
    write_json(
        directory / "candidates.manifest.json",
        {
            "scope": scope,
            "search_space": search,
            "search_space_id": "fixture-search",
            "search_reference": {"sha256": reference},
            "parameters": {"flank": 15},
        },
    )
    write_json(
        directory / "label_policy.json",
        {
            "endpoint": "observed_in_selected_search_scope",
            "units": "binary",
            "censoring": "not_applicable",
        },
    )
    observations = [
        {
            **shared,
            "peptide": peptide,
            "evidence_row_id": identity,
            "identification_type": evidence,
            "n_replicates_detected": None,
        }
        for peptide, identity, evidence in (
            ("ACDEFGHIK", "observation:direct", "By MS/MS"),
            ("LMNPQRSTK", "observation:matching", "By matching"),
            ("YVVAAAALK", "observation:ambiguous", "By MS/MS"),
        )
    ]
    candidates = []
    for peptide, parent in (
        ("ACDEFGHIK", "P1"),
        ("LMNPQRSTK", "P1"),
        ("YVVAAAALK", "P1"),
        ("YVVAAAALK", "P2"),
        ("AAAAAAAAK", "P1"),
    ):
        observation = next((row for row in observations if row["peptide"] == peptide), None)
        candidates.append(
            {
                **shared,
                "peptide": peptide,
                "uniprot_acc": parent,
                "candidate_id": (
                    f"{DATASET}:{parent}:{len(candidates) + 1}:{len(candidates) + len(peptide)}"
                ),
                "start_position": len(candidates) + 1,
                "end_position": len(candidates) + len(peptide),
                "n_flank": "" if not candidates else "AAA",
                "c_flank": "DDD",
                "observed": observation is not None,
                "source_observation_id": observation["evidence_row_id"] if observation else None,
                "label_evidence": observation["identification_type"]
                if observation
                else "not observed in scope",
                "protein_observed": True,
                "n_replicates_possible": 1,
                "n_replicates_detected": None if observation else 0,
                "first_seen_at_n_fractions": None,
            }
        )
    pq.write_table(pa.Table.from_pylist(candidates), directory / "candidates.parquet")
    pq.write_table(pa.Table.from_pylist(observations), directory / "observations.parquet")
    pq.write_table(
        pa.Table.from_pylist([{**shared, "uniprot_acc": parent} for parent in ("P1", "P2")]),
        directory / "proteins.parquet",
    )
    for name in ("validation.json", "coverage.json"):
        write_json(directory / name, {})
    (directory / "README.md").write_text(
        "Synthetic schema fixture; not biological qualification.\n"
    )
    return directory, seal(directory), tmp_path / "cache"


def load(bundle, **kwargs):
    directory, digest, cache = bundle
    return load_scoped_bulk_records(directory, bundle_sha256=digest, cache_dir=cache, **kwargs)


def test_one_vote_labels_nullable_counts_and_host_lineage(scoped_bundle):
    records, stats = load(scoped_bundle)
    assert len(records) == 4
    assert stats["full_coverage"]["n_candidate_occurrences"] == 5
    assert stats["full_coverage"]["n_ambiguous_peptides"] == 1
    preview = Path(stats["projection"]["directory"]) / "inspection.tsv"
    assert preview.exists() and "source_label_evidence" in preview.read_text()
    assert stats["n_records_by_evidence"] == {
        "By MS/MS": 2,
        "By matching": 1,
        "not observed in scope": 1,
    }
    assert all(row.n_replicates_detected is None and row.excision_label is None for row in records)
    assert {row.machinery for row in records} == {"trypsin_p"}
    by_peptide = {row.peptide: row for row in records}
    ambiguous = by_peptide["YVVAAAALK"]
    assert ambiguous.source_n_candidate_occurrences == "2"
    assert not any(
        (ambiguous.protein_id, ambiguous.flank_n, ambiguous.flank_c, ambiguous.candidate_id)
    )
    assert ambiguous.candidate_lookup_key == "YVVAAAALK"
    assert by_peptide["ACDEFGHIK"].flank_n == ""
    assert by_peptide["ACDEFGHIK"].flank_n_is_terminus
    assert not ambiguous.flank_n_is_terminus
    negative = by_peptide["AAAAAAAAK"]
    assert negative.detectability_label == 0 and not negative.evidence_row_id
    samples = PrestoDataset(bulk_ms_records=records).samples
    batch = PrestoCollator()(samples)
    assert {sample.enzymatic_digest for sample in samples} == {"trypsin_p"}
    from presto.data.vocab import EXCISION_MACHINERY_TO_IDX

    assert (batch.model_inputs()["machinery"] == EXCISION_MACHINERY_TO_IDX["trypsin_p"]).all()
    assert batch.target_masks["ms_detectability"].all()
    assert "excision" not in batch.target_masks
    assert batch.source_lineage["candidate_lookup_key"] == [row.peptide for row in records]
    assert batch.source_lineage["source_bundle_id"] == [f"sha256:{scoped_bundle[1]}"] * 4
    assert not set(batch.source_lineage) & set(batch.model_inputs())
    assert len({sample.sample_id for sample in samples}) == 4
    assert all(sample.sample_id.startswith("scoped:sha256:") for sample in samples)
    contract = build_output_contract()
    objective = contract.objectives["row:ms_detectability"]
    evidence = [label_evidence(objective, sample) for sample in samples]
    assert {item.role for item in evidence} == {"proxy"}
    assert {item.family for item in evidence} == {
        "scoped_msms",
        "scoped_matching",
        "scoped_unobserved",
    }
    with OutputCoverageCensus(contract) as census:
        census.add_batch("train", samples, batch)
        assert census.counts("train", objective.endpoint, roles=["proxy"])["observations"] == 4
        assert census.counts("train", objective.endpoint, roles=["direct"])["observations"] == 0


def test_capping_after_collapse_and_verified_cache_reuse(scoped_bundle, monkeypatch):
    records, stats = load(scoped_bundle, max_records=3, seed=17)
    assert len(records) == 3 and stats["n_dropped_by_cap"] == 1
    assert len({row.peptide for row in records}) == 3
    import presto.data.scoped_bulk as module

    original = module._batches

    def no_source_rebuild(path, columns=None):
        assert Path(path).name == "records.parquet"
        return original(path, columns)

    monkeypatch.setattr(module, "_batches", no_source_rebuild)
    again, reused = load(scoped_bundle, max_records=3, seed=17)
    assert records == again and stats["projection"] == reused["projection"]


def test_publication_cache_need_not_support_sqlite(scoped_bundle, monkeypatch):
    directory, manifest_sha256, cache = scoped_bundle
    connect = sqlite3.connect
    scratch_paths = []

    def local_sqlite(database, *args, **kwargs):
        path = Path(database).resolve()
        if path.is_relative_to(cache.resolve()):
            raise sqlite3.OperationalError("SQLite unavailable on publication filesystem")
        scratch_paths.append(path)
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", local_sqlite)
    records, stats = load_scoped_bulk_records(
        directory, bundle_sha256=manifest_sha256, cache_dir=cache
    )
    assert records
    assert stats["n_records"] == len(records)
    assert scratch_paths and all(not path.parent.exists() for path in scratch_paths)
    assert (Path(stats["projection"]["directory"]) / "records.parquet").is_file()


def test_bad_pin_and_corrupt_payload_are_rejected(scoped_bundle):
    directory, digest, cache = scoped_bundle
    with pytest.raises(ValueError, match="requires its manifest"):
        load_scoped_bulk_records(directory, cache_dir=cache)
    with pytest.raises(ValueError, match="SHA-256"):
        load_scoped_bulk_records(directory, bundle_sha256="0" * 64, cache_dir=cache)
    (directory / "searched.fasta").write_text("corrupt")
    with pytest.raises(ValueError, match="size|checksum|SHA|hash"):
        load(scoped_bundle)


@pytest.mark.parametrize(
    "table,column,value,match",
    [
        ("candidates", "protocol_id", "other-protocol", "incompatible scope"),
        ("observations", "protocol_id", "other-protocol", "incompatible scope"),
        ("proteins", "search_space_id", "other-search", "incompatible scope"),
        ("candidates", "first_seen_at_n_fractions", 14.0, "depth contract"),
        ("candidates", "source_observation_id", "wrong-observation", "source observation"),
        ("candidates", "label_evidence", "By matching", "source observation"),
        ("candidates", "protein_observed", False, "parent support"),
        ("candidates", "candidate_id", "invented", "coordinates"),
    ],
)
def test_inconsistent_scientific_contract_fails(scoped_bundle, table, column, value, match):
    directory, _, cache = scoped_bundle
    rows = pq.read_table(directory / f"{table}.parquet").to_pylist()
    rows[0][column] = value
    pq.write_table(pa.Table.from_pylist(rows), directory / f"{table}.parquet")
    digest = seal(directory)
    with pytest.raises(ValueError, match=match):
        load((directory, digest, cache))


def test_duplicate_candidate_ids_fail_instead_of_inflating_occurrences(scoped_bundle):
    directory, _, cache = scoped_bundle
    rows = pq.read_table(directory / "candidates.parquet").to_pylist()
    pq.write_table(pa.Table.from_pylist(rows + [rows[0]]), directory / "candidates.parquet")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        load((directory, seal(directory), cache))


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5])
def test_invalid_caps_fail_before_source_access(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        load_scoped_bulk_records(max_records=maximum)


def test_missing_inventory_and_damaged_projection_fail(scoped_bundle):
    _, stats = load(scoped_bundle)
    projection = Path(stats["projection"]["directory"]) / "records.parquet"
    projection.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="damaged"):
        load(scoped_bundle)
    (scoped_bundle[0] / "proteins.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="proteins.parquet"):
        load(scoped_bundle)


def test_manual_scoped_record_cannot_bypass_missing_lineage():
    from presto.data.bulk_ms import BulkMSRecord

    with pytest.raises(ValueError, match="consistent source lineage"):
        PrestoDataset(
            bulk_ms_records=[
                BulkMSRecord(
                    peptide="ACDEFGHIK", detectability_label=1, source_label_evidence="By MS/MS"
                )
            ]
        )


def test_scoped_metadata_survives_actual_prediction_exports(scoped_bundle, tmp_path):
    import csv
    from presto.data.source_lineage import source_lineage_fields
    from presto.models.presto import Presto
    from presto.training.holdout_eval import collect_holdout_predictions, write_holdout_artifacts
    from presto.scripts.train_synthetic import (
        LOSS_TASK_SPECS,
        _resolve_task_prediction,
        _get_batch_target,
        _get_batch_mask,
        _get_batch_qual,
    )

    records, _ = load(scoped_bundle)
    samples = PrestoDataset(bulk_ms_records=records).samples
    expected = {
        sample.sample_id: source_lineage_fields(sample, include_scoped=True) for sample in samples
    }
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
    with (tmp_path / "test_predictions.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    assert {row["sample_id"] for row in predictions} == set(expected)
    for row in predictions:
        for name, value in expected[row["sample_id"]].items():
            assert row[name] == value


def test_old_hitlist_fails_before_download(monkeypatch):
    import hitlist

    monkeypatch.setattr(hitlist, "__version__", "1.74.0")
    with pytest.raises(RuntimeError, match="Hitlist >=1.75.0"):
        load_scoped_bulk_records()


def test_scoped_records_pass_canonical_source_lineage_audit(scoped_bundle):
    from presto.training.data_support import audit_split_support

    records, _ = load(scoped_bundle)
    audit = audit_split_support({"admitted": PrestoDataset(bulk_ms_records=records)})
    assert audit["lineage"]["issue_count"] == 0
