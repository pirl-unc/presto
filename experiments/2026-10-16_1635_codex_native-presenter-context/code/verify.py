"""Independently verify native presenter contexts and immutable source measurements."""

import csv
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from datacache import install_bundle

FAMILY = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".hitlist/presto-presenter-context-audit/public-verification"


def pin(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def verify(publication):
    payload = json.loads(Path(publication).read_text())
    assets = {
        name: {key: record[key] for key in ("url", "sha256", "size")}
        for name, record in payload["files"].items()
    }
    paths = install_bundle(
        CACHE / pin(publication)["sha256"],
        assets,
        download_options={"show_progress": False},
    )
    audit = json.loads(Path(paths["audit.json"]).read_text())
    input_key = hashlib.sha256(
        json.dumps(audit["inputs"], sort_keys=True).encode()
    ).hexdigest()
    inputs = install_bundle(
        CACHE / "original-inputs" / input_key,
        audit["inputs"],
        download_options={"show_progress": False},
    )
    rows = read_rows(paths["presenter-context-observations.tsv"])
    changed = read_rows(paths["changed-processing-contexts.tsv"])
    membership = read_rows(inputs["membership.tsv"])
    by_id = {row["evidence_row_id"]: row for row in rows}
    assert len(by_id) == len(rows) == len(membership) == 9053
    assert [row["evidence_row_id"] for row in rows] == [
        row["evidence_row_id"] for row in membership
    ]
    for row, original in zip(rows, membership):
        for field, value in original.items():
            assert row[field] == value, (row["evidence_row_id"], field)
        alleles = json.loads(row["resolved_mhc_alleles"])
        assert alleles and all(allele.startswith("HLA-") for allele in alleles)
        assert (
            row["corrected_molecular_mhc_bucket"]
            == row["molecular_mhc_species_bucket"]
            == "human"
        )
        assert row["molecular_mhc_species_status"] == "resolved"
        assert row["molecular_mhc_species"] == "Homo sapiens"
        host = row["host"]
        assert host in {"Homo sapiens (human)", "Rattus norvegicus (brown rat)"}
        expected = "human" if host.startswith("Homo sapiens") else "murine"
        assert row["historical_host_based_mhc_bucket"] == expected
        assert row["mhc_label_changed"] == str(expected != "human")
    prior_roles = read_rows(inputs["prior-species-roles.tsv"])
    assert len(prior_roles) == len(rows)
    for row, original in zip(rows, prior_roles):
        for field, value in original.items():
            assert row[field] == value, (row["evidence_row_id"], field)
        presenter = row["presenting_species"]
        assert presenter in {"Homo sapiens", "Rattus norvegicus", ""}
        expected = {"Homo sapiens": "human", "Rattus norvegicus": "murine", "": ""}[
            presenter
        ]
        assert row["processing_species"] == presenter
        assert (
            row["processing_species_bucket"]
            == row["corrected_processing_bucket"]
            == expected
        )
        assert row["processing_species_status"] == (
            "resolved" if presenter else "unknown"
        )
        assert row["processing_species_basis"] == "native_presenting_species"
        assert row["parent_processing_species"] == ""
        host_bucket = "human" if row["host"].startswith("Homo sapiens") else "murine"
        assert row["historical_host_based_processing_bucket"] == host_bucket
        assert row["processing_context_changed"] == str(expected != host_bucket)
    expected_changed = [
        row for row in rows if row["processing_context_changed"] == "True"
    ]
    assert changed == expected_changed and len(changed) == 8936
    assert audit["n_changed_by_partition"] == dict(
        Counter(row["source_partition"] for row in changed)
    )
    original_species = read_rows(inputs["source-species.tsv"])
    for original in original_species:
        row = by_id[original["evidence_row_id"]]
        for field, value in original.items():
            assert row[field] == value, (row["evidence_row_id"], field)
    # Compare every exposed native column directly with the pinned parquet,
    # without importing Presto or its curation/normalization helpers.
    parquet = pq.ParquetFile(inputs["training.parquet"])
    columns = [name for name in rows[0] if name in parquet.schema_arrow.names]
    assert {
        "host",
        "source_organism",
        "mhc_species",
        "presenting_species",
        "introduced_mhc_species",
        "quantitative_value",
        "condition_id",
    } <= set(columns)
    seen = set()
    for batch in parquet.iter_batches(batch_size=5000, columns=columns):
        values = batch.to_pydict()
        for index, identity in enumerate(values["evidence_row_id"]):
            if identity not in by_id:
                continue
            assert identity not in seen
            seen.add(identity)
            for field in columns:
                value = values[field][index]
                assert by_id[identity][field] == (
                    "" if value is None else str(value)
                ), (
                    identity,
                    field,
                )
    assert seen == by_id.keys()
    for row in rows:
        for reported, native in {
            "reported_host_species": "host",
            "reported_antigen_species": "source_organism",
            "reported_mhc_species": "mhc_species",
            "reported_mhc_species_basis": "mhc_species_source",
            "reported_mhc_species_context_disagrees": "mhc_species_context_disagrees",
        }.items():
            assert row[reported] == row[native]
        assert row["species_context_origin"] == "native_observation"
    ic50 = read_rows(inputs["ic50-inspection.tsv"])
    assert len(ic50) == 2614
    ic50_contexts = read_rows(paths["ic50-processing-contexts.tsv"])
    assert len(ic50_contexts) == len(ic50)
    ic50_by_id = {row["evidence_row_id"]: row for row in ic50_contexts}
    assert len(ic50_by_id) == len(ic50_contexts)
    for original in ic50:
        row = by_id[original["evidence_row_id"]]
        for name in (
            "sample_id",
            "source_partition",
            "peptide",
            "source_value",
            "source_units",
            "source_inequality",
            "source_mhc_alleles",
            "resolved_mhc_alleles",
        ):
            assert row[name] == original[name]
        detail = ic50_by_id[row["evidence_row_id"]]
        for name, value in row.items():
            assert detail[name] == value
        assert (
            detail["historical_raw_target_log10_nM"] == original["raw_target_log10_nM"]
        )
        assert (
            detail["historical_effective_target_log10_nM"]
            == original["effective_target_log10_nM"]
        )
        assert float(detail["prospective_effective_target_log10_nM"]) == float(
            original["effective_target_log10_nM"]
        )
        assert float(row["quantitative_value"]) == float(original["source_value"])
        assert row["measurement_units"] == original["source_units"] == "nM"
        assert row["measurement_inequality"] == original["source_inequality"]
    policy = json.loads(Path(paths["output-policy.json"]).read_text())
    assert policy["schema_version"] == 4
    assert (
        policy["processing_species_policy"]
        == "native_presenter_and_generated_parent_context_v1"
    )
    assert audit["n_ic50_by_processing_bucket"] == dict(
        Counter(row["processing_species_bucket"] or "unknown" for row in ic50_contexts)
    )
    assert (
        policy["molecular_mhc_species_policy"]
        == "explicit_allele_identity_consensus_v1"
    )
    for axis, counts in audit["n_observations_by_axis"].items():
        assert counts == dict(Counter(row[axis] or "unknown" for row in rows))
    return {
        "verified": True,
        "publication": payload["release"],
        "n_public_assets": len(paths),
        "n_source_observations": len(rows),
        "n_changed_processing_contexts": len(changed),
        "n_ic50_values_operators_partitions_preserved": len(ic50),
        "native_columns_verified": columns,
        "historical_run_preserved": True,
        "no_model_or_new_full_input_qualification": True,
        "verifier": pin(__file__),
        "versions": {
            name: importlib.metadata.version(name) for name in ("datacache", "pyarrow")
        },
        "public_downloads": {
            name: {"path": str(path), **pin(path)} for name, path in paths.items()
        },
    }


if __name__ == "__main__":
    import sys

    result = verify(sys.argv[1])
    (FAMILY / "results").mkdir(exist_ok=True)
    (FAMILY / "results/public-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "public_downloads"},
            indent=2,
        )
    )
