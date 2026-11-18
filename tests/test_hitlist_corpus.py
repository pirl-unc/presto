"""Corpus generation pins are independent of current software and loader annotations."""

import json

import pytest

from presto.data.hitlist_corpus import corpus_contract, verify_corpus
from presto.data.modular_sources import file_sha256


def corpus(tmp_path):
    files = {}
    for name in (
        "observations.parquet",
        "binding.parquet",
        "other_assays.parquet",
        "observation_contributors.parquet",
        "bulk_proteomics.parquet",
        "line_expression.parquet",
        "peptide_mappings.parquet",
        "peptide_mappings_meta.json",
    ):
        path = tmp_path / name
        path.write_text("artificial hash fixture: " + name)
        files[name] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    manifest = tmp_path / "observations_meta.json"
    manifest.write_text(json.dumps({"files": files}))
    pin = tmp_path / "pin.json"
    pin.write_text(
        json.dumps(
            {
                "schema": "presto.hitlist-corpus.v1",
                "manifest": manifest.name,
                "manifest_sha256": file_sha256(manifest),
                "loader_version": "2.1.12",
                "producer_version": "1.78.0",
            }
        )
    )
    return pin


def test_packaged_corrected_corpus_is_explicitly_pinned():
    pin = corpus_contract()
    assert pin["manifest_sha256"] == (
        "2318bc53cdae228285218991d58fbcc2bd7266327cec1dd3ed9a03b1c944a5b7"
    )
    assert pin["producer_version"] != pin["loader_version"]


def test_verify_rejects_tampering_and_wrong_generation(tmp_path):
    pin = corpus(tmp_path)
    assert verify_corpus(tmp_path, pin=pin)["verified"]
    observations = tmp_path / "observations.parquet"
    observations.write_bytes(b"x" * observations.stat().st_size)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_corpus(tmp_path, pin=pin)
    manifest = tmp_path / "observations_meta.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        verify_corpus(tmp_path, pin=pin)
