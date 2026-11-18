"""Real pairing/split machinery with artificial source-admission fixtures."""

import gzip
import json
import random

import pandas as pd
import pytest

from presto.data import presentation_preparation as preparation
from presto.data.modular import AA, MHCInput, Observation
from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.data.presentation_identities import freeze_identities


def test_identity_sources_reject_test_before_payload_access(tmp_path):
    path = tmp_path / "identities.json"
    path.write_text(
        json.dumps(
            {
                "schema": "presto.presentation-identities.v1",
                "sources": [
                    {"path": "SEALED-NOT-PRESENT", "role": "fixed_validation", "split": "test"},
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="never TEST"):
        freeze_identities(path, tmp_path / "output")


def test_both_class_preparation_preserves_protection_and_complete_pairs(tmp_path, monkeypatch):
    rng = random.Random(42)
    raw_rows, proteins = [], []
    fields = [
        "condition_id",
        "condition_status",
        "condition_evidence",
        "condition_reference",
        "sample_attribution",
        "condition_knockout_genes",
        "condition_knockdown_genes",
        "condition_overexpression_genes",
        "condition_genetic_variants",
        "condition_transfection",
        "condition_transduction",
        "condition_cytokines",
        "condition_drugs",
        "condition_stimulation",
        "condition_background",
    ]
    for i in range(30):
        sequence = "".join(rng.choice(AA) for _ in range(80))
        peptide = sequence[20:29] if i % 2 else sequence[20:35]
        raw_rows.append(
            {
                **{name: "" for name in fields},
                "pmid": "fixture-study",
                "peptide": peptide,
                "mhc_class": "I" if i % 2 else "II",
                "mhc_allele_set": "declared",
                "evidence_row_id": f"fixture-{i}",
            }
        )
        proteins.append(f">ENSP{i}|ENST{i}|ENSG{i}\n{sequence}\n")
    files = {}
    for cls in ("I", "II"):
        path = tmp_path / f"class-{cls}.parquet"
        pd.DataFrame([row for row in raw_rows if row["mhc_class"] == cls]).to_parquet(path)
        files[f"class_{cls}"] = path
    files["samples"] = tmp_path / "samples.csv"
    files["samples"].write_text("pmid,condition_id\n")
    files["catalog"] = tmp_path / "catalog.csv"
    files["catalog"].write_text("allele\n")
    files["proteins"] = tmp_path / "proteins.fa"
    files["proteins"].write_text("".join(proteins))
    files["observed"] = tmp_path / "observed.parquet"
    pd.DataFrame({"peptide": [row["peptide"] for row in raw_rows]}).to_parquet(files["observed"])
    identities = []
    for role, peptides in {
        "sealed": ["WWWWWWWWW"],
        "fixed_validation": [raw_rows[0]["peptide"]],
        "ancestry": [raw_rows[1]["peptide"]],
    }.items():
        path = tmp_path / (role + ".txt")
        path.write_text("\n".join(peptides) + "\n")
        identities.append({"role": role, "path": str(path), "sha256": file_sha256(path)})
    manifest = {
        "schema": "presto.presentation-preparation.v1",
        "ancestry_complete": True,
        "identities": identities,
        "files": {
            name: {"path": str(path), "sha256": file_sha256(path)} for name, path in files.items()
        },
    }
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(preparation, "HumanMHCCatalog", lambda _: None)

    def admitted(raw, _catalog, **_options):
        return Observation(
            raw["evidence_row_id"],
            "hitlist_native_ms",
            "elution_observation",
            raw["peptide"],
            (MHCInput("ACDEFGHIK", "LMNPQRSTV", raw["mhc_class"]),),
            value=1.0,
            unit="binary",
            provenance={"raw_record": raw, "original_measurement_id": raw["evidence_row_id"]},
        )

    monkeypatch.setattr(preparation, "ms_observation", admitted)
    result = preparation.prepare_from_manifest(path, tmp_path / "prepared")
    assert result["status"] == "prepared"
    index = ObservationIndex(tmp_path / "prepared/manifest.json")
    rows = [index[i] for i in range(len(index))]
    assert {row.split for row in rows} == {"train", "validation"}
    assert {row.mhc[0].mhc_class for row in rows} == {"I", "II"}
    assert all(row.peptide != raw_rows[0]["peptide"] for row in rows)
    assert all(row.split == "train" for row in rows if row.peptide == raw_rows[1]["peptide"])
    pairs = {}
    for row in rows:
        pairs.setdefault(row.provenance["pair_id"], []).append(row)
        assert row.context.cell_type is None
    assert all(sorted(row.value for row in pair) == [0, 1] for pair in pairs.values())
    assert all(len({row.split for row in pair}) == 1 for pair in pairs.values())
    # Admission is recorded separately from pair construction and its exclusions.
    with gzip.open(tmp_path / "prepared/admitted.jsonl.gz", "rt") as stream:
        assert sum(1 for _ in stream) == 29
    audit = preparation.audit_prepared(tmp_path / "prepared")
    assert audit["status"] == "passed" and audit["new_validation_has_both_classes"]
