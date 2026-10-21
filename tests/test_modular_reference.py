"""Selected reference reconstruction, not a biological prediction benchmark."""

import csv
from dataclasses import replace
import zlib

import pytest

from presto.data.modular import Observation
from presto.data.modular_reference import audit_weirdo_reference
from presto.data.modular_sources import file_sha256

SALT = "category-predictor-v1"
CATEGORIES = ("human", "rodents")
PEPTIDE = "ACDEFGHI"
SECOND = "LMNPQRST"


def fold(family):
    value = zlib.crc32(f"{SALT}:{family}".encode()) % 100
    return "train" if value < 70 else "validation" if value < 85 else "test"


def bucket(family):
    return zlib.crc32(f"{SALT}:bucket:{family}".encode()) % 20


def fixture(tmp_path, *, cross_fold=False):
    family = "EXAMPLE"
    rows = [
        (f"{family}_HUMAN", "XX" + PEPTIDE * 2 + "X" + SECOND, "human"),
        (f"{family}_MOUSE", "LLLL" + PEPTIDE + "MMMM", "rodents"),
        (f"{family}_A", "ACDE", "human"),
        (f"{family}_B", "LMNP", "human"),
    ]
    if cross_fold:
        other = next(f"OTHER{i}" for i in range(100) if fold(f"OTHER{i}") != fold(family))
        rows.append((f"{other}_HUMAN", PEPTIDE, "human"))
    path = tmp_path / "reference.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identifier", "sequence", "category"])
        writer.writerows(rows)
    digest = file_sha256(path)
    observations = []
    for i, peptide in enumerate((PEPTIDE, SECOND)):
        observations.append(
            Observation(
                f"selected-{i}",
                "software",
                "identity",
                peptide,
                split=fold(family),
                group_id="software",
                identity_labels={
                    "peptide_source:human": "present",
                    "peptide_source:rodents": "present" if i == 0 else "absent",
                },
                provenance={
                    "source_reference_sha256": digest,
                    "family_split_salt": SALT,
                    "source_split": "val" if fold(family) == "validation" else fold(family),
                    "source_family_bucket": bucket(family),
                },
            )
        )
    return path, digest, observations


def audit(path, digest, rows, **kwargs):
    return audit_weirdo_reference(
        rows,
        path,
        reference_sha256=digest,
        categories=CATEGORIES,
        salt=SALT,
        **kwargs,
    )


def test_exact_memberships_occurrences_components_and_chunk_invariance(tmp_path):
    path, digest, rows = fixture(tmp_path)
    one = audit(path, digest, rows, chunk_proteins=1)
    many = audit(path, digest, rows, chunk_proteins=100)
    assert one["status"] == "passed" and one["components"] == 1
    assert one["records"] == many["records"]
    record = one["records"][0]
    assert record["reference_mask"] == 3
    assert [protein["positions"] for protein in record["proteins"]] == [[2, 10], [4]]
    assert one["records"][1]["reference_mask"] == 1
    assert one["records"][1]["proteins"][0]["positions"] == [19]


def test_audit_exposes_cross_fold_and_wrong_membership_without_relabeling(tmp_path):
    path, digest, rows = fixture(tmp_path, cross_fold=True)
    rows[0] = replace(
        rows[0],
        identity_labels={"peptide_source:human": "present", "peptide_source:rodents": "absent"},
    )
    result = audit(path, digest, rows)
    assert result["status"] == "failed"
    record = result["records"][0]
    assert record["expected_mask"] == 1 and record["reference_mask"] == 3
    assert "membership_mismatch" in record["discrepancies"]
    assert "source_fold_mismatch_or_overlap" in record["discrepancies"]
    assert rows[0].identity_labels["peptide_source:rodents"] == "absent"


def test_boundaries_missing_reference_duplicates_and_pins_fail(tmp_path):
    path, digest, rows = fixture(tmp_path)
    missing = replace(rows[0], peptide="ACDELMNP")
    result = audit(path, digest, [missing], chunk_proteins=100)
    assert "not_in_reference" in result["records"][0]["discrepancies"]
    with pytest.raises(ValueError, match="duplicate selected reference"):
        audit(path, digest, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="protein hash mismatch"):
        audit(path, "0" * 64, rows)
    with pytest.raises(TimeoutError, match="declared wall-time"):
        audit(path, digest, rows, max_seconds=1e-20)
    altered = replace(rows[0], provenance={**rows[0].provenance, "source_family_bucket": -1})
    assert "source_bucket_mismatch" in audit(path, digest, [altered])["records"][0]["discrepancies"]


def test_reference_mutation_during_scan_is_rejected(tmp_path):
    path, digest, rows = fixture(tmp_path)

    def mutate(_count):
        with path.open("a") as stream:
            stream.write("\n")

    with pytest.raises(ValueError, match="changed during audit"):
        audit(path, digest, rows, progress=mutate)
