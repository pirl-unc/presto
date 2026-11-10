"""Verify exact, core and position-sensitive near-identity semantics."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "cohort_audit", Path(__file__).with_name("cohort_audit.py")
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_hamming_position_and_length_are_preserved():
    candidates = {"ACDEFGHIKL", "CDAEFGHIKL", "ACDEYGHIKL", "ACDEFGHIKLM"}
    refs = {"ACDEFGHIKL"}
    found = audit.primary(candidates, refs)
    assert found == audit.independent(candidates, refs)
    assert found["hamming1"] == {"ACDEFGHIKL", "ACDEYGHIKL"}
    assert "ACDEFGHIKLM" in found["core8"]
    assert found["exact"] == refs


def test_midpoint_substitution_without_shared_eightmer():
    p = "ACDEFGHIKL"
    q = "ACDEYGHIKL"
    found = audit.primary({p}, {q})
    assert found == audit.independent({p}, {q})
    assert found["hamming1"] == {p} and not found["core8"] and not found["exact"]


def test_reference_noncanonical_residue_stays_in_embargo():
    candidates = {"ACDEFGHIKL"}
    refs = {"ACDEUGHIKL"}
    assert audit.primary(candidates, refs) == audit.independent(candidates, refs)
    assert audit.primary(candidates, refs)["hamming1"] == candidates
