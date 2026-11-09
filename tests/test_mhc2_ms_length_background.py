"""Boundary tests for the revised training-frequency background contract."""

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments"
PATH = BASE / "2026-10-26_2036_codex_mhc2-ms-length-background/code/cohort_rules.py"
SPEC = importlib.util.spec_from_file_location("length_background_rules", PATH)
rules = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rules)
AA = "ACDEFGHIKLMNPQRSTVWY"


def fixture():
    return {
        ("35251023", AA): ("train", "dp1"),
        ("35251023", "AAAAAAAAAAAAAAA"): ("validation", "dp2"),
        ("34433824", "VVVVVVVVVVVVVVV"): ("external", "dq1"),
    }


def test_positive_partition_code_unchanged_from_frozen_original():
    old = BASE / "2026-10-26_2030_codex_mhc2-ms-prediction/reproduce/source/cohort_rules.py"

    def functions(path):
        return {
            node.name: ast.dump(node)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)
        }

    original, revised = functions(old), functions(PATH)
    for name in ("kmers", "digest", "components", "dp_split", "partition"):
        assert revised[name] == original[name]


def test_frequencies_use_only_distinct_dp_training_peptides():
    assignments = fixture()
    frequencies = rules.training_frequencies(assignments)
    assert frequencies["counts"] == [1] * 20
    assignments[("34433824", "AAAAAAAAA")] = ("external", "dq2")
    assignments[("35251023", "YYYYYYYYY")] = ("test", "dp3")
    assert rules.training_frequencies(assignments) == frequencies
    bad = dict(frequencies, counts=[2] * 20)
    with pytest.raises(ValueError, match="frequencies disagree"):
        rules.backgrounds(assignments, set(), set(), bad)


def test_homopolymer_positive_kept_with_distinct_length_matched_control():
    assignments = fixture()
    frequencies = rules.training_frequencies(assignments)
    known = {key[1] for key in assignments}
    generated = rules.backgrounds(assignments, known, set(), frequencies)
    assert len(generated) == len(assignments)
    for parent, control in generated.items():
        assert len(parent) == len(control["peptide"])
        assert parent != control["peptide"]
        assert control["peptide"] not in known
    assert generated == rules.backgrounds(
        dict(reversed(list(assignments.items()))), known, set(), frequencies
    )


def test_known_positive_and_initializer_collisions_are_rejected():
    assignments = fixture()
    frequencies = rules.training_frequencies(assignments)
    known = {key[1] for key in assignments}
    first = rules.backgrounds(assignments, known, set(), frequencies)
    exposed = rules.kmers(first["AAAAAAAAAAAAAAA"]["peptide"])
    known.add(first[AA]["peptide"])
    changed = rules.backgrounds(assignments, known, exposed, frequencies)
    assert changed[AA]["peptide"] not in known
    for parent, control in changed.items():
        if assignments[next(key for key in assignments if key[1] == parent)][0] != "train":
            assert rules.kmers(control["peptide"]).isdisjoint(exposed)
    for left, left_control in changed.items():
        for right, right_control in changed.items():
            if left != right:
                assert rules.kmers(left_control["peptide"]).isdisjoint(
                    rules.kmers(right_control["peptide"]) | rules.kmers(right)
                )
