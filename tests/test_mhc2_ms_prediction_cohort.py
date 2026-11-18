"""Source/control boundary checks for the prospectively registered MS comparison."""

from collections import Counter
import importlib.util
from pathlib import Path

import pytest


PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-10-26_2030_codex_mhc2-ms-prediction/code/cohort_rules.py"
)
SPEC = importlib.util.spec_from_file_location("ms_prediction_cohort_rules", PATH)
rules = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rules)


def test_transitive_component_includes_distinct_end_sequences_and_ignores_order():
    left = "ACDEFGHIKLMN"
    bridge = "EFGHIKLMNPQRSTVWY"
    right = "NPQRSTVWYAAA"
    separate = "YYYYYYYYY"
    peptides = [left, bridge, right, separate]
    mapping = rules.components(peptides)
    assert mapping[left] == mapping[bridge] == mapping[right]
    assert mapping[separate] != mapping[left]
    assert rules.kmers(left).isdisjoint(rules.kmers(right))
    assert rules.components(reversed(peptides)) == mapping


def test_initializer_exposure_removes_whole_evaluation_component(monkeypatch):
    monkeypatch.setattr(rules, "dp_split", lambda _: "validation")
    left, right = "ACDEFGHIKLMN", "DEFGHIKLMNPQ"
    # Only the first peptide directly contains the exposed nine-residue segment.
    assert "ACDEFGHIK" not in rules.kmers(right)
    assignments, dispositions = rules.partition([left, right], [], {"ACDEFGHIK"})
    assert not assignments
    assert {row["reason"] for row in dispositions} == {"initializer_exposed_evaluation_component"}


def test_external_exclusion_uses_all_dp_even_excluded_dp(monkeypatch):
    monkeypatch.setattr(rules, "dp_split", lambda _: "test")
    dp, dq = "ACDEFGHIKLMN", "DEFGHIKLMNPQ"
    assignments, dispositions = rules.partition([dp], [dq], {"ACDEFGHIK"})
    assert not assignments
    assert dispositions[-1]["reason"] == "dp_source_overlap"


def test_generated_backgrounds_preserve_composition_and_reject_known_controls():
    train, test = "ACDEFGHIKLMNPQRSTVWY", "AACDEFGHIKPQRSTVWYY"
    assignments = {
        ("35251023", train): ("train", "c1"),
        ("34433824", test): ("external", "c2"),
    }
    first = rules.backgrounds(assignments, {train, test}, set())
    known = {train, test, first[train]["peptide"], first[test]["peptide"]}
    second = rules.backgrounds(assignments, known, set())
    assert second == rules.backgrounds(dict(reversed(list(assignments.items()))), known, set())
    for parent, control in second.items():
        assert Counter(control["peptide"]) == Counter(parent)
        assert control["peptide"] not in known
    assert rules.kmers(second[train]["peptide"]).isdisjoint(
        rules.kmers(second[test]["peptide"]) | rules.kmers(test)
    )
    assert rules.kmers(second[test]["peptide"]).isdisjoint(rules.kmers(train))


def test_exhausted_controls_fail_instead_of_silently_dropping_positive():
    with pytest.raises(ValueError, match="no valid background"):
        rules.backgrounds({("35251023", "AAAAAAAAA"): ("train", "c")}, {"AAAAAAAAA"}, set())


def test_same_positive_cannot_cross_partitions():
    with pytest.raises(ValueError, match="positive peptide crosses splits"):
        rules.backgrounds(
            {
                ("35251023", "ACDEFGHIK"): ("train", "c"),
                ("34433824", "ACDEFGHIK"): ("external", "d"),
            },
            {"ACDEFGHIK"},
            set(),
        )
