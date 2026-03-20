"""Tests for mhcseqs-first exact sequence resolution."""

import types

from presto.data import mhc_sequence_resolver as resolver
from presto.data.mhc_sequence_resolver import ExactMHCInput


def test_resolve_exact_mhc_sequences_prefers_mhcseqs(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: {
            "HLA-A*02:01": ExactMHCInput(
                allele="HLA-A*02:01",
                sequence="A" * 80,
                source="mhcseqs",
            ),
            "HLA-A*02:01".upper(): ExactMHCInput(
                allele="HLA-A*02:01",
                sequence="A" * 80,
                source="mhcseqs",
            ),
        },
    )
    monkeypatch.setattr(
        resolver,
        "_require_mhcseqs",
        lambda: types.SimpleNamespace(normalize_allele_name=lambda allele: allele),
    )

    mapping, stats = resolver.resolve_exact_mhc_sequences(
        ["HLA-A*02:01", "HLA-B*07:02"],
        index_csv=None,
        prefer_mhcseqs=True,
    )

    assert mapping == {"HLA-A*02:01": "A" * 80}
    assert stats["resolved"] == 1
    assert stats["resolved_mhcseqs"] == 1
    assert stats["resolved_index"] == 0
    assert stats["missing"] == 1


def test_resolve_exact_mhc_sequences_falls_back_to_index(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: {},
    )
    monkeypatch.setattr(
        resolver,
        "_require_mhcseqs",
        lambda: types.SimpleNamespace(normalize_allele_name=lambda allele: allele),
    )
    monkeypatch.setattr(
        resolver,
        "resolve_alleles",
        lambda index_csv, alleles, include_sequence: [
            {
                "input": allele,
                "found": allele == "HLA-B*07:02",
                "sequence": "B" * 80 if allele == "HLA-B*07:02" else "",
            }
            for allele in alleles
        ],
    )

    mapping, stats = resolver.resolve_exact_mhc_sequences(
        ["HLA-A*02:01", "HLA-B*07:02"],
        index_csv="dummy.csv",
        prefer_mhcseqs=True,
    )

    assert mapping == {"HLA-B*07:02": "B" * 80}
    assert stats["resolved"] == 1
    assert stats["resolved_mhcseqs"] == 0
    assert stats["resolved_index"] == 1
    assert stats["missing"] == 1


def test_resolve_exact_mhc_inputs_prefers_mhcseqs_grooves(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: {
            "HLA-A*02:01": ExactMHCInput(
                allele="HLA-A*02:01",
                sequence="A" * 181,
                groove1="A" * 90,
                groove2="B" * 93,
                mhc_class="I",
                source="mhcseqs",
            ),
            "HLA-A*02:01".upper(): ExactMHCInput(
                allele="HLA-A*02:01",
                sequence="A" * 181,
                groove1="A" * 90,
                groove2="B" * 93,
                mhc_class="I",
                source="mhcseqs",
            ),
        },
    )
    monkeypatch.setattr(
        resolver,
        "_require_mhcseqs",
        lambda: types.SimpleNamespace(normalize_allele_name=lambda allele: allele),
    )

    mapping, stats = resolver.resolve_exact_mhc_inputs(
        ["HLA-A*02:01"],
        index_csv=None,
        prefer_mhcseqs=True,
    )

    assert mapping["HLA-A*02:01"].groove1 == "A" * 90
    assert mapping["HLA-A*02:01"].groove2 == "B" * 93
    assert mapping["HLA-A*02:01"].source == "mhcseqs"
    assert stats["resolved"] == 1
    assert stats["resolved_mhcseqs"] == 1


def test_lookup_exact_mhc_input_normalizes_query(monkeypatch):
    record = ExactMHCInput(
        allele="HLA-A*02:01",
        sequence="A" * 181,
        groove1="A" * 90,
        groove2="C" * 93,
        mhc_class="I",
        source="mhcseqs",
    )
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: {
            "HLA-A*02:01": record,
            "HLA-A*02:01".upper(): record,
        },
    )
    monkeypatch.setattr(
        resolver,
        "_require_mhcseqs",
        lambda: types.SimpleNamespace(normalize_allele_name=lambda allele: "HLA-A*02:01"),
    )

    resolved = resolver.lookup_exact_mhc_input("A0201")

    assert resolved == record


def test_resolve_class_i_groove_halves_prefers_exact_mhcseqs(monkeypatch):
    record = ExactMHCInput(
        allele="HLA-A*02:01",
        sequence="A" * 181,
        groove1="G" * 90,
        groove2="H" * 93,
        mhc_class="I",
        source="mhcseqs",
    )
    monkeypatch.setattr(
        resolver,
        "lookup_exact_mhc_input",
        lambda allele, mhcseqs_search_dir=None: record,
    )

    resolved = resolver.resolve_class_i_groove_halves(
        allele="HLA-A*02:01",
        allele_sequences={"HLA-A*02:01": "S" * 181},
    )

    assert resolved == ("G" * 90, "H" * 93)


def test_resolve_class_i_groove_halves_falls_back_to_sequence_map(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "lookup_exact_mhc_input",
        lambda allele, mhcseqs_search_dir=None: None,
    )
    monkeypatch.setattr(
        resolver,
        "prepare_mhc_input",
        lambda *, mhc_a, mhc_class: types.SimpleNamespace(
            groove_half_1="A" * 90,
            groove_half_2="B" * 93,
        ),
    )

    resolved = resolver.resolve_class_i_groove_halves(
        allele="HLA-A*02:01",
        allele_sequences={"A*02:01": "S" * 181},
    )

    assert resolved == ("A" * 90, "B" * 93)
