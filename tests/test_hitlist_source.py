"""Tests for the hitlist-backed record adapter.

These run without the real ``hitlist`` package by injecting a stub module that
returns a caller-supplied frame from ``generate_training_table``.
"""

import sys
import types

import pytest

pd = pytest.importorskip("pandas")

from presto.data.hitlist_source import (  # noqa: E402
    _clean,
    _method_counts,
    _qualifier_from_inequality,
    _select_best_mapping,
    _split_allele_set,
    load_records_from_hitlist,
)

IC50 = "half maximal inhibitory concentration (IC50)"


def _binding_row(**overrides):
    row = {
        "peptide": "SIINFEKLA",
        "mhc_restriction": "HLA-A*02:01",
        "mhc_allele_set": "HLA-A*02:01",
        "mhc_class": "I",
        "host": "Homo sapiens (human)",
        "source_organism": "Homo sapiens",
        "source": "iedb",
        "n_flank": "AAAAAAAAAA",
        "c_flank": "CCCCCCCCCC",
        "evidence_row_id": "row-1",
        "is_canonical_transcript": True,
        "response_measured": IC50,
        "quantitative_value": 25.0,
        "measurement_units": "nM",
        "measurement_inequality": "=",
        "assay_method": "purified MHC/competitive/radioactivity",
    }
    row.update(overrides)
    return row


def _ms_row(**overrides):
    row = {
        "peptide": "SIINFEKLA",
        "mhc_restriction": "HLA-A*02:01",
        "mhc_allele_set": "HLA-A*02:01;HLA-B*07:02",
        "mhc_class": "I",
        "host": "Homo sapiens (human)",
        "source_organism": "Homo sapiens",
        "source": "iedb",
        "n_flank": "GGGGGGGGGG",
        "c_flank": "TTTTTTTTTT",
        "evidence_row_id": "ms-1",
        "is_canonical_transcript": True,
        "cell_line_name": "HeLa",
        "source_tissue": "Skin",
    }
    row.update(overrides)
    return row


def _install_stub_hitlist(monkeypatch, binding_rows, ms_rows):
    module = types.ModuleType("hitlist")

    def generate_training_table(*, include_evidence, columns=None, **_kwargs):
        rows = binding_rows if include_evidence == "binding" else ms_rows
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=columns or [])
        return frame

    module.generate_training_table = generate_training_table
    monkeypatch.setitem(sys.modules, "hitlist", module)


class TestHelpers:
    def test_qualifier_from_inequality(self):
        assert _qualifier_from_inequality("<") == -1
        assert _qualifier_from_inequality("<=") == -1
        assert _qualifier_from_inequality(">") == 1
        assert _qualifier_from_inequality(">=") == 1
        assert _qualifier_from_inequality("=") == 0
        assert _qualifier_from_inequality("") == 0
        assert _qualifier_from_inequality(None) == 0

    def test_split_allele_set(self):
        assert _split_allele_set("HLA-A*02:01") == ["HLA-A*02:01"]
        assert _split_allele_set("HLA-A*02:01;HLA-B*07:02") == [
            "HLA-A*02:01",
            "HLA-B*07:02",
        ]
        assert _split_allele_set(["HLA-A*02:01"]) == ["HLA-A*02:01"]
        assert _split_allele_set("") == []
        assert _split_allele_set(None) == []

    def test_clean_normalizes_missing_markers(self):
        assert _clean("nan") == ""
        assert _clean(None) == ""
        assert _clean("  X  ") == "X"

    def test_select_best_mapping_prefers_canonical_transcript(self):
        frame = pd.DataFrame(
            [
                {"evidence_row_id": "a", "is_canonical_transcript": False, "n_flank": "NO"},
                {"evidence_row_id": "a", "is_canonical_transcript": True, "n_flank": "YES"},
            ]
        )
        collapsed = _select_best_mapping(frame)
        assert len(collapsed) == 1
        assert collapsed.iloc[0]["n_flank"] == "YES"

    def test_method_counts_labels_missing_as_unspecified(self):
        class Rec:
            def __init__(self, method):
                self.assay_method = method

        counts = _method_counts([Rec("a"), Rec("a"), Rec(None)])
        assert counts == {"a": 2, "unspecified": 1}


class TestRouting:
    def test_affinity_row_becomes_binding_record_with_flanks(self, monkeypatch):
        _install_stub_hitlist(monkeypatch, [_binding_row()], [])
        binding, _, _, _, _, _, _, stats = load_records_from_hitlist()
        assert len(binding) == 1
        record = binding[0]
        assert record.value == 25.0
        assert record.measurement_type == IC50
        assert record.flank_n == "AAAAAAAAAA"
        assert record.flank_c == "CCCCCCCCCC"
        assert stats["flank_coverage"]["binding"] == 1.0

    def test_nan_value_is_skipped(self, monkeypatch):
        """float(nan) does not raise, so NaN must be tested for explicitly.

        Without this guard a missing measurement enters training as a NaN
        target and silently poisons the loss.
        """
        _install_stub_hitlist(monkeypatch, [_binding_row(quantitative_value=float("nan"))], [])
        binding, _, _, _, _, _, _, stats = load_records_from_hitlist()
        assert binding == []
        assert stats["skipped_no_numeric_value"] == 1

    def test_unexpected_unit_is_skipped(self, monkeypatch):
        _install_stub_hitlist(monkeypatch, [_binding_row(measurement_units="uM")], [])
        binding, _, _, _, _, _, _, stats = load_records_from_hitlist()
        assert binding == []
        assert stats["skipped_unexpected_unit"] == 1

    def test_unroutable_response_is_counted_not_ingested(self, monkeypatch):
        _install_stub_hitlist(
            monkeypatch,
            [_binding_row(response_measured="qualitative binding", measurement_units="")],
            [],
        )
        binding, _, _, _, _, _, _, stats = load_records_from_hitlist()
        assert binding == []
        assert stats["skipped_unroutable_response"] == 1

    def test_half_life_converts_minutes_to_hours(self, monkeypatch):
        _install_stub_hitlist(
            monkeypatch,
            [
                _binding_row(
                    response_measured="half life",
                    measurement_units="min",
                    quantitative_value=120.0,
                    assay_method="purified MHC/direct/radioactivity",
                )
            ],
            [],
        )
        _, _, stability, _, _, _, _, stats = load_records_from_hitlist()
        assert len(stability) == 1
        assert stability[0].t_half == pytest.approx(2.0)
        assert stability[0].assay_method == "purified MHC/direct/radioactivity"
        assert stats["stability_assay_methods"] == {
            "purified MHC/direct/radioactivity": 1
        }

    def test_tm_and_kinetics_routing(self, monkeypatch):
        _install_stub_hitlist(
            monkeypatch,
            [
                _binding_row(
                    response_measured="50% dissociation temperature",
                    measurement_units="°C",
                    quantitative_value=52.0,
                    evidence_row_id="tm-1",
                ),
                _binding_row(
                    response_measured="off rate",
                    measurement_units="1/s",
                    quantitative_value=0.01,
                    evidence_row_id="koff-1",
                ),
            ],
            [],
        )
        _, kinetics, stability, _, _, _, _, _ = load_records_from_hitlist()
        assert len(stability) == 1 and stability[0].tm == 52.0
        assert len(kinetics) == 1 and kinetics[0].koff == 0.01

    def test_ms_row_becomes_elution_record_with_allele_bag(self, monkeypatch):
        _install_stub_hitlist(monkeypatch, [], [_ms_row()])
        _, _, _, _, elution, _, _, stats = load_records_from_hitlist()
        assert len(elution) == 1
        assert elution[0].alleles == ["HLA-A*02:01", "HLA-B*07:02"]
        assert elution[0].flank_n == "GGGGGGGGGG"
        assert stats["flank_coverage"]["elution"] == 1.0

    def test_ms_row_without_alleles_is_dropped(self, monkeypatch):
        _install_stub_hitlist(
            monkeypatch, [], [_ms_row(mhc_allele_set="", mhc_restriction="")]
        )
        _, _, _, _, elution, _, _, _ = load_records_from_hitlist()
        assert elution == []

    def test_modalities_hitlist_does_not_carry_are_empty(self, monkeypatch):
        _install_stub_hitlist(monkeypatch, [_binding_row()], [_ms_row()])
        _, _, _, processing, _, tcell, tcr, stats = load_records_from_hitlist()
        assert processing == [] and tcell == [] and tcr == []
        assert stats["counts"]["processing"] == 0


class TestPeptideValidation:
    """Non-canonical peptides must be dropped at ingest, not at tokenization.

    IEDB ships modification and ambiguity annotations -- `NXVPMVATV`,
    `SXPSGGXGV + INDIST(X2, X7)`, `ILAETVAXV + OTH(X8)`. They describe
    chemistry the model has no representation for, and they used to reach the
    tokenizer and abort a training run mid-epoch with `Invalid amino-acid
    token ' '`. About 0.007% of rows, which is exactly why it survived every
    short smoke test.
    """

    @pytest.mark.parametrize(
        "peptide",
        [
            "GXVPFXVS + INDIST(X2, X6)",
            "ILAETVAXV + OTH(X8)",
            "NXVPMVATV",
            "SIINFEKL*",
            "",
        ],
    )
    def test_noncanonical_binding_peptide_is_skipped(self, monkeypatch, peptide):
        _install_stub_hitlist(monkeypatch, [_binding_row(peptide=peptide)], [])
        binding, _, _, _, _, _, _, stats = load_records_from_hitlist()
        assert binding == []
        assert stats["skipped_noncanonical_peptide"] == 1

    def test_noncanonical_elution_peptide_is_skipped(self, monkeypatch):
        _install_stub_hitlist(monkeypatch, [], [_ms_row(peptide="NXVPMVATV")])
        _, _, _, _, elution, _, _, stats = load_records_from_hitlist()
        assert elution == []
        assert stats["skipped_noncanonical_peptide"] == 1

    def test_canonical_peptides_still_pass(self, monkeypatch):
        _install_stub_hitlist(monkeypatch, [_binding_row()], [_ms_row()])
        binding, _, _, _, elution, _, _, stats = load_records_from_hitlist()
        assert len(binding) == 1 and len(elution) == 1
        assert stats["skipped_noncanonical_peptide"] == 0

    def test_every_ingested_peptide_is_tokenizable(self, monkeypatch):
        """The invariant the crash violated, stated directly."""
        from presto.data.hitlist_source import is_canonical_peptide

        _install_stub_hitlist(
            monkeypatch,
            [_binding_row(peptide="SIINFEKLA"), _binding_row(peptide="NXVPMVATV")],
            [_ms_row(peptide="ILAETVAXV + OTH(X8)"), _ms_row(peptide="LLDGTATLRF")],
        )
        binding, _, _, _, elution, _, _, _ = load_records_from_hitlist()
        for record in [*binding, *elution]:
            assert is_canonical_peptide(record.peptide)


class TestChunkedRowIteration:
    """Rows must stream, not materialize all at once.

    `to_dict("records")` on the full corpus builds ~750k dicts before the loop
    body runs even once, undoing the column pruning done upstream to fit the
    load in memory.
    """

    def test_yields_every_row_across_chunk_boundaries(self):
        from presto.data.hitlist_source import _iter_row_dicts

        frame = pd.DataFrame({"peptide": [f"P{i}" for i in range(25)], "n": range(25)})
        rows = list(_iter_row_dicts(frame, chunk_size=7))
        assert len(rows) == 25
        assert [r["n"] for r in rows] == list(range(25))

    def test_handles_an_empty_frame(self):
        from presto.data.hitlist_source import _iter_row_dicts

        assert list(_iter_row_dicts(pd.DataFrame({"a": []}), chunk_size=4)) == []

    def test_is_lazy(self):
        """Consuming one row must not require building all of them."""
        import itertools

        from presto.data.hitlist_source import _iter_row_dicts

        frame = pd.DataFrame({"n": range(1000)})
        first_two = list(itertools.islice(_iter_row_dicts(frame, chunk_size=2), 2))
        assert [r["n"] for r in first_two] == [0, 1]
