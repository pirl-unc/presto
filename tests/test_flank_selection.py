"""Which source protein's flanks we train on, and whether we admit it was a choice.

A peptide usually occurs in more than one protein. On hitlist 1.55.2 for
HLA-A*02:01: **80.5%** of evidence rows map to more than one protein (worst
case 2,985), and **21.6%** have mappings that disagree on a flank -- 157,423
rows where the junction the excision head trains on is one candidate among
several, presented as though it were observed.

`map_source_proteins=True` hands presto every candidate and the collapse
happens locally, so the choice is ours to make deliberately. These tests pin
the ranking and pin that the ambiguity is recorded rather than discarded.
"""

import pytest

from presto.data.flank_selection import (  # noqa: E402
    SELECTION_BASES,
    FlankChoice,
    select_source_mapping,
)


def _mapping(protein, gene, n, c, canonical=False):
    return {
        "protein_id": protein,
        "gene_name": gene,
        "n_flank": n,
        "c_flank": c,
        "is_canonical_transcript": canonical,
    }


AGREEING = [
    _mapping("P1", "GENEA", "AAAAA", "CCCCC", canonical=True),
    _mapping("P2", "GENEB", "AAAAA", "CCCCC"),
]
DISAGREEING = [
    _mapping("P2", "GENEB", "GGGGG", "TTTTT"),
    _mapping("P1", "GENEA", "AAAAA", "CCCCC", canonical=True),
]


class TestRanking:
    def test_expression_beats_canonical(self):
        """The point of the change: an expressed non-canonical isoform is
        better evidence than an unexpressed canonical one."""
        choice = select_source_mapping(DISAGREEING, expression={"GENEB": 50.0, "GENEA": 1.0})
        assert choice.basis == "expression"
        assert choice.mapping["protein_id"] == "P2"
        assert choice.expression_score == 50.0

    def test_canonical_wins_without_expression(self):
        choice = select_source_mapping(DISAGREEING)
        assert choice.basis == "canonical_transcript"
        assert choice.mapping["protein_id"] == "P1"

    def test_deterministic_order_is_the_last_resort(self):
        plain = [_mapping("P9", "G9", "A", "C"), _mapping("P3", "G3", "G", "T")]
        choice = select_source_mapping(plain)
        assert choice.basis == "deterministic_order"
        assert choice.mapping["protein_id"] == "P3", "sorted by protein id"

    def test_order_of_the_input_does_not_matter(self):
        """Frame order is an accident of the join; the pick must not depend
        on it."""
        forward = select_source_mapping(list(DISAGREEING))
        backward = select_source_mapping(list(reversed(DISAGREEING)))
        assert forward.mapping["protein_id"] == backward.mapping["protein_id"]

    def test_an_unknown_gene_does_not_score_as_zero(self):
        """A partial expression table must degrade to the canonical rule, not
        silently prefer whichever gene happens to be listed."""
        choice = select_source_mapping(DISAGREEING, expression={"GENEZ": 99.0})
        assert choice.basis == "canonical_transcript"

    def test_every_basis_is_named(self):
        for basis in ("expression", "canonical_transcript", "deterministic_order"):
            assert basis in SELECTION_BASES


class TestAmbiguityIsReported:
    def test_disagreement_is_flagged(self):
        choice = select_source_mapping(DISAGREEING)
        assert choice.n_candidates == 2
        assert choice.flanks_agree is False
        assert choice.is_ambiguous is True

    def test_agreement_is_not_ambiguous(self):
        choice = select_source_mapping(AGREEING)
        assert choice.flanks_agree is True
        assert choice.is_ambiguous is False

    def test_a_single_candidate_is_never_ambiguous(self):
        choice = select_source_mapping([AGREEING[0]])
        assert choice.n_candidates == 1
        assert choice.is_ambiguous is False

    def test_empty_input_is_an_error_not_a_silent_default(self):
        with pytest.raises(ValueError, match="at least one candidate"):
            select_source_mapping([])

    def test_the_result_carries_the_mapping_itself(self):
        """So a caller can read flanks without re-deriving the choice."""
        choice = select_source_mapping(AGREEING)
        assert isinstance(choice, FlankChoice)
        assert choice.mapping["n_flank"] == "AAAAA"


class TestIngestRecordsAmbiguity:
    """The count must reach the ingest stats; discarding it is what #34 is
    about."""

    def test_stats_are_computed_before_the_collapse(self):
        pd = pytest.importorskip("pandas")
        from presto.data.hitlist_source import (
            _select_best_mapping,
            mapping_ambiguity_stats,
        )

        frame = pd.DataFrame(
            [
                {
                    "evidence_row_id": "r1",
                    "n_flank": "AAA",
                    "c_flank": "CCC",
                    "protein_id": "P1",
                    "is_canonical_transcript": True,
                },
                {
                    "evidence_row_id": "r1",
                    "n_flank": "GGG",
                    "c_flank": "TTT",
                    "protein_id": "P2",
                    "is_canonical_transcript": False,
                },
                {
                    "evidence_row_id": "r2",
                    "n_flank": "AAA",
                    "c_flank": "CCC",
                    "protein_id": "P3",
                    "is_canonical_transcript": True,
                },
            ]
        )
        stats = mapping_ambiguity_stats(frame)
        assert stats["evidence_rows"] == 2
        assert stats["rows_with_multiple_proteins"] == 1
        assert stats["rows_with_disagreeing_flanks"] == 1
        assert stats["max_proteins_for_one_row"] == 2

        collapsed = _select_best_mapping(frame)
        assert len(collapsed) == 2, "one row per evidence row after collapse"
        assert mapping_ambiguity_stats(collapsed)["rows_with_disagreeing_flanks"] == 0, (
            "after collapsing the alternatives are gone, which is why the "
            "statistic has to be taken first"
        )

    def test_empty_frame_does_not_raise(self):
        pd = pytest.importorskip("pandas")
        from presto.data.hitlist_source import mapping_ambiguity_stats

        stats = mapping_ambiguity_stats(pd.DataFrame())
        assert stats["evidence_rows"] == 0
