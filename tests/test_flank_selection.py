"""Which source protein's flanks we train on, and whether we admit it was a choice.

A peptide usually occurs in more than one protein. On hitlist 1.55.2 for
HLA-A*02:01: **80.5%** of evidence rows map to more than one protein (worst
case 2,985), and **21.6%** have mappings that disagree on a flank -- 157,423
rows where the junction is not directly observed. Unresolved groups now carry
explicit unknown context rather than one arbitrary candidate.

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

    def test_canonical_does_not_claim_to_resolve_across_genes(self):
        choice = select_source_mapping(DISAGREEING)
        assert choice.basis == "deterministic_order"
        assert choice.mapping["protein_id"] == "P1"
        assert choice.category == "cross_gene_unresolved"
        assert choice.flank_context_resolved is False

    def test_canonical_resolves_within_one_gene(self):
        candidates = [
            _mapping("P2", "G", "GGG", "TTT"),
            _mapping("P1", "G", "AAA", "CCC", canonical=True),
        ]
        choice = select_source_mapping(candidates)
        assert choice.basis == "canonical_transcript"
        assert choice.mapping["protein_id"] == "P1"
        assert choice.category == "within_gene_canonical"
        assert choice.flank_context_resolved is True

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
        """An irrelevant table degrades to unresolved cross-gene handling."""
        choice = select_source_mapping(DISAGREEING, expression={"GENEZ": 99.0})
        assert choice.basis == "deterministic_order"
        assert choice.category == "cross_gene_unresolved"

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


class TestProductionMappingPolicy:
    """The vectorized production collapse must not invent a source junction."""

    @staticmethod
    def _row(
        evidence_row_id,
        protein,
        gene,
        transcript,
        n_flank,
        c_flank,
        *,
        canonical=False,
        position=10,
    ):
        return {
            "evidence_row_id": evidence_row_id,
            "protein_id": protein,
            "gene_name": gene,
            "gene_id": f"ID_{gene}" if gene else "",
            "transcript_id": transcript,
            "n_flank": n_flank,
            "c_flank": c_flank,
            "position": position,
            "is_canonical_transcript": canonical,
        }

    @classmethod
    def _frame(cls):
        return [
            cls._row("single", "P1", "G1", "T1", "AAA", "CCC", canonical=True),
            cls._row("agree", "P2", "G2", "T2", "AAA", "CCC", canonical=True),
            cls._row("agree", "P3", "G3", "T3", "AAA", "CCC", canonical=True),
            cls._row("within", "P4", "G4", "T4", "AAA", "CCC", canonical=True),
            cls._row("within", "P5", "G4", "T5", "GGG", "TTT"),
            cls._row("cross", "P6", "G6", "T6", "AAA", "CCC", canonical=True),
            cls._row("cross", "P7", "G7", "T7", "GGG", "TTT", canonical=True),
            cls._row("within_unresolved", "P8", "G8", "T8", "AAA", "CCC"),
            cls._row("within_unresolved", "P9", "G8", "T9", "GGG", "TTT"),
            # One canonical transcript occurring twice at different positions
            # remains ambiguous; canonical identity cannot pick the occurrence.
            cls._row("repeat", "P10", "G10", "T10", "AAA", "CCC", canonical=True),
            cls._row("repeat", "P10", "G10", "T10", "GGG", "TTT", canonical=True, position=30),
            cls._row("missing_ids", "", "", "", "AAA", "CCC"),
            cls._row("missing_ids", "", "", "", "GGG", "TTT", position=20),
        ]

    def test_categories_and_unknown_context(self):
        pd = pytest.importorskip("pandas")
        from presto.data.hitlist_source import _collapse_source_mappings

        collapsed, stats = _collapse_source_mappings(pd.DataFrame(self._frame()))
        rows = collapsed.set_index("evidence_row_id")
        assert rows.loc["single", "source_mapping_category"] == "single"
        assert rows.loc["agree", "source_mapping_category"] == "flanks_agree"
        assert rows.loc["within", "source_mapping_category"] == "within_gene_canonical"
        assert rows.loc["within", "protein_id"] == "P4"
        for evidence_id, category in (
            ("cross", "cross_gene_unresolved"),
            ("within_unresolved", "within_gene_unresolved"),
            ("repeat", "within_gene_unresolved"),
            ("missing_ids", "within_gene_unresolved"),
        ):
            assert rows.loc[evidence_id, "source_mapping_category"] == category
            assert rows.loc[evidence_id, "n_flank"] == "?"
            assert rows.loc[evidence_id, "c_flank"] == "?"
            assert bool(rows.loc[evidence_id, "flank_context_resolved"]) is False
            assert rows.loc[evidence_id, "position"] != rows.loc[evidence_id, "position"]

        assert rows.loc["agree", "source_mapping_n_candidates"] == 2
        assert rows.loc["agree", "source_mapping_n_flank_pairs"] == 1
        assert stats["category_counts"]["cross_gene_unresolved"] == 1
        assert stats["rows_with_disagreeing_flanks"] == 5

    def test_frame_order_does_not_change_the_result(self):
        pd = pytest.importorskip("pandas")
        from presto.data.hitlist_source import _collapse_source_mappings

        frame = pd.DataFrame(self._frame())
        forward, _ = _collapse_source_mappings(frame)
        backward, _ = _collapse_source_mappings(frame.iloc[::-1])
        columns = [
            "evidence_row_id",
            "protein_id",
            "n_flank",
            "c_flank",
            "source_mapping_category",
        ]
        forward_rows = forward[columns].sort_values("evidence_row_id").reset_index(drop=True)
        backward_rows = backward[columns].sort_values("evidence_row_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(forward_rows, backward_rows)


class TestUnresolvedFlanksAreDroppedNotSubstituted:
    """`X` disqualifies a row; it never redirects the choice to another protein.

    The tempting rule -- "prefer whichever candidate has a clean flank" -- was
    implemented, measured, and removed. Of the rows it changed, **91%** also
    changed which *gene* the peptide was attributed to: it was not picking a
    better transcript of the same protein, it was swapping the protein. A flank
    is only meaningful if the source protein is right, so that trades a known
    unknown for a possibly-wrong origin.

    Dropping instead costs 862 of 4,418,352 MS evidence rows (0.0195%) and 86 of
    891,685 binding rows, and buys back a single meaning for `X`.
    """

    def test_selection_ignores_flank_cleanliness(self):
        """The whole point: an X candidate does not lose *the selection*."""
        candidates = [
            _mapping("P1", "G1", "XCDEF", "CCCCC"),
            _mapping("P2", "G2", "ACDEF", "CCCCC"),
        ]
        choice = select_source_mapping(candidates)
        assert choice.basis == "deterministic_order"
        assert choice.mapping["protein_id"] == "P1"

    def test_resolved_flank_is_not_a_selection_basis(self):
        from presto.data.flank_selection import SELECTION_BASES

        assert "resolved_flank" not in SELECTION_BASES

    def test_cleanliness_and_cross_gene_canonical_status_do_not_rank(self):
        candidates = [
            _mapping("P9", "G1", "XCDEF", "CCCCC", canonical=True),
            _mapping("P2", "G2", "ACDEF", "CCCCC"),
        ]
        choice = select_source_mapping(candidates)
        assert choice.basis == "deterministic_order"
        assert choice.mapping["protein_id"] == "P2"
        assert choice.flank_context_resolved is False

    def test_expression_is_unaffected(self):
        choice = select_source_mapping(
            [
                _mapping("P1", "G1", "XCDEF", "CCCCC"),
                _mapping("P2", "G2", "ACDEF", "CCCCC"),
            ],
            expression={"G1": 99.0, "G2": 1.0},
        )
        assert choice.basis == "expression"
        assert choice.mapping["protein_id"] == "P1"

    def test_has_unresolved_flank_reads_both_sides(self):
        from presto.data.flank_selection import has_unresolved_flank

        assert has_unresolved_flank(_mapping("P", "G", "XCDEF", "CCCCC"))
        assert has_unresolved_flank(_mapping("P", "G", "ACDEF", "CCCCX"))
        assert not has_unresolved_flank(_mapping("P", "G", "ACDEF", "CCCCC"))


class TestDropUnresolvedFlankRows:
    """The bulk filter, applied after the collapse."""

    @staticmethod
    def _frame():
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(
            [
                {"evidence_row_id": "r1", "n_flank": "XCDEF", "c_flank": "CCCCC"},
                {"evidence_row_id": "r2", "n_flank": "ACDEF", "c_flank": "CCCCC"},
                {"evidence_row_id": "r3", "n_flank": "ACDEF", "c_flank": "CCCCX"},
            ]
        )

    def test_rows_with_an_unresolved_flank_are_dropped(self):
        from presto.data.hitlist_source import drop_unresolved_flank_rows

        kept, dropped = drop_unresolved_flank_rows(self._frame())
        assert dropped == 2
        assert kept["evidence_row_id"].tolist() == ["r2"]

    def test_clean_frames_are_returned_untouched(self):
        pd = pytest.importorskip("pandas")
        from presto.data.hitlist_source import drop_unresolved_flank_rows

        frame = pd.DataFrame([{"evidence_row_id": "r1", "n_flank": "AC", "c_flank": "DE"}])
        kept, dropped = drop_unresolved_flank_rows(frame)
        assert dropped == 0
        assert kept is frame

    def test_missing_flank_columns_are_tolerated(self):
        pd = pytest.importorskip("pandas")
        from presto.data.hitlist_source import drop_unresolved_flank_rows

        frame = pd.DataFrame([{"evidence_row_id": "r1"}])
        kept, dropped = drop_unresolved_flank_rows(frame)
        assert dropped == 0
        assert len(kept) == 1

    def test_null_flanks_are_not_unresolved(self):
        pd = pytest.importorskip("pandas")
        from presto.data.hitlist_source import drop_unresolved_flank_rows

        frame = pd.DataFrame([{"evidence_row_id": "r1", "n_flank": None, "c_flank": None}])
        _, dropped = drop_unresolved_flank_rows(frame)
        assert dropped == 0

    def test_the_collapse_masks_disagreement_instead_of_ranking_cleanliness(self):
        """Neither the X row nor the clean row is asserted as the source."""
        pd = pytest.importorskip("pandas")
        from presto.data.hitlist_source import _select_best_mapping

        frame = pd.DataFrame(
            [
                {
                    "evidence_row_id": "r1",
                    "n_flank": "XCDEF",
                    "c_flank": "CCCCC",
                    "protein_id": "P1",
                    "is_canonical_transcript": False,
                },
                {
                    "evidence_row_id": "r1",
                    "n_flank": "ACDEF",
                    "c_flank": "CCCCC",
                    "protein_id": "P2",
                    "is_canonical_transcript": False,
                },
            ]
        )
        kept = _select_best_mapping(frame)
        assert len(kept) == 1
        assert kept.iloc[0]["n_flank"] == "?"
        assert kept.iloc[0]["source_mapping_category"] == "within_gene_unresolved"
        assert "_unresolved_flank" not in kept.columns


class TestXIsAbsentFromTrainingData:
    """After the filter, `X` is produced by augmentation and nothing else.

    This is the property the filter exists to buy. If a future change lets a
    data-borne `X` back into a flank, the token means two things again and this
    test should fail rather than the ambiguity being rediscovered downstream.
    """

    def test_augmentation_is_the_only_source_of_x(self):
        import inspect

        from presto.data import hitlist_source

        source = inspect.getsource(hitlist_source)
        assert "drop_unresolved_flank_rows(binding_frame)" in source
        assert "drop_unresolved_flank_rows(ms_frame)" in source

    def test_the_unknown_residue_constants_agree(self):
        from presto.data.flank_selection import UNRESOLVED_RESIDUE
        from presto.data.sequence_augmentation import UNKNOWN_RESIDUE

        assert UNRESOLVED_RESIDUE == UNKNOWN_RESIDUE == "X"
