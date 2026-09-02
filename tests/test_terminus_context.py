"""A protein terminus is not a missing flank.

Both arrive as a short or empty flank string, and until now both encoded as
`<MISSING>` -- a token that means "we do not know what is here". For a peptide
at its protein's own terminus that is the opposite of true: there is nothing
upstream (or downstream), and that absence is a fact about the biology.

The C side is the sharper case. A peptide ending at the protein's C-terminus
**required no proteasomal C-terminal cut at all** -- the terminus already
existed. The excision head was being asked to score a cleavage event that never
had to happen, on 8.0% of rows, using a token asserting the context was unknown.

Measured on hitlist 1.55.2 (24,466,149 MS rows): 5.12% of rows carry an N-flank
shorter than the 15 residues hitlist extracts, 8.00% a short C-flank, and
**86.4%** of the short N-flanks sit at `position < 15` -- genuinely the protein's
start rather than an unmapped peptide.

`position` is what separates the two, which is why it is now part of the column
contract.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.data.hitlist_source import (  # noqa: E402
    HITLIST_FLANK_WIDTH,
    PROTEIN_MAPPING_COLUMNS,
    flank_context,
)
from presto.data.vocab import AA_TO_IDX, AA_VOCAB  # noqa: E402
from presto.models.presto import Presto  # noqa: E402

MISSING = AA_TO_IDX["<MISSING>"]
TERMINUS = AA_TO_IDX["<TERMINUS>"]


class TestTheTokensAreDistinct:
    def test_terminus_is_its_own_token(self):
        assert TERMINUS != MISSING
        assert AA_VOCAB[TERMINUS] == "<TERMINUS>"

    def test_terminus_was_appended(self):
        """Existing residue indices must not shift; checkpoints index by
        position."""
        assert AA_VOCAB[-1] == "<TERMINUS>"
        assert AA_VOCAB[MISSING] == "<MISSING>"

    def test_terminus_is_not_an_encodable_residue(self):
        """It must never be produced by tokenizing a sequence -- only by the
        window extractor deciding a flank ran out of protein."""
        from presto.data.vocab import ENCODABLE_RESIDUES

        assert "<TERMINUS>" not in ENCODABLE_RESIDUES


class TestFlankContext:
    """`position` present means mapped; a short flank then means the protein
    ended."""

    def test_position_is_in_the_column_contract(self):
        assert "position" in PROTEIN_MAPPING_COLUMNS

    def test_mapped_and_short_is_a_terminus(self):
        text, terminus = flank_context("ACD", 3.0)
        assert text == "ACD" and terminus is True

    def test_mapped_and_full_length_is_not(self):
        full = "A" * HITLIST_FLANK_WIDTH
        text, terminus = flank_context(full, 100.0)
        assert text == full and terminus is False

    @pytest.mark.parametrize("position", [None, float("nan")])
    def test_unmapped_is_never_a_terminus(self, position):
        """No mapping means no knowledge, which is exactly `<MISSING>`."""
        _, terminus = flank_context("ACD", position)
        assert terminus is False

    def test_an_unencodable_flank_is_not_mistaken_for_a_terminus(self):
        """The subtle one.

        `drop_unencodable_sequence` blanks a flank carrying selenocysteine or
        annotation junk. Measuring the terminus on the *cleaned* text would
        then read a full-length flank as an empty one and invent a protein
        terminus out of a tokenizer limitation. The raw length is what counts.
        """
        raw = "AUCDEFGHIKLMNPQ"  # 15 residues, U is unencodable
        assert len(raw) == HITLIST_FLANK_WIDTH
        text, terminus = flank_context(raw, 100.0)
        assert text == "", "expected the unencodable flank to be blanked"
        assert terminus is False, "a blanked flank is unknown, not a terminus"


class TestTheModelSeesTheDifference:
    @staticmethod
    def _model_with_trained_profiles():
        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        # The profiles are zero-initialized, so at init every residue index
        # scores identically and this distinction is invisible by construction.
        with torch.no_grad():
            for tensor in (
                model.excision_head.invivo_profile_c,
                model.excision_head.invivo_profile_n,
                model.excision_head.stimulus_profile_c,
            ):
                tensor.normal_(0.0, 0.5)
        model.eval()
        return model

    @staticmethod
    def _inputs(batch=2):
        return dict(
            pep_tok=torch.randint(4, 24, (batch, 10)),
            mhc_a_tok=torch.randint(4, 24, (batch, 40)),
            mhc_b_tok=torch.randint(4, 24, (batch, 40)),
            mhc_class="I",
            # Empty flanks: the case where terminus and missing are otherwise
            # indistinguishable.
            flank_n_tok=torch.zeros(batch, 8, dtype=torch.long),
            flank_c_tok=torch.zeros(batch, 8, dtype=torch.long),
            provenance={"peptide_source_idx": torch.full((batch,), 1, dtype=torch.long)},
        )

    def test_a_terminus_scores_differently_from_a_missing_flank(self):
        model = self._model_with_trained_profiles()
        inputs = self._inputs()
        with torch.no_grad():
            unknown = model(**inputs)
            terminus = model(
                **inputs,
                flank_n_is_terminus=torch.tensor([False, True]),
                flank_c_is_terminus=torch.tensor([False, True]),
            )
        assert torch.allclose(unknown["excision_logit"][0], terminus["excision_logit"][0]), (
            "the row with no terminus flag should be untouched"
        )
        assert not torch.allclose(unknown["excision_logit"][1], terminus["excision_logit"][1]), (
            "the terminus row scored identically to an unknown flank, so the "
            "distinction is not reaching the excision head"
        )

    def test_omitting_the_flags_keeps_the_previous_behaviour(self):
        """A caller that knows nothing about termini must be unaffected."""
        model = self._model_with_trained_profiles()
        inputs = self._inputs()
        with torch.no_grad():
            without = model(**inputs)
            explicit_false = model(
                **inputs,
                flank_n_is_terminus=torch.tensor([False, False]),
                flank_c_is_terminus=torch.tensor([False, False]),
            )
        assert torch.allclose(without["excision_logit"], explicit_false["excision_logit"])

    def test_the_pad_helper_selects_per_row(self):
        model = self._model_with_trained_profiles()
        pads = model._pad_for_side(torch.tensor([True, False, True]), 3, torch.device("cpu"))
        assert pads.tolist() == [TERMINUS, MISSING, TERMINUS]

    def test_a_missing_flag_falls_back_to_missing(self):
        model = self._model_with_trained_profiles()
        pads = model._pad_for_side(None, 2, torch.device("cpu"))
        assert pads.tolist() == [MISSING, MISSING]
