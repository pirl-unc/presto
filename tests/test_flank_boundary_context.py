"""A protein terminus is not a missing flank.

Both arrive as a short or empty flank string, and until now both encoded as
`?` -- a token that means "we do not know what is here". For a peptide
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
pd = pytest.importorskip("pandas")

from presto.data.hitlist_source import (  # noqa: E402
    HITLIST_FLANK_WIDTH,
    PROTEIN_MAPPING_COLUMNS,
    flank_context,
)
from presto.data.vocab import AA_TO_IDX, AA_VOCAB  # noqa: E402
from presto.models.presto import Presto  # noqa: E402

UNKNOWN_FLANK = AA_TO_IDX["?"]
BOUNDARY = AA_TO_IDX["X"]


class TestTheFlankAlphabetIsDistinct:
    """Two markers. `X` is a residue code that doubles as the boundary pad."""

    def test_boundary_and_unknown_are_different_tokens(self):
        assert BOUNDARY != UNKNOWN_FLANK

    def test_the_boundary_pad_is_x(self):
        """Not a new symbol: 4.33% of mapped flanks reaching position 0 already
        start with X, an unresolved initiator. "Ran out of protein" and
        "unresolved residue at the protein edge" are the same situation."""
        assert AA_VOCAB[BOUNDARY] == "X"

    def test_one_boundary_symbol_suffices(self):
        """Start and end need no separate symbols. The N and C junctions are
        scored by different tensors, each with a position axis, so which
        boundary was hit is already encoded by which parameter is indexed."""
        from presto.models.presto import Presto

        head = Presto(d_model=32, n_layers=2, n_heads=4).excision_head
        assert head.invivo_profile_n.shape == head.invivo_profile_c.shape
        assert head.invivo_profile_n.data_ptr() != head.invivo_profile_c.data_ptr()

    def test_markers_were_appended(self):
        """Existing residue indices must not shift; checkpoints index by
        position."""
        assert AA_VOCAB[-1] == "?"
        assert AA_VOCAB[AA_TO_IDX["X"]] == "X"

    def test_markers_are_not_encodable_residues(self):
        """They must never come from tokenizing a real sequence -- only from
        the window extractor describing an absence."""
        from presto.data.vocab import ENCODABLE_RESIDUES, FLANK_MARKERS

        assert not (FLANK_MARKERS & ENCODABLE_RESIDUES)
        assert "X" in ENCODABLE_RESIDUES, "X is a residue, not a marker"

    def test_every_marker_is_learnable(self):
        """None of these are pinned. `X` used to be held at a fixed zero vector
        with a gradient hook, on the reasoning that an ambiguous residue should
        contribute nothing -- but in this corpus X is not noise, it is the
        unresolved initiator residue of a reference protein in all 45,992
        occurrences, and that is a specific context worth representing."""
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        for symbol in ("X", "?"):
            vector = model.aa_embedding.weight[AA_TO_IDX[symbol]]
            assert vector.requires_grad
            assert float(vector.norm()) > 0.0, f"{symbol} is pinned to zero"


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
        """No mapping means no knowledge, which is exactly `?`."""
        _, terminus = flank_context("ACD", position)
        assert terminus is False

    @pytest.mark.parametrize("value", [None, float("nan"), pd.NA])
    def test_null_flank_is_not_a_terminus_with_a_mapping_position(self, value):
        text, terminus = flank_context(value, 10.0)
        assert text == ""
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
        """Boundary rows pad with X, the rest with ?."""
        model = self._model_with_trained_profiles()
        pads = model._pad_for_side(
            torch.tensor([True, False, True]),
            3,
            torch.device("cpu"),
            boundary_token=model.boundary_token_idx,
        )
        assert pads.tolist() == [BOUNDARY, UNKNOWN_FLANK, BOUNDARY]

    def test_a_missing_flag_falls_back_to_missing(self):
        model = self._model_with_trained_profiles()
        pads = model._pad_for_side(None, 2, torch.device("cpu"))
        assert pads.tolist() == [UNKNOWN_FLANK, UNKNOWN_FLANK]


class TestTheThreeStatesReachTheModel:
    """One batch, three provenances, three different windows.

    The distinction is only worth having if it survives to the tensor the
    excision head reads. These are the cases a pLM handling optional context
    around a protein has to get right, and they have to be right *per row*:
    a batch mixes peptides whose source proteins were resolved, peptides
    sitting at a protein terminus, and peptides never mapped at all.
    """

    @staticmethod
    def _window(flank_toks, pads, width=5):
        return Presto._first_valid_window(
            flank_toks, len(pads), torch.device("cpu"), torch.tensor(pads), width
        )

    def test_a_mixed_batch_pads_each_row_by_its_own_provenance(self):
        from presto.data.collate import PrestoCollator
        from presto.data.loaders import PrestoSample

        batch = PrestoCollator()(
            [
                # resolved: real residues, X among them meaning "a residue is
                # here, identity unresolved"
                PrestoSample(peptide="SIINFEKL", flank_n="MXKLL"),
                # protein terminus: nothing upstream exists
                PrestoSample(peptide="SIINFEKL", flank_n=None, flank_n_is_terminus=True),
                # never mapped: nothing is known
                PrestoSample(peptide="SIINFEKL", flank_n=None),
            ]
        )
        assert batch.flank_n_is_terminus.tolist() == [False, True, False]
        pads = [
            BOUNDARY if terminus else UNKNOWN_FLANK
            for terminus in batch.flank_n_is_terminus.tolist()
        ]
        window = self._window(batch.flank_n_tok, pads)
        assert [AA_VOCAB[t] for t in window[0].tolist()] == ["M", "X", "K", "L", "L"]
        assert set(window[1].tolist()) == {BOUNDARY}
        assert set(window[2].tolist()) == {UNKNOWN_FLANK}

    def test_an_absent_flank_tensor_still_pads_per_row(self):
        """Every row masked: the collator omits the tensor entirely."""
        window = self._window(None, [UNKNOWN_FLANK, BOUNDARY])
        assert set(window[0].tolist()) == {UNKNOWN_FLANK}
        assert set(window[1].tolist()) == {BOUNDARY}

    def test_no_flank_string_carries_the_unknown_marker(self):
        """`?` is a pad token, never a character in a sequence.

        It used to be both, and the two meanings were told apart by string
        length -- a one-character `"?"` meant the whole flank was unknown,
        while `"?"` inside a longer flank was not representable at all. The
        sequence validator has to keep rejecting it.
        """
        from presto.data.vocab import is_encodable_sequence

        assert not is_encodable_sequence("?")
        assert not is_encodable_sequence("????")
        assert is_encodable_sequence("MXKLL"), "X is a residue and must encode"


class TestTheTerminusFlagSurvivesTheRecordToSampleHop:
    """The flags were declared, collated, and read -- but never populated.

    `BindingRecord`, `ProcessingRecord` and `ElutionRecord` all carried
    `flank_n_is_terminus` / `flank_c_is_terminus`, `PrestoBatch` moved them to
    the device, and `Presto._pad_for_side` branched on them. What was missing
    was the one hop in between: no `PrestoSample(...)` construction in
    `data/loaders.py` passed them along, so every row arrived `False` and the
    `X` boundary pad was unreachable in training. A terminus is only a terminus
    if it gets there.
    """

    @staticmethod
    def _sample(**flags):
        from presto.data.loaders import BindingRecord, PrestoDataset

        record = BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=25.0,
            flank_n="MKL",
            flank_c="PQR",
            **flags,
        )
        dataset = PrestoDataset(binding_records=[record], strict_mhc_resolution=False)
        return dataset[0]

    def test_a_terminus_record_yields_a_terminus_sample(self):
        sample = self._sample(flank_n_is_terminus=True, flank_c_is_terminus=True)
        assert sample.flank_n_is_terminus is True
        assert sample.flank_c_is_terminus is True

    def test_the_two_sides_are_carried_independently(self):
        """A peptide at the protein's start is not also at its end."""
        sample = self._sample(flank_n_is_terminus=True, flank_c_is_terminus=False)
        assert sample.flank_n_is_terminus is True
        assert sample.flank_c_is_terminus is False

    def test_an_unmapped_record_claims_no_terminus(self):
        sample = self._sample()
        assert sample.flank_n_is_terminus is False
        assert sample.flank_c_is_terminus is False
