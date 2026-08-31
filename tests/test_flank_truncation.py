"""Flank truncation must keep the residues adjacent to the peptide.

The excision head reads the *last* residue of the N-flank (P1 of the
N-terminal junction) and the *first* residue of the C-flank. Truncation
defaults to keeping the left of a sequence, which is right for a C-flank and
exactly wrong for an N-flank: an over-long N-flank would lose the one residue
it was included for, and the model would score the junction from context that
does not touch it.

This is latent with today's data -- hitlist N-flanks are at most 10 residues
against a 25-residue cap -- so nothing catches it until someone extends flank
extraction or lowers the cap. Hence a test rather than a comment.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.data.tokenizer import Tokenizer  # noqa: E402
from presto.models.presto import Presto  # noqa: E402


@pytest.fixture
def tokenizer():
    return Tokenizer()


class TestTruncationSide:
    def test_left_truncation_keeps_the_tail(self, tokenizer):
        ids = tokenizer.encode("ACDEFGHIKL", max_len=3, truncate="left")
        assert ids == tokenizer.encode("IKL", max_len=3)

    def test_right_truncation_keeps_the_head(self, tokenizer):
        ids = tokenizer.encode("ACDEFGHIKL", max_len=3, truncate="right")
        assert ids == tokenizer.encode("ACD", max_len=3)

    def test_default_is_unchanged(self, tokenizer):
        assert tokenizer.encode("ACDEFGHIKL", max_len=3) == tokenizer.encode(
            "ACDEFGHIKL", max_len=3, truncate="right"
        )

    def test_truncate_side_is_not_confused_by_the_encode_cache(self, tokenizer):
        """Same sequence and length, different side, must not collide."""
        left = tokenizer.encode("ACDEFGHIKL", max_len=3, truncate="left")
        right = tokenizer.encode("ACDEFGHIKL", max_len=3, truncate="right")
        assert left != right

    def test_shorter_than_cap_is_untouched(self, tokenizer):
        assert tokenizer.encode("ACD", max_len=10, truncate="left") == (
            tokenizer.encode("ACD", max_len=10, truncate="right")
        )


class TestCollatorKeepsTheJunction:
    def test_n_flank_p1_residue_survives_truncation(self):
        """The residue the excision head reads must be present after collation."""
        from presto.data.collate import PrestoCollator
        from presto.data.loaders import ElutionRecord, PrestoDataset

        long_n_flank = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAW"  # P1 is the trailing W
        record = ElutionRecord(
            peptide="SIINFEKLA",
            alleles=["HLA-A*02:01"],
            detected=True,
            flank_n=long_n_flank,
            flank_c="CCCC",
        )
        dataset = PrestoDataset(
            elution_records=[record],
            mhc_sequences={"HLA-A*02:01": "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR"},
            strict_mhc_resolution=False,
        )
        collator = PrestoCollator(max_flank_len=8)
        batch = collator([dataset[0]])
        tokenizer = collator.tokenizer
        tokens = [t for t in batch.flank_n_tok[0].tolist() if t != 0]
        assert tokens, "N-flank did not survive collation"
        w_token = tokenizer.encode("W")[0]
        assert tokens[-1] == w_token, (
            "the N-flank's P1 residue was truncated away; the excision head "
            "reads this residue, so the junction is now scored from context "
            "that does not include it"
        )


class TestServingMatchesTraining:
    """Flank handling at serving time must match the collator exactly.

    Two literals had drifted apart. The collator keeps `DEFAULT_MAX_FLANK_LEN`
    (25) residues and truncates the N-flank from the **left**, because the
    N-flank's last residue is P1 of the N-terminal junction and is what the
    excision head reads. The predictor hard-coded 30 and truncated from the
    right, so serving kept five residues training never used and scored a
    different residue at the junction than training ever saw.

    Train/serve skew is silent by construction -- both sides run without error
    and only the numbers disagree -- so it is pinned behaviorally here rather
    than by reading the constants.
    """

    @staticmethod
    def _model():
        torch.manual_seed(0)
        return Presto(d_model=32, n_layers=2, n_heads=4)

    def test_predictor_uses_the_collator_flank_length(self):
        from presto.data.collate import DEFAULT_MAX_FLANK_LEN
        from presto.inference.predictor import Predictor

        predictor = Predictor(self._model(), device="cpu")
        assert predictor._flank_len == DEFAULT_MAX_FLANK_LEN

    def test_n_flank_keeps_the_junction_residue_on_both_paths(self):
        """The residue adjacent to the peptide must survive on both sides."""
        from presto.data.collate import DEFAULT_MAX_FLANK_LEN
        from presto.data.tokenizer import Tokenizer

        tokenizer = Tokenizer()
        # Longer than the cap, with a distinctive final residue: that residue
        # is P1 of the N-terminal junction and must be kept by both paths.
        flank = "A" * (DEFAULT_MAX_FLANK_LEN + 10) + "W"
        encoded = tokenizer.batch_encode(
            [flank], max_len=DEFAULT_MAX_FLANK_LEN, pad=True, truncate="left"
        )
        tryptophan = tokenizer.aa_to_idx["W"]
        assert tryptophan in encoded[0].tolist(), (
            "left truncation dropped the junction residue the excision head reads"
        )

        right = tokenizer.batch_encode(
            [flank], max_len=DEFAULT_MAX_FLANK_LEN, pad=True, truncate="right"
        )
        assert tryptophan not in right[0].tolist(), (
            "fixture is not discriminating: right truncation should lose it"
        )

    def test_predictor_left_truncates_the_n_flank(self):
        """`_flank_len` alone is not enough -- direction is the other half.

        Source-inspected because the encode happens deep inside a batched
        tiling path; the point is that the predictor cannot quietly go back to
        the default right truncation while still passing the length check.
        """
        import inspect

        from presto.inference import predictor as predictor_module

        source = inspect.getsource(predictor_module)
        marker = "flank_n_tok = self.tokenizer.batch_encode("
        assert marker in source, "predictor no longer encodes an N-flank here"
        call = source[source.index(marker) : source.index(marker) + 400]
        assert 'truncate="left"' in call, (
            "the predictor right-truncates the N-flank while the collator "
            "left-truncates it; serving would score a different junction "
            "residue than training"
        )
