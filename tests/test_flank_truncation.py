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
