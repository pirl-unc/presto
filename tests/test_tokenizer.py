"""Tests for tokenizer module - pins down tokenization API."""

import pytest
import torch


class TestTokenizer:
    """Test Tokenizer class."""

    def test_tokenizer_init(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        assert tok.pad_idx == 0
        assert tok.unk_idx > 0
        assert tok.bos_idx > 0
        assert tok.eos_idx > 0
        assert tok.missing_idx > 0

    def test_encode_simple_sequence(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        seq = "ACDE"
        ids = tok.encode(seq)
        assert isinstance(ids, list)
        assert len(ids) == 4
        # Should not include BOS/EOS by default
        assert ids[0] != tok.bos_idx
        assert ids[-1] != tok.eos_idx

    def test_encode_with_bos_eos(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        seq = "ACDE"
        ids = tok.encode(seq, add_bos=True, add_eos=True)
        assert len(ids) == 6  # BOS + 4 AA + EOS
        assert ids[0] == tok.bos_idx
        assert ids[-1] == tok.eos_idx

    def test_encode_with_max_len_truncates(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 AAs
        ids = tok.encode(seq, max_len=10)
        assert len(ids) == 10

    def test_encode_with_max_len_and_eos_truncates_before_eos(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 AAs
        ids = tok.encode(seq, max_len=10, add_eos=True)
        assert len(ids) == 10
        assert ids[-1] == tok.eos_idx  # EOS is preserved

    def test_encode_unknown_char_maps_to_unk(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        seq = "AC*DE"  # * is not a valid AA
        with pytest.raises(ValueError, match="Invalid amino-acid token"):
            tok.encode(seq)

    def test_encode_unknown_char_maps_to_unk_in_legacy_mode(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer(unknown_policy="unk")
        seq = "AC*DE"
        ids = tok.encode(seq)
        assert tok.unk_idx in ids

    def test_decode_simple_sequence(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        seq = "ACDE"
        ids = tok.encode(seq)
        decoded = tok.decode(ids)
        assert decoded == seq

    def test_decode_strips_special_tokens(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        seq = "ACDE"
        ids = tok.encode(seq, add_bos=True, add_eos=True)
        decoded = tok.decode(ids, strip_special=True)
        assert decoded == seq

    def test_decode_keeps_special_tokens(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        seq = "ACDE"
        ids = tok.encode(seq, add_bos=True, add_eos=True)
        decoded = tok.decode(ids, strip_special=False)
        assert "<BOS>" in decoded
        assert "<EOS>" in decoded

    def test_batch_encode(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        seqs = ["ACDE", "FGH", "IKLMNPQ"]
        batch = tok.batch_encode(seqs, max_len=10, pad=True)
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (3, 10)
        # Shorter seqs should be padded
        assert batch[1, 3:].tolist() == [tok.pad_idx] * 7

    def test_batch_encode_returns_lengths(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        seqs = ["ACDE", "FGH", "IKLMNPQ"]
        batch, lengths = tok.batch_encode(seqs, max_len=10, pad=True, return_lengths=True)
        assert lengths.tolist() == [4, 3, 7]

    def test_batch_encode_with_bos_eos(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        seqs = ["ACDE", "FGH"]
        batch, lengths = tok.batch_encode(seqs, max_len=10, pad=True, add_bos=True, add_eos=True, return_lengths=True)
        assert lengths.tolist() == [6, 5]  # BOS + seq + EOS
        assert batch[0, 0] == tok.bos_idx
        assert batch[0, 5] == tok.eos_idx


class TestTokenizerEdgeCases:
    """Test edge cases and error handling."""

    def test_encode_empty_string(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        ids = tok.encode("")
        assert ids == []

    def test_encode_empty_with_bos_eos(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        ids = tok.encode("", add_bos=True, add_eos=True)
        assert ids == [tok.bos_idx, tok.eos_idx]

    def test_decode_empty_list(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        decoded = tok.decode([])
        assert decoded == ""

    def test_lowercase_input_converted(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        ids_upper = tok.encode("ACDE")
        ids_lower = tok.encode("acde")
        assert ids_upper == ids_lower

    def test_batch_encode_empty_list(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        batch = tok.batch_encode([], max_len=10, pad=True)
        assert batch.shape == (0, 10)


class TestTokenizerConsistency:
    """Test tokenizer consistency properties."""

    def test_encode_decode_roundtrip(self):
        from presto.data.tokenizer import Tokenizer
        tok = Tokenizer()
        seqs = ["ACDEFGHIKLMNPQRSTVWY", "PEPTIDE", "MHCALLELE"]
        for seq in seqs:
            ids = tok.encode(seq)
            decoded = tok.decode(ids)
            assert decoded == seq

    def test_vocab_size_matches(self):
        from presto.data.tokenizer import Tokenizer
        from presto.data.vocab import AA_VOCAB
        tok = Tokenizer()
        assert tok.vocab_size == len(AA_VOCAB)
