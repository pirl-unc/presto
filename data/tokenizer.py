"""Tokenizer for amino acid sequences."""

from typing import Dict, List, Literal, Tuple, Union
import torch

from .vocab import AA_VOCAB, AA_TO_IDX


class Tokenizer:
    """Tokenizer for amino acid sequences.

    Handles encoding sequences to token IDs and decoding back to strings.
    Supports special tokens (PAD, UNK, BOS, EOS) and batch operations.
    """

    def __init__(self, unknown_policy: Literal["error", "unk"] = "error"):
        self.vocab = AA_VOCAB
        self.aa_to_idx = AA_TO_IDX
        self.idx_to_aa = {i: aa for i, aa in enumerate(AA_VOCAB)}
        if unknown_policy not in {"error", "unk"}:
            raise ValueError("unknown_policy must be 'error' or 'unk'")
        self.unknown_policy = str(unknown_policy)

        self.pad_idx = self.aa_to_idx["<PAD>"]
        self.unk_idx = self.aa_to_idx["<UNK>"]
        self.bos_idx = self.aa_to_idx["<BOS>"]
        self.eos_idx = self.aa_to_idx["<EOS>"]
        self.missing_idx = self.aa_to_idx["<MISSING>"]

        self._special_tokens = {"<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MISSING>"}
        self._valid_sequence_tokens = set(self.vocab) - self._special_tokens

        # Encode cache: (seq, max_len, add_bos, add_eos) → tuple of token IDs.
        # Eliminates repeated per-character Python loops for the same sequence.
        # MHC sequences (~365 chars each) dominate tokenization cost; with ~53K
        # unique alleles the cache uses ~170 MB and makes epochs 2+ near-instant.
        self._encode_cache: Dict[tuple, tuple] = {}

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def encode(
        self,
        seq: str,
        max_len: int = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode a sequence to token IDs.

        Args:
            seq: Amino acid sequence string
            max_len: Maximum length (truncates if longer). If add_bos/add_eos,
                     sequence is truncated to fit within max_len including specials.
            add_bos: Add BOS token at start
            add_eos: Add EOS token at end

        Returns:
            List of token IDs
        """
        seq = str(seq).upper()

        # Check encode cache first.
        cache_key = (seq, max_len, add_bos, add_eos)
        cached = self._encode_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        # Encode sequence with explicit unknown handling.
        ids: List[int] = []
        for idx, aa in enumerate(seq):
            token_id = self.aa_to_idx.get(aa)
            if token_id is None:
                if self.unknown_policy == "unk":
                    ids.append(self.unk_idx)
                    continue
                raise ValueError(
                    f"Invalid amino-acid token '{aa}' at position {idx} "
                    f"in sequence '{seq}'."
                )
            if aa not in self._valid_sequence_tokens:
                if self.unknown_policy == "unk":
                    ids.append(self.unk_idx)
                    continue
                raise ValueError(
                    f"Invalid sequence token '{aa}' at position {idx} "
                    f"in sequence '{seq}'."
                )
            ids.append(token_id)

        # Handle max_len with special tokens
        if max_len is not None:
            # Calculate how much space we need for special tokens
            n_special = (1 if add_bos else 0) + (1 if add_eos else 0)
            max_seq_len = max_len - n_special
            if max_seq_len < 0:
                max_seq_len = 0
            ids = ids[:max_seq_len]

        # Add special tokens
        if add_bos:
            ids = [self.bos_idx] + ids
        if add_eos:
            ids = ids + [self.eos_idx]

        self._encode_cache[cache_key] = tuple(ids)
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        """Decode token IDs back to sequence string.

        Args:
            ids: List of token IDs
            strip_special: If True, remove PAD/BOS/EOS tokens from output

        Returns:
            Decoded sequence string
        """
        if strip_special:
            tokens = [
                self.idx_to_aa.get(i, "<UNK>")
                for i in ids
                if self.idx_to_aa.get(i, "<UNK>") not in self._special_tokens
            ]
        else:
            tokens = [self.idx_to_aa.get(i, "<UNK>") for i in ids]

        return "".join(tokens)

    def batch_encode(
        self,
        seqs: List[str],
        max_len: int,
        pad: bool = True,
        add_bos: bool = False,
        add_eos: bool = False,
        return_lengths: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Batch encode multiple sequences.

        Args:
            seqs: List of sequences
            max_len: Maximum length (all sequences padded/truncated to this)
            pad: If True, pad shorter sequences
            add_bos: Add BOS token at start
            add_eos: Add EOS token at end
            return_lengths: If True, also return original lengths

        Returns:
            Tensor of shape (batch_size, max_len)
            If return_lengths, also returns lengths tensor of shape (batch_size,)
        """
        if len(seqs) == 0:
            batch = torch.zeros((0, max_len), dtype=torch.long)
            if return_lengths:
                return batch, torch.zeros(0, dtype=torch.long)
            return batch

        encoded = []
        lengths = []

        for seq in seqs:
            ids = self.encode(seq, max_len=max_len, add_bos=add_bos, add_eos=add_eos)
            lengths.append(len(ids))

            if pad and len(ids) < max_len:
                ids = ids + [self.pad_idx] * (max_len - len(ids))

            encoded.append(ids)

        batch = torch.tensor(encoded, dtype=torch.long)

        if return_lengths:
            return batch, torch.tensor(lengths, dtype=torch.long)
        return batch
