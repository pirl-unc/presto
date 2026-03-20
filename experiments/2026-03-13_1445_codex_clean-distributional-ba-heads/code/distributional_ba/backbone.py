"""Self-contained backbone for the clean distributional BA experiment."""

from __future__ import annotations

import torch
import torch.nn as nn


class FixedBackbone(nn.Module):
    """Shared 3-segment transformer encoder returning pep+groove features.

    This is intentionally local to the experiment package so the benchmark
    contract does not depend on mutable code in other experiment scripts.
    """

    def __init__(
        self,
        vocab_size: int = 26,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        max_seq_len: int = 200,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.out_dim = 3 * int(embed_dim)
        self.aa_embedding = nn.Embedding(int(vocab_size), int(embed_dim))
        self.pos_embedding = nn.Embedding(int(max_seq_len), int(embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(n_heads),
            dim_feedforward=int(ff_dim),
            batch_first=True,
            activation="gelu",
            dropout=float(dropout),
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))

    def _encode_segment(self, tok: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tok.shape
        positions = torch.arange(seq_len, device=tok.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.aa_embedding(tok) + self.pos_embedding(positions)
        pad_mask = tok == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        non_pad = (~pad_mask).float().unsqueeze(-1)
        return (x * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
    ) -> torch.Tensor:
        pep_vec = self._encode_segment(pep_tok)
        gh1_vec = self._encode_segment(mhc_a_tok)
        gh2_vec = self._encode_segment(mhc_b_tok)
        return torch.cat([pep_vec, gh1_vec, gh2_vec], dim=-1)
