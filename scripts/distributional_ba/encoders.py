"""Encoder backends for shared distributional BA experiments."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from presto.scripts.groove_baseline_probe import GrooveTransformerModel


ENCODER_BACKBONES: Tuple[str, ...] = (
    "historical_ablation",
    "groove",
)


class HistoricalAblationEncoder(nn.Module):
    """Historical EXP-16 backbone.

    This is the pre-`62e3e53` ablation encoder that powered the original
    shared-path v1-v6 BA experiments. It runs an independent transformer over
    each segment and concatenates the pooled peptide / groove-half outputs.
    """

    def __init__(
        self,
        vocab_size: int = 26,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        max_seq_len: int = 200,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = 3 * embed_dim
        self.aa_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def _encode_segment(self, tok: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tok.shape
        positions = torch.arange(seq_len, device=tok.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.aa_embedding(tok) + self.pos_embedding(positions)
        pad_mask = tok == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        non_pad = (~pad_mask).float().unsqueeze(-1)
        return (x * non_pad).sum(1) / non_pad.sum(1).clamp(min=1)

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


def build_encoder(
    *,
    encoder_backbone: str,
    embed_dim: int,
    n_heads: int,
    n_layers: int,
) -> nn.Module:
    """Instantiate one of the shared encoder backends."""
    if encoder_backbone == "historical_ablation":
        return HistoricalAblationEncoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=embed_dim,
        )
    if encoder_backbone == "groove":
        return GrooveTransformerModel(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=embed_dim,
            hidden_dim=embed_dim,
        )
    raise ValueError(
        f"Unknown encoder_backbone {encoder_backbone!r}. Expected one of {ENCODER_BACKBONES}.",
    )
