"""Sequence encoders for Presto.

Simple transformer-based sequence encoder with pooling options.
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.vocab import AA_VOCAB


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2 normalize along specified dimension."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnablePE(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pe(positions)
        return self.dropout(x)


class SequenceEncoder(nn.Module):
    """Transformer encoder for amino acid sequences.

    Encodes variable-length sequences to fixed-size representations.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
        pool: str = "mean",
        max_len: int = 512,
        vocab_size: int = None,
    ):
        """
        Args:
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension (default: 4 * d_model)
            dropout: Dropout rate
            pool: Pooling mode - "mean", "cls", or "first"
            max_len: Maximum sequence length
            vocab_size: Vocabulary size (default: len(AA_VOCAB))
        """
        super().__init__()
        self.d_model = d_model
        self.pool = pool

        vocab_size = vocab_size or len(AA_VOCAB)
        d_ff = d_ff or 4 * d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPE(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,  # keep norm_first without nested tensor warnings
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequences.

        Args:
            x: Token IDs (batch, seq_len)
            mask: Attention mask (batch, seq_len), 1 = attend, 0 = ignore
            lengths: Sequence lengths (batch,) - alternative to mask

        Returns:
            pooled: Pooled representation (batch, d_model)
            full: Full sequence representation (batch, seq_len, d_model)
        """
        B, L = x.shape

        # Build mask from lengths if provided
        if mask is None and lengths is not None:
            mask = torch.arange(L, device=x.device).expand(B, L) < lengths.unsqueeze(1)
            mask = mask.float()
        elif mask is None:
            # Infer from padding (idx=0)
            mask = (x != 0).float()

        # Check for empty sequences (all padding)
        seq_lengths = mask.sum(dim=1)  # (B,)
        has_content = seq_lengths > 0

        # Embed and add position
        h = self.embedding(x)  # (B, L, d_model)
        h = self.pos_enc(h)

        # Create attention mask for transformer (True = ignore)
        # For empty sequences, set first position to attend to avoid NaN
        attn_mask = (mask == 0)
        # Fix: ensure at least one position is attended to per sequence
        attn_mask[:, 0] = attn_mask[:, 0] & has_content

        # Transform
        h = self.transformer(h, src_key_padding_mask=attn_mask)
        h = self.norm(h)

        # Pool
        if self.pool == "cls" or self.pool == "first":
            pooled = h[:, 0]
        else:  # mean
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
            h_masked = h * mask_expanded
            pooled = h_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # Zero out pooled representation for empty sequences
        pooled = pooled * has_content.unsqueeze(-1).float()

        return pooled, h


class ProjectionHead(nn.Module):
    """MLP projection head with L2 normalization.

    Used for contrastive learning objectives.
    """

    def __init__(self, d_in: int, d_out: int, hidden: int = None):
        super().__init__()
        hidden = hidden or d_in * 2
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2 normalize.

        Args:
            x: (batch, d_in)
        Returns:
            (batch, d_out), L2 normalized
        """
        return l2_normalize(self.net(x))
