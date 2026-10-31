"""Paired-chain functional TCR–pMHC scoring, separate from repertoire recognition.

No trained weights or biological qualification are supplied by this module.
The score describes response in the training assay's scope, not molecular KD.
"""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..data.tokenizer import Tokenizer
from ..models.encoders import SequenceEncoder
from ..legacy.presto import LegacyPresto as Presto


def validate_chain(sequence: str, *, max_length: int = 512) -> str:
    """Validate supplied amino acids without truncating or reconstructing them."""
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("Each paired TCR chain must be a nonempty amino-acid string")
    sequence = sequence.upper()
    if set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError("TCR chains must contain only the 20 canonical amino acids")
    if len(sequence) > max_length:
        raise ValueError(f"TCR chain exceeds max_chain_length={max_length}; no truncation allowed")
    return sequence


class TCRSpecificityModel(nn.Module):
    """Wrap Presto with two full-chain encoders and a pair-specific response head.

    ``freeze_backbone=True`` freezes parameters and dropout on the supplied
    Presto instance. Give this wrapper its own instance if it is used elsewhere.
    Full-length, variable-domain and reconstructed inputs must be distinguished
    by the dataset contract; length alone cannot establish sequence provenance.
    """

    def __init__(
        self,
        backbone: Presto,
        *,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_chain_length: int = 512,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if d_model < 2 or d_model % 2 or n_heads < 1 or d_model % n_heads:
            raise ValueError("d_model must be positive, even, and divisible by n_heads")
        if n_layers < 1 or max_chain_length < 1 or not 0 <= dropout < 1:
            raise ValueError("Require positive layers/chain length and dropout in [0, 1)")
        self.backbone = backbone
        self.config = dict(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_chain_length=max_chain_length,
            freeze_backbone=freeze_backbone,
        )
        self.tokenizer = Tokenizer()
        encoder_args = dict(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_chain_length,
        )
        self.alpha_encoder = SequenceEncoder(**encoder_args)
        self.beta_encoder = SequenceEncoder(**encoder_args)
        self.receptor_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.pmhc_projection = nn.Sequential(
            nn.Linear(backbone.pmhc_interaction_vec_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.response_head = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.set_backbone_frozen(freeze_backbone)

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Explicitly switch between receptor-only and joint fitting."""
        self.config["freeze_backbone"] = bool(frozen)
        self.backbone.requires_grad_(not frozen)
        if frozen:
            # Optimizers inspect .grad, not requires_grad: stale gradients from
            # a preceding joint-training step must not update the frozen model.
            for parameter in self.backbone.parameters():
                parameter.grad = None
        self.backbone.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config["freeze_backbone"]:
            self.backbone.eval()
        return self

    def _tokens(self, chains: Sequence[str]) -> torch.Tensor:
        sequences = [
            validate_chain(chain, max_length=self.config["max_chain_length"]) for chain in chains
        ]
        tokens = self.tokenizer.batch_encode(sequences, max_len=max(map(len, sequences)), pad=True)
        return tokens.to(self.alpha_encoder.embedding.weight.device)

    def forward(
        self, *, tcr_alpha: Sequence[str], tcr_beta: Sequence[str], **pmhc_inputs
    ) -> dict[str, torch.Tensor]:
        """Score aligned TCR/pMHC pairs; pMHC tensors follow ``Presto.forward``.

        The sigmoid output is an uncalibrated model score until qualified on
        held-out measurements. It is not a universal probability of binding.
        """
        if isinstance(tcr_alpha, str) or isinstance(tcr_beta, str):
            raise ValueError("Pass batches of chain strings, not a single string")
        if not tcr_alpha or len(tcr_alpha) != len(tcr_beta):
            raise ValueError("Require equally sized, nonempty paired alpha/beta batches")
        peptide = pmhc_inputs.get("pep_tok")
        if peptide is None or peptide.ndim != 2 or peptide.shape[0] != len(tcr_alpha):
            raise ValueError("pep_tok batch must match the paired TCR batch")
        alpha_tokens, beta_tokens = self._tokens(tcr_alpha), self._tokens(tcr_beta)
        with torch.no_grad() if self.config["freeze_backbone"] else nullcontext():
            pmhc = self.backbone(**pmhc_inputs)["pmhc_interaction_vec"]
        alpha, _ = self.alpha_encoder(alpha_tokens)
        beta, _ = self.beta_encoder(beta_tokens)
        receptor = self.receptor_projection(torch.cat((alpha, beta), dim=-1))
        presented = self.pmhc_projection(pmhc)
        joined = torch.cat(
            (receptor, presented, receptor * presented, (receptor - presented).abs()), dim=-1
        )
        logit = self.response_head(joined).squeeze(-1)
        return {
            "tcr_functional_logit": logit,
            "tcr_functional_score": torch.sigmoid(logit),
            "tcr_vec": receptor,
            "pmhc_interaction_vec": pmhc,
        }

    def save_checkpoint(self, path: str | Path, *, metadata: dict | None = None) -> None:
        """Save architecture, both model states and caller's data/assay contract."""
        from ..training.checkpointing import build_model_config

        base = build_model_config(self.backbone)
        # Preserve optional architecture flags as well as the canonical subset.
        for name in inspect.signature(Presto).parameters:
            if name not in base:
                base[name] = getattr(self.backbone, name)
        torch.save(
            {
                "format": "presto-tcr-specificity-v1",
                "backbone_config": base,
                "wrapper_config": self.config,
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, map_location="cpu"):
        """Load this wrapper format; canonical Presto checkpoints stay separate."""
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get("format") != "presto-tcr-specificity-v1":
            raise ValueError("Not a Presto TCR specificity checkpoint")
        model = cls(Presto(**payload["backbone_config"]), **payload["wrapper_config"])
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(map_location).eval()
        return model, payload["metadata"]


def functional_response_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE for admitted, measured functional outcomes only; no implicit negatives."""
    if logits.ndim != 1 or logits.numel() == 0 or logits.shape != targets.shape:
        raise ValueError("Require nonempty, equally sized one-dimensional logits and targets")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("Unmeasured/nonfinite outcomes must be excluded before loss computation")
    if not ((targets == 0) | (targets == 1)).all():
        raise ValueError("Targets must be measured binary functional outcomes")
    return F.binary_cross_entropy_with_logits(logits, targets.to(logits))
