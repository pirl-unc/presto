"""TCR (T-cell receptor) modules for Presto.

Handles:
- TCR encoding (alpha/beta or gamma/delta chains)
- Chain type classification (TRA, TRB, TRG, TRD, IGH, IGK, IGL)
- Cell type classification (CD4_T, CD8_T, ab_T, gd_T, B_cell)
- TCR-pMHC matching with biological constraints
- Repertoire-level recognition prediction
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.vocab import (
    CHAIN_TYPES,
    CELL_TYPES,
    MHC_TYPES,
    SPECIES,
    CELL_MHC_COMPATIBILITY,
    CELL_TO_IDX,
    MHC_TO_IDX,
)
from .encoders import SequenceEncoder, l2_normalize


def get_compatibility_mask() -> torch.Tensor:
    """Get cell-MHC compatibility mask.

    Returns:
        Tensor of shape (n_cell_types, n_mhc_types) with 1.0 for valid pairs
    """
    mask = torch.zeros(len(CELL_TYPES), len(MHC_TYPES))
    for cell_type, compatible_mhc in CELL_MHC_COMPATIBILITY.items():
        cell_idx = CELL_TO_IDX[cell_type]
        for mhc_type in compatible_mhc:
            mhc_idx = MHC_TO_IDX[mhc_type]
            mask[cell_idx, mhc_idx] = 1.0
    return mask


def info_nce_loss(
    anchor_vec: torch.Tensor,
    positive_vec: torch.Tensor,
    negatives: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE contrastive loss.

    Args:
        anchor_vec: Anchor embeddings (batch, d), L2 normalized
        positive_vec: Positive embeddings (batch, d), L2 normalized
        negatives: Extra negatives (n_neg, d), L2 normalized
        temperature: Softmax temperature

    Returns:
        Scalar loss
    """
    B = anchor_vec.size(0)
    device = anchor_vec.device

    # Positive similarity
    sim_pos = (anchor_vec * positive_vec).sum(dim=-1, keepdim=True) / temperature

    # In-batch negatives: all other samples in batch
    sim_neg = anchor_vec @ positive_vec.T / temperature  # (B, B)

    if negatives is not None and negatives.size(0) > 0:
        # Add extra negatives
        sim_extra = anchor_vec @ negatives.T / temperature  # (B, n_neg)
        logits = torch.cat([sim_pos, sim_neg, sim_extra], dim=1)
    else:
        logits = torch.cat([sim_pos, sim_neg], dim=1)

    # Labels: positive is always first column
    labels = torch.zeros(B, dtype=torch.long, device=device)

    return F.cross_entropy(logits, labels)


class TCREncoder(nn.Module):
    """Encodes TCR chains into fixed-size vector representation.

    Can handle paired (alpha/beta or gamma/delta) or single chains.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.chain_encoder = SequenceEncoder(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads
        )
        # Fusion for paired chains
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def encode_chain(self, chain_tok: torch.Tensor) -> torch.Tensor:
        """Encode a single chain.

        Args:
            chain_tok: Chain tokens (batch, seq_len)

        Returns:
            Chain embedding (batch, d_model)
        """
        z, _ = self.chain_encoder(chain_tok)
        return z

    def forward(
        self,
        chain_a_tok: Optional[torch.Tensor],
        chain_b_tok: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode paired or single-chain TCR.

        Args:
            chain_a_tok: First chain (alpha/gamma) tokens or None (batch, seq_len)
            chain_b_tok: Second chain (beta/delta) tokens or None

        Returns:
            TCR embedding (batch, d_model)
        """
        if chain_a_tok is None and chain_b_tok is None:
            raise ValueError("At least one chain must be provided")

        # Handle single-chain cases
        if chain_a_tok is None:
            return self.encode_chain(chain_b_tok)

        z_a = self.encode_chain(chain_a_tok)

        if chain_b_tok is None:
            return z_a

        z_b = self.encode_chain(chain_b_tok)
        z = self.fuse(torch.cat([z_a, z_b], dim=-1))
        return z


class ChainClassifier(nn.Module):
    """Classify chain type: TRA, TRB, TRG, TRD, IGH, IGK, IGL.

    Used to identify chain type from sequence alone.
    """

    def __init__(self, d_model: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.encoder = SequenceEncoder(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads
        )
        self.classifier = nn.Linear(d_model, len(CHAIN_TYPES))

    def forward(self, chain_tok: torch.Tensor) -> torch.Tensor:
        """Classify chain type.

        Args:
            chain_tok: Chain tokens (batch, seq_len)

        Returns:
            Logits over chain types (batch, n_chain_types)
        """
        z, _ = self.encoder(chain_tok)
        return self.classifier(z)


class ChainAttributeClassifier(nn.Module):
    """Per-chain classifier for species × chain type × cell phenotype.

    Given a single chain sequence, predicts:
    - Species: human, mouse, macaque, other
    - Chain type: TRA, TRB, TRG, TRD, IGH, IGK, IGL
    - Cell phenotype: CD4_T, CD8_T, ab_T, gd_T, B_cell

    These are predicted as factorized distributions (not joint) for efficiency,
    but biological constraints link them (e.g., IGH → B_cell).
    """

    def __init__(self, d_model: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.encoder = SequenceEncoder(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads
        )
        self.species_head = nn.Linear(d_model, len(SPECIES))
        self.chain_head = nn.Linear(d_model, len(CHAIN_TYPES))
        self.phenotype_head = nn.Linear(d_model, len(CELL_TYPES))

    def forward(self, chain_tok: torch.Tensor) -> dict[str, torch.Tensor]:
        """Classify chain attributes.

        Args:
            chain_tok: Chain tokens (batch, seq_len)

        Returns:
            Dict with logits for each attribute:
                - species_logits: (batch, n_species)
                - chain_logits: (batch, n_chain_types)
                - phenotype_logits: (batch, n_cell_types)
        """
        z, _ = self.encoder(chain_tok)
        return {
            "species_logits": self.species_head(z),
            "chain_logits": self.chain_head(z),
            "phenotype_logits": self.phenotype_head(z),
        }

    def predict(self, chain_tok: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get predicted classes and probabilities.

        Args:
            chain_tok: Chain tokens (batch, seq_len)

        Returns:
            Dict with predictions for each attribute
        """
        logits = self.forward(chain_tok)
        return {
            "species": torch.argmax(logits["species_logits"], dim=-1),
            "species_probs": torch.softmax(logits["species_logits"], dim=-1),
            "chain_type": torch.argmax(logits["chain_logits"], dim=-1),
            "chain_probs": torch.softmax(logits["chain_logits"], dim=-1),
            "phenotype": torch.argmax(logits["phenotype_logits"], dim=-1),
            "phenotype_probs": torch.softmax(logits["phenotype_logits"], dim=-1),
        }


class CellTypeClassifier(nn.Module):
    """Classify cell type: CD4_T, CD8_T, ab_T, gd_T, B_cell.

    Based on TCR representation (chain combinations, etc.).
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, len(CELL_TYPES)),
        )

    def forward(self, tcr_vec: torch.Tensor) -> torch.Tensor:
        """Classify cell type from TCR embedding.

        Args:
            tcr_vec: TCR vector representation (batch, d_model)

        Returns:
            Logits over cell types (batch, n_cell_types)
        """
        return self.classifier(tcr_vec)


class TCRpMHCMatcher(nn.Module):
    """Predicts TCR-pMHC recognition.

    Takes TCR and pMHC embeddings, outputs recognition logit.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        self.proj_tcr = nn.Linear(d_model, d_model)
        self.proj_pmhc = nn.Linear(d_model, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        tcr_vec: torch.Tensor,
        pmhc_vec: torch.Tensor,
        reduce: str = "mean",
    ) -> torch.Tensor:
        """Compute recognition logit.

        Args:
            tcr_vec: TCR vector representation (batch, d_model)
            pmhc_vec: pMHC vector representation (batch, d_model) or
                (batch, n_core_windows, d_model)
            reduce: How to aggregate over core windows - "mean" or "max"

        Returns:
            Recognition logit (batch,)
        """
        projected_tcr_vec = self.proj_tcr(tcr_vec)

        if pmhc_vec.dim() == 3:
            # Multiple core windows: (batch, n_windows, d_model)
            _, n_windows, _ = pmhc_vec.shape
            projected_pmhc_vec = self.proj_pmhc(pmhc_vec)

            # Expand TCR for all core windows.
            tcr_expanded = projected_tcr_vec.unsqueeze(1).expand(-1, n_windows, -1)

            # Compute per-core-window logits.
            paired_vec = torch.cat([tcr_expanded, projected_pmhc_vec], dim=-1)
            logits = self.head(paired_vec).squeeze(-1)

            # Aggregate
            if reduce == "max":
                return logits.max(dim=-1).values
            else:
                return logits.mean(dim=-1)

        else:
            # Single pMHC vector: (batch, d_model)
            projected_pmhc_vec = self.proj_pmhc(pmhc_vec)
            paired_vec = torch.cat([projected_tcr_vec, projected_pmhc_vec], dim=-1)
            return self.head(paired_vec).squeeze(-1)


class RepertoireHead(nn.Module):
    """Amortized prediction of P(some TCR in repertoire recognizes pMHC).

    Used when no specific TCR is given - predicts based on pMHC alone
    whether the antigen is likely to be recognized by the repertoire.
    """

    def __init__(self, d_model: int = 256, n_species: int = None):
        super().__init__()
        n_species = n_species or len(SPECIES)
        self.species_embed = nn.Embedding(n_species, d_model // 4)
        self.head = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.head_no_species = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        pmhc_vec: torch.Tensor,
        species_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict repertoire recognition probability.

        Args:
            pmhc_vec: pMHC vector representation (batch, d_model)
            species_idx: Species index for repertoire context (batch,)

        Returns:
            Recognition logit (batch, 1)
        """
        if species_idx is not None:
            species_vec = self.species_embed(species_idx)
            repertoire_vec = torch.cat([pmhc_vec, species_vec], dim=-1)
            return self.head(repertoire_vec)
        else:
            return self.head_no_species(pmhc_vec)
