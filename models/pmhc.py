"""pMHC (peptide-MHC) modules for Presto.

Biological pathway:
1. Processing: MHC-INDEPENDENT (class-I and class-II pathways)
2. Binding: Class-symmetric three-latent kinetics path
3. Presentation: Combines processing and binding-derived stability

Key principle: Processing happens BEFORE MHC binding.

This module contains reusable pMHC components used by the canonical
end-to-end production architecture in `models/presto.py`.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import SequenceEncoder
from ..data.vocab import AA_TO_IDX, AA_VOCAB
from ..data.allele_resolver import (
    normalize_mhc_class,
    normalize_processing_species_label,
    PROCESSING_SPECIES_BUCKETS,
)


def stable_noisy_or(
    probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = -1,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Numerically stable Noisy-OR aggregation.

    Computes 1 - prod(1 - p_i) in log space to avoid underflow.

    Args:
        probs: Probabilities to aggregate
        mask: Optional mask (1 = include, 0 = exclude)
        dim: Dimension to aggregate over
        eps: Small constant for numerical stability

    Returns:
        Aggregated probability
    """
    if mask is not None:
        probs = probs * mask

    # Clamp to valid probability range
    probs = torch.clamp(probs, min=0.0, max=1.0 - eps)

    # Compute in log space: log(1 - p_i)
    log_one_minus_p = torch.log1p(-probs)

    # Sum logs and convert back
    log_one_minus_result = log_one_minus_p.sum(dim=dim)

    # Use expm1 for stability: 1 - exp(x) = -expm1(x)
    return -torch.expm1(log_one_minus_result)


def posterior_attribution(probs: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute posterior attribution of each instance to bag probability.

    Given instance probabilities, compute how much each contributed to
    the bag being positive.

    Args:
        probs: Per-instance probabilities

    Returns:
        Attribution scores (roughly proportional to contribution)
    """
    # Simple attribution: normalized odds
    # P(instance i caused positive | bag positive) ∝ p_i / (1 - p_i)
    odds = probs / (1 - probs + eps)
    return odds / (odds.sum() + eps)


def enumerate_core_windows(
    peptide: str,
    core_lens: Tuple[int, ...] = (9,),
) -> List[Dict[str, any]]:
    """Enumerate all possible core windows for a peptide.

    For MHC-II, the binding core can be 8-10 residues with PFRs on either side.

    Args:
        peptide: Peptide sequence
        core_lens: Possible core lengths to enumerate

    Returns:
        List of core-window dicts with: start, core_len, core, pfr_n, pfr_c
    """
    P = len(peptide)
    core_windows = []

    for core_len in core_lens:
        if P < core_len:
            continue
        for start in range(P - core_len + 1):
            core_windows.append({
                "start": start,
                "core_len": core_len,
                "core": peptide[start : start + core_len],
                "pfr_n": peptide[:start],
                "pfr_c": peptide[start + core_len :],
            })

    return core_windows


def _class_probs_from_input(
    batch_size: int,
    device: torch.device,
    mhc_class: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
    class_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Resolve class probabilities (I/II) from optional labels or logits.

    Returns tensor of shape (batch, 2) with columns [p_I, p_II].
    """
    if class_probs is not None:
        probs = class_probs.to(device=device, dtype=torch.float32)
        if probs.ndim != 2 or probs.shape[-1] != 2:
            raise ValueError(
                f"class_probs must have shape (batch, 2); got {tuple(probs.shape)}"
            )
        if probs.shape[0] != batch_size:
            if probs.shape[0] == 1:
                probs = probs.expand(batch_size, -1)
            else:
                raise ValueError(
                    f"class_probs batch mismatch: expected {batch_size}, got {probs.shape[0]}"
                )
        if probs.min().item() >= 0.0 and probs.max().item() <= 1.0:
            # Treat as probabilities (or unnormalized non-negative weights).
            return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # Otherwise interpret as logits.
        return torch.softmax(probs, dim=-1)

    if isinstance(mhc_class, str):
        cls = normalize_mhc_class(mhc_class, default="I")
        if cls == "II":
            return torch.tensor([[0.0, 1.0]], device=device).expand(batch_size, -1)
        return torch.tensor([[1.0, 0.0]], device=device).expand(batch_size, -1)

    if isinstance(mhc_class, (list, tuple)):
        probs = torch.zeros((batch_size, 2), device=device, dtype=torch.float32)
        for idx in range(batch_size):
            cls = "I"
            if idx < len(mhc_class):
                cls = normalize_mhc_class(mhc_class[idx], default="I")
            if cls == "II":
                probs[idx, 1] = 1.0
            else:
                probs[idx, 0] = 1.0
        return probs

    return torch.full((batch_size, 2), 0.5, device=device, dtype=torch.float32)


def _species_probs_from_input(
    batch_size: int,
    device: torch.device,
    species: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
    species_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Resolve species probabilities for processing conditioning.

    Returns tensor of shape (batch, n_species) with columns aligned to
    `PROCESSING_SPECIES_BUCKETS`.
    """
    n_species = len(PROCESSING_SPECIES_BUCKETS)
    if species_probs is not None:
        probs = species_probs.to(device=device, dtype=torch.float32)
        if probs.ndim != 2 or probs.shape[-1] != n_species:
            raise ValueError(
                f"species_probs must have shape (batch, {n_species}); got {tuple(probs.shape)}"
            )
        if probs.shape[0] != batch_size:
            if probs.shape[0] == 1:
                probs = probs.expand(batch_size, -1)
            else:
                raise ValueError(
                    f"species_probs batch mismatch: expected {batch_size}, got {probs.shape[0]}"
                )
        if probs.min().item() >= 0.0 and probs.max().item() <= 1.0:
            return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.softmax(probs, dim=-1)

    if isinstance(species, str):
        normalized = normalize_processing_species_label(species, default=None)
        if normalized is None:
            # Unknown species: uniform probs
            return torch.full(
                (batch_size, n_species),
                1.0 / n_species,
                device=device,
                dtype=torch.float32,
            )
        probs = torch.zeros((1, n_species), device=device, dtype=torch.float32)
        sp_idx = PROCESSING_SPECIES_BUCKETS.index(normalized) if normalized in PROCESSING_SPECIES_BUCKETS else 0
        probs[0, sp_idx] = 1.0
        return probs.expand(batch_size, -1)

    if isinstance(species, (list, tuple)):
        probs = torch.zeros((batch_size, n_species), device=device, dtype=torch.float32)
        for idx in range(batch_size):
            label = None
            if idx < len(species):
                label = normalize_processing_species_label(species[idx], default=None)
            if label is None:
                # Unknown species: uniform probs for this sample
                probs[idx, :] = 1.0 / n_species
            else:
                sp_idx = PROCESSING_SPECIES_BUCKETS.index(label) if label in PROCESSING_SPECIES_BUCKETS else 0
                probs[idx, sp_idx] = 1.0
        return probs

    return torch.full(
        (batch_size, n_species),
        1.0 / n_species,
        device=device,
        dtype=torch.float32,
    )


def _gather_prefix(prefix: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather prefix sums at per-sample indices.

    Args:
        prefix: (batch, length+1, dim)
        indices: (batch, n)
    Returns:
        Gathered values with shape (batch, n, dim).
    """
    idx = indices.unsqueeze(-1).expand(-1, -1, prefix.shape[-1])
    return torch.gather(prefix, dim=1, index=idx)


class ProcessingModule(nn.Module):
    """Antigen processing prediction.

    CRITICAL: Processing is MHC-INDEPENDENT.
    The same peptide gets processed regardless of which MHC alleles are expressed.
    Only the peptide and its source protein flanks matter.
    """

    def __init__(self, d_model: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.encoder = SequenceEncoder(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads
        )
        # Fixed-window flank path: lightweight FFN over pooled N/C flank embeddings.
        self.flank_embed = nn.Embedding(len(AA_VOCAB), d_model, padding_idx=0)
        self.flank_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.path_I_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.path_II_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Separate heads for Class I (proteasome/TAP) vs Class II (endosomal)
        self.head_I = nn.Linear(d_model, 1)
        self.head_II = nn.Linear(d_model, 1)
        # Species applies a lightweight residual modifier on top of class pathways.
        n_species = len(PROCESSING_SPECIES_BUCKETS)
        self.species_modifier_I = nn.Linear(n_species, d_model, bias=False)
        self.species_modifier_II = nn.Linear(n_species, d_model, bias=False)

    def forward(
        self,
        pep_tok: torch.Tensor,
        flank_n_tok: Optional[torch.Tensor],
        flank_c_tok: Optional[torch.Tensor],
        mhc_class: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        class_probs: Optional[torch.Tensor] = None,
        species: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        species_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict canonical mixed processing probability.

        This is the class-probability-weighted mixture of:
        - processing_class1_logit (flank-aware pathway)
        - processing_class2_logit (peptide-centric pathway)
        """
        return self.forward_components(
            pep_tok=pep_tok,
            flank_n_tok=flank_n_tok,
            flank_c_tok=flank_c_tok,
            mhc_class=mhc_class,
            class_probs=class_probs,
            species=species,
            species_probs=species_probs,
        )["processing_logit"]

    def forward_components(
        self,
        pep_tok: torch.Tensor,
        flank_n_tok: Optional[torch.Tensor],
        flank_c_tok: Optional[torch.Tensor],
        mhc_class: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        class_probs: Optional[torch.Tensor] = None,
        species: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        species_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict class-specific and mixed processing logits.

        Args:
            pep_tok: Peptide tokens (batch, seq_len)
            flank_n_tok: N-terminal flank tokens (batch, flank_len) or None
            flank_c_tok: C-terminal flank tokens (batch, flank_len) or None
            mhc_class: Optional class label(s) ("I"/"II"), used when class_probs
                is not provided.
            class_probs: Optional inferred class probabilities (batch, 2) with
                columns [p_I, p_II].
            species: Optional processing species label(s) used when
                species_probs is not provided.
            species_probs: Optional inferred species probabilities
                (batch, 4), aligned to PROCESSING_SPECIES_BUCKETS.

        Returns:
            Dict with:
                - processing_class1_logit: Class-I pathway processing logit (batch, 1)
                - processing_class2_logit: Class-II pathway processing logit (batch, 1)
                - processing_logit: weighted mixture using class probabilities (batch, 1)
        """
        # Class-conditioned mixture of two biologically distinct pathways:
        # - Class I: cytosolic proteasome/TAP processing (uses source flanks).
        # - Class II: endosomal/lysosomal processing (peptide-centric).
        batch_size = pep_tok.shape[0]
        probs = _class_probs_from_input(
            batch_size=batch_size,
            device=pep_tok.device,
            mhc_class=mhc_class,
            class_probs=class_probs,
        )
        sp_probs = _species_probs_from_input(
            batch_size=batch_size,
            device=pep_tok.device,
            species=species,
            species_probs=species_probs,
        )

        # Peptide pathway is transformer-encoded; flanks flow through a fixed-window FFN path.
        pep_vec, _ = self.encoder(pep_tok)

        def _pool_flank(tok: Optional[torch.Tensor]) -> torch.Tensor:
            if tok is None:
                return pep_vec.new_zeros((batch_size, self.d_model))
            emb = self.flank_embed(tok)
            mask = (tok != 0).unsqueeze(-1).float()
            return (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        n_flank = _pool_flank(flank_n_tok)
        c_flank = _pool_flank(flank_c_tok)
        flank_vec = self.flank_fuse(torch.cat([n_flank, c_flank], dim=-1))

        # Two independent processing pathways share inputs but use distinct fusers/heads.
        z_i = self.path_I_fuse(torch.cat([pep_vec, flank_vec], dim=-1))
        z_ii = self.path_II_fuse(torch.cat([pep_vec, flank_vec], dim=-1))
        out_I = self.head_I(z_i + self.species_modifier_I(sp_probs))
        out_II = self.head_II(z_ii + self.species_modifier_II(sp_probs))
        mixed = probs[:, :1] * out_I + probs[:, 1:2] * out_II
        return {
            "processing_class1_logit": out_I,
            "processing_class2_logit": out_II,
            "processing_logit": mixed,
        }


class BindingModule(nn.Module):
    """Peptide-MHC binding module.

    Outputs three kinetic rate latents (in log10 space):
    - log_koff: Dissociation rate log10(s⁻¹) - determines stability
    - log_kon_intrinsic: Intrinsic association rate log10(M⁻¹s⁻¹)
    - log_kon_chaperone: Chaperone-assisted association log10(M⁻¹s⁻¹)

    Physical relationships:
    - KD = koff / kon_total where kon_total combines intrinsic + chaperone
    - t1/2 = ln(2) / koff
    - kon_total ≈ kon_intrinsic + kon_chaperone (simplified model)
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        # All predict log10 of rate constants
        self.head_log_koff = nn.Linear(d_model, 1)
        self.head_log_kon_intrinsic = nn.Linear(d_model, 1)
        self.head_log_kon_chaperone = nn.Linear(d_model, 1)

    def forward(
        self,
        pmhc_vec: torch.Tensor,
        mhc_class: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        class_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute binding kinetic latents.

        Args:
            pmhc_vec: Fused peptide-MHC vector (batch, d_model)
            mhc_class: Optional class labels (ignored in latent generation).
            class_probs: Optional class probabilities (ignored in latent generation).
                Class-dependent calibration is applied downstream in
                `models/presto.py` when deriving `binding_class1_logit` and
                `binding_class2_logit` from the shared base binding logit.

        Returns:
            Dict with log_koff, log_kon_intrinsic, log_kon_chaperone (all log10 scale)
        """
        log_koff = torch.clamp(self.head_log_koff(pmhc_vec), min=-8.0, max=8.0)
        log_kon_intrinsic = torch.clamp(
            self.head_log_kon_intrinsic(pmhc_vec), min=-8.0, max=8.0
        )
        log_kon_chaperone = torch.clamp(
            self.head_log_kon_chaperone(pmhc_vec), min=-8.0, max=8.0
        )

        return {
            "log_koff": log_koff,
            "log_kon_intrinsic": log_kon_intrinsic,
            "log_kon_chaperone": log_kon_chaperone,
        }

    def derive_kd(self, latents: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Derive log10(KD in nM) from latents.

        KD = koff / kon_total
        log10(KD_nM) = log10(koff) - log10(kon_total) + 9
        (the +9 converts from M to nM)
        """
        log_koff = torch.clamp(latents["log_koff"], min=-8.0, max=8.0)
        # Combine kon_intrinsic and kon_chaperone (sum in linear space)
        kon_intrinsic = torch.pow(10, torch.clamp(latents["log_kon_intrinsic"], min=-8.0, max=8.0))
        kon_chaperone = torch.pow(10, torch.clamp(latents["log_kon_chaperone"], min=-8.0, max=8.0))
        kon_total = kon_intrinsic + kon_chaperone
        log_kon_total = torch.log10(kon_total.clamp(min=1e-10, max=1e10))

        # KD in M, convert to nM (*1e9 = +9 in log space)
        log_kd_nM = torch.clamp(log_koff - log_kon_total + 9, min=-3.0, max=8.0)
        return log_kd_nM

    def derive_half_life(self, latents: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Derive log10(t1/2 in minutes) from latents.

        t1/2 = ln(2) / koff
        log10(t1/2_min) = log10(ln(2)) - log10(koff) - log10(60)
        (the -log10(60) converts from seconds to minutes)
        """
        log_koff = torch.clamp(latents["log_koff"], min=-8.0, max=8.0)
        # ln(2) ≈ 0.693, log10(ln(2)) ≈ -0.159
        # log10(60) ≈ 1.778
        log_t_half_min = torch.clamp(-0.159 - log_koff - 1.778, min=-8.0, max=8.0)
        return log_t_half_min


class StableBindingHead(nn.Module):
    """Combines binding latents into stable binding probability logit.

    Uses learnable positive weights (via softplus) to combine latents.
    """

    def __init__(self):
        super().__init__()
        self.w_stability = nn.Parameter(torch.tensor(0.5))
        self.w_intrinsic = nn.Parameter(torch.tensor(0.3))
        self.w_chaperone = nn.Parameter(torch.tensor(0.2))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        stability: torch.Tensor,
        intrinsic: torch.Tensor,
        chaperone: torch.Tensor,
    ) -> torch.Tensor:
        """Combine latents into binding logit.

        Args:
            stability, intrinsic, chaperone: Latent strengths (batch, 1)

        Returns:
            Binding logit (batch,)
        """
        w_s = F.softplus(self.w_stability)
        w_i = F.softplus(self.w_intrinsic)
        w_c = F.softplus(self.w_chaperone)

        logit = (
            w_s * stability.squeeze(-1)
            + w_i * intrinsic.squeeze(-1)
            + w_c * chaperone.squeeze(-1)
            + self.bias
        )
        return logit


class PMHCEncoder(nn.Module):
    """Encodes peptide-MHC complex with cross-attention.

    Key insight: MHC-I binding has position-specific anchor preferences (P2, PΩ)
    that are ALLELE-SPECIFIC. The peptide encoder needs to see MHC context to
    know which positions and residues matter.

    Architecture:
    1. Encode MHC chains independently (self-attention)
    2. Encode peptide with cross-attention to MHC context
    3. Fuse all representations

    This allows peptide position 2 to "see" the B-pocket of the MHC and learn
    that A*02:01 wants hydrophobic while B*07:02 wants proline.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_cross_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        aux_layers = max(1, n_layers // 2)

        # MHC encoder (self-attention only)
        self.mhc_encoder = SequenceEncoder(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads
        )

        # Peptide embedding (shared vocab with MHC)
        from ..data.vocab import AA_VOCAB
        self.pep_embedding = nn.Embedding(len(AA_VOCAB), d_model)
        self.pep_pos_enc = nn.Embedding(64, d_model)  # Max peptide length

        # Cross-attention layers: peptide attends to MHC
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
            for _ in range(n_cross_layers)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_cross_layers)
        ])
        self.cross_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(0.1),
            ) for _ in range(n_cross_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_cross_layers)
        ])

        # Final peptide self-attention after cross-attention
        self.pep_self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True, norm_first=True,
            ),
            num_layers=aux_layers,
            enable_nested_tensor=False,  # keep norm_first without nested tensor warnings
        )

        # Fusion layer
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
        mhc_class: str = "I",
    ) -> torch.Tensor:
        """Encode pMHC complex with cross-attention.

        Args:
            pep_tok: Peptide tokens (batch, pep_len)
            mhc_a_tok: MHC alpha chain tokens (batch, mhc_len)
            mhc_b_tok: MHC beta chain tokens (batch, mhc_len)
            mhc_class: "I" or "II"

        Returns:
            Fused representation (batch, d_model)
        """
        B, pep_len = pep_tok.shape

        # 1. Encode MHC chains (self-attention)
        z_a_pooled, z_a_seq = self.mhc_encoder(mhc_a_tok)  # (B, d), (B, L, d)
        z_b_pooled, z_b_seq = self.mhc_encoder(mhc_b_tok)

        # Concatenate MHC sequences as cross-attention context
        # Shape: (B, mhc_a_len + mhc_b_len, d_model)
        mhc_context = torch.cat([z_a_seq, z_b_seq], dim=1)

        # 2. Embed peptide with positional encoding
        positions = torch.arange(pep_len, device=pep_tok.device)
        pep_emb = self.pep_embedding(pep_tok) + self.pep_pos_enc(positions)

        # 3. Cross-attention: peptide attends to MHC
        # This lets P2 see the B-pocket, PΩ see the F-pocket
        x = pep_emb
        for i, (attn, norm, ffn, ffn_norm) in enumerate(zip(
            self.cross_attention_layers, self.cross_norms,
            self.cross_ffn, self.ffn_norms
        )):
            # Cross-attention (peptide queries, MHC keys/values)
            attn_out, _ = attn(x, mhc_context, mhc_context)
            x = norm(x + attn_out)
            # FFN
            x = ffn_norm(x + ffn(x))

        # 4. Peptide self-attention (after seeing MHC context)
        x = self.pep_self_attention(x)

        # 5. Pool peptide representation
        z_pep = x.mean(dim=1)  # (B, d_model)

        # 6. Fuse all representations
        z = self.fuse(torch.cat([z_pep, z_a_pooled, z_b_pooled], dim=-1))
        return z


class CoreWindowScorer(nn.Module):
    """Enumerate peptide core windows and score them against MHC context.

    Core-window format follows:
        Npfr | core | Cpfr
    where core lengths are enumerated over a configurable range.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        core_min_len: int = 8,
        core_max_len: int = 50,
        adjacent_k: int = 5,
    ):
        super().__init__()
        self.core_min_len = int(core_min_len)
        self.core_max_len = int(core_max_len)
        self.adjacent_k = int(adjacent_k)

        self.peptide_embed = nn.Embedding(len(AA_VOCAB), d_model, padding_idx=0)
        self.core_proj = nn.Linear(d_model, d_model)
        self.groove_attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.pfr_proj = nn.Sequential(
            nn.Linear(d_model * 2 + 20 * 2 + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.core_window_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.mhc_pair_proj = nn.Linear(d_model * 2, d_model)
        self.core_window_norm = nn.LayerNorm(d_model)
        self.core_window_prior = nn.Sequential(
            nn.Linear(d_model + 5, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        aa20_lookup = torch.full((len(AA_VOCAB),), -1, dtype=torch.long)
        for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY"):
            aa20_lookup[AA_TO_IDX[aa]] = idx
        self.register_buffer("aa20_lookup", aa20_lookup, persistent=False)

    def _enumerate_core_window_grid(
        self, max_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        starts: List[int] = []
        lengths: List[int] = []
        lo = max(1, self.core_min_len)
        hi = max(lo, min(self.core_max_len, max_len))
        for core_len in range(lo, hi + 1):
            for start in range(max_len - core_len + 1):
                starts.append(start)
                lengths.append(core_len)

        if not starts:
            starts = [0]
            lengths = [max(1, max_len)]

        return (
            torch.tensor(starts, device=device, dtype=torch.long),
            torch.tensor(lengths, device=device, dtype=torch.long),
        )

    def forward(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
        mhc_a_seq: torch.Tensor,
        mhc_b_seq: torch.Tensor,
        mhc_a_pooled: torch.Tensor,
        mhc_b_pooled: torch.Tensor,
        class_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Score all candidate core windows for each peptide.

        Args:
            pep_tok: peptide tokens (batch, pep_len)
            mhc_a_tok: MHC alpha chain tokens (batch, mhc_len)
            mhc_b_tok: MHC beta chain tokens (batch, mhc_len)
            mhc_a_seq: encoded alpha chain states (batch, mhc_len, d_model)
            mhc_b_seq: encoded beta chain states (batch, mhc_len, d_model)
            mhc_a_pooled: pooled alpha embedding (batch, d_model)
            mhc_b_pooled: pooled beta embedding (batch, d_model)
            class_probs: Optional inferred class probs (batch, 2), [p_I, p_II]
        """
        batch_size, _ = pep_tok.shape
        pep_len = (pep_tok != 0).sum(dim=1)
        pep_max_len = max(1, int(pep_len.max().item()))
        pep_tok = pep_tok[:, :pep_max_len]
        starts_1d, lens_1d = self._enumerate_core_window_grid(pep_max_len, pep_tok.device)
        num_core_windows = int(starts_1d.shape[0])

        starts = starts_1d.view(1, num_core_windows).expand(batch_size, -1)
        core_lens = lens_1d.view(1, num_core_windows).expand(batch_size, -1)
        ends = starts + core_lens

        pep_len_expanded = pep_len.unsqueeze(1).expand_as(starts)
        core_window_mask = (starts < pep_len_expanded) & (ends <= pep_len_expanded)

        pep_emb = self.peptide_embed(pep_tok)
        prefix_emb = torch.cat(
            [pep_emb.new_zeros((batch_size, 1, pep_emb.shape[-1])), pep_emb.cumsum(dim=1)],
            dim=1,
        )
        core_sum = _gather_prefix(prefix_emb, ends) - _gather_prefix(prefix_emb, starts)
        core_mean = core_sum / core_lens.float().unsqueeze(-1).clamp(min=1.0)

        mhc_context = torch.cat([mhc_a_seq, mhc_b_seq], dim=1)
        mhc_padding_mask = torch.cat([mhc_a_tok == 0, mhc_b_tok == 0], dim=1)
        # MultiheadAttention returns NaN when all key positions are masked.
        # Keep one dummy key unmasked for fully empty MHC inputs.
        all_masked = mhc_padding_mask.all(dim=1)
        if all_masked.any():
            mhc_padding_mask = mhc_padding_mask.clone()
            mhc_padding_mask[all_masked, 0] = False
        groove_context, _ = self.groove_attention(
            core_mean,
            mhc_context,
            mhc_context,
            key_padding_mask=mhc_padding_mask,
        )

        # Adjacent PFR windows and distal PFR statistics.
        n_adj_start = torch.clamp(starts - self.adjacent_k, min=0)
        n_adj_end = starts
        c_adj_start = ends
        c_adj_end = torch.minimum(ends + self.adjacent_k, pep_len_expanded)

        n_adj_sum = _gather_prefix(prefix_emb, n_adj_end) - _gather_prefix(
            prefix_emb, n_adj_start
        )
        c_adj_sum = _gather_prefix(prefix_emb, c_adj_end) - _gather_prefix(
            prefix_emb, c_adj_start
        )
        n_adj_mean = n_adj_sum / (n_adj_end - n_adj_start).float().unsqueeze(-1).clamp(min=1.0)
        c_adj_mean = c_adj_sum / (c_adj_end - c_adj_start).float().unsqueeze(-1).clamp(min=1.0)

        aa_lookup = self.aa20_lookup.to(device=pep_tok.device)
        aa_idx = aa_lookup[pep_tok.clamp(min=0, max=aa_lookup.shape[0] - 1)]
        aa_valid = aa_idx >= 0
        aa_one_hot = (
            F.one_hot(aa_idx.clamp(min=0), num_classes=20).float()
            * aa_valid.unsqueeze(-1).float()
        )
        prefix_counts = torch.cat(
            [aa_one_hot.new_zeros((batch_size, 1, aa_one_hot.shape[-1])), aa_one_hot.cumsum(dim=1)],
            dim=1,
        )

        n_distal_counts = _gather_prefix(prefix_counts, n_adj_start)
        pep_total_counts = _gather_prefix(prefix_counts, pep_len_expanded)
        c_distal_counts = pep_total_counts - _gather_prefix(prefix_counts, c_adj_end)
        n_distal_len = n_adj_start.float()
        c_distal_len = (pep_len_expanded - c_adj_end).float()
        n_distal_freq = n_distal_counts / n_distal_len.unsqueeze(-1).clamp(min=1.0)
        c_distal_freq = c_distal_counts / c_distal_len.unsqueeze(-1).clamp(min=1.0)

        pfr_features = torch.cat(
            [
                n_adj_mean,
                c_adj_mean,
                n_distal_freq,
                c_distal_freq,
                (n_distal_len / 50.0).unsqueeze(-1),
                (c_distal_len / 50.0).unsqueeze(-1),
            ],
            dim=-1,
        )
        pfr_vec = self.pfr_proj(pfr_features)

        mhc_pair_vec = self.mhc_pair_proj(
            torch.cat([mhc_a_pooled, mhc_b_pooled], dim=-1)
        ).unsqueeze(1).expand(-1, num_core_windows, -1)
        core_window_vec = self.core_window_norm(
            self.core_window_proj(
                torch.cat(
                    [
                        self.core_proj(core_mean),
                        groove_context,
                        pfr_vec,
                    ],
                    dim=-1,
                )
            )
            + mhc_pair_vec
        )

        if class_probs is None:
            class_probs = torch.full(
                (batch_size, 2), 0.5, device=pep_tok.device, dtype=torch.float32
            )
        class_probs_expanded = class_probs.unsqueeze(1).expand(-1, num_core_windows, -1)
        pep_len_float = pep_len_expanded.float().clamp(min=1.0)
        core_len_norm = core_lens.float() / pep_len_float
        n_len_norm = starts.float() / pep_len_float
        c_len_norm = (pep_len_expanded - ends).float() / pep_len_float
        prior_features = torch.cat(
            [
                mhc_pair_vec,
                core_len_norm.unsqueeze(-1),
                n_len_norm.unsqueeze(-1),
                c_len_norm.unsqueeze(-1),
                class_probs_expanded,
            ],
            dim=-1,
        )
        core_window_prior_logit = self.core_window_prior(prior_features).squeeze(-1)

        core_window_vec = core_window_vec * core_window_mask.unsqueeze(-1).float()
        core_window_prior_logit = core_window_prior_logit.masked_fill(~core_window_mask, -1e4)

        return {
            "core_window_vec": core_window_vec,
            "core_window_mask": core_window_mask,
            "core_window_prior_logit": core_window_prior_logit,
            "core_window_start": starts,
            "core_window_length": core_lens,
            "npfr_length": starts,
            "cpfr_length": pep_len_expanded - ends,
        }


class PresentationBottleneck(nn.Module):
    """Combines processing and binding into presentation probability.

    Presentation requires BOTH processing AND stable binding.
    """

    def __init__(self):
        super().__init__()
        self.w_proc = nn.Parameter(torch.tensor(0.8))
        self.w_bind = nn.Parameter(torch.tensor(1.0))
        self.w_prior = nn.Parameter(torch.tensor(0.2))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        proc_logit: torch.Tensor,
        bind_logit: torch.Tensor,
        core_window_prior_logit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combine processing and binding.

        Args:
            proc_logit: Processing probability logit (batch, 1)
            bind_logit: Binding probability logit (batch,) or (batch, n_core_windows)
            core_window_prior_logit: Optional per-core-window prior logits from MHC context

        Returns:
            Presentation logit with same rank as bind_logit
        """
        proc_vec = proc_logit.squeeze(-1)
        if bind_logit.ndim > 1 and proc_vec.ndim == 1:
            proc_vec = proc_vec.unsqueeze(-1).expand_as(bind_logit)

        w_p = F.softplus(self.w_proc)
        w_b = F.softplus(self.w_bind)
        w_r = F.softplus(self.w_prior)
        combined = (
            w_p * proc_vec
            + w_b * bind_logit
            + self.bias
        )
        if core_window_prior_logit is not None:
            combined = combined + w_r * core_window_prior_logit
        return combined


