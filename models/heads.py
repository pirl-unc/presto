"""Prediction heads for IEDB measurements.

Task-specific heads that sit on top of the shared encoder:
- Binding: KD, IC50, EC50 (log10 nM)
- Kinetics: kon (log10 M⁻¹s⁻¹), koff (log10 s⁻¹)
- Stability: t1/2 (log10 min), Tm (normalized Celsius)
- T-cell: functional assay outcomes
- Elution/MS: detection probability
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Unit conversion utilities
# --------------------------------------------------------------------------

def to_log10_nM(value: torch.Tensor) -> torch.Tensor:
    """Convert nM to log10(nM)."""
    return torch.log10(value.clamp(min=1e-6))


def from_log10_nM(log_value: torch.Tensor) -> torch.Tensor:
    """Convert log10(nM) to nM."""
    return torch.pow(10, log_value)


def normalize_tm(tm_celsius: torch.Tensor, mean: float = 50.0, std: float = 15.0) -> torch.Tensor:
    """Normalize Tm from Celsius to standardized scale."""
    return (tm_celsius - mean) / std


def denormalize_tm(tm_norm: torch.Tensor, mean: float = 50.0, std: float = 15.0) -> torch.Tensor:
    """Denormalize Tm back to Celsius."""
    return tm_norm * std + mean


def smooth_upper_bound(value: torch.Tensor, max_value: float) -> torch.Tensor:
    """Apply a smooth upper bound that preserves gradient near the cap."""
    max_tensor = value.new_tensor(max_value)
    return max_tensor - F.softplus(max_tensor - value)


# --------------------------------------------------------------------------
# Individual prediction heads
# --------------------------------------------------------------------------

class KDHead(nn.Module):
    """Predicts KD in log10(nM)."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class IC50Head(nn.Module):
    """Predicts IC50 in log10(nM)."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class EC50Head(nn.Module):
    """Predicts EC50 in log10(nM)."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class KonHead(nn.Module):
    """Predicts kon in log10(M⁻¹s⁻¹)."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class KoffHead(nn.Module):
    """Predicts koff in log10(s⁻¹)."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class HalfLifeHead(nn.Module):
    """Predicts t1/2 in log10(minutes)."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class TmHead(nn.Module):
    """Predicts Tm (melting temperature) in normalized Celsius."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


# --------------------------------------------------------------------------
# Combined assay heads
# --------------------------------------------------------------------------

class AssayHeads(nn.Module):
    """Combined module for all binding/kinetics/stability assay heads.

    Uses a hybrid approach:
    - KD, koff, kon, t_half: Derived from kinetic latents (physics-based)
    - IC50, EC50: Predicted with residual correction (assay-specific factors)
    - Tm: Predicted directly (complex relationship with stability)

    This ensures the latents have proper physical interpretation while
    allowing assay-specific corrections for experimental conditions.
    """

    def __init__(self, d_model: int = 256, max_log10_nM: float = 5.0):
        super().__init__()
        self.max_log10_nM = float(max_log10_nM)
        # Residual heads for assays that need correction beyond physics
        # IC50/EC50 depend on assay conditions (peptide concentration, etc.)
        self.ic50_residual = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.ec50_residual = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # Tm has complex relationship - predict directly
        self.tm = TmHead(d_model)

    def forward(
        self,
        z: torch.Tensor,
        binding_latents: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict all assay values.

        Args:
            z: pMHC embedding (batch, d_model)
            binding_latents: Dict with log_koff, log_kon_intrinsic, log_kon_chaperone
                           If provided, KD/koff/kon/t_half are derived from these.

        Returns:
            Dict with predictions for each assay type (all in log10 scale except Tm)
        """
        results = {}

        if binding_latents is not None:
            # Derive kinetic values from latents (physics-based)
            log_koff = torch.clamp(binding_latents["log_koff"], min=-8.0, max=8.0)
            log_kon_intrinsic = torch.clamp(
                binding_latents["log_kon_intrinsic"], min=-8.0, max=8.0
            )
            log_kon_chaperone = torch.clamp(
                binding_latents["log_kon_chaperone"], min=-8.0, max=8.0
            )

            # kon_total = kon_intrinsic + kon_chaperone
            kon_intrinsic = torch.pow(10, log_kon_intrinsic)
            kon_chaperone = torch.pow(10, log_kon_chaperone)
            kon_total = kon_intrinsic + kon_chaperone
            log_kon_total = torch.log10(kon_total.clamp(min=1e-10, max=1e10))

            # KD = koff / kon (in M), convert to nM (+9).
            # Keep a smooth upper cap so weak-affinity regions still backpropagate.
            log_kd_nM = torch.clamp(log_koff - log_kon_total + 9, min=-3.0)
            affinity_obs = self.derive_affinity_observables(z, log_kd_nM)

            # t_half = ln(2) / koff, convert to minutes
            # log10(t_half_min) = log10(ln(2)/60) - log_koff = -1.937 - log_koff
            log_t_half = torch.clamp(-1.937 - log_koff, min=-8.0, max=8.0)

            results["koff"] = log_koff
            results["kon"] = log_kon_total
            results["KD_nM"] = affinity_obs["KD_nM"]
            results["t_half"] = log_t_half
            results["IC50_nM"] = affinity_obs["IC50_nM"]
            results["EC50_nM"] = affinity_obs["EC50_nM"]
        else:
            # Fallback: predict everything directly (less constrained)
            results["KD_nM"] = smooth_upper_bound(
                torch.clamp(self._direct_predict(z, "kd"), min=-3.0),
                self.max_log10_nM,
            )
            results["IC50_nM"] = smooth_upper_bound(
                torch.clamp(self._direct_predict(z, "ic50"), min=-3.0),
                self.max_log10_nM,
            )
            results["EC50_nM"] = smooth_upper_bound(
                torch.clamp(self._direct_predict(z, "ec50"), min=-3.0),
                self.max_log10_nM,
            )
            results["kon"] = self._direct_predict(z, "kon")
            results["koff"] = self._direct_predict(z, "koff")
            results["t_half"] = self._direct_predict(z, "t_half")

        # Tm always predicted directly (complex protein stability)
        results["Tm"] = self.tm(z)

        return results

    def derive_affinity_observables(
        self,
        z: torch.Tensor,
        kd_log10_nM: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Derive KD/IC50/EC50 from a shared KD latent with assay-specific bias."""
        kd_base = smooth_upper_bound(
            torch.clamp(kd_log10_nM, min=-3.0),
            self.max_log10_nM,
        )
        ic50 = smooth_upper_bound(
            torch.clamp(kd_base + self.ic50_residual(z), min=-3.0),
            self.max_log10_nM,
        )
        ec50 = smooth_upper_bound(
            torch.clamp(kd_base + self.ec50_residual(z), min=-3.0),
            self.max_log10_nM,
        )
        return {
            "KD_nM": kd_base,
            "IC50_nM": ic50,
            "EC50_nM": ec50,
        }

    def _direct_predict(self, z: torch.Tensor, assay: str) -> torch.Tensor:
        """Direct prediction fallback when latents not available."""
        # Simple linear prediction
        if not hasattr(self, f"_fallback_{assay}"):
            setattr(self, f"_fallback_{assay}",
                    nn.Linear(z.shape[-1], 1).to(z.device))
        return getattr(self, f"_fallback_{assay}")(z)


# --------------------------------------------------------------------------
# T-cell functional assay head
# --------------------------------------------------------------------------

class TCellHead(nn.Module):
    """Predicts T-cell functional assay outcome.

    Can work with specific TCR or repertoire-level features.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        # With TCR: use both pMHC and TCR
        self.with_tcr = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        # Without TCR: repertoire-level prediction
        self.without_tcr = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        pmhc_vec: torch.Tensor,
        tcr_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict T-cell response.

        Args:
            pmhc_vec: pMHC vector representation (batch, d_model)
            tcr_vec: Optional TCR vector representation (batch, d_model)

        Returns:
            Response logit (batch, 1)
        """
        if tcr_vec is not None:
            pmhc_tcr_vec = torch.cat([pmhc_vec, tcr_vec], dim=-1)
            return self.with_tcr(pmhc_tcr_vec)
        else:
            return self.without_tcr(pmhc_vec)


class TCellAssayHead(nn.Module):
    """Context-conditioned T-cell assay head.

    Models observed assay outcome as:
        biological immunogenicity latent + assay context bias.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_assay_methods: int = 11,
        n_assay_readouts: int = 15,
        n_apc_types: int = 9,
        n_culture_contexts: int = 9,
        n_stim_contexts: int = 6,
    ):
        super().__init__()
        ctx_dim = max(8, d_model // 8)

        self.assay_method_embed = nn.Embedding(n_assay_methods, ctx_dim)
        self.assay_readout_embed = nn.Embedding(n_assay_readouts, ctx_dim)
        self.apc_type_embed = nn.Embedding(n_apc_types, ctx_dim)
        self.culture_context_embed = nn.Embedding(n_culture_contexts, ctx_dim)
        self.stim_context_embed = nn.Embedding(n_stim_contexts, ctx_dim)

        self.base_with_tcr = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.base_without_tcr = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        ctx_in_dim = d_model + ctx_dim * 5
        self.context_bias = nn.Sequential(
            nn.Linear(ctx_in_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.w_bio = nn.Parameter(torch.tensor(1.0))
        self.w_base = nn.Parameter(torch.tensor(0.6))
        self.w_ctx = nn.Parameter(torch.tensor(0.6))
        self.w_lineage = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.zeros(1))
        self.lineage_projection = nn.Sequential(
            nn.Linear(4, max(8, d_model // 8)),
            nn.GELU(),
            nn.Linear(max(8, d_model // 8), 1),
        )

        # Auxiliary heads: encourage latent/embedding compatibility with assay strata.
        self.assay_method_classifier = nn.Linear(d_model, n_assay_methods)
        self.assay_readout_classifier = nn.Linear(d_model, n_assay_readouts)
        self.apc_type_classifier = nn.Linear(d_model, n_apc_types)
        self.culture_context_classifier = nn.Linear(d_model, n_culture_contexts)
        self.stim_context_classifier = nn.Linear(d_model, n_stim_contexts)

    def _context_embedding(
        self,
        assay_method_idx: Optional[torch.Tensor],
        assay_readout_idx: Optional[torch.Tensor],
        apc_type_idx: Optional[torch.Tensor],
        culture_context_idx: Optional[torch.Tensor],
        stim_context_idx: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        def _idx_or_zeros(idx: Optional[torch.Tensor]) -> torch.Tensor:
            if idx is None:
                return torch.zeros(batch_size, dtype=torch.long, device=device)
            return idx.to(device=device, dtype=torch.long).view(batch_size)

        method = self.assay_method_embed(_idx_or_zeros(assay_method_idx))
        readout = self.assay_readout_embed(_idx_or_zeros(assay_readout_idx))
        apc = self.apc_type_embed(_idx_or_zeros(apc_type_idx))
        culture = self.culture_context_embed(_idx_or_zeros(culture_context_idx))
        stim = self.stim_context_embed(_idx_or_zeros(stim_context_idx))
        return torch.cat([method, readout, apc, culture, stim], dim=-1)

    def forward(
        self,
        pmhc_vec: torch.Tensor,
        immunogenicity_logit: torch.Tensor,
        tcr_vec: Optional[torch.Tensor] = None,
        immunogenicity_cd4_logit: Optional[torch.Tensor] = None,
        immunogenicity_cd8_logit: Optional[torch.Tensor] = None,
        class_probs: Optional[torch.Tensor] = None,
        assay_method_idx: Optional[torch.Tensor] = None,
        assay_readout_idx: Optional[torch.Tensor] = None,
        apc_type_idx: Optional[torch.Tensor] = None,
        culture_context_idx: Optional[torch.Tensor] = None,
        stim_context_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Predict assay outcome and context-category logits."""
        if tcr_vec is not None:
            base = self.base_with_tcr(torch.cat([pmhc_vec, tcr_vec], dim=-1))
        else:
            base = self.base_without_tcr(pmhc_vec)

        batch_size = pmhc_vec.shape[0]
        ctx_emb = self._context_embedding(
            assay_method_idx=assay_method_idx,
            assay_readout_idx=assay_readout_idx,
            apc_type_idx=apc_type_idx,
            culture_context_idx=culture_context_idx,
            stim_context_idx=stim_context_idx,
            batch_size=batch_size,
            device=pmhc_vec.device,
        )
        ctx_bias = self.context_bias(torch.cat([pmhc_vec, ctx_emb], dim=-1))

        ig = immunogenicity_logit
        if ig.ndim == 1:
            ig = ig.unsqueeze(-1)
        logit = (
            F.softplus(self.w_bio) * ig
            + F.softplus(self.w_base) * base
            + F.softplus(self.w_ctx) * ctx_bias
            + self.bias
        )
        # Optional lineage-aware signal: lets non-sorted assays consume
        # both CD4 and CD8 immunogenicity latents with class-mixture context.
        if immunogenicity_cd4_logit is not None and immunogenicity_cd8_logit is not None:
            cd4 = immunogenicity_cd4_logit
            cd8 = immunogenicity_cd8_logit
            if cd4.ndim == 1:
                cd4 = cd4.unsqueeze(-1)
            if cd8.ndim == 1:
                cd8 = cd8.unsqueeze(-1)

            if class_probs is None:
                p_i = torch.full_like(cd4, 0.5)
                p_ii = torch.full_like(cd4, 0.5)
            else:
                p_i = class_probs[:, :1]
                p_ii = class_probs[:, 1:2]
            lineage_feat = torch.cat([cd4, cd8, p_i, p_ii], dim=-1)
            lineage_bias = self.lineage_projection(lineage_feat)
            logit = logit + F.softplus(self.w_lineage) * lineage_bias

        context_logits = {
            "assay_method": self.assay_method_classifier(pmhc_vec),
            "assay_readout": self.assay_readout_classifier(pmhc_vec),
            "apc_type": self.apc_type_classifier(pmhc_vec),
            "culture_context": self.culture_context_classifier(pmhc_vec),
            "stim_context": self.stim_context_classifier(pmhc_vec),
        }
        return logit, context_logits


# --------------------------------------------------------------------------
# Elution/MS head
# --------------------------------------------------------------------------

class ElutionHead(nn.Module):
    """Predicts elution/MS detection probability."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.context_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.w_context = nn.Parameter(torch.tensor(1.0))
        self.w_presentation = nn.Parameter(torch.tensor(1.0))
        self.w_processing = nn.Parameter(torch.tensor(0.4))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pmhc_vec: torch.Tensor,
        presentation_logit: Optional[torch.Tensor] = None,
        processing_logit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict detection logit.

        Args:
            pmhc_vec: pMHC vector representation (batch, d_model)
            presentation_logit: Optional upstream presentation logit (batch, 1)
            processing_logit: Optional upstream processing logit (batch, 1)

        Returns:
            Detection logit (batch, 1)
        """
        out = F.softplus(self.w_context) * self.context_head(pmhc_vec) + self.bias
        if presentation_logit is not None:
            out = out + F.softplus(self.w_presentation) * presentation_logit
        if processing_logit is not None:
            out = out + F.softplus(self.w_processing) * processing_logit
        return out


