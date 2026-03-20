"""Prediction heads for IEDB measurements.

Task-specific heads that sit on top of the shared encoder:
- Binding: KD, IC50, EC50 (log10 nM)
- Kinetics: kon (log10 M⁻¹s⁻¹), koff (log10 s⁻¹)
- Stability: t1/2 (log10 min), Tm (normalized Celsius)
- T-cell: functional assay outcomes
- Elution/MS: detection probability
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .affinity import (
    DEFAULT_MAX_AFFINITY_NM,
    affinity_log10_to_target_logit,
    affinity_target_logit_to_log10,
)
from ..data.vocab import (
    BINDING_ASSAY_METHODS,
    BINDING_ASSAY_PREP,
    BINDING_ASSAY_READOUT,
)


AFFINITY_DAG_FAMILY_MODES = {
    "dag_family",
    "dag_method_leaf",
    "dag_prep_readout_leaf",
}


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


def smooth_lower_bound(value: torch.Tensor, min_value: float) -> torch.Tensor:
    """Apply a smooth lower bound that preserves gradient near the floor."""
    min_tensor = value.new_tensor(min_value)
    return min_tensor + F.softplus(value - min_tensor)


def smooth_range_bound(value: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """Apply smooth lower/upper bounds sequentially."""
    return smooth_upper_bound(smooth_lower_bound(value, min_value), max_value)


def noisy_or_logit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Combine two logits via noisy-OR in probability space, return logit.

    P(A or B) = P(A) + P(B) - P(A)*P(B)
    """
    p_a = torch.sigmoid(a)
    p_b = torch.sigmoid(b)
    p_or = (p_a + p_b - p_a * p_b).clamp(1e-7, 1 - 1e-7)
    return torch.logit(p_or)


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

    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, max(1, input_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, input_dim // 2), 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class TmHead(nn.Module):
    """Predicts Tm (melting temperature) in normalized Celsius."""

    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, max(1, input_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, input_dim // 2), 1),
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

    Canonical Presto uses these as output-side heads only. Per-example assay
    identity is not an allowed predictive input feature.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_log10_nM: float = math.log10(DEFAULT_MAX_AFFINITY_NM),
        max_affinity_nM: float = DEFAULT_MAX_AFFINITY_NM,
        affinity_target_encoding: str = "log10",
        assay_context_dim: int = 0,
        affinity_assay_residual_mode: str = "legacy",
        sequence_summary_dim: int = 0,
        factorized_context_dim: int = 0,
        kd_grouping_mode: str = "merged_kd",
        class_probs_dim: int = 0,
        species_probs_dim: int = 0,
    ):
        super().__init__()
        self.max_log10_nM = float(max_log10_nM)
        self.max_affinity_nM = float(max_affinity_nM)
        self.affinity_target_encoding = str(affinity_target_encoding).strip().lower()
        self.assay_context_dim = max(int(assay_context_dim), 0)
        affinity_assay_residual_mode = str(affinity_assay_residual_mode).strip().lower()
        if affinity_assay_residual_mode not in {
            "legacy",
            "pooled_single_output",
            "shared_base_segment_residual",
            "shared_base_factorized_context_residual",
            "shared_base_factorized_context_plus_segment_residual",
            "dag_family",
            "dag_method_leaf",
            "dag_prep_readout_leaf",
        }:
            raise ValueError(
                f"Unsupported affinity_assay_residual_mode: {affinity_assay_residual_mode!r}"
            )
        self.affinity_assay_residual_mode = affinity_assay_residual_mode
        self.sequence_summary_dim = max(int(sequence_summary_dim), 0)
        self.factorized_context_dim = max(int(factorized_context_dim), 0)
        self.class_probs_dim = max(int(class_probs_dim), 0)
        self.species_probs_dim = max(int(species_probs_dim), 0)
        self._conditioning_probs_dim = self.class_probs_dim + self.species_probs_dim
        self._uses_dag_family = self.affinity_assay_residual_mode in AFFINITY_DAG_FAMILY_MODES
        kd_grouping_mode = str(kd_grouping_mode).strip().lower()
        if kd_grouping_mode not in {"merged_kd", "split_kd_proxy"}:
            raise ValueError(f"Unsupported kd_grouping_mode: {kd_grouping_mode!r}")
        self.kd_grouping_mode = kd_grouping_mode
        self.stability_score_dim = 1
        if self.affinity_assay_residual_mode == "pooled_single_output":
            residual_input_dim = 0
        elif self.affinity_assay_residual_mode == "shared_base_segment_residual":
            residual_input_dim = self.sequence_summary_dim + self.assay_context_dim + self._conditioning_probs_dim + 1
        elif self.affinity_assay_residual_mode == "shared_base_factorized_context_residual":
            residual_input_dim = self.factorized_context_dim + self._conditioning_probs_dim + 1
        elif self.affinity_assay_residual_mode in {
            "shared_base_factorized_context_plus_segment_residual",
            "dag_family",
            "dag_method_leaf",
            "dag_prep_readout_leaf",
        }:
            residual_input_dim = self.sequence_summary_dim + self.factorized_context_dim + self._conditioning_probs_dim + 1
        else:
            residual_input_dim = d_model + self.assay_context_dim
        stability_input_dim = d_model + self.stability_score_dim
        if residual_input_dim > 0:
            if self._uses_dag_family:
                self.ic50_family_residual = nn.Sequential(
                    nn.Linear(residual_input_dim, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 1),
                )
                self.ec50_family_residual = nn.Sequential(
                    nn.Linear(residual_input_dim, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 1),
                )
                self.ic50_leaf_residual = nn.Sequential(
                    nn.Linear(residual_input_dim, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 1),
                )
                self.ec50_leaf_residual = nn.Sequential(
                    nn.Linear(residual_input_dim, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 1),
                )
                if self.kd_grouping_mode == "split_kd_proxy":
                    self.kd_proxy_ic50_leaf_residual = nn.Sequential(
                        nn.Linear(residual_input_dim, d_model // 2),
                        nn.GELU(),
                        nn.Linear(d_model // 2, 1),
                    )
                    self.kd_proxy_ec50_leaf_residual = nn.Sequential(
                        nn.Linear(residual_input_dim, d_model // 2),
                        nn.GELU(),
                        nn.Linear(d_model // 2, 1),
                    )
                else:
                    self.kd_proxy_ic50_leaf_residual = None
                    self.kd_proxy_ec50_leaf_residual = None
                if self.affinity_assay_residual_mode == "dag_method_leaf":
                    self.ic50_method_leaf_residuals = nn.ModuleDict(
                        {
                            name: nn.Sequential(
                                nn.Linear(residual_input_dim, d_model // 2),
                                nn.GELU(),
                                nn.Linear(d_model // 2, 1),
                            )
                            for name in BINDING_ASSAY_METHODS
                        }
                    )
                    self.ec50_method_leaf_residuals = nn.ModuleDict(
                        {
                            name: nn.Sequential(
                                nn.Linear(residual_input_dim, d_model // 2),
                                nn.GELU(),
                                nn.Linear(d_model // 2, 1),
                            )
                            for name in BINDING_ASSAY_METHODS
                        }
                    )
                else:
                    self.ic50_method_leaf_residuals = None
                    self.ec50_method_leaf_residuals = None
                if self.affinity_assay_residual_mode == "dag_prep_readout_leaf":
                    self.ic50_prep_leaf_residuals = nn.ModuleDict(
                        {
                            name: nn.Sequential(
                                nn.Linear(residual_input_dim, d_model // 2),
                                nn.GELU(),
                                nn.Linear(d_model // 2, 1),
                            )
                            for name in BINDING_ASSAY_PREP
                        }
                    )
                    self.ic50_readout_leaf_residuals = nn.ModuleDict(
                        {
                            name: nn.Sequential(
                                nn.Linear(residual_input_dim, d_model // 2),
                                nn.GELU(),
                                nn.Linear(d_model // 2, 1),
                            )
                            for name in BINDING_ASSAY_READOUT
                        }
                    )
                    self.ec50_prep_leaf_residuals = nn.ModuleDict(
                        {
                            name: nn.Sequential(
                                nn.Linear(residual_input_dim, d_model // 2),
                                nn.GELU(),
                                nn.Linear(d_model // 2, 1),
                            )
                            for name in BINDING_ASSAY_PREP
                        }
                    )
                    self.ec50_readout_leaf_residuals = nn.ModuleDict(
                        {
                            name: nn.Sequential(
                                nn.Linear(residual_input_dim, d_model // 2),
                                nn.GELU(),
                                nn.Linear(d_model // 2, 1),
                            )
                            for name in BINDING_ASSAY_READOUT
                        }
                    )
                else:
                    self.ic50_prep_leaf_residuals = None
                    self.ic50_readout_leaf_residuals = None
                    self.ec50_prep_leaf_residuals = None
                    self.ec50_readout_leaf_residuals = None
                self.ic50_residual = None
                self.ec50_residual = None
                self.kd_proxy_ic50_residual = None
                self.kd_proxy_ec50_residual = None
            else:
                self.ic50_family_residual = None
                self.ec50_family_residual = None
                self.ic50_leaf_residual = None
                self.ec50_leaf_residual = None
                self.kd_proxy_ic50_leaf_residual = None
                self.kd_proxy_ec50_leaf_residual = None
                self.ic50_method_leaf_residuals = None
                self.ec50_method_leaf_residuals = None
                self.ic50_prep_leaf_residuals = None
                self.ic50_readout_leaf_residuals = None
                self.ec50_prep_leaf_residuals = None
                self.ec50_readout_leaf_residuals = None
                self.ic50_residual = nn.Sequential(
                    nn.Linear(residual_input_dim, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 1),
                )
                self.ec50_residual = nn.Sequential(
                    nn.Linear(residual_input_dim, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 1),
                )
                if self.kd_grouping_mode == "split_kd_proxy":
                    self.kd_proxy_ic50_residual = nn.Sequential(
                        nn.Linear(residual_input_dim, d_model // 2),
                        nn.GELU(),
                        nn.Linear(d_model // 2, 1),
                    )
                    self.kd_proxy_ec50_residual = nn.Sequential(
                        nn.Linear(residual_input_dim, d_model // 2),
                        nn.GELU(),
                        nn.Linear(d_model // 2, 1),
                    )
                else:
                    self.kd_proxy_ic50_residual = None
                    self.kd_proxy_ec50_residual = None
        else:
            self.ic50_family_residual = None
            self.ec50_family_residual = None
            self.ic50_leaf_residual = None
            self.ec50_leaf_residual = None
            self.kd_proxy_ic50_leaf_residual = None
            self.kd_proxy_ec50_leaf_residual = None
            self.ic50_method_leaf_residuals = None
            self.ec50_method_leaf_residuals = None
            self.ic50_prep_leaf_residuals = None
            self.ic50_readout_leaf_residuals = None
            self.ec50_prep_leaf_residuals = None
            self.ec50_readout_leaf_residuals = None
            self.ic50_residual = None
            self.ec50_residual = None
            self.kd_proxy_ic50_residual = None
            self.kd_proxy_ec50_residual = None
        self.t_half_residual = nn.Sequential(
            nn.Linear(stability_input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.t_half_residual_scale = nn.Parameter(torch.tensor(-2.0))
        # Tm has complex relationship - predict directly
        self.tm = TmHead(stability_input_dim)

    def forward(
        self,
        binding_affinity_vec: torch.Tensor,
        binding_stability_vec: torch.Tensor,
        binding_latents: Optional[Dict[str, torch.Tensor]] = None,
        *,
        binding_affinity_score: Optional[torch.Tensor] = None,
        binding_stability_score: Optional[torch.Tensor] = None,
        assay_context_vec: Optional[torch.Tensor] = None,
        factorized_assay_context_vec: Optional[torch.Tensor] = None,
        sequence_summary_vec: Optional[torch.Tensor] = None,
        class_probs: Optional[torch.Tensor] = None,
        species_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict all assay values.

        Args:
            binding_affinity_vec: Binding affinity latent vector (batch, d_model)
            binding_stability_vec: Binding stability latent vector (batch, d_model)
            binding_latents: Dict with log_koff, log_kon_intrinsic, log_kon_chaperone
                           If provided, KD/koff/kon/t_half are derived from these.

        Returns:
            Dict with predictions for each assay type (all in log10 scale except Tm)
        """
        results = {}
        stability_input = self._stability_input(
            binding_stability_vec,
            binding_stability_score=binding_stability_score,
        )

        if binding_latents is not None:
            # Derive kinetic values from latents (physics-based)
            log_koff = smooth_range_bound(binding_latents["log_koff"], -8.0, 8.0)
            log_kon_intrinsic = smooth_range_bound(
                binding_latents["log_kon_intrinsic"], -8.0, 8.0
            )
            log_kon_chaperone = smooth_range_bound(
                binding_latents["log_kon_chaperone"], -8.0, 8.0
            )

            # kon_total = kon_intrinsic + kon_chaperone
            kon_intrinsic = torch.pow(10, log_kon_intrinsic)
            kon_chaperone = torch.pow(10, log_kon_chaperone)
            kon_total = kon_intrinsic + kon_chaperone
            log_kon_total = torch.log10(kon_total.clamp(min=1e-10, max=1e10))

            # KD = koff / kon (in M), convert to nM (+9).
            # Keep a smooth upper cap so weak-affinity regions still backpropagate.
            log_kd_nM = smooth_lower_bound(log_koff - log_kon_total + 9, -3.0)
            affinity_obs = self.derive_affinity_observables(
                binding_affinity_vec,
                log_kd_nM,
                assay_context_vec=assay_context_vec,
                binding_affinity_score=binding_affinity_score,
                factorized_assay_context_vec=factorized_assay_context_vec,
                sequence_summary_vec=sequence_summary_vec,
                class_probs=class_probs,
                species_probs=species_probs,
            )

            # t_half = ln(2) / koff, convert to minutes
            # log10(t_half_min) = log10(ln(2)/60) - log_koff = -1.937 - log_koff
            log_t_half = smooth_range_bound(-1.937 - log_koff, -8.0, 8.0)
            t_half_bias = self._bounded_residual(
                self.t_half_residual(stability_input),
                self.t_half_residual_scale,
            )

            results["koff"] = log_koff
            results["kon"] = log_kon_total
            results.update(affinity_obs)
            results["t_half"] = smooth_range_bound(log_t_half + t_half_bias, -8.0, 8.0)
        else:
            # Fallback: predict everything directly (less constrained)
            affinity_input = self._affinity_input(
                binding_affinity_vec,
                assay_context_vec=assay_context_vec,
                binding_affinity_score=binding_affinity_score,
                factorized_assay_context_vec=factorized_assay_context_vec,
                sequence_summary_vec=sequence_summary_vec,
                class_probs=class_probs,
                species_probs=species_probs,
            )
            results["KD_nM"] = smooth_upper_bound(
                smooth_lower_bound(self._direct_predict(affinity_input, "kd"), -3.0),
                self.max_log10_nM,
            )
            results["IC50_nM"] = smooth_upper_bound(
                smooth_lower_bound(self._direct_predict(affinity_input, "ic50"), -3.0),
                self.max_log10_nM,
            )
            results["EC50_nM"] = smooth_upper_bound(
                smooth_lower_bound(self._direct_predict(affinity_input, "ec50"), -3.0),
                self.max_log10_nM,
            )
            if self.kd_grouping_mode == "split_kd_proxy":
                results["KD_proxy_ic50_nM"] = smooth_upper_bound(
                    smooth_lower_bound(self._direct_predict(affinity_input, "kd_proxy_ic50"), -3.0),
                    self.max_log10_nM,
                )
                results["KD_proxy_ec50_nM"] = smooth_upper_bound(
                    smooth_lower_bound(self._direct_predict(affinity_input, "kd_proxy_ec50"), -3.0),
                    self.max_log10_nM,
                )
            else:
                results["KD_proxy_ic50_nM"] = results["KD_nM"]
                results["KD_proxy_ec50_nM"] = results["KD_nM"]
            results["kon"] = self._direct_predict(affinity_input, "kon")
            results["koff"] = self._direct_predict(affinity_input, "koff")
            results["t_half"] = self._direct_predict(stability_input, "t_half")

        # Tm always predicted directly (complex protein stability)
        results["Tm"] = self.tm(stability_input)

        return results

    def derive_affinity_observables(
        self,
        binding_affinity_vec: torch.Tensor,
        kd_log10_nM: torch.Tensor,
        assay_context_vec: Optional[torch.Tensor] = None,
        binding_affinity_score: Optional[torch.Tensor] = None,
        factorized_assay_context_vec: Optional[torch.Tensor] = None,
        sequence_summary_vec: Optional[torch.Tensor] = None,
        class_probs: Optional[torch.Tensor] = None,
        species_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Derive KD/IC50/EC50 from a shared KD latent with assay-specific bias."""
        kd_base = smooth_upper_bound(
            smooth_lower_bound(kd_log10_nM, -3.0),
            self.max_log10_nM,
        )
        if self.affinity_assay_residual_mode == "pooled_single_output":
            if self.kd_grouping_mode == "split_kd_proxy":
                return {
                    "KD_nM": kd_base,
                    "KD_proxy_ic50_nM": kd_base,
                    "KD_proxy_ec50_nM": kd_base,
                    "IC50_nM": kd_base,
                    "EC50_nM": kd_base,
                }
            return {
                "KD_nM": kd_base,
                "KD_proxy_ic50_nM": kd_base,
                "KD_proxy_ec50_nM": kd_base,
                "IC50_nM": kd_base,
                "EC50_nM": kd_base,
            }
        residual_input = self._affinity_input(
            binding_affinity_vec,
            assay_context_vec=assay_context_vec,
            binding_affinity_score=binding_affinity_score,
            factorized_assay_context_vec=factorized_assay_context_vec,
            sequence_summary_vec=sequence_summary_vec,
            class_probs=class_probs,
            species_probs=species_probs,
        )
        if self._uses_dag_family:
            return self._dag_affinity_observables(
                kd_base=kd_base,
                residual_input=residual_input,
            )
        kd_base_target_logit = self._target_logit(kd_base)
        ic50 = self._affinity_residual_output(
            base_log10=kd_base,
            base_target_logit=kd_base_target_logit,
            residual=self.ic50_residual(residual_input),
        )
        ec50 = self._affinity_residual_output(
            base_log10=kd_base,
            base_target_logit=kd_base_target_logit,
            residual=self.ec50_residual(residual_input),
        )
        outputs = {"KD_nM": kd_base, "IC50_nM": ic50, "EC50_nM": ec50}
        if self.kd_grouping_mode == "split_kd_proxy":
            outputs["KD_proxy_ic50_nM"] = self._affinity_residual_output(
                base_log10=kd_base,
                base_target_logit=kd_base_target_logit,
                residual=self.kd_proxy_ic50_residual(residual_input),
            )
            outputs["KD_proxy_ec50_nM"] = self._affinity_residual_output(
                base_log10=kd_base,
                base_target_logit=kd_base_target_logit,
                residual=self.kd_proxy_ec50_residual(residual_input),
            )
        else:
            outputs["KD_proxy_ic50_nM"] = kd_base
            outputs["KD_proxy_ec50_nM"] = kd_base
        return outputs

    def _dag_affinity_observables(
        self,
        *,
        kd_base: torch.Tensor,
        residual_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ic50_family = self._affinity_residual_output(
            base_log10=kd_base,
            residual=self.ic50_family_residual(residual_input),
        )
        ec50_family = self._affinity_residual_output(
            base_log10=kd_base,
            residual=self.ec50_family_residual(residual_input),
        )
        outputs: Dict[str, torch.Tensor] = {
            "KD_nM": kd_base,
            "IC50_family_anchor_nM": ic50_family,
            "EC50_family_anchor_nM": ec50_family,
            "IC50_nM": self._affinity_residual_output(
                base_log10=ic50_family,
                residual=self.ic50_leaf_residual(residual_input),
            ),
            "EC50_nM": self._affinity_residual_output(
                base_log10=ec50_family,
                residual=self.ec50_leaf_residual(residual_input),
            ),
        }
        if self.kd_grouping_mode == "split_kd_proxy":
            outputs["KD_proxy_ic50_nM"] = self._affinity_residual_output(
                base_log10=ic50_family,
                residual=self.kd_proxy_ic50_leaf_residual(residual_input),
            )
            outputs["KD_proxy_ec50_nM"] = self._affinity_residual_output(
                base_log10=ec50_family,
                residual=self.kd_proxy_ec50_leaf_residual(residual_input),
            )
        else:
            outputs["KD_proxy_ic50_nM"] = kd_base
            outputs["KD_proxy_ec50_nM"] = kd_base

        if self.affinity_assay_residual_mode == "dag_method_leaf":
            outputs.update(
                self._method_leaf_outputs(
                    base_name="IC50_nM",
                    family_anchor=ic50_family,
                    residual_input=residual_input,
                    head_bank=self.ic50_method_leaf_residuals,
                )
            )
            outputs.update(
                self._method_leaf_outputs(
                    base_name="EC50_nM",
                    family_anchor=ec50_family,
                    residual_input=residual_input,
                    head_bank=self.ec50_method_leaf_residuals,
                )
            )
        elif self.affinity_assay_residual_mode == "dag_prep_readout_leaf":
            outputs.update(
                self._prep_readout_leaf_outputs(
                    base_name="IC50_nM",
                    family_anchor=ic50_family,
                    residual_input=residual_input,
                    prep_bank=self.ic50_prep_leaf_residuals,
                    readout_bank=self.ic50_readout_leaf_residuals,
                )
            )
            outputs.update(
                self._prep_readout_leaf_outputs(
                    base_name="EC50_nM",
                    family_anchor=ec50_family,
                    residual_input=residual_input,
                    prep_bank=self.ec50_prep_leaf_residuals,
                    readout_bank=self.ec50_readout_leaf_residuals,
                )
            )
        return outputs

    def _method_leaf_outputs(
        self,
        *,
        base_name: str,
        family_anchor: torch.Tensor,
        residual_input: torch.Tensor,
        head_bank: nn.ModuleDict,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for method_name, head in head_bank.items():
            outputs[self.method_output_key(base_name, method_name)] = self._affinity_residual_output(
                base_log10=family_anchor,
                residual=head(residual_input),
            )
        return outputs

    def _prep_readout_leaf_outputs(
        self,
        *,
        base_name: str,
        family_anchor: torch.Tensor,
        residual_input: torch.Tensor,
        prep_bank: nn.ModuleDict,
        readout_bank: nn.ModuleDict,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        prep_residuals = {
            prep_name: head(residual_input)
            for prep_name, head in prep_bank.items()
        }
        readout_residuals = {
            readout_name: head(residual_input)
            for readout_name, head in readout_bank.items()
        }
        for prep_name, prep_residual in prep_residuals.items():
            for readout_name, readout_residual in readout_residuals.items():
                outputs[
                    self.prep_readout_output_key(base_name, prep_name, readout_name)
                ] = self._affinity_residual_output(
                    base_log10=family_anchor,
                    residual=prep_residual + readout_residual,
                )
        return outputs

    def _target_logit(self, base_log10: torch.Tensor) -> torch.Tensor:
        return affinity_log10_to_target_logit(
            base_log10,
            encoding=self.affinity_target_encoding,
            max_affinity_nM=self.max_affinity_nM,
        )

    def _affinity_residual_output(
        self,
        *,
        base_log10: torch.Tensor,
        residual: torch.Tensor,
        base_target_logit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.affinity_target_encoding == "log10":
            return smooth_upper_bound(
                smooth_lower_bound(base_log10 + residual, -3.0),
                self.max_log10_nM,
            )
        if base_target_logit is None:
            base_target_logit = self._target_logit(base_log10)
        adjusted_target_logit = base_target_logit + residual
        adjusted_log10 = affinity_target_logit_to_log10(
            adjusted_target_logit,
            encoding=self.affinity_target_encoding,
            max_affinity_nM=self.max_affinity_nM,
        )
        return smooth_upper_bound(
            smooth_lower_bound(adjusted_log10, -3.0),
            self.max_log10_nM,
        )

    @staticmethod
    def method_output_key(base_name: str, method_name: str) -> str:
        return f"{base_name}__method__{method_name}"

    @staticmethod
    def prep_readout_output_key(base_name: str, prep_name: str, readout_name: str) -> str:
        return f"{base_name}__prep__{prep_name}__readout__{readout_name}"

    def _conditioning_probs_parts(
        self,
        batch_size: int,
        device: torch.device,
        class_probs: Optional[torch.Tensor],
        species_probs: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Build list of conditioning probability tensors for residual input."""
        parts: List[torch.Tensor] = []
        if self.class_probs_dim > 0:
            if class_probs is not None:
                parts.append(class_probs.to(device=device, dtype=torch.float32))
            else:
                parts.append(torch.zeros(batch_size, self.class_probs_dim, device=device))
        if self.species_probs_dim > 0:
            if species_probs is not None:
                parts.append(species_probs.to(device=device, dtype=torch.float32))
            else:
                parts.append(torch.zeros(batch_size, self.species_probs_dim, device=device))
        return parts

    def _affinity_input(
        self,
        binding_affinity_vec: torch.Tensor,
        *,
        assay_context_vec: Optional[torch.Tensor] = None,
        binding_affinity_score: Optional[torch.Tensor] = None,
        factorized_assay_context_vec: Optional[torch.Tensor] = None,
        sequence_summary_vec: Optional[torch.Tensor] = None,
        class_probs: Optional[torch.Tensor] = None,
        species_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = binding_affinity_vec.shape[0]
        device = binding_affinity_vec.device
        cond_parts = self._conditioning_probs_parts(batch_size, device, class_probs, species_probs)

        if self.affinity_assay_residual_mode == "shared_base_segment_residual":
            if sequence_summary_vec is None:
                sequence_summary_vec = binding_affinity_vec.new_zeros(
                    batch_size,
                    self.sequence_summary_dim,
                )
            parts = [sequence_summary_vec]
            if self.assay_context_dim > 0:
                if assay_context_vec is None:
                    assay_context_vec = binding_affinity_vec.new_zeros(
                        batch_size,
                        self.assay_context_dim,
                    )
                parts.append(assay_context_vec)
            parts.extend(cond_parts)
            if binding_affinity_score is None:
                binding_affinity_score = binding_affinity_vec.new_zeros(batch_size, 1)
            else:
                binding_affinity_score = binding_affinity_score.reshape(batch_size, 1)
            parts.append(binding_affinity_score)
            return torch.cat(parts, dim=-1)
        if self.affinity_assay_residual_mode == "shared_base_factorized_context_residual":
            if factorized_assay_context_vec is None:
                factorized_assay_context_vec = binding_affinity_vec.new_zeros(
                    batch_size,
                    self.factorized_context_dim,
                )
            parts = [factorized_assay_context_vec]
            parts.extend(cond_parts)
            if binding_affinity_score is None:
                binding_affinity_score = binding_affinity_vec.new_zeros(batch_size, 1)
            else:
                binding_affinity_score = binding_affinity_score.reshape(batch_size, 1)
            parts.append(binding_affinity_score)
            return torch.cat(parts, dim=-1)
        if self.affinity_assay_residual_mode in {
            "shared_base_factorized_context_plus_segment_residual",
            "dag_family",
            "dag_method_leaf",
            "dag_prep_readout_leaf",
        }:
            if sequence_summary_vec is None:
                sequence_summary_vec = binding_affinity_vec.new_zeros(
                    batch_size,
                    self.sequence_summary_dim,
                )
            if factorized_assay_context_vec is None:
                factorized_assay_context_vec = binding_affinity_vec.new_zeros(
                    batch_size,
                    self.factorized_context_dim,
                )
            parts = [sequence_summary_vec, factorized_assay_context_vec]
            parts.extend(cond_parts)
            if binding_affinity_score is None:
                binding_affinity_score = binding_affinity_vec.new_zeros(batch_size, 1)
            else:
                binding_affinity_score = binding_affinity_score.reshape(batch_size, 1)
            parts.append(binding_affinity_score)
            return torch.cat(parts, dim=-1)

        residual_input = binding_affinity_vec
        if self.assay_context_dim > 0:
            if assay_context_vec is None:
                assay_context_vec = binding_affinity_vec.new_zeros(
                    batch_size,
                    self.assay_context_dim,
                )
            residual_input = torch.cat([residual_input, assay_context_vec], dim=-1)
        return residual_input

    def _stability_input(
        self,
        binding_stability_vec: torch.Tensor,
        *,
        binding_stability_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if binding_stability_score is None:
            binding_stability_score = binding_stability_vec.new_zeros(
                binding_stability_vec.shape[0],
                self.stability_score_dim,
            )
        else:
            binding_stability_score = binding_stability_score.reshape(
                binding_stability_vec.shape[0],
                self.stability_score_dim,
            )
        return torch.cat([binding_stability_vec, binding_stability_score], dim=-1)

    @staticmethod
    def _bounded_residual(raw: torch.Tensor, scale_param: torch.Tensor) -> torch.Tensor:
        cap = F.softplus(scale_param)
        return F.softsign(raw) * cap

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
    """Context-conditioned T-cell assay head (S10.3).

    Receives immunogenicity latent *vectors* and upstream logits — never pmhc_vec.
    Uses gated signal projection with context-dependent bias, and noisy-OR
    ambiguity mixing for class-ambiguous assays.

    Legacy note: this remains more context-conditioned than the current
    outputs-only assay contract. Do not treat it as permission to add new
    assay-selector inputs elsewhere in Presto.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_assay_methods: int = 11,
        n_assay_readouts: int = 15,
        n_apc_types: int = 9,
        n_culture_contexts: int = 9,
        n_stim_contexts: int = 6,
        n_peptide_formats: int = 6,
    ):
        super().__init__()
        ctx_dim = max(8, d_model // 8)
        self.ctx_dim = ctx_dim
        self.d_model = d_model

        self.assay_method_embed = nn.Embedding(n_assay_methods, ctx_dim)
        self.assay_readout_embed = nn.Embedding(n_assay_readouts, ctx_dim)
        self.apc_type_embed = nn.Embedding(n_apc_types, ctx_dim)
        self.culture_context_embed = nn.Embedding(n_culture_contexts, ctx_dim)
        self.stim_context_embed = nn.Embedding(n_stim_contexts, ctx_dim)
        self.peptide_format_embed = nn.Embedding(n_peptide_formats, ctx_dim)
        self.duration_default = nn.Embedding(n_culture_contexts, 1)
        self.duration_proj = nn.Sequential(
            nn.Linear(1, ctx_dim),
            nn.GELU(),
            nn.Linear(ctx_dim, ctx_dim),
        )

        self.n_assay_methods = n_assay_methods
        self.n_assay_readouts = n_assay_readouts
        self.n_apc_types = n_apc_types
        self.n_culture_contexts = n_culture_contexts
        self.n_stim_contexts = n_stim_contexts
        self.n_peptide_formats = n_peptide_formats

        # Context-conditioned projections from shared context embedding.
        self.bias_proj = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim),
            nn.GELU(),
            nn.Linear(ctx_dim, 1),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(ctx_dim, d_model),
        )
        self.signal_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.proc_gate = nn.Sequential(
            nn.Linear(ctx_dim, 1),
        )
        self.ambiguity_gate = nn.Sequential(
            nn.Linear(ctx_dim, 1),
        )

    def _idx_or_default(
        self,
        idx: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        n_values: int,
    ) -> torch.Tensor:
        if idx is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        x = idx.to(device=device, dtype=torch.long).view(batch_size)
        return x.clamp(min=0, max=max(n_values - 1, 0))

    def _resolve_duration_vec(
        self,
        culture_idx: torch.Tensor,
        culture_duration_hours: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = culture_idx.shape[0]
        device = culture_idx.device
        if culture_duration_hours is None:
            duration_hours = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
        else:
            duration_hours = culture_duration_hours.to(device=device, dtype=torch.float32).view(batch_size, 1)
        known_duration = (duration_hours > 0.0).float()
        default_log_dur = self.duration_default(culture_idx)
        log_duration = torch.log1p(duration_hours.clamp(min=0.0))
        mixed_log_duration = known_duration * log_duration + (1.0 - known_duration) * default_log_dur
        return self.duration_proj(mixed_log_duration)

    def _context_parts(
        self,
        assay_method_idx: Optional[torch.Tensor],
        assay_readout_idx: Optional[torch.Tensor],
        apc_type_idx: Optional[torch.Tensor],
        culture_context_idx: Optional[torch.Tensor],
        stim_context_idx: Optional[torch.Tensor],
        peptide_format_idx: Optional[torch.Tensor],
        culture_duration_hours: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        method_i = self._idx_or_default(
            assay_method_idx, batch_size, device, self.n_assay_methods
        )
        readout_i = self._idx_or_default(
            assay_readout_idx, batch_size, device, self.n_assay_readouts
        )
        apc_i = self._idx_or_default(
            apc_type_idx, batch_size, device, self.n_apc_types
        )
        culture_i = self._idx_or_default(
            culture_context_idx, batch_size, device, self.n_culture_contexts
        )
        stim_i = self._idx_or_default(
            stim_context_idx, batch_size, device, self.n_stim_contexts
        )
        pepfmt_i = self._idx_or_default(
            peptide_format_idx, batch_size, device, self.n_peptide_formats
        )

        return {
            "assay_method": self.assay_method_embed(method_i),
            "assay_readout": self.assay_readout_embed(readout_i),
            "apc_type": self.apc_type_embed(apc_i),
            "culture_context": self.culture_context_embed(culture_i),
            "stim_context": self.stim_context_embed(stim_i),
            "peptide_format": self.peptide_format_embed(pepfmt_i),
            "duration": self._resolve_duration_vec(culture_i, culture_duration_hours),
        }

    @staticmethod
    def _sum_parts(parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(parts.values())

    def _predict_with_ctx(
        self,
        ctx_vec: torch.Tensor,
        immunogenicity_cd8_vec: torch.Tensor,
        immunogenicity_cd4_vec: torch.Tensor,
        presentation_class1_logit: torch.Tensor,
        presentation_class2_logit: torch.Tensor,
        binding_class1_logit: torch.Tensor,
        binding_class2_logit: torch.Tensor,
        class_probs: torch.Tensor,
    ) -> torch.Tensor:
        panel_mode = ctx_vec.ndim == 3
        if panel_mode:
            batch_size, n_panel, _ = ctx_vec.shape
            ctx_flat = ctx_vec.reshape(batch_size * n_panel, self.ctx_dim)
            def _expand(t: torch.Tensor) -> torch.Tensor:
                if t.ndim == 1:
                    t = t.unsqueeze(-1)
                return t.unsqueeze(1).expand(-1, n_panel, -1).reshape(batch_size * n_panel, -1)

            cd8_vec = _expand(immunogenicity_cd8_vec)
            cd4_vec = _expand(immunogenicity_cd4_vec)
            p1 = _expand(presentation_class1_logit)
            p2 = _expand(presentation_class2_logit)
            b1 = _expand(binding_class1_logit)
            b2 = _expand(binding_class2_logit)
            cls = _expand(class_probs)
        else:
            ctx_flat = ctx_vec
            cd8_vec = immunogenicity_cd8_vec
            cd4_vec = immunogenicity_cd4_vec
            p1 = presentation_class1_logit
            p2 = presentation_class2_logit
            b1 = binding_class1_logit
            b2 = binding_class2_logit
            cls = class_probs

        bias = self.bias_proj(ctx_flat)
        gate = torch.sigmoid(self.gate_proj(ctx_flat))
        cd8_signal = self.signal_proj(gate * cd8_vec)
        cd4_signal = self.signal_proj(gate * cd4_vec)
        proc_weight = torch.sigmoid(self.proc_gate(ctx_flat))

        cd8_upstream = proc_weight * p1 + (1 - proc_weight) * b1
        cd4_upstream = proc_weight * p2 + (1 - proc_weight) * b2
        cd8_logit = cd8_upstream + cd8_signal + bias
        cd4_logit = cd4_upstream + cd4_signal + bias

        class_ambiguity = torch.sigmoid(self.ambiguity_gate(ctx_flat))
        known = cls[:, 0:1] * cd8_logit + cls[:, 1:2] * cd4_logit
        ambiguous = noisy_or_logit(cd8_logit, cd4_logit)
        out = (1 - class_ambiguity) * known + class_ambiguity * ambiguous

        if panel_mode:
            return out.view(batch_size, n_panel)
        return out

    def predict_panel(
        self,
        immunogenicity_cd8_vec: torch.Tensor,
        immunogenicity_cd4_vec: torch.Tensor,
        presentation_class1_logit: torch.Tensor,
        presentation_class2_logit: torch.Tensor,
        binding_class1_logit: torch.Tensor,
        binding_class2_logit: torch.Tensor,
        class_probs: torch.Tensor,
        assay_method_idx: Optional[torch.Tensor] = None,
        assay_readout_idx: Optional[torch.Tensor] = None,
        apc_type_idx: Optional[torch.Tensor] = None,
        culture_context_idx: Optional[torch.Tensor] = None,
        stim_context_idx: Optional[torch.Tensor] = None,
        peptide_format_idx: Optional[torch.Tensor] = None,
        culture_duration_hours: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict panel logits across categorical assay axes."""
        batch_size = immunogenicity_cd8_vec.shape[0]
        device = immunogenicity_cd8_vec.device
        parts = self._context_parts(
            assay_method_idx=assay_method_idx,
            assay_readout_idx=assay_readout_idx,
            apc_type_idx=apc_type_idx,
            culture_context_idx=culture_context_idx,
            stim_context_idx=stim_context_idx,
            peptide_format_idx=peptide_format_idx,
            culture_duration_hours=culture_duration_hours,
            batch_size=batch_size,
            device=device,
        )

        def _axis_logits(
            axis: str,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            fixed = sum(v for k, v in parts.items() if k != axis)
            ctx_axis = fixed.unsqueeze(1) + weight.unsqueeze(0)
            return self._predict_with_ctx(
                ctx_axis,
                immunogenicity_cd8_vec=immunogenicity_cd8_vec,
                immunogenicity_cd4_vec=immunogenicity_cd4_vec,
                presentation_class1_logit=presentation_class1_logit,
                presentation_class2_logit=presentation_class2_logit,
                binding_class1_logit=binding_class1_logit,
                binding_class2_logit=binding_class2_logit,
                class_probs=class_probs,
            )

        return {
            "assay_method": _axis_logits("assay_method", self.assay_method_embed.weight),
            "assay_readout": _axis_logits("assay_readout", self.assay_readout_embed.weight),
            "apc_type": _axis_logits("apc_type", self.apc_type_embed.weight),
            "culture_context": _axis_logits("culture_context", self.culture_context_embed.weight),
            "stim_context": _axis_logits("stim_context", self.stim_context_embed.weight),
            "peptide_format": _axis_logits("peptide_format", self.peptide_format_embed.weight),
        }

    def forward(
        self,
        immunogenicity_cd8_vec: torch.Tensor,
        immunogenicity_cd4_vec: torch.Tensor,
        presentation_class1_logit: torch.Tensor,
        presentation_class2_logit: torch.Tensor,
        binding_class1_logit: torch.Tensor,
        binding_class2_logit: torch.Tensor,
        class_probs: torch.Tensor,
        assay_method_idx: Optional[torch.Tensor] = None,
        assay_readout_idx: Optional[torch.Tensor] = None,
        apc_type_idx: Optional[torch.Tensor] = None,
        culture_context_idx: Optional[torch.Tensor] = None,
        stim_context_idx: Optional[torch.Tensor] = None,
        peptide_format_idx: Optional[torch.Tensor] = None,
        culture_duration_hours: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict T-cell assay outcome from immunogenicity vecs + upstream logits.

        Args:
            immunogenicity_cd8_vec: CD8 immunogenicity latent (batch, d_model)
            immunogenicity_cd4_vec: CD4 immunogenicity latent (batch, d_model)
            presentation_class1_logit: Class-I presentation logit (batch, 1)
            presentation_class2_logit: Class-II presentation logit (batch, 1)
            binding_class1_logit: Class-I binding logit (batch, 1)
            binding_class2_logit: Class-II binding logit (batch, 1)
            class_probs: MHC class probabilities (batch, 2) [p_I, p_II]
            assay_method_idx..stim_context_idx: optional context indices

        Returns:
            T-cell assay logit (batch, 1)
        """
        batch_size = immunogenicity_cd8_vec.shape[0]
        device = immunogenicity_cd8_vec.device

        parts = self._context_parts(
            assay_method_idx=assay_method_idx,
            assay_readout_idx=assay_readout_idx,
            apc_type_idx=apc_type_idx,
            culture_context_idx=culture_context_idx,
            stim_context_idx=stim_context_idx,
            peptide_format_idx=peptide_format_idx,
            culture_duration_hours=culture_duration_hours,
            batch_size=batch_size,
            device=device,
        )
        ctx_vec = self._sum_parts(parts)
        return self._predict_with_ctx(
            ctx_vec,
            immunogenicity_cd8_vec=immunogenicity_cd8_vec,
            immunogenicity_cd4_vec=immunogenicity_cd4_vec,
            presentation_class1_logit=presentation_class1_logit,
            presentation_class2_logit=presentation_class2_logit,
            binding_class1_logit=binding_class1_logit,
            binding_class2_logit=binding_class2_logit,
            class_probs=class_probs,
        )


# --------------------------------------------------------------------------
# Elution/MS head
# --------------------------------------------------------------------------

class ElutionHead(nn.Module):
    """Elution logit = presentation + MS detectability (S9.3).

    No pmhc_vec shortcut — information flows only through designated latent paths.
    """

    def __init__(self):
        super().__init__()
        self.w_presentation = nn.Parameter(torch.tensor(1.0))
        self.w_ms_detectability = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        presentation_logit: torch.Tensor,
        ms_detectability_logit: torch.Tensor,
    ) -> torch.Tensor:
        """Predict elution/detection logit.

        Args:
            presentation_logit: Upstream presentation logit (batch, 1)
            ms_detectability_logit: MS detectability logit (batch, 1)

        Returns:
            Detection logit (batch, 1)
        """
        return (
            F.softplus(self.w_presentation) * presentation_logit
            + F.softplus(self.w_ms_detectability) * ms_detectability_logit
            + self.bias
        )
