"""Modular pMHC predictor blocks used by Presto.

These modules mirror the high-level MHCflurry decomposition while still
operating on shared trunk features from the unified Presto model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .affinity import binding_logit_from_kd_log10
from .heads import AssayHeads, smooth_lower_bound, smooth_range_bound
from .pmhc import BindingModule


@dataclass
class PrestoTrunkState:
    """Shared pMHC trunk features consumed by modular predictor heads."""

    processing_vec: torch.Tensor
    interaction_vec: torch.Tensor
    binding_affinity_vec: torch.Tensor
    binding_stability_vec: torch.Tensor
    presentation_vec: torch.Tensor
    recognition_vec: torch.Tensor
    immunogenicity_vec: torch.Tensor
    pmhc_vec: torch.Tensor
    pep_vec: torch.Tensor
    mhc_a_vec: torch.Tensor
    mhc_b_vec: torch.Tensor
    groove_vec: torch.Tensor
    class_probs: torch.Tensor


class ClassProcessingPredictor(nn.Module):
    """One class-specific processing predictor."""

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 1)

    def forward(self, processing_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        logit = self.head(processing_vec)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
        }


class ClassPresentationPredictor(nn.Module):
    """One class-specific presentation predictor."""

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 1)

    def forward(self, presentation_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        logit = self.head(presentation_vec)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
        }


class AffinityPredictor(nn.Module):
    """Shared affinity predictor operating on interaction-driven trunk features."""

    def __init__(
        self,
        *,
        interaction_dim: int,
        d_model: int,
        max_log10_nM: float,
        binding_midpoint_nM: float,
        binding_log10_scale: float,
    ) -> None:
        super().__init__()
        self.binding_midpoint_nM = float(binding_midpoint_nM)
        self.binding_midpoint_log10_nM = math.log10(max(self.binding_midpoint_nM, 1e-12))
        self.binding_log10_scale = max(float(binding_log10_scale), 1e-6)
        self.max_log10_nM = float(max_log10_nM)

        self.binding_affinity_probe = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.binding = BindingModule(d_model=interaction_dim)
        self.assay_heads = AssayHeads(
            d_model=d_model,
            max_log10_nM=self.max_log10_nM,
        )
        self.kd_assay_bias = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.kd_assay_bias_scale = nn.Parameter(torch.tensor(-1.5))
        self.binding_probe_mix_logit = nn.Parameter(torch.tensor(math.log(3.0)))
        self.w_binding_class1_calibration = nn.Parameter(torch.tensor(0.2))
        self.w_binding_class2_calibration = nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        *,
        interaction_vec: torch.Tensor,
        binding_affinity_vec: torch.Tensor,
        binding_stability_vec: torch.Tensor,
        class_probs: torch.Tensor,
        mhc_class: Optional[object] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}

        probe_kd = self.binding_affinity_probe(binding_affinity_vec)
        outputs["binding_affinity_probe_kd_raw"] = probe_kd
        outputs["binding_affinity_probe_kd"] = smooth_range_bound(probe_kd, -3.0, 8.0)

        binding_latents = self.binding(
            interaction_vec,
            mhc_class=mhc_class,
            class_probs=class_probs,
        )
        outputs["binding_latents"] = binding_latents

        log_kd_per_sample = self.binding.derive_kd(binding_latents).squeeze(-1)
        binding_logit_from_core = binding_logit_from_kd_log10(
            log_kd_per_sample,
            midpoint_nM=self.binding_midpoint_nM,
            log10_scale=self.binding_log10_scale,
        )
        binding_logit_from_core = smooth_range_bound(binding_logit_from_core, -20.0, 20.0)
        outputs["binding_logit_from_core"] = binding_logit_from_core

        kd_from_binding = (
            self.binding_midpoint_log10_nM
            - self.binding_log10_scale * binding_logit_from_core
        ).unsqueeze(-1)
        kd_bias_raw = self.kd_assay_bias(binding_affinity_vec)
        kd_bias_cap = F.softplus(self.kd_assay_bias_scale)
        kd_bias = F.softsign(kd_bias_raw) * kd_bias_cap
        outputs["binding_kd_bias_raw"] = kd_bias_raw
        outputs["binding_kd_bias"] = kd_bias
        outputs["binding_kd_bias_cap"] = kd_bias_cap.view(1)

        binding_probe_mix = torch.sigmoid(self.binding_probe_mix_logit)
        outputs["binding_probe_mix_weight"] = binding_probe_mix.view(1)

        assays = self.assay_heads(
            binding_affinity_vec,
            binding_stability_vec,
            binding_latents=binding_latents,
        )
        core_kd_log10 = smooth_lower_bound(kd_from_binding + kd_bias, -3.0)
        outputs["binding_core_kd_log10"] = core_kd_log10
        kd_log10 = (1.0 - binding_probe_mix) * core_kd_log10 + (
            binding_probe_mix * outputs["binding_affinity_probe_kd"]
        )
        outputs["binding_mixed_kd_log10"] = kd_log10
        affinity_obs = self.assay_heads.derive_affinity_observables(binding_affinity_vec, kd_log10)
        assays["KD_nM"] = affinity_obs["KD_nM"]
        assays["IC50_nM"] = affinity_obs["IC50_nM"]
        assays["EC50_nM"] = affinity_obs["EC50_nM"]
        outputs["assays"] = assays

        binding_base_logit = binding_logit_from_kd_log10(
            assays["KD_nM"].squeeze(-1),
            midpoint_nM=self.binding_midpoint_nM,
            log10_scale=self.binding_log10_scale,
        )
        binding_base_logit = smooth_range_bound(binding_base_logit, -20.0, 20.0)
        class_margin = class_probs[:, :1] - class_probs[:, 1:2]
        binding_class1_logit = (
            binding_base_logit.unsqueeze(-1)
            + F.softplus(self.w_binding_class1_calibration) * class_margin
        )
        binding_class2_logit = (
            binding_base_logit.unsqueeze(-1)
            - F.softplus(self.w_binding_class2_calibration) * class_margin
        )
        binding_logit = (
            class_probs[:, :1] * binding_class1_logit
            + class_probs[:, 1:2] * binding_class2_logit
        ).squeeze(-1)

        outputs["binding_base_logit"] = binding_base_logit
        outputs["binding_class1_logit"] = binding_class1_logit
        outputs["binding_class2_logit"] = binding_class2_logit
        outputs["binding_logit"] = binding_logit
        outputs["binding_mixed_logit"] = binding_logit.unsqueeze(-1)
        outputs["binding_class1_prob"] = torch.sigmoid(binding_class1_logit)
        outputs["binding_class2_prob"] = torch.sigmoid(binding_class2_logit)
        outputs["binding_prob"] = torch.sigmoid(binding_logit)
        outputs["binding_mixed_prob"] = outputs["binding_prob"].unsqueeze(-1)
        return outputs
