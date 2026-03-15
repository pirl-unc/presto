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

from ..data.vocab import (
    BINDING_ASSAY_GEOMETRY,
    BINDING_ASSAY_METHODS,
    BINDING_ASSAY_PREP,
    BINDING_ASSAY_READOUT,
    BINDING_ASSAY_TYPES,
)
from .affinity import binding_logit_from_kd_log10
from .heads import AssayHeads, smooth_lower_bound, smooth_range_bound
from .pmhc import BindingModule


@dataclass
class PrestoTrunkState:
    """Shared pMHC trunk features consumed by modular predictor heads."""

    processing_vec: torch.Tensor
    interaction_vec: torch.Tensor
    pep_vec: torch.Tensor
    mhc_a_vec: torch.Tensor
    mhc_b_vec: torch.Tensor
    binding_affinity_vec: torch.Tensor
    binding_stability_vec: torch.Tensor
    recognition_vec: torch.Tensor
    processing_class1_vec: torch.Tensor
    processing_class2_vec: torch.Tensor
    presentation_class1_vec: torch.Tensor
    presentation_class2_vec: torch.Tensor
    immunogenicity_cd8_vec: torch.Tensor
    immunogenicity_cd4_vec: torch.Tensor
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
        d_model: int,
        interaction_dim: Optional[int] = None,
        max_log10_nM: float,
        binding_midpoint_nM: float,
        binding_log10_scale: float,
        affinity_assay_mode: str = "score_context",
        binding_kinetic_input_mode: str = "affinity_vec",
        affinity_assay_residual_mode: str = "legacy",
        kd_grouping_mode: str = "merged_kd",
        affinity_target_encoding: str = "log10",
        max_affinity_nM: float = 50000.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.interaction_dim = int(interaction_dim or d_model)
        self.binding_midpoint_nM = float(binding_midpoint_nM)
        self.binding_midpoint_log10_nM = math.log10(max(self.binding_midpoint_nM, 1e-12))
        self.binding_log10_scale = max(float(binding_log10_scale), 1e-6)
        self.max_log10_nM = float(max_log10_nM)
        affinity_assay_mode = str(affinity_assay_mode).strip().lower()
        if affinity_assay_mode not in {"legacy", "score_context"}:
            raise ValueError(f"Unsupported affinity_assay_mode: {affinity_assay_mode!r}")
        self.affinity_assay_mode = affinity_assay_mode
        binding_kinetic_input_mode = str(binding_kinetic_input_mode).strip().lower()
        if binding_kinetic_input_mode not in {"affinity_vec", "interaction_vec", "fused"}:
            raise ValueError(
                "Unsupported binding_kinetic_input_mode: "
                f"{binding_kinetic_input_mode!r}"
            )
        self.binding_kinetic_input_mode = binding_kinetic_input_mode
        affinity_assay_residual_mode = str(affinity_assay_residual_mode).strip().lower()
        if affinity_assay_residual_mode not in {
            "legacy",
            "pooled_single_output",
            "shared_base_segment_residual",
            "shared_base_factorized_context_residual",
            "shared_base_factorized_context_plus_segment_residual",
        }:
            raise ValueError(
                "Unsupported affinity_assay_residual_mode: "
                f"{affinity_assay_residual_mode!r}"
            )
        self.affinity_assay_residual_mode = affinity_assay_residual_mode
        kd_grouping_mode = str(kd_grouping_mode).strip().lower()
        if kd_grouping_mode not in {"merged_kd", "split_kd_proxy"}:
            raise ValueError(f"Unsupported kd_grouping_mode: {kd_grouping_mode!r}")
        self.kd_grouping_mode = kd_grouping_mode
        self.affinity_target_encoding = str(affinity_target_encoding).strip().lower()
        self.max_affinity_nM = float(max_affinity_nM)
        self.context_dim = max(8, d_model // 8)

        self.binding_affinity_probe = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.binding_stability_score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        if self.binding_kinetic_input_mode == "fused":
            self.binding_input_fuse = nn.Sequential(
                nn.Linear(self.interaction_dim + d_model, self.interaction_dim),
                nn.GELU(),
                nn.Linear(self.interaction_dim, self.interaction_dim),
            )
        else:
            self.binding_input_fuse = None
        binding_input_dim = (
            d_model if self.binding_kinetic_input_mode == "affinity_vec" else self.interaction_dim
        )
        self.binding = BindingModule(d_model=binding_input_dim)
        self.assay_type_embed = nn.Embedding(len(BINDING_ASSAY_TYPES), self.context_dim)
        self.assay_method_embed = nn.Embedding(len(BINDING_ASSAY_METHODS), self.context_dim)
        self.assay_context_proj = nn.Sequential(
            nn.Linear(self.context_dim * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.assay_factor_type_embed = nn.Embedding(len(BINDING_ASSAY_TYPES), self.context_dim)
        self.assay_prep_embed = nn.Embedding(len(BINDING_ASSAY_PREP), self.context_dim)
        self.assay_geometry_embed = nn.Embedding(len(BINDING_ASSAY_GEOMETRY), self.context_dim)
        self.assay_readout_embed = nn.Embedding(len(BINDING_ASSAY_READOUT), self.context_dim)
        self.factorized_assay_context_proj = nn.Sequential(
            nn.Linear(self.context_dim * 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.segment_summary_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.assay_heads = AssayHeads(
            d_model=d_model,
            max_log10_nM=self.max_log10_nM,
            max_affinity_nM=self.max_affinity_nM,
            affinity_target_encoding=self.affinity_target_encoding,
            assay_context_dim=(d_model if self.affinity_assay_mode == "score_context" else 0),
            affinity_assay_residual_mode=self.affinity_assay_residual_mode,
            segment_summary_dim=(
                d_model
                if self.affinity_assay_residual_mode in {
                    "shared_base_segment_residual",
                    "shared_base_factorized_context_plus_segment_residual",
                }
                else 0
            ),
            factorized_context_dim=(
                d_model
                if self.affinity_assay_residual_mode in {
                    "shared_base_factorized_context_residual",
                    "shared_base_factorized_context_plus_segment_residual",
                }
                else 0
            ),
            kd_grouping_mode=self.kd_grouping_mode,
        )
        if self.affinity_assay_residual_mode == "legacy":
            kd_bias_input_dim = d_model + (d_model if self.affinity_assay_mode == "score_context" else 0)
        elif self.affinity_assay_residual_mode == "shared_base_segment_residual":
            kd_bias_input_dim = d_model + (d_model if self.affinity_assay_mode == "score_context" else 0) + 1
        elif self.affinity_assay_residual_mode == "shared_base_factorized_context_residual":
            kd_bias_input_dim = d_model + 1
        elif self.affinity_assay_residual_mode == "shared_base_factorized_context_plus_segment_residual":
            kd_bias_input_dim = d_model + d_model + 1
        else:
            kd_bias_input_dim = 0
        self.kd_assay_bias = nn.Sequential(
            nn.Linear(kd_bias_input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        ) if kd_bias_input_dim > 0 else None
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
        pep_vec: Optional[torch.Tensor] = None,
        mhc_a_vec: Optional[torch.Tensor] = None,
        mhc_b_vec: Optional[torch.Tensor] = None,
        mhc_class: Optional[object] = None,
        binding_context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}

        if self.affinity_assay_mode == "score_context":
            assay_context_vec = self._encode_binding_context(
                binding_affinity_vec=binding_affinity_vec,
                binding_context=binding_context,
            )
        else:
            assay_context_vec = binding_affinity_vec.new_zeros(binding_affinity_vec.shape)
        outputs["binding_assay_context_vec"] = assay_context_vec
        factorized_assay_context_vec = self._encode_factorized_binding_context(
            binding_affinity_vec=binding_affinity_vec,
            binding_context=binding_context,
        )
        outputs["binding_factorized_assay_context_vec"] = factorized_assay_context_vec

        probe_kd = self.binding_affinity_probe(binding_affinity_vec)
        stability_score_raw = self.binding_stability_score_head(binding_stability_vec)
        stability_score = smooth_range_bound(stability_score_raw, -8.0, 8.0)
        outputs["binding_affinity_score_raw"] = probe_kd
        outputs["binding_affinity_score"] = smooth_range_bound(probe_kd, -3.0, 8.0)
        outputs["binding_affinity_probe_kd_raw"] = probe_kd
        outputs["binding_affinity_probe_kd"] = outputs["binding_affinity_score"]
        outputs["binding_stability_score_raw"] = stability_score_raw
        outputs["binding_stability_score"] = stability_score
        if (
            pep_vec is not None
            and mhc_a_vec is not None
            and mhc_b_vec is not None
        ):
            segment_summary_vec = self.segment_summary_proj(
                torch.cat([pep_vec, mhc_a_vec, mhc_b_vec], dim=-1)
            )
        else:
            segment_summary_vec = binding_affinity_vec.new_zeros(binding_affinity_vec.shape)
        outputs["binding_segment_summary_vec"] = segment_summary_vec

        if self.binding_kinetic_input_mode == "affinity_vec":
            binding_input = binding_affinity_vec
        elif self.binding_kinetic_input_mode == "interaction_vec":
            binding_input = interaction_vec
        else:
            binding_input = self.binding_input_fuse(
                torch.cat([interaction_vec, binding_affinity_vec], dim=-1)
            )
        outputs["binding_kinetic_input_mode"] = self.binding_kinetic_input_mode

        binding_latents = self.binding(
            binding_input,
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
        if self.affinity_assay_residual_mode == "shared_base_segment_residual":
            kd_bias_parts = [segment_summary_vec]
            if self.affinity_assay_mode == "score_context":
                kd_bias_parts.append(assay_context_vec)
            kd_bias_parts.append(outputs["binding_affinity_score"])
            kd_bias_input = torch.cat(kd_bias_parts, dim=-1)
        elif self.affinity_assay_residual_mode == "shared_base_factorized_context_residual":
            kd_bias_input = torch.cat(
                [factorized_assay_context_vec, outputs["binding_affinity_score"]],
                dim=-1,
            )
        elif self.affinity_assay_residual_mode == "shared_base_factorized_context_plus_segment_residual":
            kd_bias_input = torch.cat(
                [segment_summary_vec, factorized_assay_context_vec, outputs["binding_affinity_score"]],
                dim=-1,
            )
        elif self.affinity_assay_residual_mode == "pooled_single_output":
            kd_bias_input = None
        else:
            kd_bias_input = (
                torch.cat([binding_affinity_vec, assay_context_vec], dim=-1)
                if self.affinity_assay_mode == "score_context"
                else binding_affinity_vec
            )
        if self.kd_assay_bias is not None and kd_bias_input is not None:
            kd_bias_raw = self.kd_assay_bias(kd_bias_input)
            kd_bias_cap = F.softplus(self.kd_assay_bias_scale)
            kd_bias = F.softsign(kd_bias_raw) * kd_bias_cap
        else:
            kd_bias_raw = kd_from_binding.new_zeros(kd_from_binding.shape)
            kd_bias_cap = F.softplus(self.kd_assay_bias_scale)
            kd_bias = kd_from_binding.new_zeros(kd_from_binding.shape)
        outputs["binding_kd_bias_raw"] = kd_bias_raw
        outputs["binding_kd_bias"] = kd_bias
        outputs["binding_kd_bias_cap"] = kd_bias_cap.view(1)

        binding_probe_mix = torch.sigmoid(self.binding_probe_mix_logit)
        outputs["binding_probe_mix_weight"] = binding_probe_mix.view(1)

        assays = self.assay_heads(
            binding_affinity_vec,
            binding_stability_vec,
            binding_latents=binding_latents,
            binding_affinity_score=(
                outputs["binding_affinity_score"]
                if (
                    self.affinity_assay_mode == "score_context"
                    or self.affinity_assay_residual_mode == "shared_base_segment_residual"
                    or self.affinity_assay_residual_mode
                    in {
                        "shared_base_factorized_context_residual",
                        "shared_base_factorized_context_plus_segment_residual",
                    }
                )
                else None
            ),
            binding_stability_score=(
                outputs["binding_stability_score"]
                if self.affinity_assay_mode == "score_context"
                else None
            ),
            assay_context_vec=(
                assay_context_vec
                if self.affinity_assay_mode == "score_context"
                else None
            ),
            factorized_assay_context_vec=(
                factorized_assay_context_vec
                if self.affinity_assay_residual_mode
                in {
                    "shared_base_factorized_context_residual",
                    "shared_base_factorized_context_plus_segment_residual",
                }
                else None
            ),
            segment_summary_vec=(
                segment_summary_vec
                if self.affinity_assay_residual_mode
                in {
                    "shared_base_segment_residual",
                    "shared_base_factorized_context_plus_segment_residual",
                }
                else None
            ),
        )
        core_kd_log10 = smooth_lower_bound(kd_from_binding + kd_bias, -3.0)
        outputs["binding_core_kd_log10"] = core_kd_log10
        kd_log10 = (1.0 - binding_probe_mix) * core_kd_log10 + (
            binding_probe_mix * outputs["binding_affinity_probe_kd"]
        )
        outputs["binding_mixed_kd_log10"] = kd_log10
        affinity_obs = self.assay_heads.derive_affinity_observables(
            binding_affinity_vec,
            kd_log10,
            assay_context_vec=(
                assay_context_vec
                if self.affinity_assay_mode == "score_context"
                else None
            ),
            binding_affinity_score=(
                outputs["binding_affinity_score"]
                if (
                    self.affinity_assay_mode == "score_context"
                    or self.affinity_assay_residual_mode == "shared_base_segment_residual"
                    or self.affinity_assay_residual_mode
                    in {
                        "shared_base_factorized_context_residual",
                        "shared_base_factorized_context_plus_segment_residual",
                    }
                )
                else None
            ),
            factorized_assay_context_vec=(
                factorized_assay_context_vec
                if self.affinity_assay_residual_mode
                in {
                    "shared_base_factorized_context_residual",
                    "shared_base_factorized_context_plus_segment_residual",
                }
                else None
            ),
            segment_summary_vec=(
                segment_summary_vec
                if self.affinity_assay_residual_mode
                in {
                    "shared_base_segment_residual",
                    "shared_base_factorized_context_plus_segment_residual",
                }
                else None
            ),
        )
        assays.update(affinity_obs)
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

    def _encode_binding_context(
        self,
        *,
        binding_affinity_vec: torch.Tensor,
        binding_context: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        batch_size = int(binding_affinity_vec.shape[0])
        device = binding_affinity_vec.device
        if isinstance(binding_context, dict):
            assay_type_idx = binding_context.get("assay_type_idx")
            assay_method_idx = binding_context.get("assay_method_idx")
        else:
            assay_type_idx = None
            assay_method_idx = None

        if not isinstance(assay_type_idx, torch.Tensor):
            assay_type_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            assay_type_idx = assay_type_idx.to(device=device, dtype=torch.long).reshape(batch_size)
        if not isinstance(assay_method_idx, torch.Tensor):
            assay_method_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            assay_method_idx = assay_method_idx.to(device=device, dtype=torch.long).reshape(batch_size)

        assay_type_vec = self.assay_type_embed(assay_type_idx)
        assay_method_vec = self.assay_method_embed(assay_method_idx)
        return self.assay_context_proj(torch.cat([assay_type_vec, assay_method_vec], dim=-1))

    def _encode_factorized_binding_context(
        self,
        *,
        binding_affinity_vec: torch.Tensor,
        binding_context: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        batch_size = int(binding_affinity_vec.shape[0])
        device = binding_affinity_vec.device

        def _index_from_context(name: str) -> torch.Tensor:
            raw = binding_context.get(name) if isinstance(binding_context, dict) else None
            if not isinstance(raw, torch.Tensor):
                return torch.zeros(batch_size, dtype=torch.long, device=device)
            return raw.to(device=device, dtype=torch.long).reshape(batch_size)

        assay_type_idx = _index_from_context("assay_type_idx")
        assay_prep_idx = _index_from_context("assay_prep_idx")
        assay_geometry_idx = _index_from_context("assay_geometry_idx")
        assay_readout_idx = _index_from_context("assay_readout_idx")

        parts = [
            self.assay_factor_type_embed(assay_type_idx),
            self.assay_prep_embed(assay_prep_idx),
            self.assay_geometry_embed(assay_geometry_idx),
            self.assay_readout_embed(assay_readout_idx),
        ]
        return self.factorized_assay_context_proj(torch.cat(parts, dim=-1))
