#!/usr/bin/env python
"""Assay head ablation experiments.

8 variants testing different head architectures on top of a shared
GrooveTransformerModel encoder. Determines which Presto binding head
components (kinetics, residual corrections, assay context, blend) add value.

Variants:
  a1  Single IC50 output (reference)
  a2  Measurement-type-routed multi-head
  a3  Shared base + residual correction (Presto-lite)
  a4  Physics-constrained kinetic decomposition
  a5  Assay context conditioning
  a6  Probe + kinetic blend
  a7  Multi-head + consistency regularization
  a8  Single head + type indicator feature
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from presto.data import PrestoCollator, PrestoDataset, create_dataloader
from presto.data.groove import prepare_mhc_input
from presto.data.tokenizer import Tokenizer
from presto.data.vocab import BINDING_ASSAY_TYPES, BINDING_ASSAY_METHODS
from presto.models.affinity import (
    DEFAULT_MAX_AFFINITY_NM,
    max_log10_nM,
    normalize_binding_target_log10,
)
from presto.models.heads import smooth_range_bound
from presto.scripts.focused_binding_probe import (
    DEFAULT_ALLELES,
    DEFAULT_PROBE_PEPTIDE,
    _augment_train_records_only,
    _balance_alleles,
    _create_focused_train_loader,
    _keep_binding_qualifier,
    _keep_measurement_type,
    _load_binding_records_from_merged_tsv,
    _normalize_binding_measurement,
    _require_target_allele_coverage,
    _resolve_allele_sequences,
    _select_fit_supported_probe_peptides,
    _split_csv,
    _split_records_by_peptide,
    _summarize_binding_records,
    _write_summary_artifacts,
    MEASUREMENT_PROFILES,
    MEASUREMENT_PROFILE_NUMERIC,
    NORMALIZED_MEASUREMENT_FILTERS,
    QUALIFIER_FILTERS,
)
from presto.scripts.groove_baseline_probe import (
    _find_allele_sequence,
    _verify_groove_representations,
)
from presto.scripts.train_iedb import ALL_SYNTHETIC_MODES, resolve_mhc_sequences_from_index
from presto.training.losses import censor_aware_loss


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANT_NAMES = ("a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8")
TYPED_VARIANTS = {"a2", "a3", "a7"}
CONTEXT_VARIANTS = {"a5", "a8"}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class AblationEncoder(nn.Module):
    """Shared transformer encoder returning concatenated pep+groove vector.

    Same architecture as GrooveTransformerModel but returns the intermediate
    shared_vec (B, 3*embed_dim) instead of a final scalar.
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
        B, L = tok.shape
        positions = torch.arange(L, device=tok.device).unsqueeze(0).expand(B, L)
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
        """Returns (B, 3*embed_dim) shared representation."""
        pep_vec = self._encode_segment(pep_tok)
        gh1_vec = self._encode_segment(mhc_a_tok)
        gh2_vec = self._encode_segment(mhc_b_tok)
        return torch.cat([pep_vec, gh1_vec, gh2_vec], dim=-1)


# ---------------------------------------------------------------------------
# Head variants
# ---------------------------------------------------------------------------


def _bounded_residual(raw: torch.Tensor, scale_param: torch.Tensor) -> torch.Tensor:
    """Softsign-bounded residual with learned scale (from AssayHeads)."""
    cap = F.softplus(scale_param)
    return F.softsign(raw) * cap


class A1_SingleIC50(nn.Module):
    """A1: encoder -> MLP -> log10(IC50). All types treated as interchangeable."""

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, shared_vec: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        raw = self.mlp(shared_vec)
        return {"ic50": smooth_range_bound(raw, -3.0, max_log10_nM())}


class A2_MultiTypeHead(nn.Module):
    """A2: 3 parallel MLP heads (ic50, kd, ec50). Route by measurement type."""

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.ic50_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.kd_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.ec50_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, shared_vec: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return {
            "ic50": smooth_range_bound(self.ic50_head(shared_vec), -3.0, max_log10_nM()),
            "kd": smooth_range_bound(self.kd_head(shared_vec), -3.0, max_log10_nM()),
            "ec50": smooth_range_bound(self.ec50_head(shared_vec), -3.0, max_log10_nM()),
        }


class A3_SharedBaseResidual(nn.Module):
    """A3: Shared base KD + bounded residual corrections for IC50/EC50.

    Tests the AssayHeads.derive_affinity_observables pattern in isolation.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.base_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.ic50_residual = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1),
        )
        self.ec50_residual = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1),
        )
        self.ic50_scale = nn.Parameter(torch.tensor(-2.0))
        self.ec50_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, shared_vec: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        kd_base = smooth_range_bound(self.base_head(shared_vec), -3.0, max_log10_nM())
        ic50 = smooth_range_bound(
            kd_base + _bounded_residual(self.ic50_residual(shared_vec), self.ic50_scale),
            -3.0, max_log10_nM(),
        )
        ec50 = smooth_range_bound(
            kd_base + _bounded_residual(self.ec50_residual(shared_vec), self.ec50_scale),
            -3.0, max_log10_nM(),
        )
        return {"ic50": ic50, "kd": kd_base, "ec50": ec50}


class A4_KineticDecomp(nn.Module):
    """A4: 3 kinetic heads -> physics-derived KD -> IC50 via residual.

    Tests whether the BindingModule kinetic constraint helps. All supervision
    flows through IC50; no direct kinetic supervision.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.head_log_koff = nn.Linear(in_dim, 1)
        self.head_log_kon_int = nn.Linear(in_dim, 1)
        self.head_log_kon_chap = nn.Linear(in_dim, 1)
        self.ic50_residual = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1),
        )
        self.ic50_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, shared_vec: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        log_koff = smooth_range_bound(self.head_log_koff(shared_vec), -8.0, 8.0)
        log_kon_int = smooth_range_bound(self.head_log_kon_int(shared_vec), -8.0, 8.0)
        log_kon_chap = smooth_range_bound(self.head_log_kon_chap(shared_vec), -8.0, 8.0)
        kon_total = torch.pow(10, log_kon_int) + torch.pow(10, log_kon_chap)
        log_kon_total = torch.log10(kon_total.clamp(min=1e-10, max=1e10))
        log_kd_nM = smooth_range_bound(log_koff - log_kon_total + 9, -3.0, max_log10_nM())
        resid = _bounded_residual(self.ic50_residual(shared_vec), self.ic50_scale)
        ic50 = smooth_range_bound(log_kd_nM + resid, -3.0, max_log10_nM())
        return {
            "ic50": ic50,
            "kd": log_kd_nM,
            "log_koff": log_koff,
            "log_kon": log_kon_total,
        }


class A5_AssayContext(nn.Module):
    """A5: A1 + assay type/method embeddings -> context-conditioned output.

    At eval, uses index 0 ('unknown') for both embeddings.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, ctx_dim: int = 16):
        super().__init__()
        self.type_embed = nn.Embedding(len(BINDING_ASSAY_TYPES), ctx_dim)
        self.method_embed = nn.Embedding(len(BINDING_ASSAY_METHODS), ctx_dim)
        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_dim * 2, ctx_dim * 2),
            nn.GELU(),
            nn.Linear(ctx_dim * 2, ctx_dim * 2),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + ctx_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        shared_vec: torch.Tensor,
        assay_type_idx: Optional[torch.Tensor] = None,
        assay_method_idx: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        B = shared_vec.shape[0]
        device = shared_vec.device
        if assay_type_idx is None:
            assay_type_idx = torch.zeros(B, dtype=torch.long, device=device)
        if assay_method_idx is None:
            assay_method_idx = torch.zeros(B, dtype=torch.long, device=device)
        ctx = self.ctx_proj(torch.cat([
            self.type_embed(assay_type_idx),
            self.method_embed(assay_method_idx),
        ], dim=-1))
        raw = self.mlp(torch.cat([shared_vec, ctx], dim=-1))
        return {"ic50": smooth_range_bound(raw, -3.0, max_log10_nM())}


class A6_ProbeKineticBlend(nn.Module):
    """A6: Dual-path blend of direct probe MLP and kinetic-derived KD.

    mix_logit initialized to log(3) ~= 1.099 -> sigmoid ~= 0.75 (probe-dominant).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.probe_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.head_log_koff = nn.Linear(in_dim, 1)
        self.head_log_kon_int = nn.Linear(in_dim, 1)
        self.head_log_kon_chap = nn.Linear(in_dim, 1)
        self.mix_logit = nn.Parameter(torch.tensor(math.log(3.0)))

    def forward(self, shared_vec: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        probe_ic50 = smooth_range_bound(self.probe_head(shared_vec), -3.0, max_log10_nM())
        log_koff = smooth_range_bound(self.head_log_koff(shared_vec), -8.0, 8.0)
        log_kon_int = smooth_range_bound(self.head_log_kon_int(shared_vec), -8.0, 8.0)
        log_kon_chap = smooth_range_bound(self.head_log_kon_chap(shared_vec), -8.0, 8.0)
        kon_total = torch.pow(10, log_kon_int) + torch.pow(10, log_kon_chap)
        log_kon_total = torch.log10(kon_total.clamp(min=1e-10, max=1e10))
        kinetic_kd = smooth_range_bound(log_koff - log_kon_total + 9, -3.0, max_log10_nM())
        w = torch.sigmoid(self.mix_logit)
        ic50 = w * probe_ic50 + (1 - w) * kinetic_kd
        return {
            "ic50": ic50,
            "probe_ic50": probe_ic50,
            "kinetic_kd": kinetic_kd,
            "mix_weight": w.detach(),
        }


class A7_MultiHeadConsistency(nn.Module):
    """A7: A2 + consistency regularization penalizing |kd - ic50| > threshold.

    Threshold is log10(2.0) ~= 0.301 (IC50 should be within 2x of KD).
    """

    def __init__(
        self, in_dim: int, hidden_dim: int = 128,
        consistency_threshold: float = 0.3010,
    ):
        super().__init__()
        self.consistency_threshold = consistency_threshold
        self.ic50_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.kd_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.ec50_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, shared_vec: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        ic50 = smooth_range_bound(self.ic50_head(shared_vec), -3.0, max_log10_nM())
        kd = smooth_range_bound(self.kd_head(shared_vec), -3.0, max_log10_nM())
        ec50 = smooth_range_bound(self.ec50_head(shared_vec), -3.0, max_log10_nM())
        consistency_penalty = F.relu(
            (kd - ic50).abs() - self.consistency_threshold
        ).mean()
        return {
            "ic50": ic50, "kd": kd, "ec50": ec50,
            "consistency_penalty": consistency_penalty,
        }


class A8_TypeIndicator(nn.Module):
    """A8: A1 with 7-d one-hot measurement type prepended to MLP input.

    At eval, uses IC50 indicator (index 4).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.n_types = len(BINDING_ASSAY_TYPES)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + self.n_types, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.default_type_idx = (
            BINDING_ASSAY_TYPES.index("IC50") if "IC50" in BINDING_ASSAY_TYPES else 0
        )

    def forward(
        self,
        shared_vec: torch.Tensor,
        assay_type_idx: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        B = shared_vec.shape[0]
        device = shared_vec.device
        if assay_type_idx is None:
            assay_type_idx = torch.full(
                (B,), self.default_type_idx, dtype=torch.long, device=device,
            )
        type_oh = F.one_hot(assay_type_idx, num_classes=self.n_types).float()
        raw = self.mlp(torch.cat([shared_vec, type_oh], dim=-1))
        return {"ic50": smooth_range_bound(raw, -3.0, max_log10_nM())}


HEAD_CLASSES: Dict[str, type] = {
    "a1": A1_SingleIC50,
    "a2": A2_MultiTypeHead,
    "a3": A3_SharedBaseResidual,
    "a4": A4_KineticDecomp,
    "a5": A5_AssayContext,
    "a6": A6_ProbeKineticBlend,
    "a7": A7_MultiHeadConsistency,
    "a8": A8_TypeIndicator,
}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class AblationModel(nn.Module):
    """Encoder + head wrapper for ablation experiments."""

    def __init__(self, encoder: AblationEncoder, head: nn.Module, variant: str):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.variant = variant

    def forward(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        shared_vec = self.encoder(pep_tok, mhc_a_tok, mhc_b_tok)
        return self.head(shared_vec, **kwargs)

    def predict_ic50(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
    ) -> torch.Tensor:
        """Predict IC50 for probe evaluation (pMHC inputs only)."""
        shared_vec = self.encoder(pep_tok, mhc_a_tok, mhc_b_tok)
        return self.head(shared_vec)["ic50"]


def build_ablation_model(
    variant: str,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
) -> AblationModel:
    """Build an ablation model for the given variant."""
    if variant not in HEAD_CLASSES:
        raise ValueError(f"Unknown variant: {variant!r}. Choose from {VARIANT_NAMES}")
    encoder = AblationEncoder(
        embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, ff_dim=hidden_dim,
    )
    head = HEAD_CLASSES[variant](in_dim=encoder.out_dim, hidden_dim=hidden_dim)
    return AblationModel(encoder, head, variant)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _as_float_vector(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float32).reshape(-1)


def _unified_loss(
    pred: torch.Tensor,
    batch: Any,
    device: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute loss using unified bind_target/mask/qual (all types pooled)."""
    bind_target = getattr(batch, "bind_target", None)
    bind_mask = getattr(batch, "bind_mask", None)
    bind_qual = getattr(batch, "bind_qual", None)
    if bind_target is None or bind_mask is None or bind_qual is None:
        raise RuntimeError("Batch missing binding target/mask/qual")

    pred_vec = _as_float_vector(pred)
    target_vec = normalize_binding_target_log10(
        _as_float_vector(bind_target),
        max_affinity_nM=DEFAULT_MAX_AFFINITY_NM,
        assume_log10=False,
    )
    mask_vec = _as_float_vector(bind_mask).to(device=pred_vec.device)
    qual_vec = _as_float_vector(bind_qual).to(device=pred_vec.device, dtype=torch.long)
    support = float(mask_vec.sum().item())
    if support <= 0.0:
        raise RuntimeError("Batch has no supervised samples")

    loss_vec = censor_aware_loss(pred_vec, target_vec, qual_vec, reduction="none")
    loss = (loss_vec * mask_vec).sum() / (mask_vec.sum() + 1e-8)
    return loss, {"support_binding": support}


def _typed_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Any,
    device: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute per-type-routed loss for multi-head variants (a2, a3, a7).

    Routes IC50 samples to ic50 head, KD to kd head, EC50 to ec50 head.
    Unknown-type samples route to ic50 as fallback.
    """
    targets = getattr(batch, "targets", {})
    target_masks = getattr(batch, "target_masks", {})
    target_quals = getattr(batch, "target_quals", {})

    # Map: output key -> list of batch target keys that route to it
    routing = {
        "ic50": ["binding_ic50", "binding_unknown"],
        "kd": ["binding_kd"],
        "ec50": ["binding_ec50"],
    }

    loss_parts: List[torch.Tensor] = []
    total_support = 0.0
    metrics: Dict[str, float] = {}

    for pred_key, target_keys in routing.items():
        if pred_key not in outputs:
            continue
        pred_vec = _as_float_vector(outputs[pred_key])

        for target_key in target_keys:
            if target_key not in targets:
                continue
            target_raw = _as_float_vector(targets[target_key]).to(device)
            mask = _as_float_vector(target_masks[target_key]).to(device)
            qual = _as_float_vector(target_quals[target_key]).to(device=device, dtype=torch.long)
            support = float(mask.sum().item())
            if support <= 0:
                continue

            target_norm = normalize_binding_target_log10(
                target_raw, max_affinity_nM=DEFAULT_MAX_AFFINITY_NM, assume_log10=False,
            )
            loss_vec = censor_aware_loss(pred_vec, target_norm, qual, reduction="none")
            weighted = (loss_vec * mask).sum() / (mask.sum() + 1e-8)
            loss_parts.append(weighted * support)
            total_support += support
            metrics[f"support_{target_key}"] = support

    if not loss_parts:
        # Fallback to unified target if no typed targets available
        return _unified_loss(outputs["ic50"], batch, device)

    loss = sum(loss_parts) / max(total_support, 1.0)
    metrics["support_total"] = total_support
    return loss, metrics


def _ablation_loss(
    model: AblationModel,
    batch: Any,
    device: str,
    consistency_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute loss for ablation models."""
    batch = batch.to(device)

    # Prepare context kwargs for head
    head_kwargs: Dict[str, Any] = {}
    if model.variant in CONTEXT_VARIANTS:
        binding_ctx = getattr(batch, "binding_context", {})
        if binding_ctx:
            for key in ("assay_type_idx", "assay_method_idx"):
                val = binding_ctx.get(key)
                if val is not None:
                    head_kwargs[key] = val.to(device)

    outputs = model(
        pep_tok=batch.pep_tok,
        mhc_a_tok=batch.mhc_a_tok,
        mhc_b_tok=batch.mhc_b_tok,
        **head_kwargs,
    )

    # Compute regression loss
    if model.variant in TYPED_VARIANTS:
        regression_loss, metrics = _typed_loss(outputs, batch, device)
    else:
        regression_loss, metrics = _unified_loss(outputs["ic50"], batch, device)

    total = regression_loss

    # Consistency penalty for A7
    if "consistency_penalty" in outputs and consistency_weight > 0:
        penalty = outputs["consistency_penalty"]
        total = total + consistency_weight * penalty
        metrics["consistency_penalty"] = float(penalty.detach().item())

    # Diagnostic: track mix weight for A6
    if "mix_weight" in outputs:
        metrics["probe_mix_weight"] = float(outputs["mix_weight"].item())

    return total, metrics


def _mean_ablation_loss(
    model: AblationModel,
    loader: DataLoader,
    device: str,
    consistency_weight: float = 0.1,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            loss, _ = _ablation_loss(model, batch, device, consistency_weight)
            total += float(loss.detach().item())
            batches += 1
    model.train()
    return total / max(batches, 1)


# ---------------------------------------------------------------------------
# Probe evaluation
# ---------------------------------------------------------------------------


def _evaluate_probe_panel(
    model: AblationModel,
    tokenizer: Tokenizer,
    allele_sequences: Mapping[str, str],
    peptides: Sequence[str],
    alleles: Sequence[str],
    device: str,
) -> List[Dict[str, Any]]:
    """Evaluate probe panel — identical metrics to groove_baseline_probe."""
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for peptide in peptides:
            pep = str(peptide or "").strip().upper()
            if not pep:
                continue
            pep_tok = torch.tensor(tokenizer.encode(pep, max_len=50)).unsqueeze(0).to(device)
            for allele in alleles:
                mhc_seq = _find_allele_sequence(allele_sequences, allele)
                if not mhc_seq:
                    continue
                prepared = prepare_mhc_input(mhc_a=mhc_seq, mhc_class="I")
                mhc_a_tok = torch.tensor(
                    tokenizer.encode(prepared.groove_half_1, max_len=120)
                ).unsqueeze(0).to(device)
                mhc_b_tok = torch.tensor(
                    tokenizer.encode(prepared.groove_half_2, max_len=120)
                ).unsqueeze(0).to(device)
                pred = model.predict_ic50(pep_tok, mhc_a_tok, mhc_b_tok)
                log10_val = float(pred[0, 0].item())
                rows.append({
                    "peptide": pep,
                    "allele": str(allele),
                    "ic50_log10": log10_val,
                    "ic50_nM": float(10.0 ** log10_val),
                })
    model.train()
    return rows


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Assay head ablation experiments")
    parser.add_argument(
        "--variant", type=str, required=True, choices=VARIANT_NAMES,
        help="Which ablation variant to run (a1-a8).",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument(
        "--alleles", type=str, default=",".join(DEFAULT_ALLELES),
    )
    parser.add_argument("--probe-peptide", type=str, default=DEFAULT_PROBE_PEPTIDE)
    parser.add_argument("--extra-probe-peptides", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=str, default="iedb")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--train-mhc-class-filter", type=str,
        choices=("all", "I", "II"), default="all",
    )
    parser.add_argument("--train-all-alleles", action="store_true")
    parser.add_argument(
        "--measurement-profile", type=str,
        choices=sorted(MEASUREMENT_PROFILES),
        default=MEASUREMENT_PROFILE_NUMERIC,
    )
    parser.add_argument(
        "--measurement-type-filter", type=str, default="",
        choices=sorted(NORMALIZED_MEASUREMENT_FILTERS),
    )
    parser.add_argument(
        "--qualifier-filter", type=str, default="all",
        choices=sorted(QUALIFIER_FILTERS),
    )
    parser.add_argument("--shared-peptides-only", action="store_true")
    parser.add_argument("--max-per-allele", type=int, default=-1)
    parser.add_argument("--synthetic-negatives", dest="synthetic_negatives", action="store_true")
    parser.add_argument("--no-synthetic-negatives", dest="synthetic_negatives", action="store_false")
    parser.set_defaults(synthetic_negatives=False)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument(
        "--synthetic-modes", type=str, default="",
        help=f"Comma-separated synthetic modes. Available: {','.join(ALL_SYNTHETIC_MODES)}",
    )
    parser.add_argument("--balanced-batches", action="store_true", default=True)
    parser.add_argument("--consistency-weight", type=float, default=0.1,
                        help="Weight for A7 consistency regularization.")
    parser.add_argument(
        "--probe-plot-frequency", type=str,
        choices=("epoch", "final", "off"), default="epoch",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run one batch per variant and exit.")
    args = parser.parse_args()

    variant = str(args.variant)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    random.seed(int(args.seed))

    # Output directory
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"artifacts/assay_ablation_{variant}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    merged_tsv = data_dir / "merged_deduped.tsv"
    index_csv = data_dir / "mhc_index.csv"
    if not merged_tsv.exists():
        raise FileNotFoundError(f"Merged TSV not found: {merged_tsv}")
    if not index_csv.exists():
        raise FileNotFoundError(f"MHC index not found: {index_csv}")

    probe_alleles = _split_csv(args.alleles)
    if not probe_alleles:
        raise ValueError("At least one allele is required")
    train_class_filter = (
        None if str(args.train_mhc_class_filter) == "all"
        else str(args.train_mhc_class_filter)
    )

    # ---- Load data (same pipeline as groove_baseline_probe) ----
    training_alleles = [] if bool(args.train_all_alleles) else list(probe_alleles)
    records, subset_stats = _load_binding_records_from_merged_tsv(
        merged_tsv,
        alleles=training_alleles,
        mhc_class_filter=train_class_filter,
        max_records=(None if int(args.max_records) <= 0 else int(args.max_records)),
        sampling_seed=int(args.seed) + 17,
    )

    source_filter = str(args.source or "").strip().lower()
    if source_filter:
        records = [r for r in records if str(r.source or "").strip().lower() == source_filter]
    records = [
        r for r in records
        if _keep_measurement_type(r.measurement_type, args.measurement_profile)
    ]
    if args.measurement_type_filter:
        records = [
            r for r in records
            if _normalize_binding_measurement(r.measurement_type) == str(args.measurement_type_filter)
        ]
    records = [
        r for r in records
        if _keep_binding_qualifier(getattr(r, "qualifier", 0), str(args.qualifier_filter))
    ]

    real_records = list(records)
    shared_peptide_stats: Dict[str, Any] = {}
    if args.shared_peptides_only:
        real_records, shared_peptide_stats = _filter_shared_peptides_only(
            real_records, probe_alleles,
        )
    probe_allele_counts = _require_target_allele_coverage(real_records, probe_alleles)
    balance_stats: Dict[str, Any] = {}
    if args.max_per_allele >= 0:
        real_records, balance_stats = _balance_alleles(
            real_records, probe_alleles, args.max_per_allele, rng_seed=int(args.seed),
        )
        probe_allele_counts = _require_target_allele_coverage(real_records, probe_alleles)

    train_records, val_records, split_stats = _split_records_by_peptide(
        real_records, val_fraction=0.2, seed=int(args.seed),
        alleles=(probe_alleles if not args.train_all_alleles else None),
    )
    if not train_records or not val_records:
        raise RuntimeError("Split must produce both train and val records")

    mhc_sequences, mhc_stats = resolve_mhc_sequences_from_index(
        index_csv=str(index_csv),
        alleles=sorted({
            str(r.mhc_allele or "").strip()
            for r in (train_records + val_records)
            if str(r.mhc_allele or "").strip()
        }),
    )

    synthetic_modes: Optional[Sequence[str]] = None
    if str(args.synthetic_modes or "").strip():
        synthetic_modes = [m.strip() for m in str(args.synthetic_modes).split(",") if m.strip()]
    use_synthetics = bool(args.synthetic_negatives)

    # ---- Build val dataset ----
    val_dataset = PrestoDataset(
        binding_records=val_records,
        mhc_sequences=mhc_sequences,
        strict_mhc_resolution=False,
    )
    collator = PrestoCollator()
    val_loader = create_dataloader(
        val_dataset, batch_size=int(args.batch_size), shuffle=False,
        collator=collator, balanced=False, seed=int(args.seed),
    )

    def _build_train_loader(epoch_seed: int) -> Tuple[DataLoader, Dict[str, Any]]:
        epoch_train = list(train_records)
        synth_stats: Dict[str, Any] = {"train": {}, "val": {"added": 0}}
        if use_synthetics:
            epoch_train, _, synth_stats = _augment_train_records_only(
                train_records=epoch_train,
                val_records=val_records,
                mhc_sequences=mhc_sequences,
                negative_ratio=float(args.negative_ratio),
                seed=epoch_seed,
                modes=synthetic_modes,
            )
        ds = PrestoDataset(
            binding_records=epoch_train,
            mhc_sequences=mhc_sequences,
            strict_mhc_resolution=False,
        )
        loader = _create_focused_train_loader(
            ds, collator=collator, batch_size=int(args.batch_size),
            balanced=bool(args.balanced_batches), seed=epoch_seed,
            alleles=probe_alleles,
            force_global_balance=bool(args.train_all_alleles),
        )
        return loader, synth_stats

    # ---- Verify groove representations ----
    groove_verification = _verify_groove_representations(
        train_records, mhc_sequences, probe_alleles,
    )

    # ---- Build model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_ablation_model(
        variant=variant,
        embed_dim=int(args.embed_dim),
        hidden_dim=int(args.hidden_dim),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay),
    )

    # ---- Probe setup ----
    tokenizer = Tokenizer()
    allele_sequences = _resolve_allele_sequences(index_csv)
    fit_probe_peptides = _select_fit_supported_probe_peptides(real_records, probe_alleles)
    probe_peptides = [str(args.probe_peptide).strip().upper()]
    for pep in _split_csv(str(args.extra_probe_peptides or "")):
        pep_norm = pep.strip().upper()
        if pep_norm and pep_norm not in probe_peptides:
            probe_peptides.append(pep_norm)
    for pep in fit_probe_peptides:
        if pep not in probe_peptides:
            probe_peptides.append(pep)

    # ---- Dry-run mode ----
    if args.dry_run:
        train_loader, _ = _build_train_loader(int(args.seed))
        batch = next(iter(train_loader))
        loss, metrics = _ablation_loss(
            model, batch, device,
            consistency_weight=float(args.consistency_weight),
        )
        loss.backward()
        probe_eval = _evaluate_probe_panel(
            model, tokenizer, allele_sequences, probe_peptides[:1], probe_alleles, device,
        )
        print(json.dumps({
            "event": "dry_run",
            "variant": variant,
            "n_params": n_params,
            "loss": float(loss.item()),
            "metrics": metrics,
            "probe_sample": probe_eval[:2],
        }, sort_keys=True), flush=True)
        return

    # ---- Setup logging ----
    print(json.dumps({
        "event": "assay_ablation_setup",
        "variant": variant,
        "n_params": n_params,
        "embed_dim": int(args.embed_dim),
        "hidden_dim": int(args.hidden_dim),
        "probe_alleles": probe_alleles,
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "device": device,
        "probe_peptides": probe_peptides,
        "consistency_weight": float(args.consistency_weight),
        "groove_verification": groove_verification,
    }, sort_keys=True), flush=True)

    # ---- Training loop ----
    epoch_summaries: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    model.train()

    for epoch in range(1, int(args.epochs) + 1):
        train_loader, synthetic_stats = _build_train_loader(int(args.seed) + epoch)
        train_loss_sum = 0.0
        train_batches = 0
        epoch_metrics: Dict[str, float] = {}

        for batch in train_loader:
            loss, batch_metrics = _ablation_loss(
                model, batch, device,
                consistency_weight=float(args.consistency_weight),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().item())
            train_batches += 1
            # Accumulate metrics
            for k, v in batch_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = _mean_ablation_loss(
            model, val_loader, device,
            consistency_weight=float(args.consistency_weight),
        )

        # Average epoch metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= max(train_batches, 1)

        # Probe evaluation
        probe_eval = _evaluate_probe_panel(
            model, tokenizer, allele_sequences,
            probe_peptides, probe_alleles, device,
        )
        for row in probe_eval:
            probe_rows.append({"epoch": epoch, **row})

        epoch_summary: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        epoch_summary.update(epoch_metrics)
        epoch_summaries.append(epoch_summary)

        # Write artifacts
        summary = {
            "config": {
                "variant": variant,
                "n_params": n_params,
                "embed_dim": int(args.embed_dim),
                "hidden_dim": int(args.hidden_dim),
                "probe_alleles": probe_alleles,
                "probe_peptides": probe_peptides,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
                "consistency_weight": float(args.consistency_weight),
            },
            "subset_stats": subset_stats,
            "balance_stats": balance_stats,
            "split_stats": split_stats,
            "mhc_resolve_stats": mhc_stats,
            "probe_allele_counts": probe_allele_counts,
            "real_record_summary": _summarize_binding_records(real_records),
            "train_record_summary": _summarize_binding_records(train_records),
            "val_record_summary": _summarize_binding_records(val_records),
            "synthetic_stats": synthetic_stats,
            "groove_verification": groove_verification,
            "train_real_records": len(train_records),
            "val_size": len(val_dataset),
            "epochs": epoch_summaries,
        }
        _write_summary_artifacts(
            out_dir=out_dir,
            summary=summary,
            probe_rows=probe_rows,
            write_probe_plot=(
                str(args.probe_plot_frequency) == "epoch"
                or (str(args.probe_plot_frequency) == "final" and epoch == int(args.epochs))
            ),
        )
        print(json.dumps({
            "event": "assay_ablation_epoch",
            "variant": variant,
            **epoch_summary,
            "probe_rows": probe_eval,
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
