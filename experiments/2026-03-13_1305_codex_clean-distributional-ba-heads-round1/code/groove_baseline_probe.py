#!/usr/bin/env python
"""Minimalist groove-baseline binding diagnostic.

Trains a tiny mean-pool MLP on groove sequences to predict IC50 directly,
providing a lower bound for allele-discrimination ability of the groove
representation.  Uses the same data pipeline as focused_binding_probe.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from presto.data import BindingRecord, PrestoCollator, PrestoDataset, create_dataloader
from presto.data.groove import prepare_mhc_input
from presto.data.mhc_index import build_mhc_sequence_lookup, load_mhc_index
from presto.data.tokenizer import Tokenizer
from presto.models.affinity import (
    DEFAULT_MAX_AFFINITY_NM,
    max_log10_nM,
    normalize_binding_target_log10,
)
from presto.models.heads import smooth_range_bound
from presto.scripts.focused_binding_probe import (
    DEFAULT_ALLELES,
    DEFAULT_PROBE_PEPTIDE,
    StrictAlleleBalancedBatchSampler,
    _augment_train_records_only,
    _balance_alleles,
    _collect_binding_contrastive_pairs,
    _collect_binding_peptide_ranking_pairs,
    _create_focused_train_loader,
    _filter_shared_peptides_only,
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
    _write_probe_csv,
    _write_probe_plot,
    _write_summary_artifacts,
    AFFINITY_LOSS_MODES,
    MEASUREMENT_PROFILES,
    MEASUREMENT_PROFILE_NUMERIC,
    NORMALIZED_MEASUREMENT_FILTERS,
    QUALIFIER_FILTERS,
)
from presto.scripts.train_iedb import ALL_SYNTHETIC_MODES, resolve_mhc_sequences_from_index
from presto.training.losses import censor_aware_loss


POSITION_MODES = {
    "triple",
    "triple_baseline",
    "abs_only",
    "triple_plus_abs",
    "start_only",
    "end_only",
    "start_plus_end",
    "concat_start_end",
    "concat_start_end_frac",
    "mlp_start_end",
    "mlp_start_end_frac",
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GrooveBaselineModel(nn.Module):
    """Simplest possible groove-sequence binding predictor.

    Mean-pools peptide, groove_half_1, groove_half_2 embeddings, concatenates,
    passes through a small MLP.  Output: log10(IC50_nM).
    """

    def __init__(
        self,
        vocab_size: int = 26,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_allele_classes: int = 0,
    ):
        super().__init__()
        self.aa_embedding = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Optional MHC classification head (used in curriculum pre-training)
        self.classify_head: Optional[nn.Linear] = None
        if n_allele_classes > 0:
            self.classify_head = nn.Linear(2 * embed_dim, n_allele_classes)

    def forward(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict log10(IC50_nM) from tokenized sequences.

        Args:
            pep_tok: (B, L_pep) peptide token IDs
            mhc_a_tok: (B, L_a) groove half 1 token IDs
            mhc_b_tok: (B, L_b) groove half 2 token IDs

        Returns:
            (B, 1) log10(IC50_nM) predictions
        """
        pep_vec = self._mean_pool(self.aa_embedding(pep_tok))
        gh1_vec = self._mean_pool(self.aa_embedding(mhc_a_tok))
        gh2_vec = self._mean_pool(self.aa_embedding(mhc_b_tok))
        raw = self.mlp(torch.cat([pep_vec, gh1_vec, gh2_vec], dim=-1))
        return smooth_range_bound(raw, -3.0, max_log10_nM())

    def classify_allele(
        self, mhc_a_tok: torch.Tensor, mhc_b_tok: torch.Tensor,
    ) -> torch.Tensor:
        """Classify allele from groove tokens alone. Returns (B, n_classes) logits."""
        if self.classify_head is None:
            raise RuntimeError("No classification head — set n_allele_classes > 0")
        gh1_vec = self._mean_pool(self.aa_embedding(mhc_a_tok))
        gh2_vec = self._mean_pool(self.aa_embedding(mhc_b_tok))
        return self.classify_head(torch.cat([gh1_vec, gh2_vec], dim=-1))

    @staticmethod
    def _mean_pool(embedded: torch.Tensor) -> torch.Tensor:
        """Mean-pool over sequence length, ignoring padding (token 0)."""
        mask = (embedded.abs().sum(-1) > 0).float().unsqueeze(-1)
        return (embedded * mask).sum(1) / mask.sum(1).clamp(min=1)


class GrooveTransformerModel(nn.Module):
    """Small transformer groove-sequence binding predictor.

    Adds learned positional encoding and 1-2 self-attention layers per segment
    before pooling — the minimal upgrade needed to preserve positional info
    that mean-pooling destroys.
    """

    def __init__(
        self,
        vocab_size: int = 26,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        max_seq_len: int = 200,
        hidden_dim: int = 128,
        n_allele_classes: int = 0,
        peptide_pos_mode: str = "triple",
        groove_pos_mode: str = "triple",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        peptide_pos_mode = str(peptide_pos_mode).strip().lower()
        groove_pos_mode = str(groove_pos_mode).strip().lower()
        if peptide_pos_mode not in POSITION_MODES:
            raise ValueError(f"Unsupported peptide_pos_mode: {peptide_pos_mode!r}")
        if groove_pos_mode not in POSITION_MODES:
            raise ValueError(f"Unsupported groove_pos_mode: {groove_pos_mode!r}")
        self.peptide_pos_mode = peptide_pos_mode
        self.groove_pos_mode = groove_pos_mode
        self.aa_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.end_pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.abs_pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.frac_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.concat_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.concat_frac_proj = nn.Linear(2 * embed_dim + 2, embed_dim)
        self.concat_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.concat_frac_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim + 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Optional MHC classification head (used in curriculum pre-training)
        self.classify_head: Optional[nn.Linear] = None
        if n_allele_classes > 0:
            self.classify_head = nn.Linear(2 * embed_dim, n_allele_classes)

    def _compose_positional_signal(
        self,
        *,
        mode: str,
        start_embed: torch.Tensor,
        end_embed: torch.Tensor,
        start_frac: torch.Tensor,
        end_frac: torch.Tensor,
        abs_embed: torch.Tensor,
    ) -> torch.Tensor:
        if mode in {"triple", "triple_baseline"}:
            return start_embed + end_embed + self.frac_mlp(start_frac.unsqueeze(-1))
        if mode == "abs_only":
            return abs_embed
        if mode == "triple_plus_abs":
            return start_embed + end_embed + self.frac_mlp(start_frac.unsqueeze(-1)) + abs_embed
        if mode == "start_only":
            return start_embed
        if mode == "end_only":
            return end_embed
        if mode == "start_plus_end":
            return start_embed + end_embed
        if mode == "concat_start_end":
            return self.concat_proj(torch.cat([start_embed, end_embed], dim=-1))
        frac_features = torch.cat(
            [start_frac.unsqueeze(-1), end_frac.unsqueeze(-1)],
            dim=-1,
        )
        if mode == "concat_start_end_frac":
            return self.concat_frac_proj(torch.cat([start_embed, end_embed, frac_features], dim=-1))
        if mode == "mlp_start_end":
            return self.concat_mlp(torch.cat([start_embed, end_embed], dim=-1))
        if mode == "mlp_start_end_frac":
            return self.concat_frac_mlp(torch.cat([start_embed, end_embed, frac_features], dim=-1))
        raise ValueError(f"Unsupported position mode: {mode!r}")

    def _encode_segment(self, tok: torch.Tensor, *, pos_mode: str) -> torch.Tensor:
        """Embed, add positional encoding, run through transformer, mean-pool."""
        B, L = tok.shape
        positions = torch.arange(L, device=tok.device).unsqueeze(0).expand(B, L)
        end_dist = (tok.ne(0).sum(dim=1, keepdim=True) - 1 - positions).clamp(min=0)
        denom = (tok.ne(0).sum(dim=1, keepdim=True) - 1).clamp(min=1).float()
        start_frac = positions.float() / denom
        end_frac = end_dist.float() / denom
        pos_embed = self._compose_positional_signal(
            mode=pos_mode,
            start_embed=self.pos_embedding(positions),
            end_embed=self.end_pos_embedding(end_dist.clamp(max=self.end_pos_embedding.num_embeddings - 1)),
            start_frac=start_frac,
            end_frac=end_frac,
            abs_embed=self.abs_pos_embedding(positions),
        )
        x = self.aa_embedding(tok) + pos_embed
        pad_mask = tok == 0  # PAD token = 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        # Mean-pool over non-padding positions
        non_pad = (~pad_mask).float().unsqueeze(-1)
        return (x * non_pad).sum(1) / non_pad.sum(1).clamp(min=1)

    def forward(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        pep_vec = self._encode_segment(pep_tok, pos_mode=self.peptide_pos_mode)
        gh1_vec = self._encode_segment(mhc_a_tok, pos_mode=self.groove_pos_mode)
        gh2_vec = self._encode_segment(mhc_b_tok, pos_mode=self.groove_pos_mode)
        raw = self.mlp(torch.cat([pep_vec, gh1_vec, gh2_vec], dim=-1))
        return smooth_range_bound(raw, -3.0, max_log10_nM())

    def classify_allele(
        self, mhc_a_tok: torch.Tensor, mhc_b_tok: torch.Tensor,
    ) -> torch.Tensor:
        """Classify allele from groove tokens alone. Returns (B, n_classes) logits."""
        if self.classify_head is None:
            raise RuntimeError("No classification head — set n_allele_classes > 0")
        gh1_vec = self._encode_segment(mhc_a_tok, pos_mode=self.groove_pos_mode)
        gh2_vec = self._encode_segment(mhc_b_tok, pos_mode=self.groove_pos_mode)
        return self.classify_head(torch.cat([gh1_vec, gh2_vec], dim=-1))


def _build_model(variant: str, embed_dim: int, hidden_dim: int, **kwargs: Any) -> nn.Module:
    """Construct model by variant name."""
    n_allele_classes = int(kwargs.get("n_allele_classes", 0))
    if variant == "mlp":
        return GrooveBaselineModel(
            vocab_size=26, embed_dim=embed_dim, hidden_dim=hidden_dim,
            n_allele_classes=n_allele_classes,
        )
    if variant == "transformer":
        return GrooveTransformerModel(
            vocab_size=26,
            embed_dim=embed_dim,
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("n_layers", 2),
            ff_dim=kwargs.get("ff_dim", hidden_dim),
            hidden_dim=hidden_dim,
            n_allele_classes=n_allele_classes,
            peptide_pos_mode=kwargs.get("peptide_pos_mode", "triple"),
            groove_pos_mode=kwargs.get("groove_pos_mode", "triple"),
        )
    raise ValueError(f"Unknown model variant: {variant!r}")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _as_float_vector(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float32).reshape(-1)


def _groove_baseline_loss(
    model: GrooveBaselineModel,
    batch: Any,
    device: str,
    regularization: Optional[Mapping[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute censor-aware regression loss for the groove baseline."""
    batch = batch.to(device)
    pred = model(
        pep_tok=batch.pep_tok,
        mhc_a_tok=batch.mhc_a_tok,
        mhc_b_tok=batch.mhc_b_tok,
    )

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
        raise RuntimeError("Groove baseline batch has no supervised samples")

    loss_vec = censor_aware_loss(pred_vec, target_vec, qual_vec, reduction="none")
    regression_loss = (loss_vec * mask_vec).sum() / (mask_vec.sum() + 1e-8)

    metrics: Dict[str, float] = {"support_binding": support}

    # Contrastive losses: reuse the same pair-collection logic but read from
    # model output directly instead of outputs["assays"]["KD_nM"].
    reg = dict(regularization or {})
    total = regression_loss

    # Same-peptide / different-allele ranking loss
    contrastive_weight = float(reg.get("binding_contrastive_weight", 0.0))
    if contrastive_weight > 0:
        pair_candidates, c_metrics = _collect_binding_contrastive_pairs(
            batch,
            target_gap_min=float(reg.get("binding_contrastive_target_gap_min", 0.3)),
            max_pairs=int(reg.get("binding_contrastive_max_pairs", 64)),
        )
        metrics.update(c_metrics)
        if pair_candidates:
            margin = float(reg.get("binding_contrastive_margin", 0.2))
            gap_cap = float(reg.get("binding_contrastive_target_gap_cap", 2.0)) or margin
            pair_losses: List[torch.Tensor] = []
            for gap, si, wi in pair_candidates:
                pred_gap = pred_vec[wi] - pred_vec[si]
                req = max(margin, min(float(gap), gap_cap))
                pair_losses.append(torch.relu(pred_vec.new_tensor(req) - pred_gap))
            c_loss = contrastive_weight * torch.stack(pair_losses).mean()
            total = total + c_loss

    # Same-allele / different-peptide ranking loss
    peptide_weight = float(reg.get("binding_peptide_contrastive_weight", 0.0))
    if peptide_weight > 0:
        pair_candidates_p, p_metrics = _collect_binding_peptide_ranking_pairs(
            batch,
            target_gap_min=float(reg.get("binding_peptide_contrastive_target_gap_min", 0.5)),
            max_pairs=int(reg.get("binding_peptide_contrastive_max_pairs", 128)),
        )
        metrics.update(p_metrics)
        if pair_candidates_p:
            margin_p = float(reg.get("binding_peptide_contrastive_margin", 0.2))
            gap_cap_p = float(reg.get("binding_peptide_contrastive_target_gap_cap", 2.0)) or margin_p
            pair_losses_p: List[torch.Tensor] = []
            for gap, si, wi in pair_candidates_p:
                pred_gap = pred_vec[wi] - pred_vec[si]
                req = max(margin_p, min(float(gap), gap_cap_p))
                pair_losses_p.append(torch.relu(pred_vec.new_tensor(req) - pred_gap))
            p_loss = peptide_weight * torch.stack(pair_losses_p).mean()
            total = total + p_loss

    return total, metrics


def _mean_groove_baseline_loss(
    model: GrooveBaselineModel,
    loader: DataLoader,
    device: str,
    regularization: Optional[Mapping[str, float]] = None,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            loss, _ = _groove_baseline_loss(model, batch, device, regularization=regularization)
            total += float(loss.detach().item())
            batches += 1
    model.train()
    return total / max(batches, 1)


# ---------------------------------------------------------------------------
# Probe evaluation
# ---------------------------------------------------------------------------

def _evaluate_probe_panel_baseline(
    model: GrooveBaselineModel,
    tokenizer: Tokenizer,
    allele_sequences: Mapping[str, str],
    peptides: Sequence[str],
    alleles: Sequence[str],
    device: str,
) -> List[Dict[str, Any]]:
    """Evaluate probe panel for the groove baseline model."""
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
                pred_log10 = model(pep_tok, mhc_a_tok, mhc_b_tok)
                log10_val = float(pred_log10[0, 0].item())
                nM_val = float(10.0 ** log10_val)
                rows.append({
                    "peptide": pep,
                    "allele": str(allele),
                    "ic50_log10": log10_val,
                    "ic50_nM": nM_val,
                })
    model.train()
    return rows


def _find_allele_sequence(allele_sequences: Mapping[str, str], allele: str) -> Optional[str]:
    """Find an allele sequence with fuzzy matching."""
    for key in [allele, allele.replace("HLA-", "")]:
        if key in allele_sequences:
            return allele_sequences[key]
    for key, value in allele_sequences.items():
        if allele in key or key in allele:
            return value
    return None


# ---------------------------------------------------------------------------
# Curriculum learning
# ---------------------------------------------------------------------------


@dataclass
class CurriculumPhase:
    """One phase in a curriculum training schedule."""
    epochs: int
    classify: bool = False      # MHC allele classification objective
    regress: bool = False       # IC50 regression objective
    synth: bool = False         # Add synthetic negatives
    contrastive: bool = False   # Same-peptide / different-allele ranking
    peprank: bool = False       # Same-allele / different-peptide ranking


def _parse_curriculum(spec: str) -> List[CurriculumPhase]:
    """Parse curriculum spec like '5:classify,10:regress,5:regress+synth+contrastive'."""
    phases: List[CurriculumPhase] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Curriculum phase must be '<epochs>:<modes>': {part!r}")
        epoch_str, modes_str = part.split(":", 1)
        n_epochs = int(epoch_str.strip())
        modes = {m.strip().lower() for m in modes_str.split("+")}
        phases.append(CurriculumPhase(
            epochs=n_epochs,
            classify="classify" in modes,
            regress="regress" in modes,
            synth="synth" in modes,
            contrastive="contrastive" in modes,
            peprank="peprank" in modes,
        ))
    return phases


def _allele_classification_loss(
    model: nn.Module,
    batch: Any,
    allele_to_idx: Dict[str, int],
    device: str,
) -> Tuple[torch.Tensor, float]:
    """Cross-entropy loss for allele classification from groove tokens."""
    batch = batch.to(device)
    logits = model.classify_allele(
        mhc_a_tok=batch.mhc_a_tok,
        mhc_b_tok=batch.mhc_b_tok,
    )
    # Build target labels from batch allele strings
    allele_strs = getattr(batch, "primary_alleles", None)
    if allele_strs is None:
        raise RuntimeError("Batch missing primary_alleles for classification")
    targets: List[int] = []
    mask: List[bool] = []
    for a in allele_strs:
        a_str = str(a).strip()
        if a_str in allele_to_idx:
            targets.append(allele_to_idx[a_str])
            mask.append(True)
        else:
            targets.append(0)
            mask.append(False)
    target_t = torch.tensor(targets, dtype=torch.long, device=device)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
    if mask_t.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    loss = F.cross_entropy(logits[mask_t], target_t[mask_t])
    acc = float((logits[mask_t].argmax(-1) == target_t[mask_t]).float().mean().item())
    return loss, acc


# ---------------------------------------------------------------------------
# Groove verification
# ---------------------------------------------------------------------------


def _verify_groove_representations(
    records: Sequence[Any],
    mhc_sequences: Mapping[str, str],
    probe_alleles: Sequence[str],
) -> Dict[str, Any]:
    """Verify that groove representations are correct and allele-specific.

    Checks:
    1. Each probe allele resolves to a non-empty groove sequence pair
    2. Different alleles produce different groove sequences
    3. Real records have correct allele → groove mapping
    4. Reports groove sequence lengths and uniqueness
    """
    results: Dict[str, Any] = {"allele_grooves": {}, "errors": [], "warnings": []}

    # Check each probe allele resolves to distinct groove sequences
    groove_by_allele: Dict[str, Tuple[str, str]] = {}
    for allele in probe_alleles:
        seq = mhc_sequences.get(allele, "")
        if not seq:
            results["errors"].append(f"No MHC sequence for allele {allele}")
            continue
        try:
            prepared = prepare_mhc_input(mhc_a=seq, mhc_class="I")
            gh1 = prepared.groove_half_1
            gh2 = prepared.groove_half_2
            if not gh1 and not gh2:
                results["errors"].append(f"Empty groove for {allele}")
                continue
            groove_by_allele[allele] = (gh1, gh2)
            results["allele_grooves"][allele] = {
                "gh1_len": len(gh1),
                "gh2_len": len(gh2),
                "gh1_first10": gh1[:10],
                "gh2_first10": gh2[:10],
            }
        except Exception as e:
            results["errors"].append(f"Groove extraction failed for {allele}: {e}")

    # Check that different alleles produce different grooves
    seen: Dict[Tuple[str, str], str] = {}
    for allele, groove_pair in groove_by_allele.items():
        if groove_pair in seen:
            results["errors"].append(
                f"Identical grooves for {allele} and {seen[groove_pair]}!"
            )
        else:
            seen[groove_pair] = allele
    results["n_unique_grooves"] = len(seen)
    results["n_probe_alleles"] = len(probe_alleles)

    # Spot-check a sample of real records to verify groove resolution
    n_checked = 0
    n_correct = 0
    for rec in records[:200]:
        allele = str(getattr(rec, "mhc_allele", "") or "").strip()
        if allele not in groove_by_allele:
            continue
        seq = mhc_sequences.get(allele, "")
        if not seq:
            continue
        try:
            prepared = prepare_mhc_input(mhc_a=seq, mhc_class="I")
            expected = groove_by_allele[allele]
            if (prepared.groove_half_1, prepared.groove_half_2) == expected:
                n_correct += 1
            else:
                results["warnings"].append(
                    f"Groove mismatch for record with allele {allele}"
                )
        except Exception:
            pass
        n_checked += 1
    results["records_checked"] = n_checked
    results["records_correct"] = n_correct

    return results


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Groove baseline binding diagnostic")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="artifacts/groove_baseline_probe")
    parser.add_argument(
        "--alleles", type=str, default=",".join(DEFAULT_ALLELES),
        help="Probe/evaluation allele panel.",
    )
    parser.add_argument("--probe-peptide", type=str, default=DEFAULT_PROBE_PEPTIDE)
    parser.add_argument(
        "--design-id",
        type=str,
        default="",
        help="Optional design identifier for benchmark manifests and structured logs.",
    )
    parser.add_argument(
        "--extra-probe-peptides",
        type=str,
        default="",
        help="Comma-separated additional probe peptides to evaluate each epoch.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--model-variant", type=str, default="mlp",
        choices=("mlp", "transformer"),
        help="Model architecture: 'mlp' (mean-pool) or 'transformer' (pos-enc + self-attn).",
    )
    parser.add_argument("--n-heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--n-layers", type=int, default=2, help="Transformer encoder layers.")
    parser.add_argument(
        "--peptide-pos-mode",
        type=str,
        choices=sorted(POSITION_MODES),
        default="triple",
        help="Peptide positional composition mode for transformer variants.",
    )
    parser.add_argument(
        "--groove-pos-mode",
        type=str,
        choices=sorted(POSITION_MODES),
        default="triple",
        help="Groove positional composition mode for transformer variants.",
    )
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
    parser.add_argument(
        "--class-i-anchor-strategy", type=str,
        choices=("none", "property_opposite"), default="none",
    )
    parser.add_argument("--synthetic-negatives", dest="synthetic_negatives", action="store_true")
    parser.add_argument("--no-synthetic-negatives", dest="synthetic_negatives", action="store_false")
    parser.set_defaults(synthetic_negatives=False)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument(
        "--synthetic-modes", type=str, default="",
        help=(
            "Comma-separated subset of synthetic negative modes to use. "
            f"Available: {','.join(ALL_SYNTHETIC_MODES)}. "
            "Empty string (default) uses all modes."
        ),
    )
    parser.add_argument("--balanced-batches", action="store_true", default=True)
    # Curriculum learning: comma-separated phase spec, e.g. "5:classify,10:regress,5:regress+synth+contrastive"
    # Each phase: <n_epochs>:<mode>[+<mode>...]
    # Modes: classify (MHC allele classification), regress (IC50 regression),
    #        synth (add synthetic negatives), contrastive (allele ranking loss),
    #        peprank (peptide ranking loss)
    parser.add_argument(
        "--curriculum", type=str, default="",
        help=(
            "Curriculum learning phases. Comma-separated '<epochs>:<modes>' "
            "where modes are joined with '+'. Example: "
            "'5:classify,10:regress,5:regress+synth+contrastive'. "
            "If empty, uses standard training for --epochs epochs."
        ),
    )
    parser.add_argument(
        "--binding-contrastive-weight", type=float, default=0.0,
        help="Weight for same-peptide/different-allele ranking loss.",
    )
    parser.add_argument("--binding-contrastive-margin", type=float, default=0.2)
    parser.add_argument("--binding-contrastive-target-gap-min", type=float, default=0.3)
    parser.add_argument("--binding-contrastive-target-gap-cap", type=float, default=2.0)
    parser.add_argument("--binding-contrastive-max-pairs", type=int, default=64)
    parser.add_argument(
        "--binding-peptide-contrastive-weight", type=float, default=0.0,
        help="Weight for same-allele/different-peptide ranking loss.",
    )
    parser.add_argument("--binding-peptide-contrastive-margin", type=float, default=0.2)
    parser.add_argument("--binding-peptide-contrastive-target-gap-min", type=float, default=0.5)
    parser.add_argument("--binding-peptide-contrastive-target-gap-cap", type=float, default=2.0)
    parser.add_argument("--binding-peptide-contrastive-max-pairs", type=int, default=128)
    parser.add_argument(
        "--probe-plot-frequency",
        type=str,
        choices=("epoch", "final", "off"),
        default="epoch",
        help="When to render the probe affinity plot.",
    )
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    random.seed(int(args.seed))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_tsv = data_dir / "merged_deduped.tsv"
    index_csv = data_dir / "mhc_index.csv"
    if not merged_tsv.exists():
        raise FileNotFoundError(f"Merged TSV not found: {merged_tsv}")
    if not index_csv.exists():
        raise FileNotFoundError(f"MHC index not found: {index_csv}")

    probe_alleles = _split_csv(args.alleles)
    if not probe_alleles:
        raise ValueError("At least one allele is required")
    train_class_filter = None if str(args.train_mhc_class_filter) == "all" else str(args.train_mhc_class_filter)

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
        records = [rec for rec in records if str(rec.source or "").strip().lower() == source_filter]
    records = [
        rec for rec in records if _keep_measurement_type(rec.measurement_type, args.measurement_profile)
    ]
    if args.measurement_type_filter:
        records = [
            rec for rec in records
            if _normalize_binding_measurement(rec.measurement_type) == str(args.measurement_type_filter)
        ]
    records = [
        rec for rec in records
        if _keep_binding_qualifier(getattr(rec, "qualifier", 0), str(args.qualifier_filter))
    ]

    real_records = list(records)
    shared_peptide_stats: Dict[str, Any] = {}
    if args.shared_peptides_only:
        real_records, shared_peptide_stats = _filter_shared_peptides_only(real_records, probe_alleles)
    probe_allele_counts_after_filter = _require_target_allele_coverage(real_records, probe_alleles)
    balance_stats: Dict[str, Any] = {}
    if args.max_per_allele >= 0:
        real_records, balance_stats = _balance_alleles(
            real_records, probe_alleles, args.max_per_allele, rng_seed=int(args.seed),
        )
        probe_allele_counts_after_filter = _require_target_allele_coverage(real_records, probe_alleles)
    train_records, val_records, split_stats = _split_records_by_peptide(
        real_records,
        val_fraction=0.2,
        seed=int(args.seed),
        alleles=(probe_alleles if not args.train_all_alleles else None),
    )
    if not train_records or not val_records:
        raise RuntimeError("Focused binding split must produce both train and val records")

    mhc_sequences, mhc_stats = resolve_mhc_sequences_from_index(
        index_csv=str(index_csv),
        alleles=sorted({
            str(rec.mhc_allele or "").strip()
            for rec in (train_records + val_records)
            if str(rec.mhc_allele or "").strip()
        }),
    )

    # Parse synthetic mode filter
    synthetic_modes: Optional[Sequence[str]] = None
    if str(args.synthetic_modes or "").strip():
        synthetic_modes = [m.strip() for m in str(args.synthetic_modes).split(",") if m.strip()]
        unknown = set(synthetic_modes) - set(ALL_SYNTHETIC_MODES)
        if unknown:
            raise ValueError(f"Unknown synthetic modes: {unknown}. Available: {ALL_SYNTHETIC_MODES}")

    # Per-epoch synthetic regeneration: keep real records separate, regenerate
    # synthetics each epoch with a different seed.  This avoids the model
    # memorising a fixed set of negatives.
    use_synthetics = bool(args.synthetic_negatives)

    # Build val dataset once (no synthetics in val)
    val_dataset = PrestoDataset(
        binding_records=val_records,
        mhc_sequences=mhc_sequences,
        strict_mhc_resolution=False,
    )
    collator = PrestoCollator()
    val_loader = create_dataloader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collator=collator,
        balanced=False,
        seed=int(args.seed),
    )

    def _build_train_loader(epoch_seed: int) -> Tuple[DataLoader, Dict[str, Any]]:
        """Build a fresh train loader, regenerating synthetics with epoch_seed."""
        epoch_train = list(train_records)
        synth_stats: Dict[str, Any] = {"train": {}, "val": {"added": 0}}
        if use_synthetics:
            epoch_train, _, synth_stats = _augment_train_records_only(
                train_records=epoch_train,
                val_records=val_records,
                mhc_sequences=mhc_sequences,
                negative_ratio=float(args.negative_ratio),
                seed=epoch_seed,
                class_i_anchor_strategy=str(args.class_i_anchor_strategy),
                modes=synthetic_modes,
            )
        ds = PrestoDataset(
            binding_records=epoch_train,
            mhc_sequences=mhc_sequences,
            strict_mhc_resolution=False,
        )
        loader = _create_focused_train_loader(
            ds,
            collator=collator,
            batch_size=int(args.batch_size),
            balanced=bool(args.balanced_batches),
            seed=epoch_seed,
            alleles=probe_alleles,
            force_global_balance=bool(args.train_all_alleles),
        )
        return loader, synth_stats

    # Verify groove representations on real data
    groove_verification = _verify_groove_representations(
        train_records, mhc_sequences, probe_alleles,
    )

    # Build initial train loader for epoch 1
    synthetic_stats: Dict[str, Any] = {"train": {}, "val": {}}
    train_loader, synthetic_stats = _build_train_loader(int(args.seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_variant = str(args.model_variant)

    # Parse curriculum or build default single-phase schedule
    curriculum_spec = str(args.curriculum or "").strip()
    if curriculum_spec:
        curriculum = _parse_curriculum(curriculum_spec)
    else:
        # Default: standard regression for --epochs epochs, honouring CLI flags
        curriculum = [CurriculumPhase(
            epochs=int(args.epochs),
            regress=True,
            synth=use_synthetics,
            contrastive=float(args.binding_contrastive_weight) > 0,
            peprank=float(args.binding_peptide_contrastive_weight) > 0,
        )]
    total_epochs = sum(p.epochs for p in curriculum)
    needs_classify = any(p.classify for p in curriculum)

    # Build allele → index mapping for classification head
    allele_to_idx: Dict[str, int] = {a: i for i, a in enumerate(sorted(probe_alleles))}

    model = _build_model(
        variant=model_variant,
        embed_dim=int(args.embed_dim),
        hidden_dim=int(args.hidden_dim),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        ff_dim=int(args.hidden_dim),
        n_allele_classes=len(probe_alleles) if needs_classify else 0,
        peptide_pos_mode=str(args.peptide_pos_mode),
        groove_pos_mode=str(args.groove_pos_mode),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    # Base regularization config from CLI (used as defaults; overridden per-phase)
    base_reg_cfg: Dict[str, float] = {
        "binding_contrastive_weight": float(args.binding_contrastive_weight),
        "binding_contrastive_margin": float(args.binding_contrastive_margin),
        "binding_contrastive_target_gap_min": float(args.binding_contrastive_target_gap_min),
        "binding_contrastive_target_gap_cap": float(args.binding_contrastive_target_gap_cap),
        "binding_contrastive_max_pairs": int(args.binding_contrastive_max_pairs),
        "binding_peptide_contrastive_weight": float(args.binding_peptide_contrastive_weight),
        "binding_peptide_contrastive_margin": float(args.binding_peptide_contrastive_margin),
        "binding_peptide_contrastive_target_gap_min": float(args.binding_peptide_contrastive_target_gap_min),
        "binding_peptide_contrastive_target_gap_cap": float(args.binding_peptide_contrastive_target_gap_cap),
        "binding_peptide_contrastive_max_pairs": int(args.binding_peptide_contrastive_max_pairs),
    }

    tokenizer = Tokenizer()
    allele_sequences = _resolve_allele_sequences(index_csv)
    fit_probe_peptides = _select_fit_supported_probe_peptides(real_records, probe_alleles)
    probe_peptides = [str(args.probe_peptide).strip().upper()]
    for peptide in _split_csv(str(args.extra_probe_peptides or "")):
        peptide_norm = peptide.strip().upper()
        if peptide_norm and peptide_norm not in probe_peptides:
            probe_peptides.append(peptide_norm)
    for peptide in fit_probe_peptides:
        if peptide not in probe_peptides:
            probe_peptides.append(peptide)

    curriculum_desc = [
        {"epochs": p.epochs, "modes": "+".join(
            m for m, on in [
                ("classify", p.classify), ("regress", p.regress),
                ("synth", p.synth), ("contrastive", p.contrastive),
                ("peprank", p.peprank),
            ] if on
        )}
        for p in curriculum
    ]

    print(
        json.dumps({
            "event": "groove_baseline_setup",
            "design_id": str(args.design_id),
            "model_variant": model_variant,
            "n_params": n_params,
            "embed_dim": int(args.embed_dim),
            "hidden_dim": int(args.hidden_dim),
            "peptide_pos_mode": str(args.peptide_pos_mode),
            "groove_pos_mode": str(args.groove_pos_mode),
            "probe_alleles": probe_alleles,
            "train_all_alleles": bool(args.train_all_alleles),
            "rows": len(real_records),
            "train_rows": len(train_records),
            "val_rows": len(val_records),
            "device": device,
            "probe_peptides": probe_peptides,
            "synthetic_negatives": use_synthetics,
            "synthetic_modes": list(synthetic_modes) if synthetic_modes else list(ALL_SYNTHETIC_MODES),
            "per_epoch_regeneration": use_synthetics,
            "groove_verification": groove_verification,
            "curriculum": curriculum_desc,
            "qualifier_filter": str(args.qualifier_filter),
        }, sort_keys=True),
        flush=True,
    )

    epoch_summaries: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    global_epoch = 0
    model.train()

    for phase_idx, phase in enumerate(curriculum):
        # Build per-phase regularization config
        phase_reg = dict(base_reg_cfg)
        if not phase.contrastive:
            phase_reg["binding_contrastive_weight"] = 0.0
        elif phase.contrastive and phase_reg["binding_contrastive_weight"] == 0.0:
            phase_reg["binding_contrastive_weight"] = 1.0  # default weight when curriculum enables it
        if not phase.peprank:
            phase_reg["binding_peptide_contrastive_weight"] = 0.0
        elif phase.peprank and phase_reg["binding_peptide_contrastive_weight"] == 0.0:
            phase_reg["binding_peptide_contrastive_weight"] = 0.5

        for phase_epoch in range(1, phase.epochs + 1):
            global_epoch += 1

            # Build train loader: with or without synthetics depending on phase
            if phase.synth:
                train_loader, synthetic_stats = _build_train_loader(int(args.seed) + global_epoch)
            else:
                # Real data only — build loader without synthetics
                real_ds = PrestoDataset(
                    binding_records=train_records,
                    mhc_sequences=mhc_sequences,
                    strict_mhc_resolution=False,
                )
                train_loader = _create_focused_train_loader(
                    real_ds,
                    collator=collator,
                    batch_size=int(args.batch_size),
                    balanced=bool(args.balanced_batches),
                    seed=int(args.seed) + global_epoch,
                    alleles=probe_alleles,
                    force_global_balance=bool(args.train_all_alleles),
                )
                synthetic_stats = {"train": {"added": 0}, "val": {"added": 0}}

            train_loss_sum = 0.0
            classify_acc_sum = 0.0
            train_batches = 0

            for batch in train_loader:
                losses: List[torch.Tensor] = []

                # Phase: MHC classification
                if phase.classify:
                    cls_loss, cls_acc = _allele_classification_loss(
                        model, batch, allele_to_idx, device,
                    )
                    losses.append(cls_loss)
                    classify_acc_sum += cls_acc

                # Phase: IC50 regression (optionally with contrastive/peprank)
                if phase.regress:
                    reg_loss, batch_metrics = _groove_baseline_loss(
                        model, batch, device, regularization=phase_reg,
                    )
                    losses.append(reg_loss)

                if not losses:
                    continue

                total_loss = sum(losses)
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_sum += float(total_loss.detach().item())
                train_batches += 1

            train_loss = train_loss_sum / max(train_batches, 1)
            val_loss = _mean_groove_baseline_loss(
                model, val_loader, device, regularization=phase_reg,
            )
            probe_eval = _evaluate_probe_panel_baseline(
                model, tokenizer, allele_sequences,
                probe_peptides, probe_alleles, device,
            )
            for row in probe_eval:
                probe_rows.append({"epoch": global_epoch, **row})

            phase_modes = "+".join(
                m for m, on in [
                    ("classify", phase.classify), ("regress", phase.regress),
                    ("synth", phase.synth), ("contrastive", phase.contrastive),
                    ("peprank", phase.peprank),
                ] if on
            )
            epoch_summary: Dict[str, Any] = {
                "epoch": global_epoch,
                "phase": phase_idx,
                "phase_epoch": phase_epoch,
                "phase_modes": phase_modes,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            if phase.classify:
                epoch_summary["classify_acc"] = classify_acc_sum / max(train_batches, 1)
            epoch_summaries.append(epoch_summary)

            summary = {
                "config": {
                    "design_id": str(args.design_id),
                    "model_variant": model_variant,
                    "n_params": n_params,
                    "embed_dim": int(args.embed_dim),
                    "hidden_dim": int(args.hidden_dim),
                    "peptide_pos_mode": str(args.peptide_pos_mode),
                    "groove_pos_mode": str(args.groove_pos_mode),
                    "probe_alleles": probe_alleles,
                    "probe_peptides": probe_peptides,
                    "total_epochs": total_epochs,
                    "batch_size": int(args.batch_size),
                    "seed": int(args.seed),
                    "synthetic_negatives": use_synthetics,
                    "negative_ratio": float(args.negative_ratio),
                    "synthetic_modes": list(synthetic_modes) if synthetic_modes else list(ALL_SYNTHETIC_MODES),
                    "per_epoch_regeneration": use_synthetics,
                    "class_i_anchor_strategy": str(args.class_i_anchor_strategy),
                    "qualifier_filter": str(args.qualifier_filter),
                    "curriculum": curriculum_desc,
                    "probe_plot_frequency": str(args.probe_plot_frequency),
                },
                "subset_stats": subset_stats,
                "balance_stats": balance_stats,
                "split_stats": split_stats,
                "mhc_resolve_stats": mhc_stats,
                "probe_allele_counts_after_filter": probe_allele_counts_after_filter,
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
                    or (
                        str(args.probe_plot_frequency) == "final"
                        and global_epoch == total_epochs
                    )
                ),
            )
            print(
                json.dumps({
                    "event": "groove_baseline_epoch",
                    **epoch_summary,
                    "probe_rows": probe_eval,
                }, sort_keys=True),
                flush=True,
            )


if __name__ == "__main__":
    main()
