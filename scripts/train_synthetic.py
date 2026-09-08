#!/usr/bin/env python
"""End-to-end training script with synthetic data.

This script demonstrates the full Presto training pipeline:
1. Generate synthetic training data
2. Create data loaders
3. Train the model
4. Evaluate on held-out data

Usage:
    python -m presto.scripts.train_synthetic --epochs 5 --batch_size 16
"""

import argparse
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import random_split
from tqdm.auto import tqdm

from presto.models.presto import Presto
from presto.models.affinity import (
    DEFAULT_MAX_AFFINITY_NM,
    max_log10_nM,
    normalize_binding_target_log10,
)
from presto.data import (
    PrestoDataset,
    PrestoCollator,
    create_dataloader,
    generate_synthetic_binding_data,
    generate_synthetic_elution_data,
    generate_synthetic_tcr_data,
    generate_synthetic_mhc_sequences,
    write_binding_csv,
    write_elution_csv,
    write_tcr_csv,
    write_mhc_fasta,
)
from presto.training.losses import UncertaintyWeighting
from presto.training.checkpointing import save_model_checkpoint
from presto.training.supervision import (
    TaskLossSpec as TaskLossSpec,
    LOSS_TASK_SPECS as LOSS_TASK_SPECS,
    LOSS_TASK_NAMES,
    LOSS_TASK_NAME_TO_INDEX,
    LOSS_TASK_NAME_TO_SPEC,
    PANEL_TASK_BASE_WEIGHTS,
    _ELUTION_SPEC as _ELUTION_SPEC,
    _as_float_vector,
    _resolve_output_tensor as _resolve_output_tensor,
    _resolve_task_prediction as _resolve_task_prediction,
    _get_batch_target as _get_batch_target,
    _get_batch_mask as _get_batch_mask,
    _get_batch_qual as _get_batch_qual,
    _compute_task_loss_vector as _compute_task_loss_vector,
    resolve_row_targets,
    resolve_row_predictions,
    reduce_row_predictions,
)
from presto.training.mil import (
    MIL_TASKS,
    MIL_TASK_BASE_WEIGHTS,
    get_mil_channel as _get_mil_channel,
    slice_mil_channel as _slice_mil_channel,
    run_mil_forward as _run_mil_forward,  # noqa: F401 -- compatibility export
    predict_mil_channel,
    resolve_mil_targets,
)
from presto.training.config_io import (
    load_config_file,
    merge_namespace_with_config,
    pick_train_section,
)
from presto.training.losses import PCGrad
from presto.training.run_logger import RunLogger
from presto.data.allele_resolver import (
    normalize_mhc_class,
    normalize_species_label,
)
from presto.data.vocab import (
    TCELL_CULTURE_CONTEXT_TO_IDX,
    TCELL_STIM_CONTEXT_TO_IDX,
)


SYNTHETIC_DEFAULTS = {
    "epochs": 10,
    "batch_size": 16,
    "lr": 1e-4,
    "d_model": 128,
    "n_layers": 2,
    "n_heads": 4,
    "n_binding": 200,
    "n_elution": 100,
    "n_tcr": 100,
    "data_dir": None,
    "checkpoint": None,
    "run_dir": None,
    "weight_decay": 0.01,
    "use_uncertainty_weighting": True,
    "supervised_loss_aggregation": "task_mean",
    "use_pcgrad": False,
    "seed": 42,
    "consistency_cascade_weight": 0.0,
    "consistency_assay_affinity_weight": 0.0,
    "consistency_assay_presentation_weight": 0.0,
    "consistency_no_b2m_weight": 0.0,
    "consistency_tcell_context_weight": 0.0,
    "consistency_tcell_upstream_weight": 0.0,
    "binding_orthogonality_weight": 0.01,
    "consistency_prob_margin": 0.02,
    "consistency_parent_low_threshold": 0.1,
    "consistency_presentation_high_threshold": 0.9,
    "consistency_affinity_fold_tolerance": 2.0,
    "tcell_in_vitro_margin": 0.0,
    "tcell_ex_vivo_margin": 0.0,
    "mhc_attention_sparsity_weight": 0.1,
    "mhc_attention_sparsity_min_residues": 25.0,
    "mhc_attention_sparsity_max_residues": 45.0,
    "mil_contrastive_weight": 0.0,
    "mil_contrastive_margin": 0.5,
    "mil_contrastive_max_pairs": 32,
    "binding_contrastive_weight": 0.0,
    "binding_contrastive_margin": 0.2,
    "binding_contrastive_target_gap_min": 0.3,
    "binding_contrastive_target_gap_cap": 2.0,
    "binding_contrastive_max_pairs": 64,
    "mil_bag_sparsity_weight": 0.0,
    "mil_bag_sparsity_target_sum": 1.5,
}


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    total_steps: int,
    warmup_fraction: float = 0.05,
    min_lr_scale: float = 0.1,
) -> Optional[SequentialLR]:
    """Build a linear-warmup + cosine-decay scheduler for step-wise updates."""
    total_steps = int(total_steps)
    if total_steps <= 1:
        return None

    warmup_steps = int(round(total_steps * warmup_fraction))
    warmup_steps = max(1, min(total_steps - 1, warmup_steps))
    start_factor = min(1.0, max(1e-4, 1e-6 / max(float(base_lr), 1e-12)))

    warmup = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=float(base_lr) * float(min_lr_scale),
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def _normalize_supervised_loss_aggregation(mode: Optional[str]) -> str:
    token = str(mode or "task_mean").strip().lower().replace("-", "_")
    if token in {"task", "task_mean", "equal_task"}:
        return "task_mean"
    if token in {"sample", "sample_weighted", "sample_count"}:
        return "sample_weighted"
    raise ValueError(
        "Unsupported supervised loss aggregation mode: "
        f"{mode!r}. Expected one of: task_mean, sample_weighted."
    )


def _resolve_run_args(args: argparse.Namespace) -> argparse.Namespace:
    for key, default in SYNTHETIC_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, default)
    config_path = getattr(args, "config", None)
    if not config_path:
        return args
    config = load_config_file(config_path)
    section = pick_train_section(config, "synthetic")
    return merge_namespace_with_config(args, SYNTHETIC_DEFAULTS, section)


def _regularization_config_from_args(args: argparse.Namespace) -> Dict[str, float]:
    """Extract biologic consistency/prior loss weights from run args."""
    return {
        "consistency_cascade_weight": float(getattr(args, "consistency_cascade_weight", 0.0)),
        "consistency_assay_affinity_weight": float(
            getattr(args, "consistency_assay_affinity_weight", 0.0)
        ),
        "consistency_assay_presentation_weight": float(
            getattr(args, "consistency_assay_presentation_weight", 0.0)
        ),
        "consistency_no_b2m_weight": float(getattr(args, "consistency_no_b2m_weight", 0.0)),
        "consistency_tcell_context_weight": float(
            getattr(args, "consistency_tcell_context_weight", 0.0)
        ),
        "consistency_tcell_upstream_weight": float(
            getattr(args, "consistency_tcell_upstream_weight", 0.0)
        ),
        "binding_orthogonality_weight": float(getattr(args, "binding_orthogonality_weight", 0.01)),
        "consistency_prob_margin": float(getattr(args, "consistency_prob_margin", 0.02)),
        "consistency_parent_low_threshold": float(
            getattr(args, "consistency_parent_low_threshold", 0.1)
        ),
        "consistency_presentation_high_threshold": float(
            getattr(args, "consistency_presentation_high_threshold", 0.9)
        ),
        "consistency_affinity_fold_tolerance": float(
            getattr(args, "consistency_affinity_fold_tolerance", 2.0)
        ),
        "tcell_in_vitro_margin": float(getattr(args, "tcell_in_vitro_margin", 0.0)),
        "tcell_ex_vivo_margin": float(getattr(args, "tcell_ex_vivo_margin", 0.0)),
        "mhc_attention_sparsity_weight": float(getattr(args, "mhc_attention_sparsity_weight", 0.0)),
        "mhc_attention_sparsity_min_residues": float(
            getattr(args, "mhc_attention_sparsity_min_residues", 30.0)
        ),
        "mhc_attention_sparsity_max_residues": float(
            getattr(args, "mhc_attention_sparsity_max_residues", 60.0)
        ),
        "mil_contrastive_weight": float(getattr(args, "mil_contrastive_weight", 0.0)),
        "mil_contrastive_margin": float(getattr(args, "mil_contrastive_margin", 0.5)),
        "mil_contrastive_max_pairs": float(getattr(args, "mil_contrastive_max_pairs", 32)),
        "binding_contrastive_weight": float(getattr(args, "binding_contrastive_weight", 0.0)),
        "binding_contrastive_margin": float(getattr(args, "binding_contrastive_margin", 0.2)),
        "binding_contrastive_target_gap_min": float(
            getattr(args, "binding_contrastive_target_gap_min", 0.3)
        ),
        "binding_contrastive_target_gap_cap": float(
            getattr(args, "binding_contrastive_target_gap_cap", 2.0)
        ),
        "binding_contrastive_max_pairs": float(getattr(args, "binding_contrastive_max_pairs", 64)),
        "mil_bag_sparsity_weight": float(getattr(args, "mil_bag_sparsity_weight", 0.0)),
        "mil_bag_sparsity_target_sum": float(getattr(args, "mil_bag_sparsity_target_sum", 1.5)),
    }


def _resolve_regularization_config(
    regularization: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    """Merge optional regularization overrides with defaults."""
    defaults = {
        "consistency_cascade_weight": 0.0,
        "consistency_assay_affinity_weight": 0.0,
        "consistency_assay_presentation_weight": 0.0,
        "consistency_no_b2m_weight": 0.0,
        "consistency_tcell_context_weight": 0.0,
        "consistency_tcell_upstream_weight": 0.0,
        "binding_orthogonality_weight": 0.01,
        "consistency_prob_margin": 0.02,
        "consistency_parent_low_threshold": 0.1,
        "consistency_presentation_high_threshold": 0.9,
        "consistency_affinity_fold_tolerance": 2.0,
        "tcell_in_vitro_margin": 0.0,
        "tcell_ex_vivo_margin": 0.0,
        "mhc_attention_sparsity_weight": 0.0,
        "mhc_attention_sparsity_min_residues": 30.0,
        "mhc_attention_sparsity_max_residues": 60.0,
        "mil_contrastive_weight": 0.0,
        "mil_contrastive_margin": 0.5,
        "mil_contrastive_max_pairs": 32.0,
        "binding_contrastive_weight": 0.0,
        "binding_contrastive_margin": 0.2,
        "binding_contrastive_target_gap_min": 0.3,
        "binding_contrastive_target_gap_cap": 2.0,
        "binding_contrastive_max_pairs": 64.0,
        "mil_bag_sparsity_weight": 0.0,
        "mil_bag_sparsity_target_sum": 1.5,
    }
    if regularization is None:
        return defaults
    merged = dict(defaults)
    for key, value in regularization.items():
        if key in merged:
            merged[key] = float(value)
    return merged


def create_synthetic_data(
    data_dir: Path, n_binding: int = 200, n_elution: int = 100, n_tcr: int = 100
):
    """Generate and save synthetic training data."""
    print("Generating synthetic data...")

    alleles = ["HLA-A*02:01", "HLA-A*03:01", "HLA-B*07:02", "HLA-B*08:01"]

    # Generate data
    binding_data = generate_synthetic_binding_data(n_binding, alleles)
    elution_data = generate_synthetic_elution_data(n_elution, alleles)
    tcr_data = generate_synthetic_tcr_data(n_tcr, alleles[:2])  # Fewer alleles for TCR
    mhc_sequences = generate_synthetic_mhc_sequences(alleles)

    # Save to files
    data_dir.mkdir(parents=True, exist_ok=True)
    write_binding_csv(binding_data, data_dir / "binding.csv")
    write_elution_csv(elution_data, data_dir / "elution.csv")
    write_tcr_csv(tcr_data, data_dir / "tcr.csv")
    write_mhc_fasta(mhc_sequences, data_dir / "mhc.fasta")

    print(f"  Binding samples: {len(binding_data)}")
    print(f"  Elution samples: {len(elution_data)}")
    print(f"  TCR samples: {len(tcr_data)}")
    print(f"  MHC alleles: {len(mhc_sequences)}")

    return binding_data, elution_data, tcr_data, mhc_sequences


def _flatten_output_metrics(prefix: str, value: object, metrics: Dict[str, float]) -> None:
    """Recursively summarize tensor outputs for logging."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float()
        if tensor.numel() == 0:
            return
        key = f"out_{prefix}_mean" if prefix else "out_mean"
        metrics[key] = float(tensor.mean().item())
        var_key = f"out_{prefix}_var" if prefix else "out_var"
        metrics[var_key] = float(tensor.var(unbiased=False).item())
        if prefix.endswith("_logit"):
            metrics[f"out_{prefix}_prob_mean"] = float(torch.sigmoid(tensor).mean().item())
            metrics[f"out_{prefix}_prob_var"] = float(
                torch.sigmoid(tensor).var(unbiased=False).item()
            )
        return

    if isinstance(value, dict):
        for child_key in sorted(value):
            child = value[child_key]
            child_prefix = f"{prefix}_{child_key}" if prefix else child_key
            _flatten_output_metrics(child_prefix, child, metrics)


def _summarize_outputs(outputs: Dict[str, object]) -> Dict[str, float]:
    """Summarize generated model outputs for epoch-level logging."""
    metrics: Dict[str, float] = {}
    for key in sorted(outputs):
        _flatten_output_metrics(str(key), outputs[key], metrics)
    return metrics


def _max_bucket_fraction(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    return float(max(counts.values()) / max(len(values), 1))


def _normalized_batch_species(value: Optional[str]) -> str:
    normalized = normalize_species_label(value)
    if normalized is not None:
        return normalized
    raw = str(value or "").strip()
    return raw if raw else "unknown_species"


def _batch_diversity_metrics(batch) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    alleles = [str(a).strip() for a in getattr(batch, "primary_alleles", []) if str(a).strip()]
    classes = [
        normalize_mhc_class(cls, default=None) or "unknown_class"
        for cls in getattr(batch, "mhc_class", [])
    ]
    species = [_normalized_batch_species(sp) for sp in getattr(batch, "processing_species", [])]

    if alleles:
        metrics["batch_unique_alleles"] = float(len(set(alleles)))
        metrics["batch_max_allele_fraction"] = _max_bucket_fraction(alleles)
    if classes:
        metrics["batch_unique_mhc_classes"] = float(len(set(classes)))
        metrics["batch_max_mhc_class_fraction"] = _max_bucket_fraction(classes)
    if species:
        metrics["batch_unique_species"] = float(len(set(species)))
        metrics["batch_max_species_fraction"] = _max_bucket_fraction(species)

    bind_mask = getattr(batch, "bind_mask", None)
    if (
        isinstance(bind_mask, torch.Tensor)
        and bind_mask.numel() == len(classes)
        and len(alleles) == len(classes)
        and len(species) == len(classes)
    ):
        bind_mask_bool = _as_float_vector(bind_mask) > 0
        bind_idx = bind_mask_bool.nonzero(as_tuple=False).view(-1).tolist()
        if bind_idx:
            bind_alleles = [alleles[i] for i in bind_idx if alleles[i]]
            bind_classes = [classes[i] for i in bind_idx]
            bind_species = [species[i] for i in bind_idx]
            metrics["batch_binding_samples"] = float(len(bind_idx))
            if bind_alleles:
                metrics["batch_binding_unique_alleles"] = float(len(set(bind_alleles)))
                metrics["batch_binding_max_allele_fraction"] = _max_bucket_fraction(bind_alleles)
            if bind_classes:
                metrics["batch_binding_unique_mhc_classes"] = float(len(set(bind_classes)))
            if bind_species:
                metrics["batch_binding_unique_species"] = float(len(set(bind_species)))
    return metrics


def _binding_path_diagnostic_metrics(outputs: Dict[str, object]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    latents = outputs.get("binding_latents")
    if isinstance(latents, dict):
        for name in ("log_koff", "log_kon_intrinsic", "log_kon_chaperone"):
            tensor = latents.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            vec = _as_float_vector(tensor.detach())
            metrics[f"out_binding_{name}_near_lower_rate"] = float(
                (vec <= -7.5).float().mean().item()
            )
            metrics[f"out_binding_{name}_near_upper_rate"] = float(
                (vec >= 7.5).float().mean().item()
            )

    probe_kd = outputs.get("binding_affinity_probe_kd")
    assays = outputs.get("assays")
    if isinstance(probe_kd, torch.Tensor) and isinstance(assays, dict):
        kd_tensor = assays.get("KD_nM")
        if isinstance(kd_tensor, torch.Tensor):
            probe_vec = _as_float_vector(probe_kd.detach())
            kd_vec = _as_float_vector(kd_tensor.detach())
            diff = probe_vec - kd_vec
            metrics["out_binding_probe_core_kd_l1_mean"] = float(diff.abs().mean().item())
            metrics["out_binding_probe_core_kd_signed_mean"] = float(diff.mean().item())

    kd_bias = outputs.get("binding_kd_bias")
    if isinstance(kd_bias, torch.Tensor) and kd_bias.numel() > 0:
        bias_vec = _as_float_vector(kd_bias.detach())
        metrics["out_binding_kd_bias_abs_mean"] = float(bias_vec.abs().mean().item())
        kd_bias_cap = outputs.get("binding_kd_bias_cap")
        if isinstance(kd_bias_cap, torch.Tensor) and kd_bias_cap.numel() > 0:
            cap = max(float(kd_bias_cap.detach().float().view(-1)[0].item()), 1e-6)
            metrics["out_binding_kd_bias_near_cap_rate"] = float(
                (bias_vec.abs() >= 0.9 * cap).float().mean().item()
            )
    kd_bias_raw = outputs.get("binding_kd_bias_raw")
    if isinstance(kd_bias_raw, torch.Tensor) and kd_bias_raw.numel() > 0:
        raw_vec = _as_float_vector(kd_bias_raw.detach())
        metrics["out_binding_kd_bias_raw_abs_mean"] = float(raw_vec.abs().mean().item())

    core_logit = outputs.get("binding_logit_from_core")
    if isinstance(core_logit, torch.Tensor) and core_logit.numel() > 0:
        core_vec = _as_float_vector(core_logit.detach())
        metrics["out_binding_logit_from_core_near_cap_rate"] = float(
            (core_vec.abs() >= 19.0).float().mean().item()
        )
    return metrics


def _collect_binding_contrastive_pairs(
    batch,
    *,
    target_gap_min: float,
    max_pairs: int,
) -> Tuple[List[Tuple[float, int, int]], Dict[str, float]]:
    metrics: Dict[str, float] = {
        "out_binding_same_peptide_diff_allele_pairs": 0.0,
        "out_binding_same_peptide_labeled_pairs": 0.0,
        "out_binding_same_peptide_exact_pairs": 0.0,
        "out_binding_same_peptide_rankable_pairs": 0.0,
    }
    pep_tok = getattr(batch, "pep_tok", None)
    bind_mask = getattr(batch, "bind_mask", None)
    bind_target = getattr(batch, "bind_target", None)
    bind_qual = getattr(batch, "bind_qual", None)
    primary_alleles = list(getattr(batch, "primary_alleles", []))
    if not (
        isinstance(pep_tok, torch.Tensor)
        and isinstance(bind_mask, torch.Tensor)
        and isinstance(bind_target, torch.Tensor)
        and isinstance(bind_qual, torch.Tensor)
        and len(primary_alleles) == int(pep_tok.shape[0])
    ):
        return [], metrics

    pep_rows = pep_tok.detach().cpu().tolist()
    peptide_groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for idx, row in enumerate(pep_rows):
        peptide_groups[tuple(int(v) for v in row)].append(idx)

    bind_mask_vec = (_as_float_vector(bind_mask).detach().cpu() > 0).tolist()
    bind_target_log10 = normalize_binding_target_log10(
        _as_float_vector(bind_target).detach().cpu(),
        assume_log10=False,
    ).tolist()
    bind_qual_vec = _as_float_vector(bind_qual).detach().cpu().tolist()
    min_log10 = -3.0
    max_log10 = max_log10_nM(DEFAULT_MAX_AFFINITY_NM)

    candidates: List[Tuple[float, int, int]] = []
    target_gaps: List[float] = []
    for indices in peptide_groups.values():
        if len(indices) < 2:
            continue
        for pos, idx_i in enumerate(indices):
            allele_i = str(primary_alleles[idx_i]).strip()
            if not allele_i:
                continue
            for idx_j in indices[pos + 1 :]:
                allele_j = str(primary_alleles[idx_j]).strip()
                if not allele_j or allele_i == allele_j:
                    continue
                metrics["out_binding_same_peptide_diff_allele_pairs"] += 1.0
                if not (bind_mask_vec[idx_i] and bind_mask_vec[idx_j]):
                    continue
                metrics["out_binding_same_peptide_labeled_pairs"] += 1.0
                qual_i = int(round(bind_qual_vec[idx_i]))
                qual_j = int(round(bind_qual_vec[idx_j]))
                if qual_i == 0 and qual_j == 0:
                    metrics["out_binding_same_peptide_exact_pairs"] += 1.0

                value_i = float(bind_target_log10[idx_i])
                value_j = float(bind_target_log10[idx_j])
                lower_i = min_log10 if qual_i < 0 else value_i
                upper_i = max_log10 if qual_i > 0 else value_i
                lower_j = min_log10 if qual_j < 0 else value_j
                upper_j = max_log10 if qual_j > 0 else value_j

                stronger: Optional[int] = None
                weaker: Optional[int] = None
                if upper_i + float(target_gap_min) <= lower_j:
                    stronger, weaker = idx_i, idx_j
                elif upper_j + float(target_gap_min) <= lower_i:
                    stronger, weaker = idx_j, idx_i
                if stronger is None or weaker is None:
                    continue

                gap = max(lower_j - upper_i, lower_i - upper_j)
                metrics["out_binding_same_peptide_rankable_pairs"] += 1.0
                candidates.append((gap, stronger, weaker))
                target_gaps.append(gap)

    candidates.sort(key=lambda item: item[0], reverse=True)
    if target_gaps:
        metrics["out_binding_same_peptide_target_gap_mean"] = float(
            sum(target_gaps) / len(target_gaps)
        )
    if max_pairs > 0:
        candidates = candidates[:max_pairs]
    metrics["out_binding_same_peptide_pairs_used"] = float(len(candidates))
    return candidates, metrics


def _compute_binding_contrastive_loss(
    outputs: Dict[str, object],
    batch,
    regularization: Mapping[str, float],
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    pair_candidates, metrics = _collect_binding_contrastive_pairs(
        batch,
        target_gap_min=float(regularization.get("binding_contrastive_target_gap_min", 0.3)),
        max_pairs=int(regularization.get("binding_contrastive_max_pairs", 64)),
    )
    assays = outputs.get("assays")
    if not isinstance(assays, dict):
        return None, metrics
    kd_tensor = assays.get("KD_nM")
    if not isinstance(kd_tensor, torch.Tensor):
        return None, metrics
    if not pair_candidates:
        return None, metrics

    kd_vec = _as_float_vector(kd_tensor)
    margin = float(regularization.get("binding_contrastive_margin", 0.2))
    target_gap_cap = float(regularization.get("binding_contrastive_target_gap_cap", 2.0))
    if target_gap_cap <= 0.0:
        target_gap_cap = margin
    weight = float(regularization.get("binding_contrastive_weight", 0.0))
    pair_losses: List[torch.Tensor] = []
    pred_gaps: List[float] = []
    target_gaps: List[float] = []
    required_gaps: List[float] = []
    for gap, stronger_idx, weaker_idx in pair_candidates:
        pred_gap = kd_vec[weaker_idx] - kd_vec[stronger_idx]
        required_gap = max(margin, min(float(gap), target_gap_cap))
        pair_losses.append(torch.relu(kd_vec.new_tensor(required_gap) - pred_gap))
        pred_gaps.append(float(pred_gap.detach().item()))
        target_gaps.append(float(gap))
        required_gaps.append(float(required_gap))

    metrics["out_binding_contrastive_pred_gap_mean"] = float(
        sum(pred_gaps) / max(len(pred_gaps), 1)
    )
    metrics["out_binding_contrastive_target_gap_mean"] = float(
        sum(target_gaps) / max(len(target_gaps), 1)
    )
    metrics["out_binding_contrastive_required_gap_mean"] = float(
        sum(required_gaps) / max(len(required_gaps), 1)
    )
    if weight <= 0.0:
        return None, metrics
    return weight * torch.stack(pair_losses).mean(), metrics


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        if values.numel() == 0:
            return None
        return values.mean()
    mask_vec = _as_float_vector(mask).to(device=values.device, dtype=torch.float32)
    if mask_vec.shape != values.shape:
        mask_vec = mask_vec.view(values.shape)
    denom = mask_vec.sum()
    if float(denom.item()) <= 0.0:
        return None
    return (values * mask_vec).sum() / (denom + 1e-8)


def _bag_aware_instance_cap(
    instance_to_bag: torch.Tensor,
    max_mil_instances: int,
) -> torch.Tensor:
    """Select capped MIL instances while preserving at least one per bag."""
    n_instances = int(instance_to_bag.shape[0])
    if max_mil_instances <= 0 or n_instances <= max_mil_instances:
        return torch.arange(n_instances, device=instance_to_bag.device, dtype=torch.long)

    n_bags = int(instance_to_bag.max().item()) + 1 if n_instances > 0 else 0
    target_n = max(max_mil_instances, n_bags)
    if n_instances <= target_n:
        return torch.arange(n_instances, device=instance_to_bag.device, dtype=torch.long)

    perm = torch.randperm(n_instances, device=instance_to_bag.device)
    bag_selected = torch.zeros((n_bags,), dtype=torch.bool, device=instance_to_bag.device)
    chosen = []
    for idx_t in perm:
        idx = int(idx_t.item())
        bag_idx = int(instance_to_bag[idx].item())
        if not bag_selected[bag_idx]:
            bag_selected[bag_idx] = True
            chosen.append(idx)
            if bool(bag_selected.all()):
                break

    chosen_mask = torch.zeros((n_instances,), dtype=torch.bool, device=instance_to_bag.device)
    if chosen:
        chosen_mask[torch.tensor(chosen, dtype=torch.long, device=instance_to_bag.device)] = True

    remaining = target_n - len(chosen)
    if remaining > 0:
        remainder = torch.nonzero(~chosen_mask, as_tuple=False).squeeze(-1)
        if remainder.numel() > 0:
            extra_perm = torch.randperm(remainder.numel(), device=instance_to_bag.device)
            extra = remainder[extra_perm[:remaining]]
            chosen.extend(extra.tolist())

    keep = torch.tensor(chosen, dtype=torch.long, device=instance_to_bag.device)
    keep, _ = torch.sort(keep)
    return keep


def _group_mil_instances(
    instance_to_bag: torch.Tensor,
    n_bags: int,
) -> List[List[int]]:
    grouped: List[List[int]] = [[] for _ in range(n_bags)]
    for idx in range(int(instance_to_bag.shape[0])):
        bag_idx = int(instance_to_bag[idx].item())
        if 0 <= bag_idx < n_bags:
            grouped[bag_idx].append(idx)
    return grouped


def _token_sequence_identity(a: torch.Tensor, b: torch.Tensor) -> float:
    mask = (a != 0) & (b != 0)
    if int(mask.sum().item()) <= 0:
        return 0.0
    return float((a[mask] == b[mask]).float().mean().item())


def _bag_max_mhc_identity(
    mhc_a_tok: torch.Tensor,
    bag_instances_a: Sequence[int],
    bag_instances_b: Sequence[int],
) -> float:
    max_identity = 0.0
    for idx_a in bag_instances_a:
        seq_a = mhc_a_tok[idx_a]
        for idx_b in bag_instances_b:
            identity = _token_sequence_identity(seq_a, mhc_a_tok[idx_b])
            if identity > max_identity:
                max_identity = identity
    return max_identity


def _select_mil_contrastive_pairs(
    *,
    mhc_a_tok: torch.Tensor,
    mhc_class: Sequence[str],
    bag_label: torch.Tensor,
    instance_to_bag: torch.Tensor,
    max_pairs: int,
    max_identity: float = 0.90,
) -> List[tuple[int, int, float]]:
    n_bags = int(bag_label.shape[0])
    grouped = _group_mil_instances(instance_to_bag, n_bags)
    bag_classes: List[str] = []
    for bag_idx in range(n_bags):
        if not grouped[bag_idx]:
            bag_classes.append("")
            continue
        class_name = (
            normalize_mhc_class(
                mhc_class[grouped[bag_idx][0]] if mhc_class else None,
                default=None,
            )
            or ""
        )
        bag_classes.append(class_name)

    pairs: List[tuple[int, int, float]] = []
    positive_bags = torch.nonzero(bag_label > 0.5, as_tuple=False).squeeze(-1).tolist()
    for bag_idx in positive_bags:
        if not grouped[bag_idx]:
            continue
        bag_class = bag_classes[bag_idx]
        if bag_class not in {"I", "II"}:
            continue
        best_candidate: Optional[int] = None
        best_identity = 1.0
        for cand_idx in range(n_bags):
            if cand_idx == bag_idx or not grouped[cand_idx]:
                continue
            if bag_classes[cand_idx] != bag_class:
                continue
            identity = _bag_max_mhc_identity(
                mhc_a_tok=mhc_a_tok,
                bag_instances_a=grouped[bag_idx],
                bag_instances_b=grouped[cand_idx],
            )
            if identity >= max_identity:
                continue
            if identity < best_identity:
                best_identity = identity
                best_candidate = cand_idx
        if best_candidate is not None:
            pairs.append((bag_idx, best_candidate, best_identity))

    pairs.sort(key=lambda item: item[2])
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    return pairs


def _build_contrastive_mil_channel(
    channel: Dict[str, Any],
    pairs: Sequence[tuple[int, int, float]],
) -> Optional[Dict[str, Any]]:
    if not pairs:
        return None
    instance_to_bag = channel["instance_to_bag"].to(dtype=torch.long)
    n_bags = int(channel["bag_label"].shape[0])
    grouped = _group_mil_instances(instance_to_bag, n_bags)

    anchor_instance_idx: List[int] = []
    candidate_instance_idx: List[int] = []
    contrastive_instance_to_bag: List[int] = []
    anchor_bag_indices: List[int] = []

    for new_bag_idx, (anchor_bag_idx, candidate_bag_idx, _) in enumerate(pairs):
        anchor_instances = grouped[anchor_bag_idx]
        candidate_instances = grouped[candidate_bag_idx]
        if not anchor_instances or not candidate_instances:
            continue
        anchor_seed = anchor_instances[0]
        anchor_bag_indices.append(anchor_bag_idx)
        for cand_instance in candidate_instances:
            anchor_instance_idx.append(anchor_seed)
            candidate_instance_idx.append(cand_instance)
            contrastive_instance_to_bag.append(new_bag_idx)

    if not candidate_instance_idx:
        return None

    anchor_index_t = torch.tensor(
        anchor_instance_idx,
        device=channel["pep_tok"].device,
        dtype=torch.long,
    )
    candidate_index_t = torch.tensor(
        candidate_instance_idx,
        device=channel["pep_tok"].device,
        dtype=torch.long,
    )
    contrastive: Dict[str, Any] = {
        "pep_tok": channel["pep_tok"][anchor_index_t],
        "mhc_a_tok": channel["mhc_a_tok"][candidate_index_t],
        "mhc_b_tok": channel["mhc_b_tok"][candidate_index_t],
        "mhc_class": [channel["mhc_class"][i] for i in candidate_instance_idx],
        "species": [channel["species"][i] for i in candidate_instance_idx],
        "instance_to_bag": torch.tensor(
            contrastive_instance_to_bag,
            device=channel["pep_tok"].device,
            dtype=torch.long,
        ),
        "anchor_bag_indices": torch.tensor(
            anchor_bag_indices,
            device=channel["pep_tok"].device,
            dtype=torch.long,
        ),
    }
    if isinstance(channel.get("flank_n_tok"), torch.Tensor):
        contrastive["flank_n_tok"] = channel["flank_n_tok"][anchor_index_t]
    else:
        contrastive["flank_n_tok"] = None
    if isinstance(channel.get("flank_c_tok"), torch.Tensor):
        contrastive["flank_c_tok"] = channel["flank_c_tok"][anchor_index_t]
    else:
        contrastive["flank_c_tok"] = None
    for key in ("flank_n_is_terminus", "flank_c_is_terminus"):
        value = channel.get(key)
        contrastive[key] = value[anchor_index_t] if isinstance(value, torch.Tensor) else None
    # Cellular state follows the anchor, like pep_tok and the flanks: the
    # condition belongs to the sample the peptide came from, and only the MHC
    # is swapped in to make the synthetic negative. Omitting it would run every
    # contrastive instance at the default condition while real bags run at
    # their true one, letting processing_condition_embed learn
    # "default state => negative" instead of any real biology.
    provenance = channel.get("provenance")
    if isinstance(provenance, dict):
        contrastive["provenance"] = {
            name: (tensor[anchor_index_t] if isinstance(tensor, torch.Tensor) else tensor)
            for name, tensor in provenance.items()
        }
    # Machinery follows the anchor too. `contrastive` is built fresh rather
    # than copied from `channel`, so omitting this left the contrastive pass
    # with machinery=None while the anchors it is compared against use the
    # declared value -- the two logits in the contrastive loss would then be
    # computed under different machinery-derivation rules.
    machinery = channel.get("machinery_idx")
    if isinstance(machinery, torch.Tensor):
        contrastive["machinery_idx"] = machinery[anchor_index_t]
    return contrastive


def _compute_mil_channel_losses(
    model,
    *,
    batch,
    device: str,
    channel_prefix: str,
    regularization: Mapping[str, float],
    max_mil_instances: int = 0,
    mil_chunk_size: int = 0,
    enable_contrastive: bool = False,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float], Dict[str, float]]:
    """Consume the same effective bag predictions exported by held-out eval."""
    mil_losses, mil_regularization, mil_metrics, support = {}, {}, {}, {}
    channel = _get_mil_channel(batch, channel_prefix)
    if channel is None or channel["bag_label"].numel() == 0:
        return mil_losses, mil_regularization, mil_metrics, support

    targets = resolve_mil_targets(channel, MIL_TASKS[channel_prefix])
    if max_mil_instances > 0:
        keep = _bag_aware_instance_cap(channel["instance_to_bag"].long(), max_mil_instances)
        channel = _slice_mil_channel(channel, keep)
    predictions = predict_mil_channel(
        model,
        channel=channel,
        targets=targets,
        device=device,
        chunk_size=mil_chunk_size,
    )
    bag_label = channel["bag_label"].to(device=device, dtype=torch.float32)
    instance_to_bag = channel["instance_to_bag"].to(device=device, dtype=torch.long)
    bag_probs_by_task = {}
    sparsity_weight = float(regularization.get("mil_bag_sparsity_weight", 0.0))
    sparsity_target = float(regularization.get("mil_bag_sparsity_target_sum", 1.5))
    for task_name, prediction in predictions.items():
        mask = prediction.target.mask
        bag_probs = prediction.probabilities
        mil_losses[task_name] = prediction.loss()
        support[task_name] = float(prediction.target.support)
        mil_metrics[f"batch_support_{task_name}"] = support[task_name]
        bag_probs_by_task[task_name] = bag_probs
        mil_metrics[f"out_{channel_prefix}_{task_name}_prob_mean"] = float(
            bag_probs[mask].detach().mean().item()
        )
        mil_metrics[f"out_{channel_prefix}_{task_name}_prob_var"] = float(
            bag_probs[mask].detach().var(unbiased=False).item()
        )
        if sparsity_weight > 0.0:
            bag_sum = prediction.instance_probability_sums[mask]
            mil_regularization[f"{task_name}_mil_sparsity"] = (
                sparsity_weight * F.softplus(bag_sum - sparsity_target).mean()
            )
            mil_metrics[f"out_{channel_prefix}_{task_name}_bag_sum_mean"] = float(
                bag_sum.detach().mean().item()
            )

    contrastive_weight = float(regularization.get("mil_contrastive_weight", 0.0))
    if enable_contrastive and contrastive_weight > 0.0 and "presentation" in bag_probs_by_task:
        pairs = _select_mil_contrastive_pairs(
            mhc_a_tok=channel["mhc_a_tok"],
            mhc_class=channel["mhc_class"] or [],
            bag_label=bag_label,
            instance_to_bag=instance_to_bag,
            max_pairs=int(regularization.get("mil_contrastive_max_pairs", 32)),
        )
        contrastive_channel = _build_contrastive_mil_channel(channel, pairs)
        if contrastive_channel is not None:
            contrastive_channel["bag_label"] = torch.zeros(
                len(contrastive_channel["anchor_bag_indices"]),
                device=bag_label.device,
            )
            contrastive_targets = resolve_mil_targets(
                contrastive_channel,
                [spec for spec in MIL_TASKS["mil"] if spec.name == "presentation"],
            )
            contrastive_predictions = predict_mil_channel(
                model,
                channel=contrastive_channel,
                targets=contrastive_targets,
                device=device,
                chunk_size=mil_chunk_size,
            )
            if "presentation" in contrastive_predictions:
                contrastive_bag_probs = contrastive_predictions["presentation"].probabilities
                original_bag_probs = bag_probs_by_task["presentation"][
                    contrastive_channel["anchor_bag_indices"]
                ]
                eps = 1e-6
                original_scores = torch.logit(original_bag_probs.clamp(eps, 1.0 - eps))
                contrastive_scores = torch.logit(contrastive_bag_probs.clamp(eps, 1.0 - eps))
                margin = float(regularization.get("mil_contrastive_margin", 0.5))
                contrastive_loss = F.relu(contrastive_scores - original_scores + margin).mean()
                mil_regularization["presentation_mil_contrastive"] = (
                    contrastive_weight * contrastive_loss
                )
                mil_metrics["out_mil_contrastive_pairs"] = float(
                    int(contrastive_channel["anchor_bag_indices"].shape[0])
                )
                mil_metrics["out_mil_presentation_contrastive_gap_mean"] = float(
                    (original_scores - contrastive_scores).detach().mean().item()
                )

    return mil_losses, mil_regularization, mil_metrics, support


def _compute_consistency_losses(
    outputs: Dict[str, Any],
    batch,
    regularization: Mapping[str, float],
) -> Dict[str, torch.Tensor]:
    """Compute biologic-prior consistency losses."""
    losses: Dict[str, torch.Tensor] = {}
    cascade_w = float(regularization.get("consistency_cascade_weight", 0.0))
    affinity_w = float(regularization.get("consistency_assay_affinity_weight", 0.0))
    assay_pres_w = float(regularization.get("consistency_assay_presentation_weight", 0.0))
    no_b2m_w = float(regularization.get("consistency_no_b2m_weight", 0.0))
    tcell_ctx_w = float(regularization.get("consistency_tcell_context_weight", 0.0))
    tcell_upstream_w = float(regularization.get("consistency_tcell_upstream_weight", 0.0))
    orthogonality_w = float(regularization.get("binding_orthogonality_weight", 0.01))
    mhc_attn_sparse_w = float(regularization.get("mhc_attention_sparsity_weight", 0.0))
    mhc_attn_sparse_min = float(regularization.get("mhc_attention_sparsity_min_residues", 30.0))
    mhc_attn_sparse_max = float(regularization.get("mhc_attention_sparsity_max_residues", 60.0))
    prob_margin = float(regularization.get("consistency_prob_margin", 0.02))
    parent_low_thr = float(regularization.get("consistency_parent_low_threshold", 0.1))
    pres_high_thr = float(regularization.get("consistency_presentation_high_threshold", 0.9))
    affinity_fold_tol = max(
        float(regularization.get("consistency_affinity_fold_tolerance", 2.0)),
        1.0,
    )
    affinity_log10_tol = torch.log10(
        torch.tensor(affinity_fold_tol, device=batch.pep_tok.device, dtype=torch.float32)
    )
    in_vitro_margin = float(regularization.get("tcell_in_vitro_margin", 0.0))
    ex_vivo_margin = float(regularization.get("tcell_ex_vivo_margin", 0.0))

    device = batch.pep_tok.device

    if orthogonality_w > 0:
        latent_vecs = outputs.get("latent_vecs")
        if isinstance(latent_vecs, dict):
            affinity_vec = latent_vecs.get("binding_affinity")
            stability_vec = latent_vecs.get("binding_stability")
            if isinstance(affinity_vec, torch.Tensor) and isinstance(stability_vec, torch.Tensor):
                affinity_unit = F.normalize(affinity_vec.float(), dim=-1, eps=1e-8)
                stability_unit = F.normalize(stability_vec.float(), dim=-1, eps=1e-8)
                cosine_abs = torch.abs((affinity_unit * stability_unit).sum(dim=-1))
                losses["consistency_binding_orthogonality"] = orthogonality_w * cosine_abs.mean()

    if cascade_w > 0:
        proc_prob = torch.sigmoid(_as_float_vector(outputs["processing_logit"]))
        bind_prob = torch.sigmoid(_as_float_vector(outputs["binding_logit"]))
        pres_prob = torch.sigmoid(_as_float_vector(outputs["presentation_logit"]))
        parent_min = torch.minimum(proc_prob, bind_prob)
        high_pres = torch.relu(pres_prob - pres_high_thr)
        low_parent = torch.relu(parent_low_thr - parent_min)
        losses["consistency_cascade"] = cascade_w * ((high_pres * low_parent).square().mean())

    assays = outputs.get("assays", {})
    if affinity_w > 0 and isinstance(assays, dict):
        kd = assays.get("KD_nM")
        ic50 = assays.get("IC50_nM")
        ec50 = assays.get("EC50_nM")
        if (
            isinstance(kd, torch.Tensor)
            and isinstance(ic50, torch.Tensor)
            and isinstance(ec50, torch.Tensor)
        ):
            kd_vec = _as_float_vector(kd)
            ic50_vec = _as_float_vector(ic50)
            ec50_vec = _as_float_vector(ec50)
            target_masks = getattr(batch, "target_masks", None)
            if isinstance(target_masks, dict):
                kd_supervised = target_masks.get(
                    "binding_kd",
                    torch.zeros_like(kd_vec),
                ).to(device=kd_vec.device, dtype=torch.float32)
                ic50_supervised = target_masks.get(
                    "binding_ic50",
                    torch.zeros_like(kd_vec),
                ).to(device=kd_vec.device, dtype=torch.float32)
                ec50_supervised = target_masks.get(
                    "binding_ec50",
                    torch.zeros_like(kd_vec),
                ).to(device=kd_vec.device, dtype=torch.float32)
            else:
                kd_supervised = torch.zeros_like(kd_vec)
                ic50_supervised = torch.zeros_like(kd_vec)
                ec50_supervised = torch.zeros_like(kd_vec)

            unsupervised_mask = 1.0 - torch.clamp(
                kd_supervised + ic50_supervised + ec50_supervised,
                min=0.0,
                max=1.0,
            )
            if float(unsupervised_mask.sum().item()) > 0:
                pairwise_terms = []
                for a, b in ((kd_vec, ic50_vec), (kd_vec, ec50_vec), (ic50_vec, ec50_vec)):
                    diff = torch.abs(a - b)
                    violation = torch.relu(diff - affinity_log10_tol)
                    reduced = _masked_mean(violation.square(), unsupervised_mask)
                    if reduced is not None:
                        pairwise_terms.append(reduced)
                if pairwise_terms:
                    losses["consistency_affinity_heads"] = affinity_w * (
                        sum(pairwise_terms) / len(pairwise_terms)
                    )

    if assay_pres_w > 0:
        pres_vec = _as_float_vector(outputs["presentation_logit"])
        loss_terms = []
        elut_mask = getattr(batch, "elution_mask", None)
        elution_vec = None
        if "elution_logit" in outputs:
            elution_vec = _as_float_vector(outputs["elution_logit"])
        elif "ms_logit" in outputs:
            elution_vec = _as_float_vector(outputs["ms_logit"])

        if elution_vec is not None:
            if "ms_detectability_logit" in outputs:
                ms_detect_vec = _as_float_vector(outputs["ms_detectability_logit"])
                expected_elution = pres_vec + ms_detect_vec
                mse = (elution_vec - expected_elution).square()
                reduced = _masked_mean(mse, elut_mask)
                if reduced is not None:
                    loss_terms.append(reduced)
            elif "ms_logit" in outputs:
                ms_vec = _as_float_vector(outputs["ms_logit"])
                mse = (elution_vec - ms_vec).square()
                reduced = _masked_mean(mse, elut_mask)
                if reduced is not None:
                    loss_terms.append(reduced)
            else:
                # Backward-compatible fallback for minimal output dictionaries.
                mse = (elution_vec - pres_vec).square()
                reduced = _masked_mean(mse, elut_mask)
                if reduced is not None:
                    loss_terms.append(reduced)
        if loss_terms:
            losses["consistency_assay_presentation"] = assay_pres_w * (
                sum(loss_terms) / len(loss_terms)
            )

    if (
        no_b2m_w > 0
        and getattr(batch, "mhc_a_tok", None) is not None
        and getattr(batch, "mhc_b_tok", None) is not None
    ):
        class_i_flags = []
        class_ii_flags = []
        for cls in getattr(batch, "mhc_class", []):
            normalized = normalize_mhc_class(str(cls), default="I")
            class_i_flags.append(1.0 if normalized == "I" else 0.0)
            class_ii_flags.append(1.0 if normalized == "II" else 0.0)
        if class_i_flags:
            class_i_mask = torch.tensor(class_i_flags, device=device, dtype=torch.float32)
            class_ii_mask = torch.tensor(class_ii_flags, device=device, dtype=torch.float32)
            has_alpha = (batch.mhc_a_tok != 0).any(dim=1).float()
            has_beta = (batch.mhc_b_tok != 0).any(dim=1).float()
            single_chain = (has_alpha - has_beta).abs()
            class_i_invalid = class_i_mask * single_chain
            class_ii_invalid = class_ii_mask * single_chain
            prior_mask = torch.clamp(class_i_invalid + class_ii_invalid, min=0.0, max=1.0)
            if float(prior_mask.sum().item()) > 0:
                prior_terms = []
                zeros = torch.zeros_like(prior_mask)
                for key in ("binding_logit", "presentation_logit", "elution_logit", "ms_logit"):
                    if key not in outputs:
                        continue
                    logit_vec = _as_float_vector(outputs[key])
                    bce = nn.functional.binary_cross_entropy_with_logits(
                        logit_vec,
                        zeros,
                        reduction="none",
                    )
                    reduced = _masked_mean(bce, prior_mask)
                    if reduced is not None:
                        prior_terms.append(reduced)
                if prior_terms:
                    losses["consistency_chain_assembly"] = no_b2m_w * (
                        sum(prior_terms) / len(prior_terms)
                    )

    if tcell_ctx_w > 0 and "tcell_logit" in outputs and "immunogenicity_logit" in outputs:
        tcell_mask = getattr(batch, "tcell_mask", None)
        if tcell_mask is not None and float(tcell_mask.sum().item()) > 0:
            stim_idx = None
            culture_idx = None
            if isinstance(getattr(batch, "tcell_context", None), dict):
                stim_idx = batch.tcell_context.get("stim_context_idx")
                culture_idx = batch.tcell_context.get("culture_context_idx")
            if isinstance(stim_idx, torch.Tensor) and isinstance(culture_idx, torch.Tensor):
                tcell_vec = _as_float_vector(outputs["tcell_logit"])
                ig_vec = _as_float_vector(outputs["immunogenicity_logit"])
                delta = tcell_vec - ig_vec
                valid_mask = _as_float_vector(tcell_mask) > 0

                in_vitro_mask = (
                    (stim_idx == TCELL_STIM_CONTEXT_TO_IDX["IN_VITRO_STIM"])
                    | (culture_idx == TCELL_CULTURE_CONTEXT_TO_IDX["IN_VITRO"])
                    | (culture_idx == TCELL_CULTURE_CONTEXT_TO_IDX["SHORT_RESTIM"])
                ) & valid_mask
                ex_vivo_mask = (
                    (stim_idx == TCELL_STIM_CONTEXT_TO_IDX["EX_VIVO"])
                    | (culture_idx == TCELL_CULTURE_CONTEXT_TO_IDX["DIRECT_EX_VIVO"])
                ) & valid_mask

                ctx_terms = []
                if bool(in_vitro_mask.any()):
                    ctx_terms.append(
                        torch.relu(in_vitro_margin - delta[in_vitro_mask]).square().mean()
                    )
                if bool(ex_vivo_mask.any()):
                    ctx_terms.append(
                        torch.relu(delta[ex_vivo_mask] - ex_vivo_margin).square().mean()
                    )
                if ctx_terms:
                    losses["consistency_tcell_context"] = tcell_ctx_w * (
                        sum(ctx_terms) / len(ctx_terms)
                    )

    if (
        tcell_upstream_w > 0
        and "tcell_logit" in outputs
        and "binding_logit" in outputs
        and "presentation_logit" in outputs
    ):
        tcell_prob = torch.sigmoid(_as_float_vector(outputs["tcell_logit"]))
        bind_prob = torch.sigmoid(_as_float_vector(outputs["binding_logit"]))
        pres_prob = torch.sigmoid(_as_float_vector(outputs["presentation_logit"]))
        upstream_cap = torch.minimum(bind_prob, pres_prob) + prob_margin
        upstream_cap = torch.clamp(upstream_cap, min=0.0, max=1.0)
        upstream_violation = torch.relu(tcell_prob - upstream_cap).square()
        tcell_mask = getattr(batch, "tcell_mask", None)
        reduced = _masked_mean(upstream_violation, tcell_mask)
        if reduced is not None:
            losses["consistency_tcell_upstream"] = tcell_upstream_w * reduced

    if mhc_attn_sparse_w > 0:
        effective = outputs.get("binding_mhc_attention_effective_residues")
        valid_mask = outputs.get("binding_mhc_attention_valid_mask")
        if isinstance(effective, torch.Tensor):
            if not isinstance(valid_mask, torch.Tensor):
                valid_mask = torch.ones_like(effective)
            lower = torch.relu(
                torch.tensor(mhc_attn_sparse_min, device=effective.device) - effective
            )
            upper = torch.relu(
                effective - torch.tensor(mhc_attn_sparse_max, device=effective.device)
            )
            penalty = lower.square() + upper.square()
            reduced = _masked_mean(penalty, valid_mask)
            if reduced is not None:
                losses["consistency_binding_mhc_attention_sparsity"] = mhc_attn_sparse_w * reduced

    return losses


def compute_loss(
    model,
    batch,
    device,
    uncertainty_weighting=None,
    regularization: Optional[Mapping[str, float]] = None,
    supervised_loss_aggregation: str = "task_mean",
    profile_performance: bool = False,
    non_blocking_transfer: bool = False,
    use_amp: bool = False,
    max_mil_instances: int = 0,
    mil_chunk_size: int = 0,
):
    """Compute multi-task loss for a batch."""
    # Move batch to device
    try:
        batch = batch.to(device, non_blocking=non_blocking_transfer)
    except TypeError:
        batch = batch.to(device)
    regularization_cfg = _resolve_regularization_config(regularization)
    aggregation_mode = _normalize_supervised_loss_aggregation(supervised_loss_aggregation)
    perf_start = time.perf_counter() if profile_performance else 0.0
    perf_metrics: Dict[str, float] = {}

    # bf16 AMP: autocast covers forward pass + loss computation.
    # When enabled=False this is a no-op (safe for CPU tests).
    amp_enabled = use_amp and str(device).startswith("cuda")
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled)

    # Forward pass
    forward_start = time.perf_counter() if profile_performance else 0.0
    return_binding_attention = bool(
        float(regularization_cfg.get("mhc_attention_sparsity_weight", 0.0)) > 0.0
    )
    with amp_ctx:
        outputs = model(
            **batch.model_inputs(),
            return_binding_attention=return_binding_attention,
        )
        if profile_performance:
            perf_metrics["perf_forward_main_sec"] = float(time.perf_counter() - forward_start)
        output_metrics = _summarize_outputs(outputs)
        output_metrics.update(_batch_diversity_metrics(batch))
        output_metrics.update(_binding_path_diagnostic_metrics(outputs))
        has_mil_elution = (
            getattr(batch, "mil_bag_label", None) is not None
            and getattr(batch, "mil_instance_to_bag", None) is not None
        )
        has_tcell_mil = (
            getattr(batch, "tcell_mil_bag_label", None) is not None
            and getattr(batch, "tcell_mil_instance_to_bag", None) is not None
        )
        if profile_performance:
            mil_instance_to_bag = getattr(batch, "mil_instance_to_bag", None)
            mil_bag_label = getattr(batch, "mil_bag_label", None)
            perf_metrics["perf_mil_instances"] = float(
                int(mil_instance_to_bag.numel())
                if isinstance(mil_instance_to_bag, torch.Tensor)
                else 0
            )
            perf_metrics["perf_mil_bags"] = float(
                int(mil_bag_label.numel()) if isinstance(mil_bag_label, torch.Tensor) else 0
            )
            tcell_mil_instance_to_bag = getattr(batch, "tcell_mil_instance_to_bag", None)
            tcell_mil_bag_label = getattr(batch, "tcell_mil_bag_label", None)
            perf_metrics["perf_tcell_mil_instances"] = float(
                int(tcell_mil_instance_to_bag.numel())
                if isinstance(tcell_mil_instance_to_bag, torch.Tensor)
                else 0
            )
            perf_metrics["perf_tcell_mil_bags"] = float(
                int(tcell_mil_bag_label.numel())
                if isinstance(tcell_mil_bag_label, torch.Tensor)
                else 0
            )

        supervised_start = time.perf_counter() if profile_performance else 0.0
        row_targets = resolve_row_targets(batch)
        row_predictions = resolve_row_predictions(outputs, row_targets)
        supervised_losses, supervised_loss_support = reduce_row_predictions(row_predictions)
        for task_name, support in supervised_loss_support.items():
            output_metrics[f"batch_support_{task_name}"] = support
        if profile_performance:
            perf_metrics["perf_supervised_loss_sec"] = float(time.perf_counter() - supervised_start)

        if has_mil_elution or has_tcell_mil:
            mil_start = time.perf_counter() if profile_performance else 0.0
            if has_mil_elution:
                mil_losses, mil_regularization, mil_metrics, mil_support = (
                    _compute_mil_channel_losses(
                        model=model,
                        batch=batch,
                        device=device,
                        channel_prefix="mil",
                        regularization=regularization_cfg,
                        max_mil_instances=max_mil_instances,
                        mil_chunk_size=mil_chunk_size,
                        enable_contrastive=True,
                    )
                )
                supervised_losses.update(mil_losses)
                supervised_loss_support.update(mil_support)
                output_metrics.update(mil_metrics)
            else:
                mil_regularization = {}

            tcell_mil_regularization: Dict[str, torch.Tensor] = {}
            if has_tcell_mil:
                tcell_mil_losses, tcell_mil_regularization, tcell_mil_metrics, tcell_support = (
                    _compute_mil_channel_losses(
                        model=model,
                        batch=batch,
                        device=device,
                        channel_prefix="tcell_mil",
                        regularization=regularization_cfg,
                        max_mil_instances=max_mil_instances,
                        mil_chunk_size=mil_chunk_size,
                        enable_contrastive=False,
                    )
                )
                supervised_losses.update(tcell_mil_losses)
                supervised_loss_support.update(tcell_support)
                output_metrics.update(tcell_mil_metrics)

            if profile_performance:
                perf_metrics["perf_mil_sec"] = float(time.perf_counter() - mil_start)
        else:
            mil_regularization = {}
            tcell_mil_regularization = {}
        if not (has_mil_elution or has_tcell_mil) and profile_performance:
            perf_metrics["perf_mil_sec"] = 0.0

        regularization_start = time.perf_counter() if profile_performance else 0.0
        regularization_losses = _compute_consistency_losses(
            outputs=outputs,
            batch=batch,
            regularization=regularization_cfg,
        )
        binding_contrastive_loss, binding_contrastive_metrics = _compute_binding_contrastive_loss(
            outputs=outputs,
            batch=batch,
            regularization=regularization_cfg,
        )
        output_metrics.update(binding_contrastive_metrics)
        if binding_contrastive_loss is not None:
            regularization_losses["binding_contrastive"] = binding_contrastive_loss
        regularization_losses.update(mil_regularization)
        regularization_losses.update(tcell_mil_regularization)
        if profile_performance:
            perf_metrics["perf_regularization_sec"] = float(
                time.perf_counter() - regularization_start
            )
        losses: Dict[str, torch.Tensor] = {}
        losses.update(supervised_losses)
        losses.update(regularization_losses)

        # Combine losses
        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True), {}, output_metrics

        if supervised_losses:
            weighted_terms = []
            total_weight = 0.0
            supervised_task_weights: Dict[str, float] = {}
            for task_name, task_loss in supervised_losses.items():
                spec = LOSS_TASK_NAME_TO_SPEC.get(task_name)
                # Panel axes reduce to declared groups, separate from the
                # established uncertainty-parameter indices in LOSS_TASK_SPECS.
                base_weight = (
                    max(float(spec.base_weight), 0.0)
                    if spec is not None
                    else PANEL_TASK_BASE_WEIGHTS.get(
                        task_name, MIL_TASK_BASE_WEIGHTS.get(task_name, 1.0)
                    )
                )
                if base_weight <= 0.0:
                    continue
                if aggregation_mode == "task_mean":
                    task_weight = base_weight
                else:
                    task_weight = base_weight * max(
                        float(supervised_loss_support.get(task_name, 1.0)), 1e-6
                    )
                total_weight += task_weight
                supervised_task_weights[task_name] = task_weight

                if uncertainty_weighting is not None:
                    task_idx = LOSS_TASK_NAME_TO_INDEX.get(task_name)
                    if task_idx is not None and task_idx < uncertainty_weighting.log_vars.shape[0]:
                        log_var = uncertainty_weighting.log_vars[task_idx]
                        task_term = torch.exp(-log_var) * task_loss + log_var
                    else:
                        task_term = task_loss
                else:
                    task_term = task_loss
                weighted_terms.append(task_term * task_weight)

            if weighted_terms:
                supervised_total = sum(weighted_terms) / max(total_weight, 1e-8)
                for task_name, task_weight in supervised_task_weights.items():
                    output_metrics[f"batch_supervised_weight_{task_name}"] = float(
                        task_weight / max(total_weight, 1e-8)
                    )
            else:
                supervised_total = torch.tensor(0.0, device=device)
        else:
            supervised_total = torch.tensor(0.0, device=device)

        if regularization_losses:
            regularization_total = sum(regularization_losses.values())
        else:
            regularization_total = torch.tensor(0.0, device=device)

        total_loss = supervised_total + regularization_total

    # Perf metrics collected outside autocast
    if profile_performance:
        perf_metrics["perf_compute_total_sec"] = float(time.perf_counter() - perf_start)
    output_metrics.update(perf_metrics)

    return total_loss, losses, output_metrics


def summarize_uncertainty_weights(
    uncertainty_weighting: Optional[UncertaintyWeighting],
) -> Dict[str, float]:
    """Return per-task uncertainty parameters as scalar metrics."""
    if uncertainty_weighting is None:
        return {}
    metrics: Dict[str, float] = {}
    with torch.no_grad():
        log_vars = uncertainty_weighting.log_vars.detach().cpu()
    n_tasks = min(len(LOSS_TASK_NAMES), int(log_vars.shape[0]))
    for idx in range(n_tasks):
        task = LOSS_TASK_NAMES[idx]
        log_var = float(log_vars[idx].item())
        metrics[f"uw_log_var_{task}"] = log_var
        metrics[f"uw_weight_{task}"] = float(torch.exp(-log_vars[idx]).item())
    return metrics


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    scheduler=None,
    uncertainty_weighting=None,
    pcgrad: PCGrad = None,
    regularization: Optional[Mapping[str, float]] = None,
    show_progress: bool = True,
    profile_performance: bool = False,
    supervised_loss_aggregation: str = "task_mean",
    non_blocking_transfer: bool = False,
    perf_log_interval_batches: int = 0,
    use_amp: bool = False,
    max_mil_instances: int = 0,
    max_batches: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_samples = 0
    start_time = time.perf_counter()
    task_loss_totals: Dict[str, float] = {}
    task_loss_counts: Dict[str, int] = {}
    output_totals: Dict[str, float] = {}
    output_counts: Dict[str, int] = {}
    perf_data_wait_sec = 0.0
    perf_compute_loss_sec = 0.0
    perf_backward_sec = 0.0
    perf_optimizer_sec = 0.0
    perf_log_interval_batches = max(0, int(perf_log_interval_batches))
    window_start = start_time
    window_batches = 0
    window_samples = 0
    window_data_wait_sec = 0.0
    window_compute_loss_sec = 0.0
    window_backward_sec = 0.0
    window_optimizer_sec = 0.0
    window_forward_main_sec = 0.0
    window_mil_sec = 0.0
    window_regularization_sec = 0.0

    iterator = train_loader
    if show_progress:
        total_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
        iterator = tqdm(train_loader, total=total_batches, desc="train", leave=False, unit="batch")
    prev_batch_end = time.perf_counter()

    for batch in iterator:
        batch_start = time.perf_counter()
        perf_data_wait_sec += batch_start - prev_batch_end
        compute_start = time.perf_counter()
        loss, loss_dict, output_dict = compute_loss(
            model,
            batch,
            device,
            uncertainty_weighting,
            regularization=regularization,
            supervised_loss_aggregation=supervised_loss_aggregation,
            profile_performance=profile_performance,
            non_blocking_transfer=non_blocking_transfer,
            use_amp=use_amp,
            max_mil_instances=max_mil_instances,
        )
        compute_elapsed = time.perf_counter() - compute_start
        perf_compute_loss_sec += compute_elapsed
        batch_samples = 0
        pep_tok = getattr(batch, "pep_tok", None)
        if isinstance(pep_tok, torch.Tensor) and pep_tok.ndim >= 1:
            batch_samples = int(pep_tok.shape[0])

        stepped_optimizer = False
        if pcgrad is not None and len(loss_dict) > 1:
            step_start = time.perf_counter()
            stepped_optimizer = (
                pcgrad.step(list(loss_dict.values()), model.parameters()) is not None
            )
            backward_elapsed = time.perf_counter() - step_start
            optimizer_elapsed = 0.0
            perf_backward_sec += backward_elapsed
        else:
            optimizer.zero_grad()
            backward_start = time.perf_counter()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            backward_elapsed = time.perf_counter() - backward_start
            perf_backward_sec += backward_elapsed
            optim_start = time.perf_counter()
            optimizer.step()
            stepped_optimizer = True
            optimizer_elapsed = time.perf_counter() - optim_start
            perf_optimizer_sec += optimizer_elapsed
        if stepped_optimizer and scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        for name, value in loss_dict.items():
            task_loss_totals[name] = task_loss_totals.get(name, 0.0) + float(value.detach().item())
            task_loss_counts[name] = task_loss_counts.get(name, 0) + 1
        for name, value in output_dict.items():
            output_totals[name] = output_totals.get(name, 0.0) + float(value)
            output_counts[name] = output_counts.get(name, 0) + 1
        n_batches += 1
        total_samples += batch_samples
        window_batches += 1
        window_samples += batch_samples
        window_data_wait_sec += batch_start - prev_batch_end
        window_compute_loss_sec += compute_elapsed
        window_backward_sec += backward_elapsed
        window_optimizer_sec += optimizer_elapsed
        if profile_performance:
            window_forward_main_sec += float(output_dict.get("perf_forward_main_sec", 0.0))
            window_mil_sec += float(output_dict.get("perf_mil_sec", 0.0))
            window_regularization_sec += float(output_dict.get("perf_regularization_sec", 0.0))

        if show_progress and hasattr(iterator, "set_postfix"):
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            iterator.set_postfix(
                {
                    "loss": f"{(total_loss / max(n_batches, 1)):.4f}",
                    "sps": f"{(total_samples / elapsed):.1f}",
                },
                refresh=False,
            )
        if (
            show_progress
            and perf_log_interval_batches > 0
            and n_batches % perf_log_interval_batches == 0
            and window_batches > 0
        ):
            window_elapsed = max(time.perf_counter() - window_start, 1e-6)
            msg = (
                "Perf window "
                f"{n_batches - window_batches + 1}-{n_batches}: "
                f"wait={window_data_wait_sec / window_batches:.3f}s, "
                f"compute={window_compute_loss_sec / window_batches:.3f}s, "
                f"backward={window_backward_sec / window_batches:.3f}s, "
                f"optim={window_optimizer_sec / window_batches:.3f}s per batch | "
                f"wait={100.0 * window_data_wait_sec / window_elapsed:.1f}%, "
                f"compute={100.0 * window_compute_loss_sec / window_elapsed:.1f}%, "
                f"backward={100.0 * window_backward_sec / window_elapsed:.1f}%, "
                f"optim={100.0 * window_optimizer_sec / window_elapsed:.1f}%"
            )
            if profile_performance:
                msg += (
                    " | inner: "
                    f"forward={window_forward_main_sec / window_batches:.3f}s, "
                    f"mil={window_mil_sec / window_batches:.3f}s, "
                    f"reg={window_regularization_sec / window_batches:.3f}s"
                )
            msg += f" | sps={window_samples / window_elapsed:.1f}"
            if hasattr(iterator, "write"):
                iterator.write(msg)
            else:
                print(msg)
            window_start = time.perf_counter()
            window_batches = 0
            window_samples = 0
            window_data_wait_sec = 0.0
            window_compute_loss_sec = 0.0
            window_backward_sec = 0.0
            window_optimizer_sec = 0.0
            window_forward_main_sec = 0.0
            window_mil_sec = 0.0
            window_regularization_sec = 0.0
        prev_batch_end = time.perf_counter()
        if max_batches > 0 and n_batches >= max_batches:
            break

    if show_progress and hasattr(iterator, "close"):
        iterator.close()

    epoch_loss = total_loss / max(n_batches, 1)
    task_means = {
        f"loss_{name}": task_loss_totals[name] / max(task_loss_counts[name], 1)
        for name in sorted(task_loss_totals)
    }
    output_means = {
        name: output_totals[name] / max(output_counts[name], 1) for name in sorted(output_totals)
    }
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    runtime_metrics = {
        "train_samples": float(total_samples),
        "train_samples_per_sec": float(total_samples / elapsed),
        "train_batches": float(n_batches),
        "train_sec": float(elapsed),
        "perf_data_wait_sec_total": float(perf_data_wait_sec),
        "perf_compute_loss_sec_total": float(perf_compute_loss_sec),
        "perf_backward_sec_total": float(perf_backward_sec),
        "perf_optimizer_sec_total": float(perf_optimizer_sec),
        "perf_data_wait_sec_per_batch": float(perf_data_wait_sec / max(n_batches, 1)),
        "perf_compute_loss_sec_per_batch": float(perf_compute_loss_sec / max(n_batches, 1)),
        "perf_backward_sec_per_batch": float(perf_backward_sec / max(n_batches, 1)),
        "perf_optimizer_sec_per_batch": float(perf_optimizer_sec / max(n_batches, 1)),
        "perf_data_wait_pct_epoch": float(100.0 * perf_data_wait_sec / elapsed),
        "perf_compute_loss_pct_epoch": float(100.0 * perf_compute_loss_sec / elapsed),
        "perf_backward_pct_epoch": float(100.0 * perf_backward_sec / elapsed),
        "perf_optimizer_pct_epoch": float(100.0 * perf_optimizer_sec / elapsed),
    }
    return epoch_loss, {**task_means, **output_means, **runtime_metrics}


def evaluate(
    model,
    val_loader,
    device,
    regularization: Optional[Mapping[str, float]] = None,
    show_progress: bool = True,
    supervised_loss_aggregation: str = "task_mean",
    use_amp: bool = False,
    max_mil_instances: int = 0,
    max_batches: int = 0,
    mil_chunk_size: int = 0,
    batch_receipts: Optional[list] = None,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_samples = 0
    start_time = time.perf_counter()
    task_loss_totals: Dict[str, float] = {}
    task_loss_counts: Dict[str, int] = {}
    output_totals: Dict[str, float] = {}
    output_counts: Dict[str, int] = {}

    iterator = val_loader
    if show_progress:
        total_batches = len(val_loader) if hasattr(val_loader, "__len__") else None
        iterator = tqdm(val_loader, total=total_batches, desc="eval", leave=False, unit="batch")

    with torch.no_grad():
        for batch in iterator:
            loss, loss_dict, output_dict = compute_loss(
                model,
                batch,
                device,
                regularization=regularization,
                supervised_loss_aggregation=supervised_loss_aggregation,
                use_amp=use_amp,
                max_mil_instances=max_mil_instances,
                mil_chunk_size=mil_chunk_size,
            )
            if batch_receipts is not None:
                from presto.training.evaluation_ledger import evaluation_receipt

                batch_receipts.append(evaluation_receipt(n_batches, loss, loss_dict, output_dict))
            pep_tok = getattr(batch, "pep_tok", None)
            if isinstance(pep_tok, torch.Tensor) and pep_tok.ndim >= 1:
                total_samples += int(pep_tok.shape[0])
            total_loss += loss.item()
            for name, value in loss_dict.items():
                task_loss_totals[name] = task_loss_totals.get(name, 0.0) + float(
                    value.detach().item()
                )
                task_loss_counts[name] = task_loss_counts.get(name, 0) + 1
            for name, value in output_dict.items():
                output_totals[name] = output_totals.get(name, 0.0) + float(value)
                output_counts[name] = output_counts.get(name, 0) + 1
            n_batches += 1
            if show_progress and hasattr(iterator, "set_postfix"):
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                iterator.set_postfix(
                    {
                        "loss": f"{(total_loss / max(n_batches, 1)):.4f}",
                        "sps": f"{(total_samples / elapsed):.1f}",
                    },
                    refresh=False,
                )
            if max_batches > 0 and n_batches >= max_batches:
                break

    if show_progress and hasattr(iterator, "close"):
        iterator.close()

    epoch_loss = total_loss / max(n_batches, 1)
    task_means = {
        f"loss_{name}": task_loss_totals[name] / max(task_loss_counts[name], 1)
        for name in sorted(task_loss_totals)
    }
    output_means = {
        name: output_totals[name] / max(output_counts[name], 1) for name in sorted(output_totals)
    }
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    runtime_metrics = {
        "eval_samples": float(total_samples),
        "eval_samples_per_sec": float(total_samples / elapsed),
        "eval_batches": float(n_batches),
        "eval_sec": float(elapsed),
    }
    return epoch_loss, {**task_means, **output_means, **runtime_metrics}


def run(args: argparse.Namespace) -> None:
    """Run synthetic training with parsed arguments."""
    args = _resolve_run_args(args)

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    run_dir_arg = getattr(args, "run_dir", None)
    run_dir = Path(run_dir_arg) if run_dir_arg else None
    if run_dir is None and args.checkpoint:
        run_dir = Path(args.checkpoint).resolve().parent
    run_logger = RunLogger(run_dir, config=vars(args)) if run_dir is not None else None

    # Create data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(tempfile.mkdtemp()) / "presto_data"

    # Generate synthetic data
    binding_data, elution_data, tcr_data, mhc_sequences = create_synthetic_data(
        data_dir, args.n_binding, args.n_elution, args.n_tcr
    )

    # Create dataset
    dataset = PrestoDataset(
        binding_records=binding_data,
        elution_records=elution_data,
        tcr_records=tcr_data,
        mhc_sequences=mhc_sequences,
    )
    print(f"Total samples: {len(dataset)}")

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    collator = PrestoCollator()
    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collator=collator
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collator=collator
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = Presto(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer and uncertainty weighting
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    uncertainty_weighting = None
    if args.use_uncertainty_weighting:
        uncertainty_weighting = UncertaintyWeighting(n_tasks=len(LOSS_TASK_NAMES)).to(device)
        optimizer.add_param_group({"params": uncertainty_weighting.parameters()})
    total_steps = max(1, int(args.epochs) * max(len(train_loader), 1))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        base_lr=args.lr,
        total_steps=total_steps,
    )
    pcgrad = PCGrad(optimizer) if args.use_pcgrad else None
    regularization_cfg = _regularization_config_from_args(args)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    try:
        for epoch in range(args.epochs):
            train_loss, train_task_losses = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                scheduler=scheduler,
                uncertainty_weighting=uncertainty_weighting,
                pcgrad=pcgrad,
                regularization=regularization_cfg,
                supervised_loss_aggregation=args.supervised_loss_aggregation,
            )
            val_loss, val_task_losses = evaluate(
                model,
                val_loader,
                device,
                regularization=regularization_cfg,
                supervised_loss_aggregation=args.supervised_loss_aggregation,
            )
            uw_metrics = summarize_uncertainty_weights(uncertainty_weighting)
            current_lr = float(optimizer.param_groups[0]["lr"])

            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6g}"
            )
            if run_logger is not None:
                run_logger.log(
                    epoch + 1,
                    "train",
                    {"loss": train_loss, "lr": current_lr, **train_task_losses, **uw_metrics},
                )
                run_logger.log(epoch + 1, "val", {"loss": val_loss, **val_task_losses})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.checkpoint:
                    save_model_checkpoint(
                        args.checkpoint,
                        model=model,
                        optimizer_state_dict=optimizer.state_dict(),
                        epoch=epoch + 1,
                        metrics={"train_loss": train_loss, "val_loss": val_loss},
                        train_config=vars(args),
                        extra={"best_val_loss": best_val_loss},
                    )
                    print(f"  Saved checkpoint to {args.checkpoint}")
        if run_logger is not None:
            run_logger.log(args.epochs, "summary", {"best_val_loss": best_val_loss})
    finally:
        if run_logger is not None:
            run_logger.close()

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")

    # Quick inference test
    print("\nRunning inference test...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        batch = batch.to(device)
        outputs = model(
            pep_tok=batch.pep_tok,
            mhc_a_tok=batch.mhc_a_tok,
            mhc_b_tok=batch.mhc_b_tok,
            mhc_class="I",
            species=batch.processing_species,
        )
        pres_prob = torch.sigmoid(outputs["presentation_logit"])
        print(f"Sample presentation probabilities: {pres_prob[:5].cpu().numpy().flatten()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Presto on synthetic data")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON/YAML config file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_binding", type=int, default=200, help="Number of binding samples")
    parser.add_argument("--n_elution", type=int, default=100, help="Number of elution samples")
    parser.add_argument("--n_tcr", type=int, default=100, help="Number of TCR samples")
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Data directory (temp if not specified)"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Save checkpoint path")
    parser.add_argument(
        "--run-dir", dest="run_dir", type=str, default=None, help="Run artifact directory"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--use-uncertainty-weighting",
        dest="use_uncertainty_weighting",
        action="store_true",
        default=True,
        help="Use learned uncertainty weighting over task losses",
    )
    parser.add_argument(
        "--no-uncertainty-weighting",
        dest="use_uncertainty_weighting",
        action="store_false",
        help="Disable learned uncertainty weighting",
    )
    parser.add_argument(
        "--supervised-loss-aggregation",
        type=str,
        choices=["task_mean", "sample_weighted"],
        default="task_mean",
        help=(
            "How to combine supervised task losses: "
            "task_mean (equal per task) or sample_weighted "
            "(weight by in-batch labeled sample count per task)"
        ),
    )
    parser.add_argument(
        "--use-pcgrad", action="store_true", help="Use PCGrad for multi-task gradient conflicts"
    )
    parser.add_argument(
        "--consistency-cascade-weight",
        type=float,
        default=0.0,
        help="Weight for anti-saturation cascade prior (high presentation with low parent)",
    )
    parser.add_argument(
        "--consistency-assay-affinity-weight",
        type=float,
        default=0.0,
        help="Weight for KD/IC50/EC50 closeness regularization",
    )
    parser.add_argument(
        "--consistency-assay-presentation-weight",
        type=float,
        default=0.0,
        help="Weight for elution/MS vs presentation consistency",
    )
    parser.add_argument(
        "--consistency-no-b2m-weight",
        type=float,
        default=0.0,
        help="Weight for invalid chain-assembly prior (class I/II single-chain cases)",
    )
    parser.add_argument(
        "--consistency-tcell-context-weight",
        type=float,
        default=0.0,
        help="Weight for in-vitro >= ex-vivo T-cell context prior",
    )
    parser.add_argument(
        "--consistency-tcell-upstream-weight",
        type=float,
        default=0.0,
        help="Weight for T-cell outputs requiring strong upstream binding/presentation",
    )
    parser.add_argument(
        "--binding-orthogonality-weight",
        type=float,
        default=0.01,
        help="Weight for |cos(binding_affinity_vec, binding_stability_vec)| regularization",
    )
    parser.add_argument(
        "--consistency-prob-margin",
        type=float,
        default=0.02,
        help="Shared margin used in probabilistic consistency constraints",
    )
    parser.add_argument(
        "--consistency-parent-low-threshold",
        type=float,
        default=0.1,
        help="Low-parent threshold used by anti-saturation presentation prior",
    )
    parser.add_argument(
        "--consistency-presentation-high-threshold",
        type=float,
        default=0.9,
        help="High-presentation threshold used by anti-saturation presentation prior",
    )
    parser.add_argument(
        "--consistency-affinity-fold-tolerance",
        type=float,
        default=2.0,
        help="Allowed KD/IC50/EC50 discrepancy fold before penalty (2.0 = within 2x)",
    )
    parser.add_argument(
        "--mhc-attention-sparsity-weight",
        type=float,
        default=0.0,
        help="Weight for binding latent MHC-attention support regularization",
    )
    parser.add_argument(
        "--mhc-attention-sparsity-min-residues",
        type=float,
        default=30.0,
        help="Lower target bound for effective attended MHC residues",
    )
    parser.add_argument(
        "--mhc-attention-sparsity-max-residues",
        type=float,
        default=60.0,
        help="Upper target bound for effective attended MHC residues",
    )
    parser.add_argument(
        "--mil-contrastive-weight",
        type=float,
        default=0.0,
        help="Weight for presentation-vs-substituted-genotype MIL contrastive loss",
    )
    parser.add_argument(
        "--mil-contrastive-margin",
        type=float,
        default=0.5,
        help="Required logit margin between true and substituted-genotype MIL bags",
    )
    parser.add_argument(
        "--mil-contrastive-max-pairs",
        type=int,
        default=32,
        help="Maximum positive MIL bags per batch used for genotype-substitution contrastive loss",
    )
    parser.add_argument(
        "--binding-contrastive-weight",
        type=float,
        default=0.0,
        help="Weight for same-peptide/different-allele binding ranking loss",
    )
    parser.add_argument(
        "--binding-contrastive-margin",
        type=float,
        default=0.2,
        help="Required predicted log10(KD) margin for stronger-vs-weaker allele pairs",
    )
    parser.add_argument(
        "--binding-contrastive-target-gap-min",
        type=float,
        default=0.3,
        help="Minimum observed log10(KD) gap required before a binding pair is used for ranking",
    )
    parser.add_argument(
        "--binding-contrastive-max-pairs",
        type=int,
        default=64,
        help="Maximum same-peptide/different-allele binding pairs per batch used for ranking",
    )
    parser.add_argument(
        "--mil-bag-sparsity-weight",
        type=float,
        default=0.0,
        help="Weight for MIL bag sparsity prior on summed instance probabilities",
    )
    parser.add_argument(
        "--mil-bag-sparsity-target-sum",
        type=float,
        default=1.5,
        help="MIL bag summed-probability target before sparsity penalty activates",
    )
    parser.add_argument(
        "--tcell-in-vitro-margin",
        type=float,
        default=0.0,
        help="Required tcell-immunogenicity logit margin for in-vitro contexts",
    )
    parser.add_argument(
        "--tcell-ex-vivo-margin",
        type=float,
        default=0.0,
        help="Maximum tcell-immunogenicity logit margin for ex-vivo contexts",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    run(args)


if __name__ == "__main__":
    main()
