#!/usr/bin/env python
"""Train Presto on unified multi-source immunology datasets.

Primary supervision comes from IEDB/CEDAR exports, with optional VDJdb and
10x VDJ chain data. Synthetic negatives and consistency regularizers are
integrated into one unified mixed-source loop with time-varying
weights.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import random_split

from presto.data import (
    BindingRecord,
    KineticsRecord,
    StabilityRecord,
    ProcessingRecord,
    ElutionRecord,
    PrestoCollator,
    PrestoDataset,
    PrestoSample,
    TCellRecord,
    Sc10xVDJRecord,
    VDJdbRecord,
    Tokenizer,
    HUMAN_B2M_SEQUENCE,
    class_i_beta2m_sequence,
    create_dataloader,
    infer_mhc_class,
    infer_species,
    normalize_mhc_class,
    normalize_processing_species_label,
    normalize_species_label,
    load_iedb_binding,
    load_iedb_kinetics,
    load_iedb_stability,
    load_iedb_processing,
    load_iedb_elution,
    load_iedb_tcell,
    load_10x_vdj,
    load_vdjdb,
)
from presto.data.cross_source_dedup import (
    UnifiedRecord,
    classify_assay_type,
    parse_iedb_binding,
    parse_iedb_tcell,
)
from presto.data.loaders import (
    MHC_ALLOWED_AA,
    MIN_MHC_CHAIN_LENGTH,
    UniProtProtein,
    load_uniprot_proteins,
)
from presto.data.mhc_index import classify_unresolved_allele, load_mhc_index, resolve_alleles
from presto.data.vocab import FOREIGN_CATEGORIES, normalize_organism
from presto.models.presto import Presto
from presto.models.affinity import (
    DEFAULT_MAX_AFFINITY_NM,
    binding_prob_from_kd_log10,
)
from presto.scripts.train_synthetic import (
    LOSS_TASK_NAMES,
    _regularization_config_from_args,
    evaluate,
    train_epoch,
)
from presto.training.checkpointing import save_model_checkpoint
from presto.training.config_io import (
    load_config_file,
    merge_namespace_with_config,
)
from presto.training.run_logger import RunLogger

try:
    from presto.training.losses import PCGrad, UncertaintyWeighting
except ImportError:
    class UncertaintyWeighting(nn.Module):
        """Fallback uncertainty weighting for older runtime environments."""

        def __init__(self, n_tasks: int):
            super().__init__()
            self.log_vars = nn.Parameter(torch.zeros(n_tasks))

        def forward(self, losses):
            total = 0.0
            for i, loss in enumerate(losses):
                precision = torch.exp(-self.log_vars[i])
                total = total + precision * loss + self.log_vars[i]
            return total

    class PCGrad:
        """Fallback PCGrad shim that defaults to mean-loss backprop."""

        def __init__(self, optimizer: torch.optim.Optimizer):
            self._optimizer = optimizer

        def step(self, losses, parameters):
            valid = [loss for loss in losses if loss is not None]
            if not valid:
                return None
            self._optimizer.zero_grad(set_to_none=True)
            total = sum(valid) / len(valid)
            total.backward()
            self._optimizer.step()
            return float(total.detach().item())


IEDB_DEFAULTS = {
    "profile": "full",
    "epochs": 5,
    "batch_size": 512,
    "num_workers": 4,
    "pin_memory": True,
    # LR scaled by sqrt(512/64)=2.83 from base 1e-4 for larger batch size
    "lr": 2.8e-4,
    "d_model": 128,
    "n_layers": 2,
    "n_heads": 4,
    "data_dir": "./data",
    "merged_tsv": None,
    "require_merged_input": True,
    "binding_file": None,
    "tcell_file": None,
    "cedar_binding_file": None,
    "cedar_tcell_file": None,
    "vdjdb_file": None,
    "sc10x_file": None,
    "index_csv": None,
    "strict_mhc_resolution": True,
    "max_binding": 0,
    "max_kinetics": 0,
    "max_stability": 0,
    "max_processing": 0,
    "max_elution": 0,
    "max_tcell": 0,
    "max_vdjdb": 0,
    "max_10x": 0,
    "cap_sampling": "reservoir",
    "synthetic_pmhc_negative_ratio": 1.0,
    "synthetic_class_i_no_mhc_beta_negative_ratio": 0.25,
    "synthetic_processing_negative_ratio": 0.5,
    "synthetic_negative_min_nM": DEFAULT_MAX_AFFINITY_NM * 0.5,
    "synthetic_negative_max_nM": DEFAULT_MAX_AFFINITY_NM,
    "consistency_cascade_weight": 0.2,
    "consistency_assay_affinity_weight": 0.1,
    "consistency_assay_presentation_weight": 0.1,
    "consistency_no_b2m_weight": 0.5,
    "consistency_tcell_context_weight": 0.05,
    "consistency_tcell_upstream_weight": 0.2,
    "consistency_prob_margin": 0.02,
    "consistency_parent_low_threshold": 0.1,
    "consistency_presentation_high_threshold": 0.9,
    "consistency_affinity_fold_tolerance": 2.0,
    "mhc_attention_sparsity_weight": 0.1,
    "mhc_attention_sparsity_min_residues": 25.0,
    "mhc_attention_sparsity_max_residues": 45.0,
    "tcell_in_vitro_margin": 0.1,
    "tcell_ex_vivo_margin": 0.0,
    "val_frac": 0.2,
    "checkpoint": None,
    "run_dir": None,
    "weight_decay": 0.01,
    "use_uncertainty_weighting": True,
    "supervised_loss_aggregation": "sample_weighted",
    "use_pcgrad": False,
    "profile_performance": True,
    "perf_log_interval_batches": 100,
    "track_probe_affinity": True,
    "probe_peptide": "SLLQHLIGL",
    "probe_alleles": "HLA-A*02:01,HLA-A*01:01,HLA-A*03:01,HLA-A*11:01,HLA-A*24:02,HLA-B*07:02,HLA-B*08:01,HLA-B*15:01,HLA-B*35:01,HLA-B*44:02",
    "probe_plot_file": "probe_affinity_over_epochs.png",
    "track_probe_motif_scan": True,
    "motif_scan_positions": "1,2,3,4,5,6,7,8,9",
    "motif_scan_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
    "track_pmhc_flow": True,
    "pmhc_flow_batches": 2,
    "pmhc_flow_max_samples": 512,
    "track_output_latent_stats": True,
    "output_latent_stats_batches": 2,
    "output_latent_stats_max_samples": 512,
    "filter_unresolved_mhc": False,
    "mhc_augmentation_samples": 60000,
    "mhc_augmentation_max_fraction": 0.05,
    "seed": 42,
    "device": None,
}

CANARY_PROFILE_OVERRIDES = {
    "epochs": 1,
    "batch_size": 128,
    "balanced_batches": True,
    "max_binding": 512,
    "max_kinetics": 256,
    "max_stability": 256,
    "max_processing": 256,
    "max_elution": 512,
    "max_tcell": 512,
    "max_vdjdb": 256,
    "max_10x": 256,
    # Canary prioritizes speed over representativeness.
    "cap_sampling": "head",
    "track_probe_motif_scan": False,
}

DIAGNOSTIC_PROFILE_OVERRIDES = {
    "epochs": 10,
    "batch_size": 512,
    "balanced_batches": True,
    "track_probe_affinity": True,
    "track_pmhc_flow": True,
    "pmhc_flow_batches": 8,
    "pmhc_flow_max_samples": 2048,
    "track_output_latent_stats": True,
    "output_latent_stats_batches": 8,
    "output_latent_stats_max_samples": 2048,
    "filter_unresolved_mhc": True,
    "profile_performance": True,
    "strict_mhc_resolution": True,
}

CANARY_BOOTSTRAP_PER_MODALITY = 32

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SYNTHETIC_ELUTION_NEGATIVE_SCALE = 0.5
SYNTHETIC_CASCADE_ELUTION_NEGATIVE_SCALE = 0.5
SYNTHETIC_CASCADE_TCELL_NEGATIVE_SCALE = 0.5
MHC_SEQUENCE_ALLOWED_AA = set(MHC_ALLOWED_AA)
MIN_MHC_SEQUENCE_LEN = int(MIN_MHC_CHAIN_LENGTH)

# Keep uncertainty-weighting task indexing aligned with the actual
# supervised loss registry used by train/eval (`train_synthetic`).
IEDB_LOSS_TASK_NAMES = tuple(LOSS_TASK_NAMES)


def summarize_uncertainty_weights(
    uncertainty_weighting: Optional[UncertaintyWeighting],
) -> Dict[str, float]:
    """Return per-task uncertainty parameters as scalar metrics."""
    if uncertainty_weighting is None:
        return {}
    metrics: Dict[str, float] = {}
    with torch.no_grad():
        log_vars = uncertainty_weighting.log_vars.detach().cpu()
    n_tasks = min(len(IEDB_LOSS_TASK_NAMES), int(log_vars.shape[0]))
    for idx in range(n_tasks):
        task = IEDB_LOSS_TASK_NAMES[idx]
        log_var = float(log_vars[idx].item())
        metrics[f"uw_log_var_{task}"] = log_var
        metrics[f"uw_weight_{task}"] = float(torch.exp(-log_vars[idx]).item())
    return metrics


def _call_train_epoch_compat(
    model,
    train_loader,
    optimizer,
    device: str,
    uncertainty_weighting,
    pcgrad,
    regularization_config: Optional[Dict[str, float]] = None,
    supervised_loss_aggregation: str = "sample_weighted",
    profile_performance: bool = False,
    non_blocking_transfer: bool = False,
    perf_log_interval_batches: int = 0,
    use_amp: bool = False,
    max_mil_instances: int = 0,
    max_batches: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Call train_epoch across old/new script signatures."""
    kwargs = {}
    params = inspect.signature(train_epoch).parameters
    if "uncertainty_weighting" in params:
        kwargs["uncertainty_weighting"] = uncertainty_weighting
    if "pcgrad" in params:
        kwargs["pcgrad"] = pcgrad
    if "regularization" in params:
        kwargs["regularization"] = regularization_config
    if "supervised_loss_aggregation" in params:
        kwargs["supervised_loss_aggregation"] = str(
            supervised_loss_aggregation or "sample_weighted"
        )
    if "profile_performance" in params:
        kwargs["profile_performance"] = bool(profile_performance)
    if "non_blocking_transfer" in params:
        kwargs["non_blocking_transfer"] = bool(non_blocking_transfer)
    if "perf_log_interval_batches" in params:
        kwargs["perf_log_interval_batches"] = max(0, int(perf_log_interval_batches))
    if "use_amp" in params:
        kwargs["use_amp"] = bool(use_amp)
    if "max_mil_instances" in params:
        kwargs["max_mil_instances"] = int(max_mil_instances)
    if "max_batches" in params:
        kwargs["max_batches"] = int(max_batches)

    result = train_epoch(model, train_loader, optimizer, device, **kwargs)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return float(result[0]), result[1]
    return float(result), {}


def _call_evaluate_compat(
    model,
    val_loader,
    device: str,
    regularization_config: Optional[Dict[str, float]] = None,
    supervised_loss_aggregation: str = "sample_weighted",
    use_amp: bool = False,
    max_mil_instances: int = 0,
    max_val_batches: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Call evaluate across old/new script signatures."""
    kwargs = {}
    params = inspect.signature(evaluate).parameters
    if "regularization" in params:
        kwargs["regularization"] = regularization_config
    if "supervised_loss_aggregation" in params:
        kwargs["supervised_loss_aggregation"] = str(
            supervised_loss_aggregation or "sample_weighted"
        )
    if "use_amp" in params:
        kwargs["use_amp"] = bool(use_amp)
    if "max_mil_instances" in params:
        kwargs["max_mil_instances"] = int(max_mil_instances)
    if "max_batches" in params:
        kwargs["max_batches"] = int(max_val_batches)
    result = evaluate(model, val_loader, device, **kwargs)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return float(result[0]), result[1]
    return float(result), {}


def _pick_unified_train_section(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Pick train.unified config section with train.iedb fallback."""
    if "train" in config and isinstance(config["train"], Mapping):
        train_cfg = config["train"]
        if "unified" in train_cfg and isinstance(train_cfg["unified"], Mapping):
            return dict(train_cfg["unified"])
        if "iedb" in train_cfg and isinstance(train_cfg["iedb"], Mapping):
            return dict(train_cfg["iedb"])
        return dict(train_cfg)
    if "unified" in config and isinstance(config["unified"], Mapping):
        return dict(config["unified"])
    if "iedb" in config and isinstance(config["iedb"], Mapping):
        return dict(config["iedb"])
    return dict(config)


def _apply_profile_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """Apply profile presets after config merge."""
    profile = str(getattr(args, "profile", "full") or "full").lower()
    if profile not in {"full", "canary", "diagnostic"}:
        raise ValueError(f"Unsupported profile: {profile}")
    args.profile = profile
    if profile == "full":
        return args

    overrides = (
        CANARY_PROFILE_OVERRIDES
        if profile == "canary"
        else DIAGNOSTIC_PROFILE_OVERRIDES
    )
    for key, value in overrides.items():
        if not hasattr(args, key):
            setattr(args, key, value)
            continue
        current = getattr(args, key)
        default = IEDB_DEFAULTS.get(key)
        if current == default:
            setattr(args, key, value)
    return args


def _resolve_run_args(args: argparse.Namespace) -> argparse.Namespace:
    for key, default in IEDB_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, default)
    config_path = getattr(args, "config", None)
    if config_path:
        config = load_config_file(config_path)
        section = _pick_unified_train_section(config)
        args = merge_namespace_with_config(args, IEDB_DEFAULTS, section)
    return _apply_profile_overrides(args)


def _regularization_for_epoch(
    base_regularization: Dict[str, float],
    epoch_idx: int,
    total_epochs: int,
) -> Dict[str, float]:
    """Smooth ramp schedule for semi-supervised consistency terms."""
    if total_epochs <= 1:
        return dict(base_regularization)

    progress = float(epoch_idx) / float(max(total_epochs - 1, 1))
    consistency_factor = min(1.0, progress / 0.5)  # ramp over first 50%
    tcell_factor = min(1.0, progress / 0.7)  # ramp slightly slower

    cfg = dict(base_regularization)
    for key in (
        "consistency_cascade_weight",
        "consistency_assay_affinity_weight",
        "consistency_assay_presentation_weight",
        "consistency_no_b2m_weight",
    ):
        cfg[key] = float(cfg.get(key, 0.0)) * consistency_factor
    for key in ("consistency_tcell_context_weight", "consistency_tcell_upstream_weight"):
        cfg[key] = float(cfg.get(key, 0.0)) * tcell_factor
    cfg["schedule_consistency_factor"] = consistency_factor
    cfg["schedule_tcell_factor"] = tcell_factor
    return cfg


def _metric_safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return token or "unknown"


def _resolve_probe_specs(
    *,
    probe_peptide: str,
    probe_alleles: Sequence[str],
    mhc_sequences: Mapping[str, str],
    index_csv: Optional[str],
    device: str,
    max_mhc_len: int = 400,
) -> List[Dict[str, Any]]:
    """Prepare fixed probe tensors for per-epoch affinity tracking."""
    peptide = str(probe_peptide or "").strip().upper()
    if not peptide:
        return []

    alleles = [a for a in (_split_allele_list(",".join(probe_alleles))) if a]
    if not alleles:
        return []

    resolved_mhc: Dict[str, str] = {}
    unresolved: List[str] = []
    for allele in alleles:
        seq = str(mhc_sequences.get(allele, "")).strip().upper()
        if seq:
            resolved_mhc[allele] = seq
        else:
            unresolved.append(allele)

    if unresolved and index_csv:
        seq_map, _ = resolve_mhc_sequences_from_index(index_csv=index_csv, alleles=unresolved)
        for allele in unresolved:
            seq = str(seq_map.get(allele, "")).strip().upper()
            if seq:
                resolved_mhc[allele] = seq

    tokenizer = Tokenizer()
    pep_tok = tokenizer.batch_encode([peptide], max_len=max(50, len(peptide), 1), pad=True).to(device)
    specs: List[Dict[str, Any]] = []

    for allele in alleles:
        mhc_a_seq = resolved_mhc.get(allele, "")
        if not mhc_a_seq:
            continue

        cls = normalize_mhc_class(infer_mhc_class(allele), default="I")
        species = normalize_species_label(infer_species(allele))
        if cls == "I":
            mhc_b_seq = class_i_beta2m_sequence(species) or HUMAN_B2M_SEQUENCE
        else:
            mhc_b_seq = ""

        mhc_a_tok = tokenizer.batch_encode([mhc_a_seq], max_len=max_mhc_len, pad=True).to(device)
        mhc_b_tok = tokenizer.batch_encode([mhc_b_seq], max_len=max_mhc_len, pad=True).to(device)
        tag = f"probe_{_metric_safe_token(peptide)}_{_metric_safe_token(allele)}"
        specs.append(
            {
                "peptide": peptide,
                "allele": allele,
                "tag": tag,
                "pep_tok": pep_tok,
                "mhc_a_tok": mhc_a_tok,
                "mhc_b_tok": mhc_b_tok,
            }
        )

    missing = [allele for allele in alleles if allele not in {spec["allele"] for spec in specs}]
    if missing:
        print(
            "Probe tracking warning: unresolved probe alleles skipped: "
            + ", ".join(missing)
        )
    return specs


@torch.no_grad()
def _evaluate_probe_affinity(
    model: Presto,
    probe_specs: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Evaluate fixed probe pMHC pairs and return scalar metrics + raw rows."""
    if not probe_specs:
        return {}, []

    was_training = bool(model.training)
    model.eval()

    midpoint = float(getattr(model, "binding_midpoint_nM", 500.0))
    scale = float(getattr(model, "binding_log10_scale", 0.35))
    scale = max(scale, 1e-6)

    metrics: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    by_allele: Dict[str, Dict[str, float]] = {}
    for spec in probe_specs:
        outputs = model(
            pep_tok=spec["pep_tok"],
            mhc_a_tok=spec["mhc_a_tok"],
            mhc_b_tok=spec["mhc_b_tok"],
            mhc_class=None,
            species=None,
        )
        kd_log10 = float(outputs["assays"]["KD_nM"][0].item())
        kd_nM = float(10.0 ** kd_log10)
        binding_prob = float(
            binding_prob_from_kd_log10(
                kd_log10,
                midpoint_nM=midpoint,
                log10_scale=scale,
            )
        )
        presentation_prob = float(torch.sigmoid(outputs["presentation_logit"])[0].item())
        processing_prob = float(torch.sigmoid(outputs["processing_logit"])[0].item())

        tag = str(spec["tag"])
        metrics[f"{tag}_kd_log10"] = kd_log10
        metrics[f"{tag}_kd_nM"] = kd_nM
        metrics[f"{tag}_binding_prob"] = binding_prob
        metrics[f"{tag}_processing_prob"] = processing_prob
        metrics[f"{tag}_presentation_prob"] = presentation_prob

        by_allele[str(spec["allele"])] = {
            "kd_log10": kd_log10,
            "kd_nM": kd_nM,
            "binding_prob": binding_prob,
            "presentation_prob": presentation_prob,
        }
        rows.append(
            {
                "peptide": str(spec["peptide"]),
                "allele": str(spec["allele"]),
                "tag": tag,
                "kd_log10": kd_log10,
                "kd_nM": kd_nM,
                "binding_prob": binding_prob,
                "processing_prob": processing_prob,
                "presentation_prob": presentation_prob,
            }
        )

    # Explicit A*02:01 vs A*24:02 deltas for requested sanity control.
    a0201 = by_allele.get("HLA-A*02:01")
    a2402 = by_allele.get("HLA-A*24:02")
    if a0201 is not None and a2402 is not None:
        metrics["probe_sllqhligl_a0201_minus_a2402_kd_log10"] = (
            a0201["kd_log10"] - a2402["kd_log10"]
        )
        metrics["probe_sllqhligl_a0201_minus_a2402_binding_prob"] = (
            a0201["binding_prob"] - a2402["binding_prob"]
        )
        metrics["probe_sllqhligl_a0201_minus_a2402_presentation_prob"] = (
            a0201["presentation_prob"] - a2402["presentation_prob"]
        )

    if was_training:
        model.train()
    return metrics, rows


def _parse_motif_scan_positions(raw: str, peptide_len: int) -> List[int]:
    """Parse comma-separated 1-based scan positions, filtered to peptide bounds."""
    text = str(raw or "").strip()
    if not text:
        return []
    out: List[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            pos = int(token)
        except ValueError:
            continue
        if 1 <= pos <= peptide_len:
            out.append(pos)
    # Stable dedupe preserving first appearance.
    seen = set()
    unique: List[int] = []
    for pos in out:
        if pos in seen:
            continue
        seen.add(pos)
        unique.append(pos)
    return unique


@torch.no_grad()
def _evaluate_probe_motif_scan(
    model: Presto,
    probe_specs: Sequence[Mapping[str, Any]],
    *,
    positions_1based: Sequence[int],
    amino_acids: str = AMINO_ACIDS,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Evaluate per-position amino-acid substitutions for probe peptide/allele pairs."""
    if not probe_specs:
        return {}, []
    aa_list = [aa for aa in str(amino_acids or "").strip().upper() if aa in AMINO_ACIDS]
    if not aa_list:
        aa_list = list(AMINO_ACIDS)

    was_training = bool(model.training)
    model.eval()
    midpoint = float(getattr(model, "binding_midpoint_nM", 500.0))
    scale = float(getattr(model, "binding_log10_scale", 0.35))
    scale = max(scale, 1e-6)

    metrics: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    for spec in probe_specs:
        peptide = str(spec["peptide"]).strip().upper()
        if not peptide:
            continue
        positions = _parse_motif_scan_positions(
            ",".join(str(p) for p in positions_1based),
            peptide_len=len(peptide),
        )
        if not positions:
            continue
        device = spec["mhc_a_tok"].device
        tokenizer = Tokenizer()
        allele = str(spec["allele"])
        allele_tag = _metric_safe_token(allele)

        for pos_1b in positions:
            pos_idx = pos_1b - 1
            wt_aa = peptide[pos_idx]
            rank_pairs: List[Tuple[str, float, float]] = []
            for aa in aa_list:
                mutated = peptide[:pos_idx] + aa + peptide[pos_idx + 1 :]
                pep_tok = tokenizer.batch_encode(
                    [mutated],
                    max_len=max(50, len(mutated), 1),
                    pad=True,
                ).to(device)
                outputs = model(
                    pep_tok=pep_tok,
                    mhc_a_tok=spec["mhc_a_tok"],
                    mhc_b_tok=spec["mhc_b_tok"],
                    mhc_class=None,
                    species=None,
                )
                kd_log10 = float(outputs["assays"]["KD_nM"][0].item())
                binding_prob = float(
                    binding_prob_from_kd_log10(
                        kd_log10,
                        midpoint_nM=midpoint,
                        log10_scale=scale,
                    )
                )
                kd_nM = float(10.0 ** kd_log10)
                rank_pairs.append((aa, binding_prob, kd_nM))
                rows.append(
                    {
                        "peptide": peptide,
                        "allele": allele,
                        "position_1based": int(pos_1b),
                        "wt_aa": wt_aa,
                        "sub_aa": aa,
                        "binding_prob": binding_prob,
                        "kd_nM": kd_nM,
                    }
                )

            rank_pairs.sort(key=lambda t: t[1], reverse=True)
            best_aa, best_prob, best_kd = rank_pairs[0]
            wt_rank = 1 + next((idx for idx, (aa, _, _) in enumerate(rank_pairs) if aa == wt_aa), len(rank_pairs))
            wt_prob = next((p for aa, p, _ in rank_pairs if aa == wt_aa), 0.0)
            wt_kd = next((k for aa, _, k in rank_pairs if aa == wt_aa), float("nan"))
            metrics[f"motif_{allele_tag}_p{pos_1b}_wt_rank"] = float(wt_rank)
            metrics[f"motif_{allele_tag}_p{pos_1b}_wt_binding_prob"] = float(wt_prob)
            metrics[f"motif_{allele_tag}_p{pos_1b}_best_binding_prob"] = float(best_prob)
            metrics[f"motif_{allele_tag}_p{pos_1b}_wt_minus_best"] = float(wt_prob - best_prob)
            # Keep best-AA identity numeric in metrics via amino-acid index.
            metrics[f"motif_{allele_tag}_p{pos_1b}_best_aa_idx"] = float(AMINO_ACIDS.find(best_aa))
            metrics[f"motif_{allele_tag}_p{pos_1b}_best_kd_nM"] = float(best_kd)
            metrics[f"motif_{allele_tag}_p{pos_1b}_wt_kd_nM"] = float(wt_kd)

    if was_training:
        model.train()
    return metrics, rows


def _write_probe_motif_artifacts(
    run_dir: Path,
    motif_history: Sequence[Mapping[str, Any]],
) -> Dict[str, Path]:
    """Persist epoch-wise probe motif-scan rows as CSV/JSON artifacts."""
    if not motif_history:
        return {}

    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "probe_motif_scan_over_epochs.csv"
    fieldnames = [
        "epoch",
        "peptide",
        "allele",
        "position_1based",
        "wt_aa",
        "sub_aa",
        "binding_prob",
        "kd_nM",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in motif_history:
            writer.writerow(
                {
                    "epoch": row.get("epoch"),
                    "peptide": row.get("peptide"),
                    "allele": row.get("allele"),
                    "position_1based": row.get("position_1based"),
                    "wt_aa": row.get("wt_aa"),
                    "sub_aa": row.get("sub_aa"),
                    "binding_prob": row.get("binding_prob"),
                    "kd_nM": row.get("kd_nM"),
                }
            )
    json_path = run_dir / "probe_motif_scan_over_epochs.json"
    json_path.write_text(json.dumps(list(motif_history), indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}


def _compute_discrimination_metrics(
    probe_rows: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute inter-allele discrimination metrics from probe affinity rows.

    Returns metrics that quantify how differently the model treats different
    MHC alleles for the same peptide — higher values mean better discrimination.
    """
    if not probe_rows:
        return {}
    kd_log10_vals = [float(r["kd_log10"]) for r in probe_rows if "kd_log10" in r]
    if len(kd_log10_vals) < 2:
        return {}
    mean_kd = sum(kd_log10_vals) / len(kd_log10_vals)
    variance = sum((v - mean_kd) ** 2 for v in kd_log10_vals) / len(kd_log10_vals)
    kd_range = max(kd_log10_vals) - min(kd_log10_vals)
    # Discrimination ratio: range in log10 space (orders of magnitude spread)
    return {
        "probe_inter_allele_kd_log10_variance": variance,
        "probe_inter_allele_kd_log10_range": kd_range,
        "probe_discrimination_ratio": kd_range,
    }


def _compute_motif_specificity(
    motif_rows: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute motif specificity metrics from probe motif-scan rows.

    Measures how allele-specific the position-weight matrices are:
    - Lower per-allele mean entropy → more specific motifs
    - Higher inter-allele cosine distance → more distinct motifs
    """
    if not motif_rows:
        return {}

    # Group by (allele, position) → list of binding_prob per amino acid
    from collections import defaultdict
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in motif_rows:
        key = (str(row["allele"]), int(row["position_1based"]))
        grouped[key].append(float(row["binding_prob"]))

    # Per-allele mean entropy of position-weight matrices
    allele_entropies: Dict[str, List[float]] = defaultdict(list)
    allele_vectors: Dict[str, List[List[float]]] = defaultdict(list)
    for (allele, _pos), probs in grouped.items():
        # Normalize to distribution
        total = sum(probs)
        if total <= 0:
            continue
        dist = [p / total for p in probs]
        # Shannon entropy
        entropy = -sum(p * math.log(p + 1e-12) for p in dist)
        allele_entropies[allele].append(entropy)
        allele_vectors[allele].append(dist)

    if not allele_entropies:
        return {}

    metrics: Dict[str, float] = {}
    # Mean entropy across all alleles and positions (lower = more specific)
    all_entropies = [e for ents in allele_entropies.values() for e in ents]
    metrics["motif_mean_entropy"] = sum(all_entropies) / len(all_entropies)

    # Inter-allele cosine distance: compare flattened PWM vectors across alleles
    allele_names = sorted(allele_vectors.keys())
    if len(allele_names) >= 2:
        # Build flat vector per allele (concatenate all position distributions)
        flat: Dict[str, List[float]] = {}
        for a in allele_names:
            flat[a] = [v for pos_dist in allele_vectors[a] for v in pos_dist]
        cosine_dists: List[float] = []
        for i in range(len(allele_names)):
            for j in range(i + 1, len(allele_names)):
                va, vb = flat[allele_names[i]], flat[allele_names[j]]
                min_len = min(len(va), len(vb))
                if min_len == 0:
                    continue
                va, vb = va[:min_len], vb[:min_len]
                dot = sum(a * b for a, b in zip(va, vb))
                norm_a = math.sqrt(sum(a * a for a in va) + 1e-12)
                norm_b = math.sqrt(sum(b * b for b in vb) + 1e-12)
                cosine_sim = dot / (norm_a * norm_b)
                cosine_dists.append(1.0 - cosine_sim)
        if cosine_dists:
            metrics["motif_inter_allele_mean_cosine_distance"] = (
                sum(cosine_dists) / len(cosine_dists)
            )
    return metrics


def _as_float_vector(tensor: torch.Tensor) -> torch.Tensor:
    vec = tensor.float()
    if vec.ndim > 1 and vec.shape[-1] == 1:
        vec = vec.squeeze(-1)
    return vec


def _batch_index_tensor(
    tensor: Optional[torch.Tensor],
    index: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if tensor is None or index is None:
        return tensor
    if tensor.ndim < 1:
        return tensor
    if tensor.shape[0] != index.shape[0]:
        return tensor
    return tensor.index_select(0, index)


def _batch_index_list(
    values: Optional[Sequence[Any]],
    index: Optional[torch.Tensor],
) -> Optional[List[Any]]:
    if values is None or index is None:
        return list(values) if values is not None else None
    items = list(values)
    if len(items) != int(index.shape[0]):
        return items
    order = index.detach().cpu().tolist()
    return [items[int(i)] for i in order]


def _forward_pmhc_variant(
    model: Presto,
    batch,
    *,
    peptide_index: Optional[torch.Tensor],
    mhc_index: Optional[torch.Tensor],
) -> Dict[str, Any]:
    pep_tok = _batch_index_tensor(batch.pep_tok, peptide_index)
    mhc_a_tok = _batch_index_tensor(batch.mhc_a_tok, mhc_index)
    mhc_b_tok = _batch_index_tensor(batch.mhc_b_tok, mhc_index)
    if pep_tok is None or mhc_a_tok is None or mhc_b_tok is None:
        raise ValueError("Missing required peptide/MHC tensors for pMHC flow diagnostics")

    return model(
        pep_tok=pep_tok,
        mhc_a_tok=mhc_a_tok,
        mhc_b_tok=mhc_b_tok,
        mhc_class=_batch_index_list(getattr(batch, "mhc_class", None), mhc_index),
        species=_batch_index_list(getattr(batch, "processing_species", None), mhc_index),
        tcr_a_tok=_batch_index_tensor(getattr(batch, "tcr_a_tok", None), peptide_index),
        tcr_b_tok=_batch_index_tensor(getattr(batch, "tcr_b_tok", None), peptide_index),
        flank_n_tok=_batch_index_tensor(getattr(batch, "flank_n_tok", None), peptide_index),
        flank_c_tok=_batch_index_tensor(getattr(batch, "flank_c_tok", None), peptide_index),
        tcell_context=None,
    )


def _extract_output_vector(outputs: Mapping[str, Any], key: str) -> Optional[torch.Tensor]:
    if key == "kd_log10":
        assays = outputs.get("assays")
        if isinstance(assays, Mapping):
            kd = assays.get("KD_nM")
            if isinstance(kd, torch.Tensor):
                return _as_float_vector(kd)
        return None

    value = outputs.get(key)
    if isinstance(value, torch.Tensor):
        return _as_float_vector(value)
    return None


@torch.no_grad()
def _evaluate_pmhc_information_flow(
    model: Presto,
    val_loader,
    device: str,
    *,
    n_batches: int = 2,
    max_samples: int = 512,
    non_blocking_transfer: bool = False,
) -> Dict[str, float]:
    """Estimate peptide/MHC contribution and interaction via counterfactual shuffles."""
    n_batches = max(0, int(n_batches))
    max_samples = max(0, int(max_samples))
    if n_batches <= 0 or max_samples <= 1:
        return {}

    was_training = bool(model.training)
    model.eval()

    score_keys = (
        "binding_logit",
        "binding_prob",
        "presentation_logit",
        "presentation_prob",
        "processing_logit",
        "kd_log10",
    )
    totals: Dict[str, float] = {
        "pmhc_flow_batches": 0.0,
        "pmhc_flow_samples": 0.0,
    }

    batches_used = 0
    samples_used = 0

    try:
        for batch in val_loader:
            if batches_used >= n_batches or samples_used >= max_samples:
                break

            try:
                batch = batch.to(device, non_blocking=non_blocking_transfer)
            except TypeError:
                batch = batch.to(device)

            pep_tok = getattr(batch, "pep_tok", None)
            if not isinstance(pep_tok, torch.Tensor) or pep_tok.ndim < 2:
                continue
            batch_size = int(pep_tok.shape[0])
            if batch_size < 2:
                continue

            keep = min(batch_size, max_samples - samples_used)
            if keep < 2:
                break

            if keep < batch_size:
                idx_keep = torch.arange(keep, device=pep_tok.device, dtype=torch.long)
                real_outputs = _forward_pmhc_variant(
                    model,
                    batch,
                    peptide_index=idx_keep,
                    mhc_index=idx_keep,
                )
            else:
                idx_keep = None
                real_outputs = _forward_pmhc_variant(
                    model,
                    batch,
                    peptide_index=None,
                    mhc_index=None,
                )

            bsz_eff = keep
            base_index = (
                idx_keep
                if idx_keep is not None
                else torch.arange(bsz_eff, device=pep_tok.device, dtype=torch.long)
            )
            if bsz_eff <= 1:
                continue

            mhc_shift = 1
            pep_shift = 2 if bsz_eff > 2 else 1
            mhc_perm = torch.roll(base_index, shifts=mhc_shift, dims=0)
            pep_perm = torch.roll(base_index, shifts=pep_shift, dims=0)

            mhc_shuf_outputs = _forward_pmhc_variant(
                model,
                batch,
                peptide_index=base_index,
                mhc_index=mhc_perm,
            )
            pep_shuf_outputs = _forward_pmhc_variant(
                model,
                batch,
                peptide_index=pep_perm,
                mhc_index=base_index,
            )
            both_shuf_outputs = _forward_pmhc_variant(
                model,
                batch,
                peptide_index=pep_perm,
                mhc_index=mhc_perm,
            )

            for score_key in score_keys:
                real = _extract_output_vector(real_outputs, score_key)
                mhc_shuf = _extract_output_vector(mhc_shuf_outputs, score_key)
                pep_shuf = _extract_output_vector(pep_shuf_outputs, score_key)
                both_shuf = _extract_output_vector(both_shuf_outputs, score_key)
                if (
                    real is None
                    or mhc_shuf is None
                    or pep_shuf is None
                    or both_shuf is None
                ):
                    continue

                if real.shape != mhc_shuf.shape or real.shape != pep_shuf.shape:
                    continue

                delta_mhc_abs = torch.mean(torch.abs(real - mhc_shuf))
                delta_pep_abs = torch.mean(torch.abs(real - pep_shuf))
                delta_both_abs = torch.mean(torch.abs(real - both_shuf))
                interaction_abs = torch.mean(torch.abs(real - mhc_shuf - pep_shuf + both_shuf))
                real_std = torch.std(real, unbiased=False)
                norm_denom = real_std + 1e-6
                interaction_ratio = interaction_abs / (delta_mhc_abs + delta_pep_abs + 1e-6)

                prefix = f"pmhc_flow_{score_key}"
                per_batch_metrics = {
                    f"{prefix}_delta_mhc_abs": delta_mhc_abs,
                    f"{prefix}_delta_peptide_abs": delta_pep_abs,
                    f"{prefix}_delta_both_abs": delta_both_abs,
                    f"{prefix}_interaction_abs": interaction_abs,
                    f"{prefix}_delta_mhc_norm": delta_mhc_abs / norm_denom,
                    f"{prefix}_delta_peptide_norm": delta_pep_abs / norm_denom,
                    f"{prefix}_interaction_norm": interaction_abs / norm_denom,
                    f"{prefix}_interaction_ratio": interaction_ratio,
                    f"{prefix}_real_minus_mhc_mean": torch.mean(real - mhc_shuf),
                    f"{prefix}_real_minus_peptide_mean": torch.mean(real - pep_shuf),
                    f"{prefix}_real_std": real_std,
                }
                for key, value in per_batch_metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value.item()) * float(bsz_eff)

            totals["pmhc_flow_batches"] += 1.0
            totals["pmhc_flow_samples"] += float(bsz_eff)
            batches_used += 1
            samples_used += bsz_eff
    finally:
        if was_training:
            model.train()

    total_weight = max(float(totals.get("pmhc_flow_samples", 0.0)), 1.0)
    averaged: Dict[str, float] = {}
    for key, value in totals.items():
        if key in {"pmhc_flow_batches", "pmhc_flow_samples"}:
            averaged[key] = float(value)
        else:
            averaged[key] = float(value) / total_weight

    mhc_norm = float(averaged.get("pmhc_flow_binding_logit_delta_mhc_norm", 0.0))
    pep_norm = float(averaged.get("pmhc_flow_binding_logit_delta_peptide_norm", 0.0))
    interaction_norm = float(averaged.get("pmhc_flow_binding_logit_interaction_norm", 0.0))
    if mhc_norm < 0.03 and pep_norm < 0.03:
        status = 0.0  # near-invariant binding head
    elif mhc_norm < 0.03 and pep_norm >= 0.03:
        status = 1.0  # peptide-dominant, weak MHC usage
    elif interaction_norm < 0.02 and min(mhc_norm, pep_norm) >= 0.03:
        status = 2.0  # both inputs used, weak pairwise interaction
    else:
        status = 3.0  # joint peptide-MHC interaction signal present
    averaged["pmhc_flow_status_code"] = status
    return averaged


def _update_scalar_moments(
    stats: Dict[str, Dict[str, float]],
    key: str,
    values: torch.Tensor,
) -> None:
    vec = values.detach().float().reshape(-1)
    if vec.numel() <= 0:
        return
    entry = stats.setdefault(key, {"count": 0.0, "sum": 0.0, "sum_sq": 0.0})
    entry["count"] += float(vec.numel())
    entry["sum"] += float(vec.sum().item())
    entry["sum_sq"] += float((vec * vec).sum().item())


def _summarize_scalar_moments(
    stats: Mapping[str, Mapping[str, float]],
    prefix: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, entry in stats.items():
        count = float(entry.get("count", 0.0))
        if count <= 0:
            continue
        mean = float(entry.get("sum", 0.0)) / count
        var = float(entry.get("sum_sq", 0.0)) / count - mean * mean
        if var < 0:
            var = 0.0
        out[f"{prefix}_{key}_mean"] = mean
        out[f"{prefix}_{key}_var"] = var
        out[f"{prefix}_{key}_std"] = var ** 0.5
    return out


def _update_vector_moments(
    stats: Dict[str, Dict[str, Any]],
    key: str,
    values: torch.Tensor,
) -> None:
    vec = values.detach().float().reshape(values.shape[0], -1)
    if vec.numel() <= 0:
        return
    vec_cpu = vec.cpu()
    norms = vec_cpu.norm(dim=1)
    entry = stats.get(key)
    if entry is None:
        entry = {
            "count": 0.0,
            "sum": torch.zeros(vec_cpu.shape[1], dtype=torch.float32),
            "sum_sq": torch.zeros(vec_cpu.shape[1], dtype=torch.float32),
            "norm_count": 0.0,
            "norm_sum": 0.0,
            "norm_sum_sq": 0.0,
        }
        stats[key] = entry

    entry["count"] += float(vec_cpu.shape[0])
    entry["sum"] = entry["sum"] + vec_cpu.sum(dim=0)
    entry["sum_sq"] = entry["sum_sq"] + (vec_cpu * vec_cpu).sum(dim=0)
    entry["norm_count"] += float(norms.numel())
    entry["norm_sum"] += float(norms.sum().item())
    entry["norm_sum_sq"] += float((norms * norms).sum().item())


def _summarize_vector_moments(
    stats: Mapping[str, Mapping[str, Any]],
    prefix: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, entry in stats.items():
        count = float(entry.get("count", 0.0))
        if count <= 0:
            continue
        sum_vec = entry.get("sum")
        sum_sq_vec = entry.get("sum_sq")
        if not isinstance(sum_vec, torch.Tensor) or not isinstance(sum_sq_vec, torch.Tensor):
            continue
        mean_vec = sum_vec / count
        var_vec = sum_sq_vec / count - mean_vec * mean_vec
        var_vec = torch.clamp(var_vec, min=0.0)
        out[f"{prefix}_{key}_feature_var_mean"] = float(var_vec.mean().item())
        out[f"{prefix}_{key}_feature_var_max"] = float(var_vec.max().item())

        norm_count = float(entry.get("norm_count", 0.0))
        if norm_count > 0:
            norm_mean = float(entry.get("norm_sum", 0.0)) / norm_count
            norm_var = float(entry.get("norm_sum_sq", 0.0)) / norm_count - norm_mean * norm_mean
            if norm_var < 0:
                norm_var = 0.0
            out[f"{prefix}_{key}_norm_mean"] = norm_mean
            out[f"{prefix}_{key}_norm_var"] = norm_var
            out[f"{prefix}_{key}_norm_std"] = norm_var ** 0.5
    return out


@torch.no_grad()
def _evaluate_output_and_latent_statistics(
    model: Presto,
    val_loader,
    device: str,
    *,
    n_batches: int = 2,
    max_samples: int = 512,
    non_blocking_transfer: bool = False,
) -> Dict[str, float]:
    """Track output-head and latent statistics over validation batches."""
    n_batches = max(0, int(n_batches))
    max_samples = max(0, int(max_samples))
    if n_batches <= 0 or max_samples <= 0:
        return {}

    was_training = bool(model.training)
    model.eval()

    scalar_stats: Dict[str, Dict[str, float]] = {}
    vector_stats: Dict[str, Dict[str, Any]] = {}
    batches_used = 0
    samples_used = 0

    scalar_output_keys = (
        "binding_logit",
        "binding_prob",
        "processing_logit",
        "processing_prob",
        "presentation_logit",
        "presentation_prob",
        "elution_logit",
        "elution_prob",
        "ms_logit",
        "ms_prob",
        "recognition_repertoire_logit",
        "recognition_repertoire_prob",
        "immunogenicity_logit",
        "immunogenicity_prob",
        "tcell_logit",
        "tcell_prob",
    )
    assay_scalar_keys = (
        "KD_nM",
        "IC50_nM",
        "EC50_nM",
        "kon",
        "koff",
        "t_half",
        "Tm",
    )
    vector_output_keys = (
        "pmhc_vec",
        "mhc_class_logits",
        "mhc_species_logits",
        "chain_species_logits",
        "chain_type_logits",
        "chain_phenotype_logits",
    )

    try:
        for batch in val_loader:
            if batches_used >= n_batches or samples_used >= max_samples:
                break

            try:
                batch = batch.to(device, non_blocking=non_blocking_transfer)
            except TypeError:
                batch = batch.to(device)

            pep_tok = getattr(batch, "pep_tok", None)
            if not isinstance(pep_tok, torch.Tensor) or pep_tok.ndim < 2:
                continue
            batch_size = int(pep_tok.shape[0])
            if batch_size <= 0:
                continue

            outputs = model(
                pep_tok=batch.pep_tok,
                mhc_a_tok=batch.mhc_a_tok,
                mhc_b_tok=batch.mhc_b_tok,
                mhc_class=batch.mhc_class,
                species=batch.processing_species,
                tcr_a_tok=batch.tcr_a_tok,
                tcr_b_tok=batch.tcr_b_tok,
                flank_n_tok=batch.flank_n_tok,
                flank_c_tok=batch.flank_c_tok,
                tcell_context=batch.tcell_context if batch.tcell_context else None,
            )
            take = min(batch_size, max_samples - samples_used)
            if take <= 0:
                break
            idx = slice(0, take)

            for key in scalar_output_keys:
                value = outputs.get(key)
                if isinstance(value, torch.Tensor) and value.ndim >= 1:
                    _update_scalar_moments(scalar_stats, key, value[idx])

            assays = outputs.get("assays")
            if isinstance(assays, Mapping):
                for assay_key in assay_scalar_keys:
                    value = assays.get(assay_key)
                    if isinstance(value, torch.Tensor) and value.ndim >= 1:
                        _update_scalar_moments(
                            scalar_stats,
                            f"assay_{assay_key}",
                            value[idx],
                        )

            for key in vector_output_keys:
                value = outputs.get(key)
                if isinstance(value, torch.Tensor) and value.ndim >= 2:
                    _update_vector_moments(vector_stats, key, value[idx])

            latent_vecs = outputs.get("latent_vecs")
            if isinstance(latent_vecs, Mapping):
                for latent_name, tensor in latent_vecs.items():
                    if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                        _update_vector_moments(
                            vector_stats,
                            f"latent_{latent_name}",
                            tensor[idx],
                        )

            binding_latents = outputs.get("binding_latents")
            if isinstance(binding_latents, Mapping):
                for latent_name, tensor in binding_latents.items():
                    if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
                        key = f"binding_latent_{latent_name}"
                        if tensor.ndim == 1:
                            _update_scalar_moments(scalar_stats, key, tensor[idx])
                        else:
                            _update_vector_moments(vector_stats, key, tensor[idx])

            samples_used += take
            batches_used += 1
    finally:
        if was_training:
            model.train()

    metrics = {
        "output_latent_batches": float(batches_used),
        "output_latent_samples": float(samples_used),
    }
    metrics.update(_summarize_scalar_moments(scalar_stats, prefix="diag"))
    metrics.update(_summarize_vector_moments(vector_stats, prefix="diag"))
    return metrics


def _write_probe_artifacts(
    *,
    run_dir: Path,
    probe_history: Sequence[Mapping[str, Any]],
    plot_file: str,
) -> Optional[Path]:
    """Persist probe history CSV + affinity-over-epoch plot."""
    if not probe_history:
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "probe_affinity_over_epochs.csv"
    fields = [
        "epoch",
        "peptide",
        "allele",
        "tag",
        "kd_log10",
        "kd_nM",
        "binding_prob",
        "processing_prob",
        "presentation_prob",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in probe_history:
            writer.writerow({key: row.get(key, "") for key in fields})

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - defensive for minimal envs
        print(f"Probe plot skipped (matplotlib unavailable): {exc}")
        return None

    by_tag: Dict[str, List[Mapping[str, Any]]] = {}
    for row in probe_history:
        tag = str(row.get("tag", "probe"))
        by_tag.setdefault(tag, []).append(row)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for tag, rows in sorted(by_tag.items()):
        rows_sorted = sorted(rows, key=lambda r: int(r.get("epoch", 0)))
        epochs = [int(r["epoch"]) for r in rows_sorted]
        kd_nM = [float(r["kd_nM"]) for r in rows_sorted]
        allele = str(rows_sorted[0].get("allele", "unknown"))
        ax.plot(epochs, kd_nM, marker="o", linewidth=2.0, label=allele)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Predicted KD (nM)")
    ax.set_yscale("log")
    ax.set_title("Probe Affinity Over Epochs")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_path = run_dir / str(plot_file or "probe_affinity_over_epochs.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def find_iedb_export_file(
    root: Path,
    keywords: Sequence[str],
    required_keywords: Sequence[str] = (),
) -> Path:
    """Locate the best-matching IEDB export file under a directory tree."""
    if not root.exists():
        raise FileNotFoundError(f"IEDB directory not found: {root}")

    lowered_keywords = tuple(k.lower() for k in keywords)
    lowered_required = tuple(k.lower() for k in required_keywords)
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            continue
        name = path.name.lower()
        if lowered_required and not all(keyword in name for keyword in lowered_required):
            continue
        candidates.append(path)

    if not candidates:
        if lowered_required:
            raise FileNotFoundError(
                f"No CSV/TSV/TXT files matched required keywords {lowered_required} under: {root}"
            )
        raise FileNotFoundError(f"No CSV/TSV/TXT files found under: {root}")

    def _score(path: Path) -> Tuple[int, int]:
        name = path.name.lower()
        keyword_hits = sum(1 for keyword in lowered_keywords if keyword in name)
        size = int(path.stat().st_size)
        return keyword_hits, size

    best = max(candidates, key=_score)
    if _score(best)[0] == 0 and lowered_keywords:
        raise FileNotFoundError(
            f"No export file matched keywords {lowered_keywords} under {root}"
        )
    return best


def _take_records(records: Iterable, max_records: Optional[int]) -> List:
    limit = max_records if max_records is not None and max_records > 0 else None
    out = []
    for rec in records:
        out.append(rec)
        if limit is not None and len(out) >= limit:
            break
    return out


def _merge_records_with_limit(
    record_lists: Sequence[Sequence],
    max_records: Optional[int],
    seed: int,
) -> List:
    """Merge record groups and enforce one total cap across groups."""
    merged: List = []
    for records in record_lists:
        merged.extend(list(records))

    limit = _normalize_limit(max_records)
    if limit is None or len(merged) <= limit:
        return merged

    rng = random.Random(seed)
    selected = sorted(rng.sample(range(len(merged)), limit))
    return [merged[i] for i in selected]


def _normalize_limit(max_records: Optional[int]) -> Optional[int]:
    """Normalize CLI limits so <=0 consistently means no limit."""
    if max_records is None:
        return None
    return max_records if max_records > 0 else None


def _parse_binary_response(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in {"positive", "pos", "1", "true", "yes", "+"}:
        return 1.0
    if normalized in {"negative", "neg", "0", "false", "no", "-"}:
        return 0.0
    if "positive" in normalized or "reactive" in normalized:
        return 1.0
    if "negative" in normalized or "non-reactive" in normalized:
        return 0.0
    try:
        return 1.0 if float(normalized) > 0 else 0.0
    except ValueError:
        pass
    return None


def _random_peptide(rng: random.Random, length: int) -> str:
    """Generate a random amino-acid peptide."""
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(max(1, int(length))))


def _scramble_sequence(rng: random.Random, sequence: str) -> str:
    """Return a shuffled permutation of the provided sequence."""
    seq = (sequence or "").strip().upper()
    if len(seq) <= 1:
        return seq
    chars = list(seq)
    rng.shuffle(chars)
    scrambled = "".join(chars)
    if scrambled == seq:
        # Avoid no-op permutations when possible.
        return _random_peptide(rng, len(seq))
    return scrambled


def _random_mhc_sequence_like(
    rng: random.Random,
    mhc_sequence: Optional[str],
    mhc_class: str,
) -> str:
    """Generate random AA MHC-like sequence matching observed/default length."""
    seq = (mhc_sequence or "").strip().upper()
    if seq:
        return _random_peptide(rng, len(seq))
    default_len = 180 if mhc_class == "I" else 220
    return _random_peptide(rng, default_len)


def _class_default_peptide_length(mhc_class: str, rng: random.Random) -> int:
    if mhc_class == "II":
        return rng.randint(12, 20)
    return rng.randint(8, 11)


def _prefer_sequence_backed_alleles(
    alleles: Sequence[str],
    mhc_sequences: Dict[str, str],
) -> List[str]:
    """Prefer allele pools with explicit resolved sequences when available."""
    seq_backed = [a for a in alleles if a in mhc_sequences]
    return seq_backed if seq_backed else list(alleles)


def bootstrap_missing_modalities_for_canary(
    *,
    binding_records: Sequence[BindingRecord],
    kinetics_records: Sequence[KineticsRecord],
    stability_records: Sequence[StabilityRecord],
    processing_records: Sequence[ProcessingRecord],
    seed: int,
    max_per_modality: int = CANARY_BOOTSTRAP_PER_MODALITY,
) -> Tuple[List[KineticsRecord], List[StabilityRecord], List[ProcessingRecord], Dict[str, int]]:
    """Backfill sparse modalities for fast canary runs.

    This is only used for smoke-iteration feedback loops. It preserves the
    normal training code path while guaranteeing that kinetics/stability/
    processing batches can be constructed even when tiny head-files are used.
    """
    kinetics = list(kinetics_records)
    stability = list(stability_records)
    processing = list(processing_records)

    if not binding_records:
        return kinetics, stability, processing, {"kinetics": 0, "stability": 0, "processing": 0}

    rng = random.Random(seed + 101)
    n_bootstrap = max(0, int(max_per_modality))
    stats = {"kinetics": 0, "stability": 0, "processing": 0}

    binding_pool = list(binding_records)

    def _sample_binding() -> BindingRecord:
        return rng.choice(binding_pool)

    if not kinetics and n_bootstrap > 0:
        for _ in range(n_bootstrap):
            src = _sample_binding()
            kinetics.append(
                KineticsRecord(
                    peptide=src.peptide,
                    mhc_allele=src.mhc_allele,
                    kon=rng.uniform(1e4, 1e6),
                    koff=rng.uniform(1e-4, 5e-2),
                    assay_type="canary_bootstrap",
                    mhc_sequence=src.mhc_sequence,
                    mhc_class=src.mhc_class,
                    species=src.species,
                    source="canary_bootstrap",
                )
            )
        stats["kinetics"] = n_bootstrap

    if not stability and n_bootstrap > 0:
        for _ in range(n_bootstrap):
            src = _sample_binding()
            stability.append(
                StabilityRecord(
                    peptide=src.peptide,
                    mhc_allele=src.mhc_allele,
                    t_half=rng.uniform(0.25, 24.0),
                    tm=rng.uniform(35.0, 70.0),
                    assay_type="canary_bootstrap",
                    mhc_sequence=src.mhc_sequence,
                    mhc_class=src.mhc_class,
                    species=src.species,
                    source="canary_bootstrap",
                )
            )
        stats["stability"] = n_bootstrap

    if not processing and n_bootstrap > 0:
        for _ in range(n_bootstrap):
            src = _sample_binding()
            mhc_class = src.mhc_class if src.mhc_class in {"I", "II"} else infer_mhc_class(src.mhc_allele)
            processing.append(
                ProcessingRecord(
                    peptide=src.peptide,
                    flank_n=_random_peptide(rng, 10),
                    flank_c=_random_peptide(rng, 10),
                    label=1.0,
                    processing_type="processing",
                    mhc_allele=src.mhc_allele,
                    mhc_class=mhc_class,
                    species=src.species,
                    source="canary_bootstrap",
                )
            )
        stats["processing"] = n_bootstrap

    return kinetics, stability, processing, stats


def augment_binding_records_with_synthetic_negatives(
    binding_records: Sequence[BindingRecord],
    mhc_sequences: Dict[str, str],
    negative_ratio: float,
    weak_value_min_nM: float,
    weak_value_max_nM: float,
    seed: int,
    class_i_no_mhc_beta_ratio: float = 0.0,
) -> Tuple[List[BindingRecord], Dict[str, int]]:
    """Add synthetic weak-affinity binding negatives.

    Mode semantics:
    - `peptide_*`: peptide perturbation with MHC retained.
    - `mhc_*`: MHC perturbation with peptide retained.
    - `random` means generated de novo AA sequence.
    - `scramble` means a shuffled permutation of an existing sequence.
    """
    mode_cycle = (
        "peptide_scramble",
        "peptide_random",
        "mhc_scramble",
        "mhc_random",
        "no_mhc_alpha",
        "no_mhc_beta",
    )

    def _empty_stats() -> Dict[str, int]:
        return {
            "added": 0,
            "added_general": 0,
            "no_mhc_beta": 0,
            **{mode: 0 for mode in mode_cycle},
        }

    records = list(binding_records)
    if not records or (negative_ratio <= 0 and class_i_no_mhc_beta_ratio <= 0):
        return records, _empty_stats()

    rng = random.Random(seed)
    neg_min = min(float(weak_value_min_nM), float(weak_value_max_nM))
    neg_max = max(float(weak_value_min_nM), float(weak_value_max_nM))
    n_to_add = max(0, int(round(len(records) * float(negative_ratio))))
    if n_to_add == 0 and class_i_no_mhc_beta_ratio <= 0:
        return records, _empty_stats()

    by_class: Dict[str, List[BindingRecord]] = {"I": [], "II": []}
    alleles_by_class: Dict[str, List[str]] = {"I": [], "II": []}
    for rec in records:
        mhc_class = rec.mhc_class if rec.mhc_class in {"I", "II"} else infer_mhc_class(rec.mhc_allele)
        by_class[mhc_class].append(rec)
        if rec.mhc_allele:
            alleles_by_class[mhc_class].append(rec.mhc_allele)

    for mhc_class in ("I", "II"):
        deduped = sorted(set(alleles_by_class[mhc_class]))
        alleles_by_class[mhc_class] = _prefer_sequence_backed_alleles(deduped, mhc_sequences)

    mode_counts = {mode: 0 for mode in mode_cycle}

    synthetic_records: List[BindingRecord] = []
    for idx in range(n_to_add):
        mode = mode_cycle[idx % len(mode_cycle)]
        source = rng.choice(records)
        mhc_class = source.mhc_class if source.mhc_class in {"I", "II"} else infer_mhc_class(source.mhc_allele)

        class_records = by_class[mhc_class] or records
        class_alleles = alleles_by_class[mhc_class]
        if not class_alleles:
            class_alleles = sorted({rec.mhc_allele for rec in class_records if rec.mhc_allele})

        peptide = source.peptide or _random_peptide(rng, _class_default_peptide_length(mhc_class, rng))
        allele = source.mhc_allele
        source_mhc_seq = (source.mhc_sequence or mhc_sequences.get(source.mhc_allele) or "").strip().upper()
        direct_mhc_sequence: Optional[str] = None
        source_label = f"synthetic_negative_{mode}"

        if mode == "peptide_scramble":
            peptide = _scramble_sequence(rng, peptide) or _random_peptide(
                rng, _class_default_peptide_length(mhc_class, rng)
            )
            allele = source.mhc_allele
        elif mode == "peptide_random":
            pep_len = len(peptide) if peptide else _class_default_peptide_length(mhc_class, rng)
            peptide = _random_peptide(rng, pep_len)
            allele = source.mhc_allele
        elif mode == "mhc_scramble":
            peptide = source.peptide or _random_peptide(rng, _class_default_peptide_length(mhc_class, rng))
            allele = source.mhc_allele
            direct_mhc_sequence = _scramble_sequence(rng, source_mhc_seq) if source_mhc_seq else _random_mhc_sequence_like(
                rng, source_mhc_seq, mhc_class
            )
        elif mode == "mhc_random":
            peptide = source.peptide or _random_peptide(rng, _class_default_peptide_length(mhc_class, rng))
            allele = source.mhc_allele
            direct_mhc_sequence = _random_mhc_sequence_like(rng, source_mhc_seq, mhc_class)
        elif mode == "no_mhc_alpha":
            peptide = source.peptide or _random_peptide(rng, _class_default_peptide_length(mhc_class, rng))
            allele = source.mhc_allele
            source_label = "synthetic_negative_no_mhc_alpha"
        else:
            peptide = source.peptide or _random_peptide(rng, _class_default_peptide_length(mhc_class, rng))
            allele = source.mhc_allele
            source_label = "synthetic_negative_no_mhc_beta"

        if not allele:
            continue

        mode_counts[mode] += 1
        synthetic_records.append(
            BindingRecord(
                peptide=peptide,
                mhc_allele=allele,
                value=rng.uniform(neg_min, neg_max),
                qualifier=0,
                measurement_type=source.measurement_type or "IC50",
                unit="nM",
                assay_type=source_label,
                mhc_sequence=direct_mhc_sequence,
                mhc_class=mhc_class,
                species="synthetic",
                source=source_label,
            )
        )

    no_mhc_beta_added = 0
    if class_i_no_mhc_beta_ratio > 0:
        class_i_records = [
            rec
            for rec in records
            if normalize_mhc_class(
                rec.mhc_class,
                default=infer_mhc_class(rec.mhc_allele),
            )
            == "I"
        ]
        n_no_mhc_beta = max(0, int(round(len(class_i_records) * float(class_i_no_mhc_beta_ratio))))
        for _ in range(n_no_mhc_beta):
            if not class_i_records:
                break
            source = rng.choice(class_i_records)
            if not source.mhc_allele:
                continue
            synthetic_records.append(
                BindingRecord(
                    peptide=source.peptide,
                    mhc_allele=source.mhc_allele,
                    value=rng.uniform(neg_min, neg_max),
                    qualifier=0,
                    measurement_type=source.measurement_type or "IC50",
                    unit="nM",
                    assay_type="synthetic_negative_no_mhc_beta",
                    mhc_sequence=source.mhc_sequence or mhc_sequences.get(source.mhc_allele),
                    mhc_class="I",
                    species=source.species,
                    source="synthetic_negative_no_mhc_beta",
                )
            )
            no_mhc_beta_added += 1

    records.extend(synthetic_records)
    stats = {
        "added": len(synthetic_records),
        "added_general": sum(mode_counts.values()),
        **mode_counts,
        "no_mhc_beta": no_mhc_beta_added + mode_counts.get("no_mhc_beta", 0),
    }
    return records, stats


def augment_elution_records_with_synthetic_negatives(
    elution_records: Sequence[ElutionRecord],
    negative_ratio: float,
    seed: int,
) -> Tuple[List[ElutionRecord], Dict[str, int]]:
    """Add synthetic elution negatives, including allele-mismatch negatives.

    Random semantics:
    - peptide_random: generated random AA sequence, class-length matched.
    - mhc_random: allele sampled from other known restricting alleles in-class.
    """
    mode_cycle = (
        "peptide_random_mhc_real",
        "peptide_real_mhc_random",
        "peptide_random_mhc_random",
    )

    def _empty_stats() -> Dict[str, int]:
        return {
            "added": 0,
            "hard_pair": 0,
            **{mode: 0 for mode in mode_cycle},
        }

    records = list(elution_records)
    if not records:
        return records, _empty_stats()

    rng = random.Random(seed + 17)
    n_to_add = max(0, int(round(len(records) * max(float(negative_ratio), 0.0))))

    by_class: Dict[str, List[ElutionRecord]] = {"I": [], "II": []}
    alleles_by_class: Dict[str, List[str]] = {"I": [], "II": []}
    for rec in records:
        mhc_class = rec.mhc_class if rec.mhc_class in {"I", "II"} else "I"
        by_class[mhc_class].append(rec)
        for allele in rec.alleles:
            if allele:
                alleles_by_class[mhc_class].append(allele)

    for mhc_class in ("I", "II"):
        alleles_by_class[mhc_class] = sorted(set(alleles_by_class[mhc_class]))

    known_negative_alleles_by_peptide: Dict[str, List[str]] = {}
    positives_by_peptide: Dict[str, List[ElutionRecord]] = {}
    for rec in records:
        peptide = rec.peptide.strip()
        if not peptide:
            continue
        if rec.detected:
            positives_by_peptide.setdefault(peptide, []).append(rec)
            continue
        negs = known_negative_alleles_by_peptide.setdefault(peptide, [])
        for allele in rec.alleles:
            if allele and allele not in negs:
                negs.append(allele)

    def _sample_allele(
        pool: Sequence[str],
        exclude: Optional[str] = None,
    ) -> Optional[str]:
        if not pool:
            return None
        if exclude:
            options = [a for a in pool if a != exclude]
            if options:
                return rng.choice(options)
        return rng.choice(list(pool))

    mode_counts = {mode: 0 for mode in mode_cycle}
    synthetic_records: List[ElutionRecord] = []
    for idx in range(n_to_add):
        mode = mode_cycle[idx % len(mode_cycle)]
        source = rng.choice(records)
        mhc_class = source.mhc_class if source.mhc_class in {"I", "II"} else "I"
        class_records = by_class[mhc_class] or records
        class_alleles = alleles_by_class[mhc_class]
        if not class_alleles:
            class_alleles = sorted(
                {
                    allele
                    for rec in class_records
                    for allele in rec.alleles
                    if allele
                }
            )

        pep_len = len(source.peptide) if source.peptide else _class_default_peptide_length(source.mhc_class, rng)
        peptide = source.peptide
        alleles = list(source.alleles)

        # mode: peptide_random_mhc_real
        if mode == "peptide_random_mhc_real":
            peptide = _random_peptide(rng, pep_len)

        # mode: peptide_real_mhc_random
        elif mode == "peptide_real_mhc_random":
            peptide = source.peptide
            sampled: List[str] = []
            for allele in alleles:
                preferred = [
                    cand
                    for cand in known_negative_alleles_by_peptide.get(peptide, [])
                    if cand != allele
                ]
                replacement = (
                    rng.choice(preferred)
                    if preferred
                    else _sample_allele(class_alleles, exclude=allele)
                )
                if replacement:
                    sampled.append(replacement)
            if sampled:
                deduped = []
                for allele in sampled:
                    if allele not in deduped:
                        deduped.append(allele)
                alleles = deduped

        # mode: peptide_random_mhc_random
        else:
            peptide = _random_peptide(rng, pep_len)
            sampled = []
            n_alleles = max(1, len(alleles))
            for _ in range(n_alleles):
                replacement = _sample_allele(class_alleles)
                if replacement:
                    sampled.append(replacement)
            if sampled:
                deduped = []
                for allele in sampled:
                    if allele not in deduped:
                        deduped.append(allele)
                alleles = deduped

        if not alleles:
            continue

        mode_counts[mode] += 1
        synthetic_records.append(
            ElutionRecord(
                peptide=peptide,
                alleles=alleles,
                detected=False,
                cell_type=source.cell_type,
                tissue=source.tissue,
                mhc_class=source.mhc_class,
                species=source.species,
                source="synthetic_negative",
            )
        )

    # Build deterministic hard negatives when both positive and negative labels
    # exist for the same peptide but with different restricting alleles.
    hard_pair_added = 0
    for peptide, neg_alleles in known_negative_alleles_by_peptide.items():
        positive_records = positives_by_peptide.get(peptide, [])
        if not positive_records:
            continue
        for source in positive_records:
            for neg_allele in neg_alleles:
                if neg_allele in source.alleles:
                    continue
                synthetic_records.append(
                    ElutionRecord(
                        peptide=peptide,
                        alleles=[neg_allele],
                        detected=False,
                        cell_type=source.cell_type,
                        tissue=source.tissue,
                        mhc_class=source.mhc_class,
                        species=source.species,
                        source="synthetic_negative_hard_pair",
                    )
                )
                hard_pair_added += 1

    records.extend(synthetic_records)
    return records, {
        "added": len(synthetic_records),
        "hard_pair": hard_pair_added,
        **mode_counts,
    }


def augment_processing_records_with_synthetic_negatives(
    processing_records: Sequence[ProcessingRecord],
    negative_ratio: float,
    seed: int,
) -> Tuple[List[ProcessingRecord], Dict[str, int]]:
    """Add synthetic processing negatives from sequence-context corruption.

    Processing negatives avoid MHC ablation modes and instead corrupt local
    peptide/flank context to model upstream cleavage/transport failure.
    """
    records = list(processing_records)
    if not records or negative_ratio <= 0:
        return records, {"added": 0, "flank_shuffle": 0, "peptide_scramble": 0}

    rng = random.Random(seed + 29)
    n_to_add = max(0, int(round(len(records) * float(negative_ratio))))
    mode_cycle = ("flank_shuffle", "peptide_scramble")
    mode_counts = {"flank_shuffle": 0, "peptide_scramble": 0}
    synthetic_records: List[ProcessingRecord] = []
    for idx in range(n_to_add):
        source = rng.choice(records)
        mode = mode_cycle[idx % len(mode_cycle)]
        flank_n_len = max(6, len(source.flank_n) if source.flank_n else 10)
        flank_c_len = max(6, len(source.flank_c) if source.flank_c else 10)
        peptide = source.peptide or _random_peptide(rng, _class_default_peptide_length(source.mhc_class, rng))
        flank_n = source.flank_n or _random_peptide(rng, flank_n_len)
        flank_c = source.flank_c or _random_peptide(rng, flank_c_len)
        source_label = f"synthetic_negative_processing_{mode}"

        if mode == "flank_shuffle":
            donor = rng.choice(records)
            flank_n = donor.flank_n or _random_peptide(rng, flank_n_len)
            flank_c = donor.flank_c or _random_peptide(rng, flank_c_len)
            if donor is source:
                flank_n = _scramble_sequence(rng, flank_n) or _random_peptide(rng, flank_n_len)
                flank_c = _scramble_sequence(rng, flank_c) or _random_peptide(rng, flank_c_len)
        else:
            peptide = _scramble_sequence(rng, peptide) or _random_peptide(
                rng, len(peptide) if peptide else _class_default_peptide_length(source.mhc_class, rng)
            )

        mode_counts[mode] += 1
        synthetic_records.append(
            ProcessingRecord(
                peptide=peptide,
                flank_n=flank_n,
                flank_c=flank_c,
                label=0.0,
                processing_type=source.processing_type,
                mhc_allele=source.mhc_allele,
                mhc_class=source.mhc_class,
                species=source.species,
                source=source_label,
            )
        )

    records.extend(synthetic_records)
    return records, {"added": len(synthetic_records), **mode_counts}


def cascade_binding_negatives_to_downstream(
    binding_records: Sequence[BindingRecord],
    elution_records: Sequence[ElutionRecord],
    tcell_records: Sequence[TCellRecord],
    elution_ratio: float,
    tcell_ratio: float,
    seed: int,
) -> Tuple[List[ElutionRecord], List[TCellRecord], Dict[str, int]]:
    """Project synthetic binding negatives into downstream assay negatives."""
    elution = list(elution_records)
    tcell = list(tcell_records)

    synthetic_binding = [
        rec
        for rec in binding_records
        if (rec.source or "").startswith("synthetic_negative")
        and rec.mhc_allele
        and rec.peptide
    ]
    if not synthetic_binding:
        return elution, tcell, {"elution_added": 0, "tcell_added": 0}

    rng = random.Random(seed + 53)
    n_elution = max(0, int(round(len(synthetic_binding) * max(float(elution_ratio), 0.0))))
    n_tcell = max(0, int(round(len(synthetic_binding) * max(float(tcell_ratio), 0.0))))

    elution_keys = {
        (rec.peptide, tuple(rec.alleles), bool(rec.detected), rec.mhc_class)
        for rec in elution
    }
    tcell_keys = {
        (rec.peptide, rec.mhc_allele, float(rec.response), rec.mhc_class)
        for rec in tcell
    }

    elution_added = 0
    for _ in range(n_elution):
        source = rng.choice(synthetic_binding)
        key = (
            source.peptide,
            (source.mhc_allele,),
            False,
            source.mhc_class,
        )
        if key in elution_keys:
            continue
        elution.append(
            ElutionRecord(
                peptide=source.peptide,
                alleles=[source.mhc_allele],
                detected=False,
                mhc_class=source.mhc_class,
                species=source.species,
                source="synthetic_negative_from_binding",
            )
        )
        elution_keys.add(key)
        elution_added += 1

    tcell_added = 0
    for _ in range(n_tcell):
        source = rng.choice(synthetic_binding)
        key = (
            source.peptide,
            source.mhc_allele,
            0.0,
            source.mhc_class,
        )
        if key in tcell_keys:
            continue
        tcell.append(
            TCellRecord(
                peptide=source.peptide,
                mhc_allele=source.mhc_allele,
                response=0.0,
                assay_type="synthetic_negative",
                assay_method="synthetic_negative",
                effector_culture_condition="Direct ex vivo",
                mhc_class=source.mhc_class,
                species=source.species,
                source="synthetic_negative_from_binding",
            )
        )
        tcell_keys.add(key)
        tcell_added += 1

    return elution, tcell, {"elution_added": elution_added, "tcell_added": tcell_added}


def load_iedb_binding_and_elution_records(
    binding_file: Path,
    max_binding: Optional[int],
    max_elution: Optional[int],
) -> Tuple[List[BindingRecord], List[ElutionRecord]]:
    """Load binding + elution from IEDB exports, with fallback parsers."""
    binding_limit = _normalize_limit(max_binding)
    elution_limit = _normalize_limit(max_elution)

    binding_records = _take_records(load_iedb_binding(binding_file), binding_limit)
    elution_records = _take_records(load_iedb_elution(binding_file), elution_limit)
    if binding_records or elution_records:
        return binding_records, elution_records

    # Full IEDB exports use multi-row headers; parse via cross-source parser.
    binding_records = []
    elution_records = []
    for rec in parse_iedb_binding(binding_file):
        if rec.value is not None and (
            binding_limit is None or len(binding_records) < binding_limit
        ):
            binding_records.append(
                BindingRecord(
                    peptide=rec.peptide,
                    mhc_allele=rec.mhc_allele,
                    value=float(rec.value),
                    qualifier=rec.qualifier,
                    measurement_type=rec.value_type or "IC50",
                    mhc_class=rec.mhc_class or infer_mhc_class(rec.mhc_allele),
                    species=rec.species or infer_species(rec.mhc_allele),
                    source="iedb",
                )
            )
        elif rec.mhc_allele and (
            elution_limit is None or len(elution_records) < elution_limit
        ):
            elution_records.append(
                ElutionRecord(
                    peptide=rec.peptide,
                    alleles=[rec.mhc_allele],
                    detected=True,
                    mhc_class=rec.mhc_class or infer_mhc_class(rec.mhc_allele),
                    species=rec.species or infer_species(rec.mhc_allele),
                    source="iedb",
                )
            )
        if (
            binding_limit is not None
            and elution_limit is not None
            and len(binding_records) >= binding_limit
            and len(elution_records) >= elution_limit
        ):
            break
    return binding_records, elution_records


def load_iedb_additional_records(
    binding_file: Path,
    max_kinetics: Optional[int],
    max_stability: Optional[int],
    max_processing: Optional[int],
) -> Tuple[List[KineticsRecord], List[StabilityRecord], List[ProcessingRecord]]:
    """Load optional kinetics/stability/processing records from IEDB exports."""
    kinetics_records = _take_records(
        load_iedb_kinetics(binding_file),
        _normalize_limit(max_kinetics),
    )
    stability_records = _take_records(
        load_iedb_stability(binding_file),
        _normalize_limit(max_stability),
    )
    processing_records = _take_records(
        load_iedb_processing(binding_file),
        _normalize_limit(max_processing),
    )
    return kinetics_records, stability_records, processing_records


def load_iedb_tcell_records(
    tcell_file: Path,
    max_tcell: Optional[int],
) -> List[TCellRecord]:
    """Load T-cell records from IEDB exports, with fallback parsers."""
    tcell_limit = _normalize_limit(max_tcell)
    tcell_records = _take_records(load_iedb_tcell(tcell_file), tcell_limit)
    if tcell_records:
        return tcell_records

    parsed_records: List[TCellRecord] = []
    for rec in parse_iedb_tcell(tcell_file):
        response = _parse_binary_response(rec.response)
        if response is None or not rec.mhc_allele:
            continue
        parsed_records.append(
            TCellRecord(
                peptide=rec.peptide,
                mhc_allele=rec.mhc_allele,
                response=response,
                mhc_class=rec.mhc_class or infer_mhc_class(rec.mhc_allele),
                species=rec.species or infer_species(rec.mhc_allele),
                source="iedb",
            )
        )
        if tcell_limit is not None and len(parsed_records) >= tcell_limit:
            break
    return parsed_records


def load_vdjdb_records(
    vdjdb_file: Path,
    max_vdjdb: Optional[int],
) -> List[VDJdbRecord]:
    """Load VDJdb records for TCR-supervised training."""
    return _take_records(load_vdjdb(vdjdb_file), _normalize_limit(max_vdjdb))


def load_10x_records(
    sc10x_file: Path,
    max_10x: Optional[int],
) -> List[Sc10xVDJRecord]:
    """Load 10x VDJ contig records for chain-label supervision."""
    return _take_records(load_10x_vdj(sc10x_file), _normalize_limit(max_10x))


def _generate_mhc_only_samples(
    index_csv: str,
    max_samples: int = 5000,
    seed: int = 42,
) -> List[PrestoSample]:
    """Generate MHC-only training samples from the MHC index.

    These samples have a polyalanine placeholder peptide and real MHC
    sequences. They contribute zero supervised binding/tcell/elution loss
    (no targets) but the MHC type/species classification losses train the
    base encoder's MHC representations for discriminative allele encoding.
    """
    records = load_mhc_index(index_csv)
    # Filter to records with valid sequences (no X, ?, or other non-canonical chars)
    valid = []
    rejected: list[tuple[str, str, str]] = []  # (allele, reason, bad_chars)
    for rec in records.values():
        if not rec.sequence or len(rec.sequence) < 50:
            continue
        if "X" in rec.sequence:
            rejected.append((rec.normalized or rec.allele, "contains_X", "X"))
            continue
        bad_chars = set(rec.sequence.upper()) - MHC_SEQUENCE_ALLOWED_AA
        if bad_chars:
            rejected.append((
                rec.normalized or rec.allele,
                "non_canonical_chars",
                "".join(sorted(bad_chars)),
            ))
            continue
        valid.append(rec)

    if rejected:
        print(
            f"MHC augmentation: rejected {len(rejected)} alleles with invalid sequences:"
        )
        for allele, reason, chars in rejected[:50]:
            print(f"  {allele:30s}  reason={reason:20s}  chars={chars}")
        if len(rejected) > 50:
            print(f"  ... and {len(rejected) - 50} more")

    if not valid:
        return []

    rng = random.Random(seed)
    if len(valid) > max_samples:
        valid = rng.sample(valid, max_samples)

    samples: List[PrestoSample] = []
    for rec in valid:
        mhc_class = normalize_mhc_class(rec.mhc_class, default="I")
        species = rec.species or infer_species(rec.normalized) or None

        # Assign chain correctly based on MHC class and gene
        gene = (rec.gene or "").upper()
        is_class_ii_beta = any(
            tag in gene for tag in ("DRB", "DQB", "DPB", "AB", "EB")
        )
        is_class_ii_alpha = any(
            tag in gene for tag in ("DRA", "DQA", "DPA", "AA", "EA")
        )

        if is_class_ii_beta:
            # Class II beta chain → mhc_b slot
            mhc_a_seq = ""
            mhc_b_seq = rec.sequence
        elif mhc_class == "II" and is_class_ii_alpha:
            # Class II alpha chain → mhc_a slot
            mhc_a_seq = rec.sequence
            mhc_b_seq = ""
        else:
            # Class I alpha or unknown → mhc_a slot, pair with species B2M
            mhc_a_seq = rec.sequence
            mhc_b_seq = class_i_beta2m_sequence(species)

        samples.append(PrestoSample(
            peptide="AAAAAAAAA",  # polyalanine placeholder
            mhc_a=mhc_a_seq,
            mhc_b=mhc_b_seq,
            mhc_class=mhc_class,
            species=species,
            sample_source="mhc_augmentation",
            assay_group="mhc_aux",
            synthetic_kind="mhc_only",
            primary_allele=rec.normalized,
            sample_id=f"mhc_aug_{len(samples)}",
        ))
    return samples


def _effective_mhc_augmentation_sample_limit(
    requested_samples: int,
    current_dataset_size: int,
    max_fraction: float,
) -> int:
    """Bound fixed-size MHC-only augmentation so capped runs stay label-dense."""
    requested = max(0, int(requested_samples))
    current_size = max(0, int(current_dataset_size))
    fraction = max(0.0, float(max_fraction))
    if requested <= 0 or current_size <= 0:
        return 0
    if fraction <= 0.0:
        return requested
    fraction_cap = max(1, int(round(current_size * fraction)))
    return min(requested, fraction_cap)


def generate_uniprot_samples(
    proteins: List[UniProtProtein],
    mhc_sequences: Dict[str, str],
    n_samples: int,
    seed: int = 42,
) -> List[PrestoSample]:
    """Generate UniProt-derived negative samples for species/foreignness supervision.

    Each sample draws a random protein (balanced 50/50 between foreign and
    non-foreign categories), extracts a random subsequence as a pseudo-peptide,
    and pairs it with a randomly chosen MHC allele. These samples carry very
    high (weak) binding values so they contribute negligible binding signal,
    but they provide species_of_origin and foreignness_label supervision.
    """
    if not proteins or not mhc_sequences or n_samples <= 0:
        return []

    rng = random.Random(seed)

    # Split proteins into foreign vs non-foreign pools
    foreign = [p for p in proteins if p.category in FOREIGN_CATEGORIES]
    non_foreign = [p for p in proteins if p.category not in FOREIGN_CATEGORIES]

    if not foreign and not non_foreign:
        return []

    allele_keys = list(mhc_sequences.keys())
    samples: List[PrestoSample] = []

    for i in range(n_samples):
        # 50/50 balance between foreign and non-foreign
        if foreign and non_foreign:
            pool = foreign if (i % 2 == 0) else non_foreign
        elif foreign:
            pool = foreign
        else:
            pool = non_foreign

        protein = rng.choice(pool)

        # Extract random subsequence of length 8-50
        seq = protein.sequence
        max_len = min(50, len(seq))
        min_len = min(8, len(seq))
        if min_len > max_len:
            min_len = max_len
        pep_len = rng.randint(min_len, max_len)
        start = rng.randint(0, len(seq) - pep_len)
        peptide = seq[start : start + pep_len]

        # Pick random MHC allele
        allele_key = rng.choice(allele_keys)
        mhc_seq = mhc_sequences[allele_key]

        # Random weak binding value (50000-100000 nM)
        bind_value = rng.uniform(50000, 100000)

        is_foreign = protein.category in FOREIGN_CATEGORIES

        samples.append(PrestoSample(
            peptide=peptide,
            mhc_a=mhc_seq,
            mhc_b="",
            mhc_class="I",
            bind_value=bind_value,
            bind_qual=0,
            processing_label=0.0,
            elution_label=0.0,
            tcell_label=0.0,
            species_of_origin=protein.category,
            foreignness_label=1.0 if is_foreign else 0.0,
            sample_source="uniprot_negative",
            synthetic_kind="uniprot_negative",
            species="human",
            sample_id=f"uniprot_{i}",
        ))

    return samples


def _collect_unique_alleles(
    binding_records: Sequence,
    kinetics_records: Sequence,
    stability_records: Sequence,
    processing_records: Sequence,
    elution_records: Sequence,
    tcell_records: Sequence,
    vdjdb_records: Sequence,
) -> List[str]:
    alleles: List[str] = []
    for rec in binding_records:
        if getattr(rec, "mhc_allele", None):
            alleles.append(rec.mhc_allele.strip())
    for rec in kinetics_records:
        if getattr(rec, "mhc_allele", None):
            alleles.append(rec.mhc_allele.strip())
    for rec in stability_records:
        if getattr(rec, "mhc_allele", None):
            alleles.append(rec.mhc_allele.strip())
    for rec in processing_records:
        if getattr(rec, "mhc_allele", None):
            alleles.append(rec.mhc_allele.strip())
    for rec in tcell_records:
        if getattr(rec, "mhc_allele", None):
            alleles.append(rec.mhc_allele.strip())
    for rec in elution_records:
        for allele in getattr(rec, "alleles", []) or []:
            if allele:
                alleles.append(allele.strip())
    for rec in vdjdb_records:
        if getattr(rec, "mhc_a", None):
            alleles.append(rec.mhc_a.strip())
    return sorted({a for a in alleles if a})


def resolve_mhc_sequences_from_index(
    index_csv: str,
    alleles: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Resolve allele names to sequences using a built MHC index."""
    unique_alleles = sorted({a.strip() for a in alleles if a and a.strip()})
    if not unique_alleles:
        return {}, {"total": 0, "resolved": 0, "missing": 0}

    results = resolve_alleles(
        index_csv=index_csv,
        alleles=unique_alleles,
        include_sequence=True,
    )
    mapping: Dict[str, str] = {}
    resolved = 0
    for row in results:
        if row.get("found") and row.get("sequence"):
            mapping[str(row["input"])] = str(row["sequence"])
            resolved += 1
    return mapping, {
        "total": len(unique_alleles),
        "resolved": resolved,
        "missing": len(unique_alleles) - resolved,
    }


def audit_loaded_mhc_sequence_quality(
    mhc_sequences: Mapping[str, str],
) -> Dict[str, Any]:
    """Validate loaded MHC sequence alphabet and summarize ambiguous residues."""
    noncanonical_examples: List[Tuple[str, str]] = []
    x_examples: List[Tuple[str, int]] = []
    short_examples: List[Tuple[str, int]] = []
    x_residue_total = 0
    noncanonical_count = 0
    x_sequence_count = 0
    short_count = 0

    for allele, raw_seq in mhc_sequences.items():
        seq = str(raw_seq or "").strip().upper()
        if not seq:
            continue
        bad = sorted({ch for ch in seq if ch not in MHC_SEQUENCE_ALLOWED_AA})
        if bad:
            noncanonical_count += 1
            if len(noncanonical_examples) < 12:
                noncanonical_examples.append((str(allele), "".join(bad)))
        if "X" in seq:
            x_sequence_count += 1
            n_x = int(seq.count("X"))
            x_residue_total += n_x
            if len(x_examples) < 12:
                x_examples.append((str(allele), n_x))
        if len(seq) < MIN_MHC_SEQUENCE_LEN:
            short_count += 1
            if len(short_examples) < 12:
                short_examples.append((str(allele), len(seq)))

    return {
        "total_sequences": int(len(mhc_sequences)),
        "noncanonical_count": int(noncanonical_count),
        "noncanonical_examples": noncanonical_examples,
        "x_sequence_count": int(x_sequence_count),
        "x_residue_total": int(x_residue_total),
        "x_examples": x_examples,
        "short_count": int(short_count),
        "short_examples": short_examples,
    }


def _is_mhc_sequence_resolved(
    *,
    allele: Optional[str],
    direct_seq: Optional[str],
    mhc_sequences: Mapping[str, str],
) -> bool:
    if direct_seq and str(direct_seq).strip():
        return True
    token = str(allele or "").strip()
    if not token:
        return True
    seq = str(mhc_sequences.get(token, "")).strip()
    return bool(seq)


def _audit_unresolved_mhc_resolution(
    *,
    binding_records: Sequence[BindingRecord],
    kinetics_records: Sequence[KineticsRecord],
    stability_records: Sequence[StabilityRecord],
    processing_records: Sequence[ProcessingRecord],
    elution_records: Sequence[ElutionRecord],
    tcell_records: Sequence[TCellRecord],
    vdjdb_records: Sequence[VDJdbRecord],
    mhc_sequences: Mapping[str, str],
) -> Dict[str, Any]:
    """Summarize unresolved MHC alleles before dataset construction."""
    by_allele: Counter[str] = Counter()
    by_modality: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    detail: Counter[Tuple[str, str, str, str]] = Counter()
    allele_meta: Dict[str, Dict[str, str]] = {}

    def _record(modality: str, source: str, allele: Optional[str]) -> None:
        token = str(allele or "").strip()
        if not token:
            return
        meta = allele_meta.get(token)
        if meta is None:
            meta = classify_unresolved_allele(token)
            allele_meta[token] = meta
        category = str(meta.get("category") or "unclassified")
        src = source or "unknown"
        by_allele[token] += 1
        by_modality[modality] += 1
        by_source[src] += 1
        by_category[category] += 1
        detail[(modality, src, token, category)] += 1

    for rec in binding_records:
        source = str(rec.source or "unknown")
        # Intentional synthetic ablation: class-I no-alpha negatives have no MHCa by design.
        if source == "synthetic_negative_no_mhc_alpha":
            continue
        if not _is_mhc_sequence_resolved(
            allele=rec.mhc_allele,
            direct_seq=rec.mhc_sequence,
            mhc_sequences=mhc_sequences,
        ):
            _record("binding", source, rec.mhc_allele)

    for rec in kinetics_records:
        if not _is_mhc_sequence_resolved(
            allele=rec.mhc_allele,
            direct_seq=rec.mhc_sequence,
            mhc_sequences=mhc_sequences,
        ):
            _record("kinetics", str(rec.source or "unknown"), rec.mhc_allele)

    for rec in stability_records:
        if not _is_mhc_sequence_resolved(
            allele=rec.mhc_allele,
            direct_seq=rec.mhc_sequence,
            mhc_sequences=mhc_sequences,
        ):
            _record("stability", str(rec.source or "unknown"), rec.mhc_allele)

    for rec in processing_records:
        if not _is_mhc_sequence_resolved(
            allele=rec.mhc_allele,
            direct_seq=None,
            mhc_sequences=mhc_sequences,
        ):
            _record("processing", str(rec.source or "unknown"), rec.mhc_allele)

    for rec in elution_records:
        source = str(rec.source or "unknown")
        for allele in rec.alleles or []:
            if not _is_mhc_sequence_resolved(
                allele=allele,
                direct_seq=None,
                mhc_sequences=mhc_sequences,
            ):
                _record("elution", source, allele)

    for rec in tcell_records:
        if not _is_mhc_sequence_resolved(
            allele=rec.mhc_allele,
            direct_seq=rec.mhc_sequence,
            mhc_sequences=mhc_sequences,
        ):
            _record("tcell", str(rec.source or "unknown"), rec.mhc_allele)

    for rec in vdjdb_records:
        if not _is_mhc_sequence_resolved(
            allele=rec.mhc_a,
            direct_seq=None,
            mhc_sequences=mhc_sequences,
        ):
            _record("vdjdb", str(rec.source or "unknown"), rec.mhc_a)

    return {
        "total_unresolved": int(sum(by_allele.values())),
        "by_allele": by_allele,
        "by_modality": by_modality,
        "by_source": by_source,
        "by_category": by_category,
        "allele_meta": allele_meta,
        "detail": detail,
    }


def _write_unresolved_mhc_report(
    *,
    run_dir: Optional[Path],
    audit: Mapping[str, Any],
) -> Dict[str, Optional[Path]]:
    """Write unresolved MHC summaries to run-dir for edge-case triage."""
    outputs: Dict[str, Optional[Path]] = {"alleles": None, "detail": None}
    if run_dir is None:
        return outputs
    by_allele = audit.get("by_allele")
    allele_meta = audit.get("allele_meta")
    detail = audit.get("detail")
    if not isinstance(by_allele, Counter) or not by_allele:
        return outputs

    run_dir.mkdir(parents=True, exist_ok=True)
    allele_path = run_dir / "unresolved_mhc_alleles.csv"
    with allele_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "allele",
                "count",
                "category",
                "parsed_type",
                "normalized",
                "species",
                "mhc_class",
                "gene",
                "parse_error",
            ]
        )
        for allele, count in by_allele.most_common():
            meta: Mapping[str, str] = {}
            if isinstance(allele_meta, Mapping):
                current = allele_meta.get(allele, {})
                if isinstance(current, Mapping):
                    meta = current
            writer.writerow(
                [
                    allele,
                    count,
                    meta.get("category", ""),
                    meta.get("parsed_type", ""),
                    meta.get("normalized", ""),
                    meta.get("species", ""),
                    meta.get("mhc_class", ""),
                    meta.get("gene", ""),
                    meta.get("parse_error", ""),
                ]
            )
    outputs["alleles"] = allele_path

    if isinstance(detail, Counter) and detail:
        detail_path = run_dir / "unresolved_mhc_detail.csv"
        with detail_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["modality", "source", "allele", "category", "count"])
            for (modality, source, allele, category), count in detail.most_common():
                writer.writerow([modality, source, allele, category, count])
        outputs["detail"] = detail_path

    return outputs


def _coarse_species_bucket(
    species: Optional[str],
    allele: Optional[str],
) -> str:
    bucket = normalize_processing_species_label(species)
    if bucket in {None, "", "other"}:
        inferred = normalize_processing_species_label(infer_species(str(allele or "")))
        if inferred:
            bucket = inferred
    if bucket not in {"human", "murine", "nhp", "other"}:
        bucket = "other"
    return str(bucket)


def _audit_mhc_sequence_coverage(
    *,
    binding_records: Sequence[BindingRecord],
    kinetics_records: Sequence[KineticsRecord],
    stability_records: Sequence[StabilityRecord],
    processing_records: Sequence[ProcessingRecord],
    elution_records: Sequence[ElutionRecord],
    tcell_records: Sequence[TCellRecord],
    vdjdb_records: Sequence[VDJdbRecord],
    mhc_sequences: Mapping[str, str],
) -> Dict[str, Any]:
    """Summarize resolved-vs-missing MHC coverage across training rows."""
    overall_state = Counter()
    modality_state: Dict[str, Counter[str]] = defaultdict(Counter)
    species_by_state: Dict[str, Counter[str]] = {
        "resolved": Counter(),
        "missing": Counter(),
    }
    modality_species_by_state: Dict[str, Dict[str, Counter[str]]] = defaultdict(
        lambda: {"resolved": Counter(), "missing": Counter()}
    )

    def _record(
        *,
        modality: str,
        allele: Optional[str],
        direct_seq: Optional[str],
        species: Optional[str],
        source: Optional[str] = None,
    ) -> None:
        # Synthetic class-I no-alpha negatives intentionally ablate MHCa.
        if str(source or "") == "synthetic_negative_no_mhc_alpha":
            return
        resolved = _is_mhc_sequence_resolved(
            allele=allele,
            direct_seq=direct_seq,
            mhc_sequences=mhc_sequences,
        )
        state = "resolved" if resolved else "missing"
        sp = _coarse_species_bucket(species=species, allele=allele)
        overall_state[state] += 1
        modality_state[modality][state] += 1
        species_by_state[state][sp] += 1
        modality_species_by_state[modality][state][sp] += 1

    for rec in binding_records:
        _record(
            modality="binding",
            allele=rec.mhc_allele,
            direct_seq=rec.mhc_sequence,
            species=rec.species,
            source=rec.source,
        )
    for rec in kinetics_records:
        _record(
            modality="kinetics",
            allele=rec.mhc_allele,
            direct_seq=rec.mhc_sequence,
            species=rec.species,
            source=rec.source,
        )
    for rec in stability_records:
        _record(
            modality="stability",
            allele=rec.mhc_allele,
            direct_seq=rec.mhc_sequence,
            species=rec.species,
            source=rec.source,
        )
    for rec in processing_records:
        _record(
            modality="processing",
            allele=rec.mhc_allele,
            direct_seq=None,
            species=rec.species,
            source=rec.source,
        )
    for rec in elution_records:
        alleles = list(rec.alleles or [])
        if not alleles:
            alleles = [None]
        for allele in alleles:
            _record(
                modality="elution",
                allele=allele,
                direct_seq=None,
                species=rec.species,
                source=rec.source,
            )
    for rec in tcell_records:
        _record(
            modality="tcell",
            allele=rec.mhc_allele,
            direct_seq=rec.mhc_sequence,
            species=rec.species,
            source=rec.source,
        )
    for rec in vdjdb_records:
        _record(
            modality="vdjdb",
            allele=rec.mhc_a,
            direct_seq=None,
            species=getattr(rec, "species", None),
            source=rec.source,
        )

    total_rows = int(overall_state.get("resolved", 0) + overall_state.get("missing", 0))
    resolved_rows = int(overall_state.get("resolved", 0))
    missing_rows = int(overall_state.get("missing", 0))
    denom = max(total_rows, 1)

    by_modality: Dict[str, Dict[str, float]] = {}
    for modality, counts in modality_state.items():
        mod_total = int(counts.get("resolved", 0) + counts.get("missing", 0))
        mod_denom = max(mod_total, 1)
        by_modality[modality] = {
            "rows": mod_total,
            "resolved_rows": int(counts.get("resolved", 0)),
            "missing_rows": int(counts.get("missing", 0)),
            "resolved_fraction": float(counts.get("resolved", 0)) / float(mod_denom),
            "missing_fraction": float(counts.get("missing", 0)) / float(mod_denom),
        }

    def _species_counter_dict(counter: Counter[str]) -> Dict[str, int]:
        return {
            "human": int(counter.get("human", 0)),
            "murine": int(counter.get("murine", 0)),
            "nhp": int(counter.get("nhp", 0)),
            "other": int(counter.get("other", 0)),
        }

    species_summary = {
        "resolved": _species_counter_dict(species_by_state["resolved"]),
        "missing": _species_counter_dict(species_by_state["missing"]),
    }

    modality_species_summary: Dict[str, Dict[str, Dict[str, int]]] = {}
    for modality, state_map in modality_species_by_state.items():
        modality_species_summary[modality] = {
            "resolved": _species_counter_dict(state_map.get("resolved", Counter())),
            "missing": _species_counter_dict(state_map.get("missing", Counter())),
        }

    return {
        "overall": {
            "rows_considered": total_rows,
            "resolved_rows": resolved_rows,
            "missing_rows": missing_rows,
            "resolved_fraction": float(resolved_rows) / float(denom),
            "missing_fraction": float(missing_rows) / float(denom),
        },
        "species_by_state": species_summary,
        "by_modality": by_modality,
        "by_modality_species": modality_species_summary,
    }


def _filter_records_to_resolved_mhc(
    *,
    binding_records: Sequence[BindingRecord],
    kinetics_records: Sequence[KineticsRecord],
    stability_records: Sequence[StabilityRecord],
    processing_records: Sequence[ProcessingRecord],
    elution_records: Sequence[ElutionRecord],
    tcell_records: Sequence[TCellRecord],
    vdjdb_records: Sequence[VDJdbRecord],
    mhc_sequences: Mapping[str, str],
) -> Tuple[
    List[BindingRecord],
    List[KineticsRecord],
    List[StabilityRecord],
    List[ProcessingRecord],
    List[ElutionRecord],
    List[TCellRecord],
    List[VDJdbRecord],
    Dict[str, int],
]:
    """Drop unresolved-MHC rows; for elution keep only resolved alleles per row."""
    stats: Dict[str, int] = {
        "binding_dropped": 0,
        "kinetics_dropped": 0,
        "stability_dropped": 0,
        "processing_dropped": 0,
        "elution_rows_dropped": 0,
        "elution_alleles_dropped": 0,
        "tcell_dropped": 0,
        "vdjdb_dropped": 0,
    }

    def _keep(allele: Optional[str], direct_seq: Optional[str], source: Optional[str] = None) -> bool:
        if str(source or "") == "synthetic_negative_no_mhc_alpha":
            return True
        return _is_mhc_sequence_resolved(
            allele=allele,
            direct_seq=direct_seq,
            mhc_sequences=mhc_sequences,
        )

    binding_filtered: List[BindingRecord] = []
    for rec in binding_records:
        if _keep(rec.mhc_allele, rec.mhc_sequence, rec.source):
            binding_filtered.append(rec)
        else:
            stats["binding_dropped"] += 1

    kinetics_filtered: List[KineticsRecord] = []
    for rec in kinetics_records:
        if _keep(rec.mhc_allele, rec.mhc_sequence, rec.source):
            kinetics_filtered.append(rec)
        else:
            stats["kinetics_dropped"] += 1

    stability_filtered: List[StabilityRecord] = []
    for rec in stability_records:
        if _keep(rec.mhc_allele, rec.mhc_sequence, rec.source):
            stability_filtered.append(rec)
        else:
            stats["stability_dropped"] += 1

    processing_filtered: List[ProcessingRecord] = []
    for rec in processing_records:
        if _keep(rec.mhc_allele, None, rec.source):
            processing_filtered.append(rec)
        else:
            stats["processing_dropped"] += 1

    elution_filtered: List[ElutionRecord] = []
    for rec in elution_records:
        alleles = list(rec.alleles or [])
        if not alleles:
            stats["elution_rows_dropped"] += 1
            continue
        resolved_alleles = [
            allele
            for allele in alleles
            if _keep(allele, None, rec.source)
        ]
        dropped = len(alleles) - len(resolved_alleles)
        if dropped > 0:
            stats["elution_alleles_dropped"] += dropped
        if not resolved_alleles:
            stats["elution_rows_dropped"] += 1
            continue
        if resolved_alleles == alleles:
            elution_filtered.append(rec)
        else:
            elution_filtered.append(replace(rec, alleles=resolved_alleles))

    tcell_filtered: List[TCellRecord] = []
    for rec in tcell_records:
        if _keep(rec.mhc_allele, rec.mhc_sequence, rec.source):
            tcell_filtered.append(rec)
        else:
            stats["tcell_dropped"] += 1

    vdjdb_filtered: List[VDJdbRecord] = []
    for rec in vdjdb_records:
        if _keep(rec.mhc_a, None, rec.source):
            vdjdb_filtered.append(rec)
        else:
            stats["vdjdb_dropped"] += 1

    return (
        binding_filtered,
        kinetics_filtered,
        stability_filtered,
        processing_filtered,
        elution_filtered,
        tcell_filtered,
        vdjdb_filtered,
        stats,
    )


def _write_mhc_sequence_coverage_report(
    *,
    run_dir: Optional[Path],
    coverage: Mapping[str, Any],
    basename: str = "mhc_sequence_coverage",
) -> Dict[str, Optional[Path]]:
    outputs: Dict[str, Optional[Path]] = {"json": None, "csv": None}
    if run_dir is None:
        return outputs
    run_dir.mkdir(parents=True, exist_ok=True)

    stem = str(basename or "mhc_sequence_coverage").strip() or "mhc_sequence_coverage"
    json_path = run_dir / f"{stem}.json"
    json_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    outputs["json"] = json_path

    overall = coverage.get("overall", {}) if isinstance(coverage, Mapping) else {}
    by_modality = coverage.get("by_modality", {}) if isinstance(coverage, Mapping) else {}
    species_by_state = coverage.get("species_by_state", {}) if isinstance(coverage, Mapping) else {}
    by_modality_species = (
        coverage.get("by_modality_species", {}) if isinstance(coverage, Mapping) else {}
    )
    total_rows = int(overall.get("rows_considered", 0) or 0)
    denom_total = max(total_rows, 1)

    csv_path = run_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scope",
                "modality",
                "state",
                "species",
                "count",
                "fraction_of_total_rows",
                "fraction_within_scope",
            ]
        )

        for state in ("resolved", "missing"):
            count = int(overall.get(f"{state}_rows", 0) or 0)
            writer.writerow(
                [
                    "overall",
                    "all",
                    state,
                    "all",
                    count,
                    float(count) / float(denom_total),
                    float(count) / float(denom_total),
                ]
            )

        for state in ("resolved", "missing"):
            state_counts = species_by_state.get(state, {}) if isinstance(species_by_state, Mapping) else {}
            state_total = int(sum(int(state_counts.get(sp, 0) or 0) for sp in ("human", "murine", "nhp", "other")))
            state_denom = max(state_total, 1)
            for species in ("human", "murine", "nhp", "other"):
                count = int(state_counts.get(species, 0) or 0)
                writer.writerow(
                    [
                        "overall_species",
                        "all",
                        state,
                        species,
                        count,
                        float(count) / float(denom_total),
                        float(count) / float(state_denom),
                    ]
                )

        if isinstance(by_modality, Mapping):
            for modality in sorted(by_modality.keys()):
                mod_stats = by_modality.get(modality, {})
                if not isinstance(mod_stats, Mapping):
                    continue
                mod_total = int(mod_stats.get("rows", 0) or 0)
                mod_denom = max(mod_total, 1)
                for state in ("resolved", "missing"):
                    count = int(mod_stats.get(f"{state}_rows", 0) or 0)
                    writer.writerow(
                        [
                            "modality",
                            modality,
                            state,
                            "all",
                            count,
                            float(count) / float(denom_total),
                            float(count) / float(mod_denom),
                        ]
                    )
                modality_species = (
                    by_modality_species.get(modality, {})
                    if isinstance(by_modality_species, Mapping)
                    else {}
                )
                for state in ("resolved", "missing"):
                    state_species = (
                        modality_species.get(state, {})
                        if isinstance(modality_species, Mapping)
                        else {}
                    )
                    state_total = int(sum(int(state_species.get(sp, 0) or 0) for sp in ("human", "murine", "nhp", "other")))
                    state_denom = max(state_total, 1)
                    for species in ("human", "murine", "nhp", "other"):
                        count = int(state_species.get(species, 0) or 0)
                        writer.writerow(
                            [
                                "modality_species",
                                modality,
                                state,
                                species,
                                count,
                                float(count) / float(denom_total),
                                float(count) / float(state_denom),
                            ]
                        )

    outputs["csv"] = csv_path
    return outputs


def _print_mhc_sequence_coverage_summary(coverage: Mapping[str, Any]) -> None:
    overall = coverage.get("overall", {}) if isinstance(coverage, Mapping) else {}
    rows = int(overall.get("rows_considered", 0) or 0)
    resolved = int(overall.get("resolved_rows", 0) or 0)
    missing = int(overall.get("missing_rows", 0) or 0)
    resolved_pct = 100.0 * float(overall.get("resolved_fraction", 0.0) or 0.0)
    missing_pct = 100.0 * float(overall.get("missing_fraction", 0.0) or 0.0)
    print(
        "MHC sequence coverage: "
        f"resolved={resolved}/{rows} ({resolved_pct:.2f}%), "
        f"missing={missing}/{rows} ({missing_pct:.2f}%)"
    )
    species_by_state = coverage.get("species_by_state", {}) if isinstance(coverage, Mapping) else {}
    for state in ("resolved", "missing"):
        counts = species_by_state.get(state, {}) if isinstance(species_by_state, Mapping) else {}
        if not isinstance(counts, Mapping):
            continue
        text = ", ".join(
            f"{species}={int(counts.get(species, 0) or 0)}"
            for species in ("human", "murine", "nhp", "other")
        )
        print(f"  {state} species buckets: {text}")

def _split_allele_list(raw_value: Optional[str]) -> List[str]:
    if raw_value is None:
        return []
    parts = re.split(r"[,;/|]", str(raw_value))
    return [part.strip() for part in parts if part and part.strip()]


def _parse_float_or_none(raw_value: Optional[str]) -> Optional[float]:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int_or_default(raw_value: Optional[str], default: int = 0) -> int:
    if raw_value is None:
        return default
    text = str(raw_value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


VALID_AA_TOKENS = set("ACDEFGHIKLMNPQRSTVWYX")
MISSING_SEQ_TOKENS = {"NA", "N/A", "NONE", "NULL", "-", "?"}


def _normalize_required_aa_sequence(raw_value: Optional[str]) -> Optional[str]:
    text = str(raw_value or "").strip().upper()
    if not text:
        return None
    if any(ch not in VALID_AA_TOKENS for ch in text):
        return None
    return text


def _normalize_optional_aa_sequence(raw_value: Optional[str]) -> Tuple[str, bool]:
    raw_text = str(raw_value or "").strip()
    text = raw_text.upper()
    if not text:
        return "", False
    if text in MISSING_SEQ_TOKENS:
        return "", True
    if any(ch not in VALID_AA_TOKENS for ch in text):
        return "", True
    return text, False


def _append_with_cap_sampling(
    records: List[Any],
    item: Any,
    limit: Optional[int],
    *,
    sampling: str,
    rng: random.Random,
    seen: int,
) -> int:
    """Append or reservoir-replace while honoring an optional cap."""
    if limit is None:
        records.append(item)
        return seen

    if sampling == "head":
        if len(records) < limit:
            records.append(item)
        return seen

    # Reservoir sampling over all valid candidates seen for this modality.
    seen += 1
    if len(records) < limit:
        records.append(item)
    else:
        replace_idx = rng.randrange(seen)
        if replace_idx < limit:
            records[replace_idx] = item
    return seen


def load_records_from_merged_tsv(
    merged_tsv: Path,
    *,
    max_binding: Optional[int],
    max_kinetics: Optional[int],
    max_stability: Optional[int],
    max_processing: Optional[int],
    max_elution: Optional[int],
    max_tcell: Optional[int],
    max_vdjdb: Optional[int],
    cap_sampling: str = "reservoir",
    sampling_seed: int = 42,
) -> Tuple[
    List[BindingRecord],
    List[KineticsRecord],
    List[StabilityRecord],
    List[ProcessingRecord],
    List[ElutionRecord],
    List[TCellRecord],
    List[VDJdbRecord],
    Dict[str, Any],
]:
    """Load canonical training records directly from merged TSV output."""
    if not merged_tsv.exists():
        raise FileNotFoundError(f"Merged TSV not found: {merged_tsv}")

    binding_records: List[BindingRecord] = []
    kinetics_records: List[KineticsRecord] = []
    stability_records: List[StabilityRecord] = []
    processing_records: List[ProcessingRecord] = []
    elution_records: List[ElutionRecord] = []
    tcell_records: List[TCellRecord] = []
    vdjdb_records: List[VDJdbRecord] = []
    by_assay: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    rows_dropped_invalid_peptide = 0
    rows_sanitized_optional_sequences = 0

    binding_limit = _normalize_limit(max_binding)
    kinetics_limit = _normalize_limit(max_kinetics)
    stability_limit = _normalize_limit(max_stability)
    processing_limit = _normalize_limit(max_processing)
    elution_limit = _normalize_limit(max_elution)
    tcell_limit = _normalize_limit(max_tcell)
    vdjdb_limit = _normalize_limit(max_vdjdb)
    sampling_mode = str(cap_sampling or "reservoir").strip().lower()
    if sampling_mode not in {"head", "reservoir"}:
        raise ValueError(
            f"Unsupported cap sampling mode: {cap_sampling!r}. "
            "Expected one of: head, reservoir."
        )
    sampling_rng = random.Random(int(sampling_seed))
    binding_seen = 0
    kinetics_seen = 0
    stability_seen = 0
    processing_seen = 0
    elution_seen = 0
    tcell_seen = 0
    vdjdb_seen = 0

    def _all_requested_limits_satisfied() -> bool:
        """Return true when every explicitly requested limit has been met."""
        limits_and_counts = (
            (binding_limit, len(binding_records)),
            (kinetics_limit, len(kinetics_records)),
            (stability_limit, len(stability_records)),
            (processing_limit, len(processing_records)),
            (elution_limit, len(elution_records)),
            (tcell_limit, len(tcell_records)),
            (vdjdb_limit, len(vdjdb_records)),
        )
        any_bounded = False
        for limit, count in limits_and_counts:
            if limit is None:
                continue
            any_bounded = True
            if count < limit:
                return False
        return any_bounded

    with merged_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            peptide = _normalize_required_aa_sequence(row.get("peptide"))
            if not peptide:
                rows_dropped_invalid_peptide += 1
                continue

            mhc_allele = (row.get("mhc_allele") or "").strip()
            mhc_class = normalize_mhc_class(row.get("mhc_class"), default=infer_mhc_class(mhc_allele))
            source = (row.get("source") or "").strip() or "unknown"
            species = (row.get("species") or "").strip() or infer_species(mhc_allele) or None
            antigen_species_col = (row.get("antigen_species") or "").strip() or None
            value = _parse_float_or_none(row.get("value"))
            qualifier = _parse_int_or_default(row.get("qualifier"), default=0)
            value_type = (row.get("value_type") or "").strip()
            response = (row.get("response") or "").strip()
            assay_type_col = (row.get("assay_type") or "").strip()
            assay_method_col = (row.get("assay_method") or "").strip()
            apc_name_col = (row.get("apc_name") or "").strip()
            effector_culture_col = (row.get("effector_culture_condition") or "").strip()
            apc_culture_col = (row.get("apc_culture_condition") or "").strip()
            in_vitro_process_col = (row.get("in_vitro_process_type") or "").strip()
            in_vitro_responder_col = (row.get("in_vitro_responder_cell") or "").strip()
            in_vitro_stimulator_col = (row.get("in_vitro_stimulator_cell") or "").strip()
            record_type = (row.get("record_type") or "").strip()
            cdr3_alpha, alpha_sanitized = _normalize_optional_aa_sequence(row.get("cdr3_alpha"))
            cdr3_beta, beta_sanitized = _normalize_optional_aa_sequence(row.get("cdr3_beta"))
            rows_sanitized_optional_sequences += int(alpha_sanitized) + int(beta_sanitized)
            trav = (row.get("trav") or "").strip()
            trbv = (row.get("trbv") or "").strip()

            unified = UnifiedRecord(
                peptide=peptide,
                mhc_allele=mhc_allele,
                mhc_class=mhc_class,
                source=source,
                record_type=record_type,
                value=value,
                value_type=value_type,
                qualifier=qualifier,
                response=response,
                assay_type=assay_type_col or None,
                assay_method=assay_method_col or None,
                apc_name=apc_name_col or None,
                effector_culture_condition=effector_culture_col or None,
                apc_culture_condition=apc_culture_col or None,
                in_vitro_process_type=in_vitro_process_col or None,
                in_vitro_responder_cell=in_vitro_responder_col or None,
                in_vitro_stimulator_cell=in_vitro_stimulator_col or None,
                cdr3_alpha=cdr3_alpha or None,
                cdr3_beta=cdr3_beta or None,
                trav=trav or None,
                trbv=trbv or None,
                species=species,
                antigen_species=antigen_species_col,
            )
            assay_type = classify_assay_type(unified)
            by_assay[assay_type] = by_assay.get(assay_type, 0) + 1
            by_source[source] = by_source.get(source, 0) + 1

            if assay_type == "binding_affinity":
                if value is None:
                    continue
                binding_seen = _append_with_cap_sampling(
                    binding_records,
                    BindingRecord(
                        peptide=peptide,
                        mhc_allele=mhc_allele,
                        value=value,
                        qualifier=qualifier,
                        measurement_type=value_type or "IC50",
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                    ),
                    binding_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=binding_seen,
                )
                continue

            if assay_type == "binding_kon":
                if value is None:
                    continue
                kinetics_seen = _append_with_cap_sampling(
                    kinetics_records,
                    KineticsRecord(
                        peptide=peptide,
                        mhc_allele=mhc_allele,
                        kon=value,
                        koff=None,
                        kon_qualifier=qualifier,
                        assay_type=value_type or "kon",
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                    ),
                    kinetics_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=kinetics_seen,
                )
                continue

            if assay_type == "binding_koff":
                if value is None:
                    continue
                kinetics_seen = _append_with_cap_sampling(
                    kinetics_records,
                    KineticsRecord(
                        peptide=peptide,
                        mhc_allele=mhc_allele,
                        kon=None,
                        koff=value,
                        koff_qualifier=qualifier,
                        assay_type=value_type or "koff",
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                    ),
                    kinetics_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=kinetics_seen,
                )
                continue

            if assay_type == "binding_t_half":
                if value is None:
                    continue
                stability_seen = _append_with_cap_sampling(
                    stability_records,
                    StabilityRecord(
                        peptide=peptide,
                        mhc_allele=mhc_allele,
                        t_half=value,
                        tm=None,
                        t_half_qualifier=qualifier,
                        assay_type=value_type or "t_half",
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                    ),
                    stability_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=stability_seen,
                )
                continue

            if assay_type == "binding_tm":
                if value is None:
                    continue
                stability_seen = _append_with_cap_sampling(
                    stability_records,
                    StabilityRecord(
                        peptide=peptide,
                        mhc_allele=mhc_allele,
                        t_half=None,
                        tm=value,
                        tm_qualifier=qualifier,
                        assay_type=value_type or "Tm",
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                    ),
                    stability_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=stability_seen,
                )
                continue

            if assay_type.startswith("elution_ms"):
                detected = _parse_binary_response(response)
                if detected is None:
                    detected = 0.0 if source.startswith("synthetic_negative") else 1.0
                alleles = _split_allele_list(mhc_allele)
                if not alleles and mhc_allele:
                    alleles = [mhc_allele]
                if not alleles:
                    continue
                elution_seen = _append_with_cap_sampling(
                    elution_records,
                    ElutionRecord(
                        peptide=peptide,
                        alleles=alleles,
                        detected=bool(detected > 0.5),
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                    ),
                    elution_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=elution_seen,
                )
                continue

            if assay_type == "tcell_response":
                response_value = _parse_binary_response(response)
                if response_value is None:
                    continue
                tcell_seen = _append_with_cap_sampling(
                    tcell_records,
                    TCellRecord(
                        peptide=peptide,
                        mhc_allele=mhc_allele,
                        response=response_value,
                        assay_type=assay_type_col or value_type or None,
                        assay_method=assay_method_col or None,
                        apc_name=apc_name_col or None,
                        effector_culture_condition=effector_culture_col or None,
                        apc_culture_condition=apc_culture_col or None,
                        in_vitro_process_type=in_vitro_process_col or None,
                        in_vitro_responder_cell=in_vitro_responder_col or None,
                        in_vitro_stimulator_cell=in_vitro_stimulator_col or None,
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                    ),
                    tcell_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=tcell_seen,
                )
                continue

            if assay_type == "tcr_pmhc":
                if not mhc_allele:
                    continue
                gene = "TRB" if cdr3_beta else "TRA"
                vdjdb_seen = _append_with_cap_sampling(
                    vdjdb_records,
                    VDJdbRecord(
                        peptide=peptide,
                        mhc_a=mhc_allele,
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                        gene=gene,
                        cdr3_alpha=cdr3_alpha or None,
                        cdr3_beta=cdr3_beta or None,
                        v_alpha=trav or None,
                        v_beta=trbv or None,
                    ),
                    vdjdb_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=vdjdb_seen,
                )
                continue

            if assay_type == "processing" or record_type == "processing":
                label = _parse_binary_response(response)
                if label is None:
                    label = 1.0
                processing_seen = _append_with_cap_sampling(
                    processing_records,
                    ProcessingRecord(
                        peptide=peptide,
                        label=label,
                        processing_type=value_type or "processing",
                        mhc_allele=mhc_allele or None,
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                    ),
                    processing_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=processing_seen,
                )

            if sampling_mode == "head" and _all_requested_limits_satisfied():
                break

    stats = {
        "rows_scanned": sum(by_assay.values()),
        "rows_by_assay": dict(sorted(by_assay.items(), key=lambda item: (-item[1], item[0]))),
        "rows_by_source": dict(sorted(by_source.items(), key=lambda item: (-item[1], item[0]))),
        "rows_dropped_invalid_peptide": rows_dropped_invalid_peptide,
        "rows_sanitized_optional_sequences": rows_sanitized_optional_sequences,
        "cap_sampling": sampling_mode,
        "records_loaded": {
            "binding": len(binding_records),
            "kinetics": len(kinetics_records),
            "stability": len(stability_records),
            "processing": len(processing_records),
            "elution": len(elution_records),
            "tcell": len(tcell_records),
            "tcr": len(vdjdb_records),
        },
    }
    return (
        binding_records,
        kinetics_records,
        stability_records,
        processing_records,
        elution_records,
        tcell_records,
        vdjdb_records,
        stats,
    )


def run(args: argparse.Namespace) -> None:
    """Run unified multi-source training with parsed arguments."""
    args = _resolve_run_args(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    profile = str(getattr(args, "profile", "full") or "full").lower()
    if profile == "canary":
        print("Using profile: canary (fast all-source smoke run)")
    elif profile == "diagnostic":
        print("Using profile: diagnostic (coverage + probe + pMHC flow + latent diagnostics)")

    run_dir_arg = getattr(args, "run_dir", None)
    run_dir = Path(run_dir_arg) if run_dir_arg else None
    if run_dir is None and args.checkpoint:
        run_dir = Path(args.checkpoint).resolve().parent
    run_logger = RunLogger(run_dir, config=vars(args)) if run_dir is not None else None

    data_dir = Path(args.data_dir)
    merged_tsv = Path(args.merged_tsv) if getattr(args, "merged_tsv", None) else data_dir / "merged_deduped.tsv"

    binding_records: List[BindingRecord] = []
    kinetics_records: List[KineticsRecord] = []
    stability_records: List[StabilityRecord] = []
    processing_records: List[ProcessingRecord] = []
    elution_records: List[ElutionRecord] = []
    tcell_records: List[TCellRecord] = []
    vdjdb_records: List[VDJdbRecord] = []
    sc10x_records: List[Sc10xVDJRecord] = []

    if merged_tsv.exists():
        print(f"Merged input: {merged_tsv}")
        (
            binding_records,
            kinetics_records,
            stability_records,
            processing_records,
            elution_records,
            tcell_records,
            vdjdb_records,
            merged_stats,
        ) = load_records_from_merged_tsv(
            merged_tsv=merged_tsv,
            max_binding=args.max_binding,
            max_kinetics=args.max_kinetics,
            max_stability=args.max_stability,
            max_processing=args.max_processing,
            max_elution=args.max_elution,
            max_tcell=args.max_tcell,
            max_vdjdb=args.max_vdjdb,
            cap_sampling=getattr(args, "cap_sampling", "reservoir"),
            sampling_seed=args.seed + 17,
        )
        print(f"  Merged rows scanned: {merged_stats['rows_scanned']}")
        print(f"  Cap sampling: {merged_stats.get('cap_sampling', 'unknown')}")
        print("  Merged rows by assay:")
        for assay, count in merged_stats["rows_by_assay"].items():
            print(f"    {assay}: {count}")
        print("  Merged rows by source:")
        for source, count in merged_stats["rows_by_source"].items():
            print(f"    {source}: {count}")
        dropped_invalid = int(merged_stats.get("rows_dropped_invalid_peptide", 0) or 0)
        sanitized_optional = int(
            merged_stats.get("rows_sanitized_optional_sequences", 0) or 0
        )
        if dropped_invalid > 0 or sanitized_optional > 0:
            print(
                "  Sequence cleanup: "
                f"dropped_invalid_peptide={dropped_invalid}, "
                f"sanitized_optional_seq_fields={sanitized_optional}"
            )

        if args.sc10x_file:
            sc10x_file = Path(args.sc10x_file)
        else:
            try:
                sc10x_file = find_iedb_export_file(
                    data_dir / "10x",
                    keywords=("contig", "10x", "vdj", "tcr"),
                )
            except FileNotFoundError:
                sc10x_file = None
        print(f"10x VDJ export: {sc10x_file or '(not found)'}")
        if sc10x_file is not None:
            sc10x_records = load_10x_records(
                sc10x_file=sc10x_file,
                max_10x=args.max_10x,
            )
    else:
        if getattr(args, "require_merged_input", True):
            raise FileNotFoundError(
                f"Merged TSV not found at {merged_tsv}. "
                "Run `python -m presto data merge --datadir ./data` first "
                "or pass --merged-tsv. "
                "Use --allow-raw-fallback only when intentionally debugging raw exports."
            )

        print("Merged TSV not found; falling back to raw source exports.")
        iedb_dir = data_dir / "iedb"
        binding_file = Path(args.binding_file) if args.binding_file else find_iedb_export_file(
            iedb_dir, keywords=("mhc", "ligand")
        )
        tcell_file = Path(args.tcell_file) if args.tcell_file else find_iedb_export_file(
            iedb_dir, keywords=("tcell",)
        )
        cedar_binding_file: Optional[Path] = (
            Path(args.cedar_binding_file) if getattr(args, "cedar_binding_file", None) else None
        )
        cedar_tcell_file: Optional[Path] = (
            Path(args.cedar_tcell_file) if getattr(args, "cedar_tcell_file", None) else None
        )
        if cedar_binding_file is None:
            try:
                candidate = find_iedb_export_file(
                    iedb_dir,
                    keywords=("mhc", "ligand"),
                    required_keywords=("cedar",),
                )
                if candidate != binding_file:
                    cedar_binding_file = candidate
            except FileNotFoundError:
                cedar_binding_file = None
        if cedar_tcell_file is None:
            try:
                candidate = find_iedb_export_file(
                    iedb_dir,
                    keywords=("tcell",),
                    required_keywords=("cedar",),
                )
                if candidate != tcell_file:
                    cedar_tcell_file = candidate
            except FileNotFoundError:
                cedar_tcell_file = None
        if args.vdjdb_file:
            vdjdb_file = Path(args.vdjdb_file)
        else:
            try:
                vdjdb_file = find_iedb_export_file(data_dir / "vdjdb", keywords=("vdjdb",))
            except FileNotFoundError:
                vdjdb_file = None
        if args.sc10x_file:
            sc10x_file = Path(args.sc10x_file)
        else:
            try:
                sc10x_file = find_iedb_export_file(
                    data_dir / "10x",
                    keywords=("contig", "10x", "vdj", "tcr"),
                )
            except FileNotFoundError:
                sc10x_file = None
        print(f"Binding/elution export: {binding_file}")
        if cedar_binding_file is not None:
            print(f"CEDAR binding/elution export: {cedar_binding_file}")
        print(f"T-cell export: {tcell_file}")
        if cedar_tcell_file is not None:
            print(f"CEDAR T-cell export: {cedar_tcell_file}")
        print(f"VDJdb export: {vdjdb_file or '(not found)'}")
        print(f"10x VDJ export: {sc10x_file or '(not found)'}")

        print("Loading unified multi-source records...")
        primary_binding_records, primary_elution_records = load_iedb_binding_and_elution_records(
            binding_file=binding_file,
            max_binding=args.max_binding,
            max_elution=args.max_elution,
        )
        primary_kinetics_records, primary_stability_records, primary_processing_records = load_iedb_additional_records(
            binding_file=binding_file,
            max_kinetics=args.max_kinetics,
            max_stability=args.max_stability,
            max_processing=args.max_processing,
        )
        primary_tcell_records = load_iedb_tcell_records(
            tcell_file=tcell_file,
            max_tcell=args.max_tcell,
        )
        binding_record_groups = [primary_binding_records]
        elution_record_groups = [primary_elution_records]
        kinetics_record_groups = [primary_kinetics_records]
        stability_record_groups = [primary_stability_records]
        processing_record_groups = [primary_processing_records]
        tcell_record_groups = [primary_tcell_records]

        if cedar_binding_file is not None:
            cedar_binding_records, cedar_elution_records = load_iedb_binding_and_elution_records(
                binding_file=cedar_binding_file,
                max_binding=args.max_binding,
                max_elution=args.max_elution,
            )
            cedar_kinetics_records, cedar_stability_records, cedar_processing_records = load_iedb_additional_records(
                binding_file=cedar_binding_file,
                max_kinetics=args.max_kinetics,
                max_stability=args.max_stability,
                max_processing=args.max_processing,
            )
            binding_record_groups.append(cedar_binding_records)
            elution_record_groups.append(cedar_elution_records)
            kinetics_record_groups.append(cedar_kinetics_records)
            stability_record_groups.append(cedar_stability_records)
            processing_record_groups.append(cedar_processing_records)
        if cedar_tcell_file is not None:
            cedar_tcell_records = load_iedb_tcell_records(
                tcell_file=cedar_tcell_file,
                max_tcell=args.max_tcell,
            )
            tcell_record_groups.append(cedar_tcell_records)

        binding_records = _merge_records_with_limit(
            binding_record_groups,
            max_records=args.max_binding,
            seed=args.seed + 11,
        )
        elution_records = _merge_records_with_limit(
            elution_record_groups,
            max_records=args.max_elution,
            seed=args.seed + 12,
        )
        kinetics_records = _merge_records_with_limit(
            kinetics_record_groups,
            max_records=args.max_kinetics,
            seed=args.seed + 13,
        )
        stability_records = _merge_records_with_limit(
            stability_record_groups,
            max_records=args.max_stability,
            seed=args.seed + 14,
        )
        processing_records = _merge_records_with_limit(
            processing_record_groups,
            max_records=args.max_processing,
            seed=args.seed + 15,
        )
        tcell_records = _merge_records_with_limit(
            tcell_record_groups,
            max_records=args.max_tcell,
            seed=args.seed + 16,
        )
        if vdjdb_file is not None:
            vdjdb_records = load_vdjdb_records(
                vdjdb_file=vdjdb_file,
                max_vdjdb=args.max_vdjdb,
            )
        if sc10x_file is not None:
            sc10x_records = load_10x_records(
                sc10x_file=sc10x_file,
                max_10x=args.max_10x,
            )
    if getattr(args, "profile", "full") == "canary":
        kinetics_records, stability_records, processing_records, canary_stats = (
            bootstrap_missing_modalities_for_canary(
                binding_records=binding_records,
                kinetics_records=kinetics_records,
                stability_records=stability_records,
                processing_records=processing_records,
                seed=args.seed,
            )
        )
        if any(canary_stats.values()):
            print(
                "Canary bootstrap records added: "
                f"kinetics={canary_stats['kinetics']}, "
                f"stability={canary_stats['stability']}, "
                f"processing={canary_stats['processing']}"
            )
    print(f"  Binding: {len(binding_records)}")
    print(f"  Kinetics: {len(kinetics_records)}")
    print(f"  Stability: {len(stability_records)}")
    print(f"  Processing: {len(processing_records)}")
    print(f"  Elution: {len(elution_records)}")
    print(f"  T-cell: {len(tcell_records)}")
    print(f"  VDJdb: {len(vdjdb_records)}")
    print(f"  10x VDJ: {len(sc10x_records)}")
    if (
        not binding_records
        and not kinetics_records
        and not stability_records
        and not processing_records
        and not elution_records
        and not tcell_records
        and not vdjdb_records
        and not sc10x_records
    ):
        raise RuntimeError("No usable unified training records were loaded.")

    strict_mhc_resolution = bool(getattr(args, "strict_mhc_resolution", True))
    filter_unresolved_mhc = bool(getattr(args, "filter_unresolved_mhc", False))

    mhc_sequences: Dict[str, str] = {}
    if args.index_csv:
        unique_alleles = _collect_unique_alleles(
            binding_records,
            kinetics_records,
            stability_records,
            processing_records,
            elution_records,
            tcell_records,
            vdjdb_records,
        )
        mhc_sequences, stats = resolve_mhc_sequences_from_index(args.index_csv, unique_alleles)
        print(
            "Resolved MHC sequences from index: "
            f"{stats['resolved']}/{stats['total']} alleles"
        )
        mhc_quality = audit_loaded_mhc_sequence_quality(mhc_sequences)
        print(
            "Loaded MHC sequence quality: "
            f"total={mhc_quality['total_sequences']}, "
            f"with_X={mhc_quality['x_sequence_count']} "
            f"(residues={mhc_quality['x_residue_total']})"
        )
        # Build the COMPLETE set of invalid alleles by scanning all sequences,
        # not just the capped examples list from the quality audit.
        invalid_alleles: Dict[str, str] = {}
        for allele, raw_seq in list(mhc_sequences.items()):
            seq = str(raw_seq or "").strip().upper()
            if not seq:
                continue
            bad = sorted({ch for ch in seq if ch not in MHC_SEQUENCE_ALLOWED_AA})
            if bad:
                invalid_alleles[str(allele)] = f"noncanonical={''.join(bad)}"
            elif len(seq) < MIN_MHC_SEQUENCE_LEN:
                invalid_alleles[str(allele)] = f"short_len={len(seq)}"

        if invalid_alleles:
            example_text = ", ".join(
                f"{allele}:{reason}"
                for allele, reason in list(invalid_alleles.items())[:8]
            )
            if filter_unresolved_mhc:
                for allele in invalid_alleles:
                    mhc_sequences.pop(allele, None)
                print(
                    "Dropped invalid MHC sequences from index prior to training: "
                    f"count={len(invalid_alleles)}, examples={example_text}"
                )
            else:
                raise ValueError(
                    "Invalid loaded MHC sequences found (non-canonical residues "
                    "and/or shorter than the minimum accepted groove-bearing "
                    f"fragment ({MIN_MHC_SEQUENCE_LEN} aa)). "
                    f"examples={example_text}"
                )

    coverage_audit = _audit_mhc_sequence_coverage(
        binding_records=binding_records,
        kinetics_records=kinetics_records,
        stability_records=stability_records,
        processing_records=processing_records,
        elution_records=elution_records,
        tcell_records=tcell_records,
        vdjdb_records=vdjdb_records,
        mhc_sequences=mhc_sequences,
    )
    _print_mhc_sequence_coverage_summary(coverage_audit)
    coverage_reports = _write_mhc_sequence_coverage_report(
        run_dir=run_dir,
        coverage=coverage_audit,
    )
    coverage_json = coverage_reports.get("json")
    coverage_csv = coverage_reports.get("csv")
    if isinstance(coverage_json, Path) or isinstance(coverage_csv, Path):
        refs: List[str] = []
        if isinstance(coverage_json, Path):
            refs.append(f"json={coverage_json}")
        if isinstance(coverage_csv, Path):
            refs.append(f"csv={coverage_csv}")
        print("MHC coverage report written: " + ", ".join(refs))

    unresolved_audit = _audit_unresolved_mhc_resolution(
        binding_records=binding_records,
        kinetics_records=kinetics_records,
        stability_records=stability_records,
        processing_records=processing_records,
        elution_records=elution_records,
        tcell_records=tcell_records,
        vdjdb_records=vdjdb_records,
        mhc_sequences=mhc_sequences,
    )
    unresolved_total = int(unresolved_audit.get("total_unresolved", 0))
    if unresolved_total > 0:
        by_modality = unresolved_audit.get("by_modality", Counter())
        by_source = unresolved_audit.get("by_source", Counter())
        by_category = unresolved_audit.get("by_category", Counter())
        if isinstance(by_modality, Counter) and by_modality:
            by_modality_text = ", ".join(
                f"{modality}={count}" for modality, count in by_modality.most_common()
            )
            print(f"Unresolved MHC by modality: {by_modality_text}")
        if isinstance(by_source, Counter) and by_source:
            by_source_text = ", ".join(
                f"{source}={count}" for source, count in by_source.most_common(10)
            )
            print(f"Unresolved MHC by source (top 10): {by_source_text}")
        if isinstance(by_category, Counter) and by_category:
            by_category_text = ", ".join(
                f"{category}={count}" for category, count in by_category.most_common(10)
            )
            print(f"Unresolved MHC by category (top 10): {by_category_text}")

    if unresolved_total > 0 and filter_unresolved_mhc:
        print("Filtering unresolved MHC rows before training (resolved-only mode).")
        (
            binding_records,
            kinetics_records,
            stability_records,
            processing_records,
            elution_records,
            tcell_records,
            vdjdb_records,
            filter_stats,
        ) = _filter_records_to_resolved_mhc(
            binding_records=binding_records,
            kinetics_records=kinetics_records,
            stability_records=stability_records,
            processing_records=processing_records,
            elution_records=elution_records,
            tcell_records=tcell_records,
            vdjdb_records=vdjdb_records,
            mhc_sequences=mhc_sequences,
        )
        print(
            "Resolved-only drop stats: "
            f"binding={filter_stats['binding_dropped']}, "
            f"kinetics={filter_stats['kinetics_dropped']}, "
            f"stability={filter_stats['stability_dropped']}, "
            f"processing={filter_stats['processing_dropped']}, "
            f"elution_rows={filter_stats['elution_rows_dropped']}, "
            f"elution_alleles={filter_stats['elution_alleles_dropped']}, "
            f"tcell={filter_stats['tcell_dropped']}, "
            f"vdjdb={filter_stats['vdjdb_dropped']}"
        )
        coverage_post = _audit_mhc_sequence_coverage(
            binding_records=binding_records,
            kinetics_records=kinetics_records,
            stability_records=stability_records,
            processing_records=processing_records,
            elution_records=elution_records,
            tcell_records=tcell_records,
            vdjdb_records=vdjdb_records,
            mhc_sequences=mhc_sequences,
        )
        print("Post-filter coverage:")
        _print_mhc_sequence_coverage_summary(coverage_post)
        post_reports = _write_mhc_sequence_coverage_report(
            run_dir=run_dir,
            coverage=coverage_post,
            basename="mhc_sequence_coverage_post_filter",
        )
        post_json = post_reports.get("json")
        post_csv = post_reports.get("csv")
        if isinstance(post_json, Path) or isinstance(post_csv, Path):
            refs: List[str] = []
            if isinstance(post_json, Path):
                refs.append(f"json={post_json}")
            if isinstance(post_csv, Path):
                refs.append(f"csv={post_csv}")
            print("Post-filter MHC coverage report written: " + ", ".join(refs))

        unresolved_audit = _audit_unresolved_mhc_resolution(
            binding_records=binding_records,
            kinetics_records=kinetics_records,
            stability_records=stability_records,
            processing_records=processing_records,
            elution_records=elution_records,
            tcell_records=tcell_records,
            vdjdb_records=vdjdb_records,
            mhc_sequences=mhc_sequences,
        )
        unresolved_total = int(unresolved_audit.get("total_unresolved", 0))

    if unresolved_total > 0:
        reports = _write_unresolved_mhc_report(run_dir=run_dir, audit=unresolved_audit)
        by_allele = unresolved_audit.get("by_allele", Counter())
        if strict_mhc_resolution:
            top_alleles = ""
            if isinstance(by_allele, Counter) and by_allele:
                top_alleles = ", ".join(
                    f"{allele} ({count})" for allele, count in by_allele.most_common(10)
                )
            report_refs: List[str] = []
            allele_report = reports.get("alleles")
            if isinstance(allele_report, Path):
                report_refs.append(f"alleles={allele_report}")
            detail_report = reports.get("detail")
            if isinstance(detail_report, Path):
                report_refs.append(f"detail={detail_report}")
            report_msg = f" Reports: {', '.join(report_refs)}." if report_refs else ""
            top_msg = f" Top unresolved alleles: {top_alleles}." if top_alleles else ""
            raise RuntimeError(
                "Unresolved MHC alleles are present in training data "
                f"({unresolved_total} unresolved rows)."
                + top_msg
                + report_msg
            )
        print(
            "WARNING: unresolved MHC alleles found in training data and strict mode is disabled."
        )

    synthetic_pmhc_ratio = max(float(args.synthetic_pmhc_negative_ratio), 0.0)
    synthetic_elution_ratio = synthetic_pmhc_ratio * SYNTHETIC_ELUTION_NEGATIVE_SCALE
    synthetic_cascade_elution_ratio = (
        synthetic_pmhc_ratio * SYNTHETIC_CASCADE_ELUTION_NEGATIVE_SCALE
    )
    synthetic_cascade_tcell_ratio = (
        synthetic_pmhc_ratio * SYNTHETIC_CASCADE_TCELL_NEGATIVE_SCALE
    )
    print(
        "Synthetic negative config: "
        f"pmhc={synthetic_pmhc_ratio:.3f}, "
        f"processing={max(float(args.synthetic_processing_negative_ratio), 0.0):.3f}, "
        f"derived_elution={synthetic_elution_ratio:.3f}, "
        f"derived_cascade_elution={synthetic_cascade_elution_ratio:.3f}, "
        f"derived_cascade_tcell={synthetic_cascade_tcell_ratio:.3f}"
    )

    if (
        synthetic_pmhc_ratio > 0
        or args.synthetic_class_i_no_mhc_beta_negative_ratio > 0
    ):
        binding_records, binding_aug_stats = augment_binding_records_with_synthetic_negatives(
            binding_records=binding_records,
            mhc_sequences=mhc_sequences,
            negative_ratio=synthetic_pmhc_ratio,
            weak_value_min_nM=args.synthetic_negative_min_nM,
            weak_value_max_nM=args.synthetic_negative_max_nM,
            seed=args.seed,
            class_i_no_mhc_beta_ratio=args.synthetic_class_i_no_mhc_beta_negative_ratio,
        )
        print(
            "Added synthetic binding negatives: "
            f"{binding_aug_stats['added']} "
            f"(peptide_scramble={binding_aug_stats['peptide_scramble']}, "
            f"peptide_random={binding_aug_stats['peptide_random']}, "
            f"mhc_scramble={binding_aug_stats['mhc_scramble']}, "
            f"mhc_random={binding_aug_stats['mhc_random']}, "
            f"no_mhc_alpha={binding_aug_stats['no_mhc_alpha']}, "
            f"no_mhc_beta={binding_aug_stats['no_mhc_beta']})"
        )

    if synthetic_elution_ratio > 0:
        elution_records, elution_aug_stats = augment_elution_records_with_synthetic_negatives(
            elution_records=elution_records,
            negative_ratio=synthetic_elution_ratio,
            seed=args.seed,
        )
        print(f"Added synthetic elution negatives: {elution_aug_stats['added']}")

    if args.synthetic_processing_negative_ratio > 0:
        processing_records, processing_aug_stats = augment_processing_records_with_synthetic_negatives(
            processing_records=processing_records,
            negative_ratio=args.synthetic_processing_negative_ratio,
            seed=args.seed,
        )
        print(f"Added synthetic processing negatives: {processing_aug_stats['added']}")

    if synthetic_cascade_elution_ratio > 0 or synthetic_cascade_tcell_ratio > 0:
        elution_records, tcell_records, cascade_stats = cascade_binding_negatives_to_downstream(
            binding_records=binding_records,
            elution_records=elution_records,
            tcell_records=tcell_records,
            elution_ratio=synthetic_cascade_elution_ratio,
            tcell_ratio=synthetic_cascade_tcell_ratio,
            seed=args.seed,
        )
        print(
            "Cascaded binding negatives downstream: "
            f"elution={cascade_stats['elution_added']}, "
            f"tcell={cascade_stats['tcell_added']}"
        )

    dataset = PrestoDataset(
        binding_records=binding_records,
        kinetics_records=kinetics_records,
        stability_records=stability_records,
        processing_records=processing_records,
        elution_records=elution_records,
        tcell_records=tcell_records,
        vdjdb_records=vdjdb_records,
        sc10x_records=sc10x_records,
        mhc_sequences=mhc_sequences,
        strict_mhc_resolution=strict_mhc_resolution,
    )
    mhc_augmentation_samples = int(getattr(args, "mhc_augmentation_samples", 0))
    mhc_augmentation_max_fraction = float(
        getattr(args, "mhc_augmentation_max_fraction", 0.05)
    )
    effective_mhc_augmentation_samples = _effective_mhc_augmentation_sample_limit(
        requested_samples=mhc_augmentation_samples,
        current_dataset_size=len(dataset),
        max_fraction=mhc_augmentation_max_fraction,
    )
    if (
        mhc_augmentation_samples > 0
        and effective_mhc_augmentation_samples < mhc_augmentation_samples
    ):
        print(
            "MHC augmentation: capped fixed sample request to preserve label-dense batches "
            f"(requested={mhc_augmentation_samples}, "
            f"cap_fraction={mhc_augmentation_max_fraction:.3f}, "
            f"effective={effective_mhc_augmentation_samples}, "
            f"base_dataset={len(dataset)})"
        )
    if effective_mhc_augmentation_samples > 0 and args.index_csv:
        mhc_only = _generate_mhc_only_samples(
            index_csv=args.index_csv,
            max_samples=effective_mhc_augmentation_samples,
            seed=args.seed,
        )
        if mhc_only:
            dataset.samples.extend(mhc_only)
            print(f"MHC augmentation: added {len(mhc_only)} MHC-only samples")

    # UniProt negative samples (species_of_origin + foreignness supervision)
    uniprot_ratio = float(getattr(args, "uniprot_negative_ratio", 0.1))
    max_uniprot = int(getattr(args, "max_uniprot", 0))
    if uniprot_ratio > 0:
        uniprot_tsv = data_dir / "uniprot" / "proteins.tsv"
        if uniprot_tsv.exists():
            proteins = load_uniprot_proteins(uniprot_tsv)
            n_uniprot = int(len(dataset) * uniprot_ratio)
            if max_uniprot > 0:
                n_uniprot = min(n_uniprot, max_uniprot)
            if proteins and n_uniprot > 0:
                uniprot_samples = generate_uniprot_samples(
                    proteins, mhc_sequences, n_uniprot, seed=args.seed,
                )
                dataset.samples.extend(uniprot_samples)
                print(f"UniProt negatives: added {len(uniprot_samples)} samples")
        else:
            print(f"UniProt proteins.tsv not found at {uniprot_tsv}; skipping UniProt negatives")

    print(f"Total samples: {len(dataset)}")
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 samples for train/val split.")

    val_size = max(1, int(len(dataset) * float(args.val_frac)))
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size
    split_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_gen,
    )

    collator = PrestoCollator()
    use_pin_memory = bool(getattr(args, "pin_memory", True)) and str(device).startswith("cuda")
    use_non_blocking_transfer = bool(use_pin_memory and str(device).startswith("cuda"))
    num_workers = max(0, int(getattr(args, "num_workers", 0)))
    perf_log_interval_batches = max(
        0,
        int(getattr(args, "perf_log_interval_batches", 0)),
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        collator=collator,
        balanced=bool(getattr(args, "balanced_batches", True)),
        seed=args.seed,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        collator=collator,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Balanced train batches: {bool(getattr(args, 'balanced_batches', True))}")
    print(
        "DataLoader config: "
        f"num_workers={num_workers}, "
        f"pin_memory={use_pin_memory}, "
        f"non_blocking_transfer={use_non_blocking_transfer}"
    )

    model = Presto(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Performance: AMP (bf16) and torch.compile
    use_amp = bool(getattr(args, "use_amp", True)) and str(device).startswith("cuda")
    use_compile = bool(getattr(args, "use_compile", True)) and str(device).startswith("cuda")
    max_mil_instances = int(getattr(args, "max_mil_instances", 128))
    if use_amp:
        print("AMP enabled: bf16 autocast on CUDA")
    if use_compile:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled (mode=reduce-overhead)")
    if max_mil_instances > 0:
        print(f"MIL instance cap: {max_mil_instances} per batch")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    uncertainty_weighting = None
    if args.use_uncertainty_weighting:
        uncertainty_weighting = UncertaintyWeighting(n_tasks=len(IEDB_LOSS_TASK_NAMES)).to(device)
        optimizer.add_param_group({"params": uncertainty_weighting.parameters()})
    pcgrad = PCGrad(optimizer) if args.use_pcgrad else None
    regularization_cfg = _regularization_config_from_args(args)
    track_probe_affinity = bool(getattr(args, "track_probe_affinity", True))
    track_probe_motif_scan = bool(getattr(args, "track_probe_motif_scan", True))
    motif_scan_amino_acids = str(getattr(args, "motif_scan_amino_acids", AMINO_ACIDS) or AMINO_ACIDS)
    motif_scan_positions_text = str(getattr(args, "motif_scan_positions", "2,9") or "2,9")
    track_pmhc_flow = bool(getattr(args, "track_pmhc_flow", True))
    pmhc_flow_batches = max(0, int(getattr(args, "pmhc_flow_batches", 2)))
    pmhc_flow_max_samples = max(0, int(getattr(args, "pmhc_flow_max_samples", 512)))
    track_output_latent_stats = bool(getattr(args, "track_output_latent_stats", True))
    output_latent_stats_batches = max(
        0,
        int(getattr(args, "output_latent_stats_batches", 2)),
    )
    output_latent_stats_max_samples = max(
        0,
        int(getattr(args, "output_latent_stats_max_samples", 512)),
    )
    probe_specs: List[Dict[str, Any]] = []
    probe_history: List[Dict[str, Any]] = []
    motif_history: List[Dict[str, Any]] = []
    if track_probe_affinity:
        probe_alleles = _split_allele_list(str(getattr(args, "probe_alleles", "")))
        if probe_alleles:
            probe_specs = _resolve_probe_specs(
                probe_peptide=str(getattr(args, "probe_peptide", "SLLQHLIGL")),
                probe_alleles=probe_alleles,
                mhc_sequences=mhc_sequences,
                index_csv=getattr(args, "index_csv", None),
                device=device,
            )
            if probe_specs:
                print(
                    "Probe tracking enabled for "
                    + ", ".join(f"{spec['peptide']}|{spec['allele']}" for spec in probe_specs)
                )
            else:
                print("Probe tracking disabled: no resolvable probe allele sequences found.")
        else:
            print("Probe tracking disabled: no probe alleles configured.")
    if track_pmhc_flow and pmhc_flow_batches > 0 and pmhc_flow_max_samples > 1:
        print(
            "pMHC flow diagnostics enabled: "
            f"batches={pmhc_flow_batches}, max_samples={pmhc_flow_max_samples}"
        )
    else:
        print("pMHC flow diagnostics disabled.")
    if (
        track_output_latent_stats
        and output_latent_stats_batches > 0
        and output_latent_stats_max_samples > 0
    ):
        print(
            "Output/latent diagnostics enabled: "
            f"batches={output_latent_stats_batches}, "
            f"max_samples={output_latent_stats_max_samples}"
        )
    else:
        print("Output/latent diagnostics disabled.")

    print("\nStarting training...")
    best_val_loss = float("inf")
    cuda_stats_device: Optional[torch.device] = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        cuda_stats_device = torch.device(device)

    try:
        for epoch in range(args.epochs):
            if cuda_stats_device is not None:
                torch.cuda.reset_peak_memory_stats(cuda_stats_device)
            epoch_regularization_cfg = _regularization_for_epoch(
                base_regularization=regularization_cfg,
                epoch_idx=epoch,
                total_epochs=args.epochs,
            )
            max_batches_per_epoch = int(getattr(args, "max_batches", 0))
            max_val_batches_per_epoch = int(getattr(args, "max_val_batches", 0))
            train_loss, train_task_losses = _call_train_epoch_compat(
                model,
                train_loader,
                optimizer,
                device,
                uncertainty_weighting=uncertainty_weighting,
                pcgrad=pcgrad,
                regularization_config=epoch_regularization_cfg,
                supervised_loss_aggregation=str(
                    getattr(args, "supervised_loss_aggregation", "sample_weighted")
                ),
                profile_performance=bool(getattr(args, "profile_performance", True)),
                non_blocking_transfer=use_non_blocking_transfer,
                perf_log_interval_batches=perf_log_interval_batches,
                use_amp=use_amp,
                max_mil_instances=max_mil_instances,
                max_batches=max_batches_per_epoch,
            )
            val_loss, val_task_losses = _call_evaluate_compat(
                model,
                val_loader,
                device,
                regularization_config=epoch_regularization_cfg,
                supervised_loss_aggregation=str(
                    getattr(args, "supervised_loss_aggregation", "sample_weighted")
                ),
                use_amp=use_amp,
                max_mil_instances=max_mil_instances,
                max_val_batches=max_val_batches_per_epoch,
            )
            probe_metrics: Dict[str, float] = {}
            probe_rows: List[Dict[str, Any]] = []
            if probe_specs:
                probe_metrics, probe_rows = _evaluate_probe_affinity(model, probe_specs)
                for row in probe_rows:
                    probe_history.append({"epoch": epoch + 1, **row})
            motif_metrics: Dict[str, float] = {}
            motif_rows: List[Dict[str, Any]] = []
            if probe_specs and track_probe_motif_scan:
                motif_pos = _parse_motif_scan_positions(
                    motif_scan_positions_text,
                    peptide_len=len(str(getattr(args, "probe_peptide", "SLLQHLIGL")).strip()),
                )
                if motif_pos:
                    motif_metrics, motif_rows = _evaluate_probe_motif_scan(
                        model,
                        probe_specs,
                        positions_1based=motif_pos,
                        amino_acids=motif_scan_amino_acids,
                    )
                    for row in motif_rows:
                        motif_history.append({"epoch": epoch + 1, **row})
            if probe_rows:
                probe_metrics.update(_compute_discrimination_metrics(probe_rows))
            if motif_rows:
                motif_metrics.update(_compute_motif_specificity(motif_rows))
            pmhc_flow_metrics: Dict[str, float] = {}
            if track_pmhc_flow and pmhc_flow_batches > 0 and pmhc_flow_max_samples > 1:
                pmhc_flow_metrics = _evaluate_pmhc_information_flow(
                    model,
                    val_loader,
                    device,
                    n_batches=pmhc_flow_batches,
                    max_samples=pmhc_flow_max_samples,
                    non_blocking_transfer=use_non_blocking_transfer,
                )
            output_latent_metrics: Dict[str, float] = {}
            if (
                track_output_latent_stats
                and output_latent_stats_batches > 0
                and output_latent_stats_max_samples > 0
            ):
                output_latent_metrics = _evaluate_output_and_latent_statistics(
                    model,
                    val_loader,
                    device,
                    n_batches=output_latent_stats_batches,
                    max_samples=output_latent_stats_max_samples,
                    non_blocking_transfer=use_non_blocking_transfer,
                )
            uw_metrics = summarize_uncertainty_weights(uncertainty_weighting)
            gpu_metrics: Dict[str, float] = {}
            if cuda_stats_device is not None:
                peak_allocated = float(torch.cuda.max_memory_allocated(cuda_stats_device))
                peak_reserved = float(torch.cuda.max_memory_reserved(cuda_stats_device))
                rows = max(int(args.batch_size), 1)
                gpu_metrics = {
                    "gpu_peak_allocated_bytes": peak_allocated,
                    "gpu_peak_reserved_bytes": peak_reserved,
                    "gpu_peak_allocated_gib": peak_allocated / float(1024**3),
                    "gpu_peak_reserved_gib": peak_reserved / float(1024**3),
                    "gpu_peak_allocated_bytes_per_batch_row": peak_allocated / float(rows),
                }

            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )
            perf_wait = float(train_task_losses.get("perf_data_wait_sec_per_batch", 0.0))
            perf_compute = float(train_task_losses.get("perf_compute_loss_sec_per_batch", 0.0))
            perf_backward = float(train_task_losses.get("perf_backward_sec_per_batch", 0.0))
            perf_optim = float(train_task_losses.get("perf_optimizer_sec_per_batch", 0.0))
            if perf_wait > 0.0 or perf_compute > 0.0 or perf_backward > 0.0:
                print(
                    "  Perf (sec/batch): "
                    f"wait={perf_wait:.3f}, "
                    f"compute={perf_compute:.3f}, "
                    f"backward={perf_backward:.3f}, "
                    f"optim={perf_optim:.3f}"
                )
                print(
                    "  Perf (% epoch): "
                    f"wait={float(train_task_losses.get('perf_data_wait_pct_epoch', 0.0)):.1f}%, "
                    f"compute={float(train_task_losses.get('perf_compute_loss_pct_epoch', 0.0)):.1f}%, "
                    f"backward={float(train_task_losses.get('perf_backward_pct_epoch', 0.0)):.1f}%, "
                    f"optim={float(train_task_losses.get('perf_optimizer_pct_epoch', 0.0)):.1f}%"
                )
                if perf_wait > max(perf_compute + perf_backward + perf_optim, 0.0):
                    print("  Bottleneck hint: data wait dominates batch time.")
                elif perf_compute > max(perf_wait + perf_backward + perf_optim, 0.0):
                    print("  Bottleneck hint: forward/loss compute dominates batch time.")
            if gpu_metrics:
                print(
                    "  GPU peak: "
                    f"allocated={gpu_metrics['gpu_peak_allocated_gib']:.2f} GiB, "
                    f"reserved={gpu_metrics['gpu_peak_reserved_gib']:.2f} GiB, "
                    f"alloc/row≈{gpu_metrics['gpu_peak_allocated_bytes_per_batch_row'] / (1024**2):.2f} MiB"
                )
            if probe_rows:
                probe_text = ", ".join(
                    f"{row['allele']}: KD≈{row['kd_nM']:.1f} nM (bind={row['binding_prob']:.6f})"
                    for row in probe_rows
                )
                print(f"  Probe affinity: {probe_text}")
            if motif_rows:
                # Print compact per-allele per-position motif summary.
                summary_chunks: List[str] = []
                grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
                for row in motif_rows:
                    grouped[(str(row["allele"]), int(row["position_1based"]))].append(row)
                for (allele, pos), rows in sorted(grouped.items()):
                    ranked = sorted(rows, key=lambda r: float(r["binding_prob"]), reverse=True)
                    top = ranked[0]
                    wt = next((r for r in ranked if str(r["sub_aa"]) == str(r["wt_aa"])), None)
                    wt_rank = (
                        1 + next((i for i, r in enumerate(ranked) if str(r["sub_aa"]) == str(r["wt_aa"])), len(ranked))
                    )
                    if wt is not None:
                        summary_chunks.append(
                            f"{allele} P{pos} top={top['sub_aa']}({float(top['binding_prob']):.3f}) "
                            f"wt={wt['wt_aa']} rank={wt_rank}"
                        )
                if summary_chunks:
                    print("  Probe motif scan: " + "; ".join(summary_chunks))
            if pmhc_flow_metrics:
                mhc_norm = float(
                    pmhc_flow_metrics.get("pmhc_flow_binding_logit_delta_mhc_norm", 0.0)
                )
                pep_norm = float(
                    pmhc_flow_metrics.get("pmhc_flow_binding_logit_delta_peptide_norm", 0.0)
                )
                int_norm = float(
                    pmhc_flow_metrics.get("pmhc_flow_binding_logit_interaction_norm", 0.0)
                )
                status_code = int(pmhc_flow_metrics.get("pmhc_flow_status_code", -1.0))
                status_text = {
                    0: "near-invariant binding head",
                    1: "peptide-dominant, weak MHC usage",
                    2: "inputs used, weak pairwise interaction",
                    3: "joint peptide-MHC interaction present",
                }.get(status_code, "unknown")
                print(
                    "  pMHC flow (binding logit norm): "
                    f"mhc={mhc_norm:.4f}, peptide={pep_norm:.4f}, interaction={int_norm:.4f} "
                    f"[{status_text}]"
                )
            if output_latent_metrics:
                pmhc_var = float(
                    output_latent_metrics.get("diag_pmhc_vec_feature_var_mean", 0.0)
                )
                bind_var = float(output_latent_metrics.get("diag_binding_logit_var", 0.0))
                pres_var = float(
                    output_latent_metrics.get("diag_presentation_logit_var", 0.0)
                )
                print(
                    "  Output/latent diagnostics: "
                    f"pmhc_vec_var={pmhc_var:.4f}, "
                    f"binding_logit_var={bind_var:.4f}, "
                    f"presentation_logit_var={pres_var:.4f}"
                )
            if run_logger is not None:
                run_logger.log(
                    epoch + 1,
                    "train",
                    {
                        "loss": train_loss,
                        **train_task_losses,
                        **uw_metrics,
                        "schedule_consistency_factor": epoch_regularization_cfg.get(
                            "schedule_consistency_factor",
                            1.0,
                        ),
                        "schedule_tcell_factor": epoch_regularization_cfg.get(
                            "schedule_tcell_factor",
                            1.0,
                        ),
                        **gpu_metrics,
                    },
                )
                run_logger.log(epoch + 1, "val", {"loss": val_loss, **val_task_losses})
                if probe_metrics:
                    run_logger.log(epoch + 1, "probe", probe_metrics)
                if motif_metrics:
                    run_logger.log(epoch + 1, "probe_motif", motif_metrics)
                if pmhc_flow_metrics:
                    run_logger.log(epoch + 1, "pmhc_flow", pmhc_flow_metrics)
                if output_latent_metrics:
                    run_logger.log(epoch + 1, "output_latent", output_latent_metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.checkpoint:
                    # Unwrap compiled model for checkpoint config extraction
                    raw_model = getattr(model, "_orig_mod", model)
                    save_model_checkpoint(
                        args.checkpoint,
                        model=raw_model,
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
        if run_dir is not None and probe_history:
            probe_plot = _write_probe_artifacts(
                run_dir=run_dir,
                probe_history=probe_history,
                plot_file=str(getattr(args, "probe_plot_file", "probe_affinity_over_epochs.png")),
            )
            if probe_plot is not None:
                print(f"Probe plot saved to {probe_plot}")
            probe_json = run_dir / "probe_affinity_over_epochs.json"
            probe_json.write_text(json.dumps(probe_history, indent=2), encoding="utf-8")
        if run_dir is not None and motif_history:
            motif_artifacts = _write_probe_motif_artifacts(run_dir=run_dir, motif_history=motif_history)
            motif_csv = motif_artifacts.get("csv")
            if motif_csv is not None:
                print(f"Probe motif scan saved to {motif_csv}")
        if run_logger is not None:
            run_logger.close()

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train Presto on unified multi-source immunology data"
    )
    parser.add_argument("--config", type=str, default=None, help="Optional JSON/YAML config file")
    parser.add_argument(
        "--profile",
        type=str,
        choices=["full", "canary", "diagnostic"],
        default="full",
        help=(
            "Training profile preset "
            "(canary: fast smoke run; diagnostic: richer coverage/flow/latent diagnostics)"
        ),
    )
    parser.add_argument("--data-dir", dest="data_dir", type=str, default="./data", help="Data directory with downloaded datasets")
    parser.add_argument(
        "--merged-tsv",
        type=str,
        default=None,
        help="Path to merged deduplicated TSV (default: <data-dir>/merged_deduped.tsv)",
    )
    parser.add_argument(
        "--require-merged-input",
        dest="require_merged_input",
        action="store_true",
        default=True,
        help="Require merged TSV input (default: true)",
    )
    parser.add_argument(
        "--allow-raw-fallback",
        dest="require_merged_input",
        action="store_false",
        help="Allow fallback to raw source exports when merged TSV is unavailable",
    )
    parser.add_argument("--binding-file", type=str, default=None, help="Override path to IEDB MHC ligand export")
    parser.add_argument("--tcell-file", type=str, default=None, help="Override path to IEDB T-cell export")
    parser.add_argument("--cedar-binding-file", type=str, default=None, help="Optional path to CEDAR MHC ligand export")
    parser.add_argument("--cedar-tcell-file", type=str, default=None, help="Optional path to CEDAR T-cell export")
    parser.add_argument("--vdjdb-file", type=str, default=None, help="Override path to VDJdb export")
    parser.add_argument(
        "--10x-file",
        dest="sc10x_file",
        type=str,
        default=None,
        help="Override path to 10x VDJ contig CSV/TSV",
    )
    parser.add_argument("--index-csv", type=str, default=None, help="Optional built MHC index CSV for allele->sequence resolution")
    parser.add_argument(
        "--strict-mhc-resolution",
        dest="strict_mhc_resolution",
        action="store_true",
        default=True,
        help="Require all non-ablation MHC alleles to resolve to amino-acid sequences (default: true)",
    )
    parser.add_argument(
        "--allow-unresolved-mhc",
        dest="strict_mhc_resolution",
        action="store_false",
        help="Allow unresolved MHC alleles (debug only; unresolved MHC chains become empty sequences)",
    )
    parser.add_argument(
        "--filter-unresolved-mhc",
        dest="filter_unresolved_mhc",
        action="store_true",
        default=False,
        help=(
            "Drop unresolved-MHC rows before dataset construction "
            "(resolved-only training subset)"
        ),
    )
    parser.add_argument(
        "--no-filter-unresolved-mhc",
        dest="filter_unresolved_mhc",
        action="store_false",
        help="Disable unresolved-MHC row filtering",
    )
    parser.add_argument("--max-binding", type=int, default=0, help="Max binding records to load (<=0 means no limit)")
    parser.add_argument("--max-kinetics", type=int, default=0, help="Max kinetics records to load (<=0 means no limit)")
    parser.add_argument("--max-stability", type=int, default=0, help="Max stability records to load (<=0 means no limit)")
    parser.add_argument("--max-processing", type=int, default=0, help="Max processing records to load (<=0 means no limit)")
    parser.add_argument("--max-elution", type=int, default=0, help="Max elution records to load (<=0 means no limit)")
    parser.add_argument("--max-tcell", type=int, default=0, help="Max T-cell records to load (<=0 means no limit)")
    parser.add_argument("--max-vdjdb", type=int, default=0, help="Max VDJdb records to load (<=0 means no limit)")
    parser.add_argument(
        "--cap-sampling",
        dest="cap_sampling",
        type=str,
        choices=["head", "reservoir"],
        default="reservoir",
        help=(
            "Sampling strategy when modality caps are set "
            "(head=first-N rows, reservoir=representative one-pass sample)"
        ),
    )
    parser.add_argument(
        "--max-10x",
        dest="max_10x",
        type=int,
        default=0,
        help="Max 10x VDJ records to load (<=0 means no limit)",
    )
    parser.add_argument(
        "--synthetic-pmhc-negative-ratio",
        type=float,
        default=1.0,
        help=(
            "Primary synthetic non-binding pMHC ratio per real binding sample "
            "(also drives downstream elution/T-cell synthetic negatives)"
        ),
    )
    parser.add_argument(
        "--synthetic-class-i-no-mhc-beta-negative-ratio",
        dest="synthetic_class_i_no_mhc_beta_negative_ratio",
        type=float,
        default=0.25,
        help=(
            "Additional class-I negatives without MHC beta chain (beta2m) "
            "per real class-I binding sample (0 disables)"
        ),
    )
    parser.add_argument(
        "--synthetic-processing-negative-ratio",
        type=float,
        default=0.5,
        help="Synthetic processing negatives to add per real processing sample (0 disables)",
    )
    parser.add_argument(
        "--synthetic-negative-min-nM",
        type=float,
        default=DEFAULT_MAX_AFFINITY_NM * 0.5,
        help="Minimum synthetic weak-affinity value (nM)",
    )
    parser.add_argument(
        "--synthetic-negative-max-nM",
        type=float,
        default=DEFAULT_MAX_AFFINITY_NM,
        help="Maximum synthetic weak-affinity value (nM)",
    )
    parser.add_argument(
        "--mhc-augmentation-samples",
        dest="mhc_augmentation_samples",
        type=int,
        default=60000,
        help=(
            "Requested count of MHC-only auxiliary samples drawn from the MHC index "
            "(capped by --mhc-augmentation-max-fraction)"
        ),
    )
    parser.add_argument(
        "--mhc-augmentation-max-fraction",
        dest="mhc_augmentation_max_fraction",
        type=float,
        default=0.05,
        help=(
            "Maximum fraction of the current non-MHC-only dataset that fixed-size "
            "MHC augmentation may add (0 disables the cap)"
        ),
    )
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (0 disables worker parallelism)",
    )
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="Enable pinned host memory in DataLoader (default: true)",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned host memory in DataLoader",
    )
    parser.add_argument(
        "--balanced-batches",
        dest="balanced_batches",
        action="store_true",
        default=True,
        help="Balance train mini-batches by assay/source/label/allele strata (default: true)",
    )
    parser.add_argument(
        "--no-balanced-batches",
        dest="balanced_batches",
        action="store_false",
        help="Disable balanced mini-batch sampling",
    )
    parser.add_argument("--lr", type=float, default=2.8e-4, help="Learning rate")
    # Performance: AMP, compile, MIL cap
    parser.add_argument(
        "--amp",
        dest="use_amp",
        action="store_true",
        default=True,
        help="Enable bf16 automatic mixed precision on CUDA (default: true)",
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable bf16 automatic mixed precision",
    )
    parser.add_argument(
        "--compile",
        dest="use_compile",
        action="store_true",
        default=False,
        help="Enable torch.compile for kernel fusion (default: false)",
    )
    parser.add_argument(
        "--no-compile",
        dest="use_compile",
        action="store_false",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--max-mil-instances",
        dest="max_mil_instances",
        type=int,
        default=128,
        help="Max MIL instances per batch (0=unlimited, default: 128)",
    )
    parser.add_argument(
        "--max-batches",
        dest="max_batches",
        type=int,
        default=0,
        help="Max training batches per epoch (0=unlimited, default: 0)",
    )
    parser.add_argument(
        "--max-val-batches",
        dest="max_val_batches",
        type=int,
        default=0,
        help="Max validation batches per epoch (0=unlimited, default: 0)",
    )
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--checkpoint", type=str, default=None, help="Save checkpoint path")
    parser.add_argument("--run-dir", dest="run_dir", type=str, default=None, help="Run artifact directory")
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
        default="sample_weighted",
        help=(
            "How to combine supervised task losses: "
            "task_mean (equal per task) or sample_weighted "
            "(weight by in-batch labeled sample count per task)"
        ),
    )
    parser.add_argument(
        "--profile-performance",
        dest="profile_performance",
        action="store_true",
        default=True,
        help="Record epoch performance breakdown (data wait/compute/backward/optimizer)",
    )
    parser.add_argument(
        "--no-profile-performance",
        dest="profile_performance",
        action="store_false",
        help="Disable epoch performance breakdown instrumentation",
    )
    parser.add_argument(
        "--perf-log-interval-batches",
        type=int,
        default=100,
        help=(
            "Emit rolling in-epoch perf breakdown every N train batches "
            "(0 disables rolling perf logs)"
        ),
    )
    parser.add_argument("--use-pcgrad", action="store_true", help="Use PCGrad for multi-task gradient conflicts")
    parser.add_argument(
        "--consistency-cascade-weight",
        type=float,
        default=0.2,
        help="Weight for anti-saturation cascade prior (high presentation with low parent)",
    )
    parser.add_argument(
        "--consistency-assay-affinity-weight",
        type=float,
        default=0.1,
        help="Weight for KD/IC50/EC50 closeness regularization",
    )
    parser.add_argument(
        "--consistency-assay-presentation-weight",
        type=float,
        default=0.1,
        help="Weight for elution/MS vs presentation consistency",
    )
    parser.add_argument(
        "--consistency-no-b2m-weight",
        type=float,
        default=0.5,
        help="Weight for invalid chain-assembly prior (class I/II single-chain cases)",
    )
    parser.add_argument(
        "--consistency-tcell-context-weight",
        type=float,
        default=0.05,
        help="Weight for in-vitro >= ex-vivo T-cell context prior",
    )
    parser.add_argument(
        "--consistency-tcell-upstream-weight",
        type=float,
        default=0.2,
        help="Weight for T-cell outputs requiring strong upstream binding/presentation",
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
        "--tcell-in-vitro-margin",
        type=float,
        default=0.1,
        help="Required tcell-immunogenicity logit margin for in-vitro contexts",
    )
    parser.add_argument(
        "--tcell-ex-vivo-margin",
        type=float,
        default=0.0,
        help="Maximum tcell-immunogenicity logit margin for ex-vivo contexts",
    )
    parser.add_argument(
        "--track-probe-affinity",
        dest="track_probe_affinity",
        action="store_true",
        default=True,
        help=(
            "Track fixed probe pMHC affinities every epoch and emit "
            "probe_affinity_over_epochs.{csv,json,png}"
        ),
    )
    parser.add_argument(
        "--no-track-probe-affinity",
        dest="track_probe_affinity",
        action="store_false",
        help="Disable fixed probe affinity tracking and plotting",
    )
    parser.add_argument(
        "--probe-peptide",
        type=str,
        default="SLLQHLIGL",
        help="Peptide sequence used for fixed per-epoch probe tracking",
    )
    parser.add_argument(
        "--probe-alleles",
        type=str,
        default="HLA-A*02:01,HLA-A*24:02",
        help="Comma-separated alleles for probe tracking",
    )
    parser.add_argument(
        "--probe-plot-file",
        type=str,
        default="probe_affinity_over_epochs.png",
        help="Filename (inside run-dir) for probe affinity plot",
    )
    parser.add_argument(
        "--track-probe-motif-scan",
        dest="track_probe_motif_scan",
        action="store_true",
        default=True,
        help=(
            "Track probe peptide single-residue substitution scans at selected positions "
            "and log motif-oriented metrics per epoch"
        ),
    )
    parser.add_argument(
        "--no-track-probe-motif-scan",
        dest="track_probe_motif_scan",
        action="store_false",
        help="Disable probe motif substitution-scan diagnostics",
    )
    parser.add_argument(
        "--motif-scan-positions",
        type=str,
        default="2,9",
        help="Comma-separated 1-based peptide positions for probe motif scanning",
    )
    parser.add_argument(
        "--motif-scan-amino-acids",
        type=str,
        default=AMINO_ACIDS,
        help="Amino-acid alphabet used for probe motif substitutions",
    )
    parser.add_argument(
        "--track-pmhc-flow",
        dest="track_pmhc_flow",
        action="store_true",
        default=True,
        help=(
            "Track peptide-MHC information flow with counterfactual shuffles "
            "(real vs MHC-shuffled vs peptide-shuffled vs both)"
        ),
    )
    parser.add_argument(
        "--no-track-pmhc-flow",
        dest="track_pmhc_flow",
        action="store_false",
        help="Disable peptide-MHC information-flow diagnostics",
    )
    parser.add_argument(
        "--pmhc-flow-batches",
        type=int,
        default=2,
        help="Validation batches per epoch to use for pMHC information-flow diagnostics",
    )
    parser.add_argument(
        "--pmhc-flow-max-samples",
        type=int,
        default=512,
        help="Max validation samples per epoch to evaluate for pMHC information-flow diagnostics",
    )
    parser.add_argument(
        "--track-output-latent-stats",
        dest="track_output_latent_stats",
        action="store_true",
        default=True,
        help=(
            "Track validation statistics for output heads and latent vectors "
            "(means/variances/norms)"
        ),
    )
    parser.add_argument(
        "--no-track-output-latent-stats",
        dest="track_output_latent_stats",
        action="store_false",
        help="Disable output/latent diagnostic tracking",
    )
    parser.add_argument(
        "--output-latent-stats-batches",
        type=int,
        default=2,
        help="Validation batches per epoch for output/latent diagnostics",
    )
    parser.add_argument(
        "--output-latent-stats-max-samples",
        type=int,
        default=512,
        help="Max validation samples per epoch for output/latent diagnostics",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
