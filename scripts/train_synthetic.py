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
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import random_split
from tqdm.auto import tqdm

from presto.models.presto import Presto
from presto.models.affinity import (
    DEFAULT_MAX_AFFINITY_NM,
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
from presto.training.losses import censor_aware_loss, mil_bag_loss, UncertaintyWeighting
from presto.training.checkpointing import save_model_checkpoint
from presto.training.config_io import (
    load_config_file,
    merge_namespace_with_config,
    pick_train_section,
)
from presto.training.losses import PCGrad
from presto.training.run_logger import RunLogger
from presto.data.allele_resolver import (
    PROCESSING_SPECIES_TO_IDX,
    infer_gene,
    normalize_mhc_class,
    normalize_processing_species_label,
)
from presto.data.mhc_index import infer_fine_chain_type
from presto.data.vocab import (
    MHC_CHAIN_FINE_TO_IDX,
    N_MHC_CHAIN_FINE_TYPES,
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
    "supervised_loss_aggregation": "sample_weighted",
    "use_pcgrad": False,
    "seed": 42,
    "consistency_cascade_weight": 0.0,
    "consistency_assay_affinity_weight": 0.0,
    "consistency_assay_presentation_weight": 0.0,
    "consistency_no_b2m_weight": 0.0,
    "consistency_tcell_context_weight": 0.0,
    "consistency_tcell_upstream_weight": 0.0,
    "consistency_prob_margin": 0.02,
    "consistency_parent_low_threshold": 0.1,
    "consistency_presentation_high_threshold": 0.9,
    "consistency_affinity_fold_tolerance": 2.0,
    "tcell_in_vitro_margin": 0.0,
    "tcell_ex_vivo_margin": 0.0,
    "mhc_attention_sparsity_weight": 0.1,
    "mhc_attention_sparsity_min_residues": 25.0,
    "mhc_attention_sparsity_max_residues": 45.0,
}


@dataclass(frozen=True)
class TaskLossSpec:
    """Specification for a supervised training loss."""

    name: str
    target_key: str
    mask_key: str
    pred_paths: Tuple[Tuple[str, ...], ...]
    loss_type: str  # one of: "censor", "bce", "mse"
    target_attr: Optional[str] = None
    mask_attr: Optional[str] = None
    qual_key: Optional[str] = None
    qual_attr: Optional[str] = None
    target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


LOSS_TASK_SPECS: Tuple[TaskLossSpec, ...] = (
    TaskLossSpec(
        name="binding",
        target_key="binding_unknown",
        mask_key="binding_unknown",
        pred_paths=(("assays", "KD_nM"),),
        loss_type="censor",
        target_attr="bind_target",
        mask_attr="bind_mask",
        qual_key="binding_unknown",
        qual_attr="bind_qual",
        target_transform=lambda t: normalize_binding_target_log10(
            t,
            max_affinity_nM=DEFAULT_MAX_AFFINITY_NM,
            assume_log10=False,
        ),
    ),
    TaskLossSpec(
        name="binding_kd",
        target_key="binding_kd",
        mask_key="binding_kd",
        pred_paths=(("assays", "KD_nM"),),
        loss_type="censor",
        qual_key="binding_kd",
        target_transform=lambda t: normalize_binding_target_log10(
            t,
            max_affinity_nM=DEFAULT_MAX_AFFINITY_NM,
            assume_log10=False,
        ),
    ),
    TaskLossSpec(
        name="binding_ic50",
        target_key="binding_ic50",
        mask_key="binding_ic50",
        pred_paths=(("assays", "IC50_nM"),),
        loss_type="censor",
        qual_key="binding_ic50",
        target_transform=lambda t: normalize_binding_target_log10(
            t,
            max_affinity_nM=DEFAULT_MAX_AFFINITY_NM,
            assume_log10=False,
        ),
    ),
    TaskLossSpec(
        name="binding_ec50",
        target_key="binding_ec50",
        mask_key="binding_ec50",
        pred_paths=(("assays", "EC50_nM"),),
        loss_type="censor",
        qual_key="binding_ec50",
        target_transform=lambda t: normalize_binding_target_log10(
            t,
            max_affinity_nM=DEFAULT_MAX_AFFINITY_NM,
            assume_log10=False,
        ),
    ),
    TaskLossSpec(
        name="elution",
        target_key="elution",
        mask_key="elution",
        pred_paths=(("elution_logit",),),
        loss_type="bce",
        target_attr="elution_label",
        mask_attr="elution_mask",
    ),
    TaskLossSpec(
        name="presentation",
        target_key="elution",
        mask_key="elution",
        pred_paths=(("presentation_logit",),),
        loss_type="bce",
        target_attr="elution_label",
        mask_attr="elution_mask",
    ),
    TaskLossSpec(
        name="tcell",
        target_key="tcell",
        mask_key="tcell",
        pred_paths=(("tcell_logit",), ("recognition_repertoire_logit",)),
        loss_type="bce",
        target_attr="tcell_label",
        mask_attr="tcell_mask",
    ),
    TaskLossSpec(
        name="immunogenicity",
        target_key="tcell",
        mask_key="tcell",
        pred_paths=(("immunogenicity_logit",),),
        loss_type="bce",
        target_attr="tcell_label",
        mask_attr="tcell_mask",
    ),
    TaskLossSpec(
        name="tcell_assay_method",
        target_key="tcell_assay_method",
        mask_key="tcell_assay_method",
        pred_paths=(
            ("tcell_panel_logits", "assay_method"),
            ("tcell_context_logits", "assay_method"),
        ),
        loss_type="ce",
    ),
    TaskLossSpec(
        name="tcell_assay_readout",
        target_key="tcell_assay_readout",
        mask_key="tcell_assay_readout",
        pred_paths=(
            ("tcell_panel_logits", "assay_readout"),
            ("tcell_context_logits", "assay_readout"),
        ),
        loss_type="ce",
    ),
    TaskLossSpec(
        name="tcell_apc_type",
        target_key="tcell_apc_type",
        mask_key="tcell_apc_type",
        pred_paths=(
            ("tcell_panel_logits", "apc_type"),
            ("tcell_context_logits", "apc_type"),
        ),
        loss_type="ce",
    ),
    TaskLossSpec(
        name="tcell_culture_context",
        target_key="tcell_culture_context",
        mask_key="tcell_culture_context",
        pred_paths=(
            ("tcell_panel_logits", "culture_context"),
            ("tcell_context_logits", "culture_context"),
        ),
        loss_type="ce",
    ),
    TaskLossSpec(
        name="tcell_stim_context",
        target_key="tcell_stim_context",
        mask_key="tcell_stim_context",
        pred_paths=(
            ("tcell_panel_logits", "stim_context"),
            ("tcell_context_logits", "stim_context"),
        ),
        loss_type="ce",
    ),
    TaskLossSpec(
        name="tcell_peptide_format",
        target_key="tcell_peptide_format",
        mask_key="tcell_peptide_format",
        pred_paths=(
            ("tcell_panel_logits", "peptide_format"),
            ("tcell_context_logits", "peptide_format"),
        ),
        loss_type="ce",
    ),
    TaskLossSpec(
        name="kon",
        target_key="kon",
        mask_key="kon",
        pred_paths=(("assays", "kon"),),
        loss_type="mse",
        target_attr="kon_target",
        mask_attr="kon_mask",
    ),
    TaskLossSpec(
        name="koff",
        target_key="koff",
        mask_key="koff",
        pred_paths=(("assays", "koff"),),
        loss_type="mse",
        target_attr="koff_target",
        mask_attr="koff_mask",
    ),
    TaskLossSpec(
        name="t_half",
        target_key="t_half",
        mask_key="t_half",
        pred_paths=(("assays", "t_half"),),
        loss_type="mse",
        target_attr="t_half_target",
        mask_attr="t_half_mask",
    ),
    TaskLossSpec(
        name="tm",
        target_key="tm",
        mask_key="tm",
        pred_paths=(("assays", "Tm"),),
        loss_type="mse",
        target_attr="tm_target",
        mask_attr="tm_mask",
    ),
    TaskLossSpec(
        name="binding_affinity_probe",
        target_key="binding_unknown",
        mask_key="binding_unknown",
        pred_paths=(("binding_affinity_probe_kd",),),
        loss_type="censor",
        target_attr="bind_target",
        mask_attr="bind_mask",
        qual_key="binding_unknown",
        qual_attr="bind_qual",
        target_transform=lambda t: normalize_binding_target_log10(
            t,
            max_affinity_nM=DEFAULT_MAX_AFFINITY_NM,
            assume_log10=False,
        ),
    ),
    TaskLossSpec(
        name="processing",
        target_key="processing",
        mask_key="processing",
        pred_paths=(("processing_logit",),),
        loss_type="bce",
        target_attr="processing_label",
        mask_attr="processing_mask",
    ),
    TaskLossSpec(
        name="mhc_class",
        target_key="mhc_class",
        mask_key="mhc_class",
        pred_paths=(("mhc_class_logits",),),
        loss_type="ce",
    ),
    TaskLossSpec(
        name="mhc_species",
        target_key="mhc_species",
        mask_key="mhc_species",
        pred_paths=(("mhc_species_logits",),),
        loss_type="ce",
    ),
    TaskLossSpec(
        name="mhc_a_fine_type",
        target_key="mhc_a_fine_type",
        mask_key="mhc_a_fine_type",
        pred_paths=(("mhc_a_type_logits",),),
        loss_type="ce",
    ),
    TaskLossSpec(
        name="mhc_b_fine_type",
        target_key="mhc_b_fine_type",
        mask_key="mhc_b_fine_type",
        pred_paths=(("mhc_b_type_logits",),),
        loss_type="ce",
    ),
    TaskLossSpec(
        name="chain_species",
        target_key="chain_species",
        mask_key="chain_species",
        pred_paths=(("chain_species_logits",),),
        loss_type="ce",
        target_attr="chain_species_label",
        mask_attr="chain_species_mask",
    ),
    TaskLossSpec(
        name="chain_type",
        target_key="chain_type",
        mask_key="chain_type",
        pred_paths=(("chain_type_logits",),),
        loss_type="ce",
        target_attr="chain_type_label",
        mask_attr="chain_type_mask",
    ),
    TaskLossSpec(
        name="chain_phenotype",
        target_key="chain_phenotype",
        mask_key="chain_phenotype",
        pred_paths=(("chain_phenotype_logits",),),
        loss_type="ce",
        target_attr="chain_phenotype_label",
        mask_attr="chain_phenotype_mask",
    ),
    TaskLossSpec(
        name="species_of_origin",
        target_key="species_of_origin",
        mask_key="species_of_origin",
        pred_paths=(("species_of_origin_logits",),),
        loss_type="ce",
    ),
    TaskLossSpec(
        name="foreignness",
        target_key="foreignness",
        mask_key="foreignness",
        pred_paths=(("foreignness_logit",),),
        loss_type="bce",
    ),
)

LOSS_TASK_NAMES: Tuple[str, ...] = tuple(spec.name for spec in LOSS_TASK_SPECS)
LOSS_TASK_NAME_TO_INDEX: Dict[str, int] = {
    name: idx for idx, name in enumerate(LOSS_TASK_NAMES)
}


def _normalize_supervised_loss_aggregation(mode: Optional[str]) -> str:
    token = str(mode or "sample_weighted").strip().lower().replace("-", "_")
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
        "mhc_attention_sparsity_weight": float(
            getattr(args, "mhc_attention_sparsity_weight", 0.0)
        ),
        "mhc_attention_sparsity_min_residues": float(
            getattr(args, "mhc_attention_sparsity_min_residues", 30.0)
        ),
        "mhc_attention_sparsity_max_residues": float(
            getattr(args, "mhc_attention_sparsity_max_residues", 60.0)
        ),
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
        "consistency_prob_margin": 0.02,
        "consistency_parent_low_threshold": 0.1,
        "consistency_presentation_high_threshold": 0.9,
        "consistency_affinity_fold_tolerance": 2.0,
        "tcell_in_vitro_margin": 0.0,
        "tcell_ex_vivo_margin": 0.0,
        "mhc_attention_sparsity_weight": 0.0,
        "mhc_attention_sparsity_min_residues": 30.0,
        "mhc_attention_sparsity_max_residues": 60.0,
    }
    if regularization is None:
        return defaults
    merged = dict(defaults)
    for key, value in regularization.items():
        if key in merged:
            merged[key] = float(value)
    return merged


def create_synthetic_data(data_dir: Path, n_binding: int = 200, n_elution: int = 100, n_tcr: int = 100):
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


def _as_float_vector(tensor: torch.Tensor) -> torch.Tensor:
    vec = tensor.float()
    if vec.ndim > 1 and vec.shape[-1] == 1:
        vec = vec.squeeze(-1)
    return vec


def _resolve_output_tensor(
    outputs: Dict[str, object],
    pred_paths: Sequence[Tuple[str, ...]],
) -> Optional[torch.Tensor]:
    for path in pred_paths:
        current: object = outputs
        valid = True
        for part in path:
            if not isinstance(current, dict) or part not in current:
                valid = False
                break
            current = current[part]
        if valid and isinstance(current, torch.Tensor):
            return current
    return None


def _batch_mapping(batch, attr_name: str) -> Optional[Dict[str, torch.Tensor]]:
    value = getattr(batch, attr_name, None)
    return value if isinstance(value, dict) else None


def _infer_fine_chain_types_for_batch(batch) -> Optional[list]:
    """Infer fine MHC chain types for alpha and beta chains from batch metadata."""
    classes = getattr(batch, "mhc_class", None)
    alleles = getattr(batch, "primary_alleles", None)
    if not isinstance(classes, (list, tuple)):
        return None
    n = len(classes)
    if not isinstance(alleles, (list, tuple)) or len(alleles) != n:
        alleles = [""] * n

    a_labels: list = []
    b_labels: list = []
    a_masks: list = []
    b_masks: list = []
    for i in range(n):
        mc = str(classes[i]).strip().upper()
        allele = str(alleles[i]).strip()
        gene = infer_gene(allele) if allele else ""

        # Alpha chain fine type
        if mc == "II":
            a_ft = infer_fine_chain_type(gene, "II")
            # For class II, alpha goes in slot a
            if a_ft in ("MHC_IIb",):
                # Gene-inferred as beta but in alpha slot — keep as-is
                pass
            a_labels.append(MHC_CHAIN_FINE_TO_IDX.get(a_ft, MHC_CHAIN_FINE_TO_IDX["unknown"]))
            a_masks.append(1.0 if a_ft != "unknown" else 0.0)
            # Beta chain for class II
            b_ft = "MHC_IIb"
            b_labels.append(MHC_CHAIN_FINE_TO_IDX[b_ft])
            b_masks.append(1.0)
        elif mc in ("I", ""):
            a_ft = infer_fine_chain_type(gene, "I")
            a_labels.append(MHC_CHAIN_FINE_TO_IDX.get(a_ft, MHC_CHAIN_FINE_TO_IDX["unknown"]))
            a_masks.append(1.0 if a_ft != "unknown" else 0.0)
            # Beta chain for class I is B2M
            b_labels.append(MHC_CHAIN_FINE_TO_IDX["B2M"])
            b_masks.append(1.0)
        else:
            a_labels.append(MHC_CHAIN_FINE_TO_IDX["unknown"])
            a_masks.append(0.0)
            b_labels.append(MHC_CHAIN_FINE_TO_IDX["unknown"])
            b_masks.append(0.0)

    return a_labels, b_labels, a_masks, b_masks


def _get_batch_target(batch, spec: TaskLossSpec) -> Optional[torch.Tensor]:
    if spec.name == "mhc_class":
        classes = getattr(batch, "mhc_class", None)
        if not isinstance(classes, (list, tuple)):
            return None
        labels = []
        for cls in classes:
            normalized = str(cls).strip().upper()
            labels.append(1 if normalized == "II" else 0)
        return torch.tensor(labels, dtype=torch.long, device=batch.pep_tok.device)
    if spec.name == "mhc_species":
        species_values = getattr(batch, "processing_species", None)
        if not isinstance(species_values, (list, tuple)):
            return None
        labels = []
        for raw in species_values:
            bucket = normalize_processing_species_label(raw, default=None)
            if bucket is None:
                # Unknown species: use placeholder label (masked out by _get_batch_mask)
                labels.append(0)
            else:
                labels.append(PROCESSING_SPECIES_TO_IDX[bucket])
        return torch.tensor(labels, dtype=torch.long, device=batch.pep_tok.device)
    if spec.name == "mhc_a_fine_type":
        result = _infer_fine_chain_types_for_batch(batch)
        if result is None:
            return None
        a_labels, _, _, _ = result
        return torch.tensor(a_labels, dtype=torch.long, device=batch.pep_tok.device)
    if spec.name == "mhc_b_fine_type":
        result = _infer_fine_chain_types_for_batch(batch)
        if result is None:
            return None
        _, b_labels, _, _ = result
        return torch.tensor(b_labels, dtype=torch.long, device=batch.pep_tok.device)

    targets = _batch_mapping(batch, "targets")
    if targets is not None and spec.target_key in targets:
        return targets[spec.target_key]
    if spec.target_attr:
        return getattr(batch, spec.target_attr, None)
    return None


def _get_batch_mask(batch, spec: TaskLossSpec) -> Optional[torch.Tensor]:
    if spec.name == "mhc_class":
        classes = getattr(batch, "mhc_class", None)
        if not isinstance(classes, (list, tuple)):
            return None
        mask = []
        for cls in classes:
            normalized = str(cls).strip().upper()
            mask.append(1.0 if normalized in {"I", "II"} else 0.0)
        return torch.tensor(mask, dtype=torch.float32, device=batch.pep_tok.device)
    if spec.name == "mhc_species":
        species_values = getattr(batch, "processing_species", None)
        if not isinstance(species_values, (list, tuple)):
            return None
        mask = []
        for raw in species_values:
            bucket = normalize_processing_species_label(raw, default=None)
            mask.append(1.0 if bucket is not None else 0.0)
        return torch.tensor(mask, dtype=torch.float32, device=batch.pep_tok.device)
    if spec.name == "mhc_a_fine_type":
        result = _infer_fine_chain_types_for_batch(batch)
        if result is None:
            return None
        _, _, a_masks, _ = result
        return torch.tensor(a_masks, dtype=torch.float32, device=batch.pep_tok.device)
    if spec.name == "mhc_b_fine_type":
        result = _infer_fine_chain_types_for_batch(batch)
        if result is None:
            return None
        _, _, _, b_masks = result
        return torch.tensor(b_masks, dtype=torch.float32, device=batch.pep_tok.device)

    target_masks = _batch_mapping(batch, "target_masks")
    if target_masks is not None and spec.mask_key in target_masks:
        return target_masks[spec.mask_key]
    if spec.mask_attr:
        return getattr(batch, spec.mask_attr, None)
    return None


def _get_batch_qual(batch, spec: TaskLossSpec) -> Optional[torch.Tensor]:
    target_quals = _batch_mapping(batch, "target_quals")
    if target_quals is not None and spec.qual_key and spec.qual_key in target_quals:
        return target_quals[spec.qual_key]
    if spec.qual_attr:
        return getattr(batch, spec.qual_attr, None)
    return None


def _compute_task_loss_vector(
    spec: TaskLossSpec,
    pred: torch.Tensor,
    target: torch.Tensor,
    qual_tensor: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if spec.loss_type == "ce":
        target_idx = target.long().view(-1)
        return nn.functional.cross_entropy(pred, target_idx, reduction="none")

    pred_vec = _as_float_vector(pred)
    target_vec = _as_float_vector(target)

    if spec.target_transform is not None:
        target_vec = spec.target_transform(target_vec)

    if spec.loss_type == "bce":
        return nn.functional.binary_cross_entropy_with_logits(
            pred_vec, target_vec, reduction="none"
        )
    if spec.loss_type == "mse":
        return nn.functional.mse_loss(pred_vec, target_vec, reduction="none")
    if spec.loss_type == "censor":
        if qual_tensor is None:
            return None
        qual_vec = _as_float_vector(qual_tensor).to(dtype=torch.long)
        return censor_aware_loss(
            pred_vec,
            target_vec,
            qual_vec,
            reduction="none",
        )
    raise ValueError(f"Unknown loss type: {spec.loss_type}")


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


def _build_mil_prob_matrix(
    inst_probs: torch.Tensor,
    instance_to_bag: torch.Tensor,
    n_bags: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack flat instance probabilities into a dense (bag, instance) matrix."""
    if inst_probs.ndim != 1:
        raise ValueError(f"Expected flat probabilities, got shape={tuple(inst_probs.shape)}")
    if instance_to_bag.ndim != 1:
        raise ValueError(
            f"Expected flat bag indices, got shape={tuple(instance_to_bag.shape)}"
        )
    if inst_probs.shape[0] != instance_to_bag.shape[0]:
        raise ValueError(
            "Instance probability length and bag-index length differ: "
            f"{inst_probs.shape[0]} vs {instance_to_bag.shape[0]}"
        )
    if n_bags <= 0:
        raise ValueError("n_bags must be positive")

    counts = torch.bincount(instance_to_bag, minlength=n_bags)
    max_instances = int(counts.max().item()) if counts.numel() > 0 else 0
    if max_instances <= 0:
        empty = torch.zeros((n_bags, 1), dtype=inst_probs.dtype, device=inst_probs.device)
        return empty, empty

    probs = torch.zeros(
        (n_bags, max_instances),
        dtype=inst_probs.dtype,
        device=inst_probs.device,
    )
    mask = torch.zeros_like(probs)
    offsets = torch.zeros((n_bags,), dtype=torch.long, device=inst_probs.device)

    for idx in range(inst_probs.shape[0]):
        bag_idx = int(instance_to_bag[idx].item())
        pos = int(offsets[bag_idx].item())
        probs[bag_idx, pos] = inst_probs[idx]
        mask[bag_idx, pos] = 1.0
        offsets[bag_idx] = offsets[bag_idx] + 1

    return probs, mask


def _compute_mil_elution_losses(
    model,
    batch,
    device: str,
    max_mil_instances: int = 0,
) -> tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Compute bag-level MIL losses for elution/presentation/MS outputs."""
    mil_losses: Dict[str, torch.Tensor] = {}
    mil_metrics: Dict[str, float] = {}

    if (
        getattr(batch, "mil_pep_tok", None) is None
        or getattr(batch, "mil_mhc_a_tok", None) is None
        or getattr(batch, "mil_mhc_b_tok", None) is None
        or getattr(batch, "mil_instance_to_bag", None) is None
        or getattr(batch, "mil_bag_label", None) is None
    ):
        return mil_losses, mil_metrics

    mil_bag_label = batch.mil_bag_label.to(device=device, dtype=torch.float32)
    if mil_bag_label.numel() == 0:
        return mil_losses, mil_metrics

    # Subsample MIL instances when exceeding cap to bound GPU cost.
    mil_pep_tok = batch.mil_pep_tok
    mil_mhc_a_tok = batch.mil_mhc_a_tok
    mil_mhc_b_tok = batch.mil_mhc_b_tok
    mil_mhc_class = batch.mil_mhc_class
    mil_species = batch.mil_species
    mil_flank_n_tok = getattr(batch, "mil_flank_n_tok", None)
    mil_flank_c_tok = getattr(batch, "mil_flank_c_tok", None)
    mil_instance_to_bag = batch.mil_instance_to_bag

    n_instances = mil_pep_tok.shape[0]
    if max_mil_instances > 0 and n_instances > max_mil_instances:
        keep = torch.randperm(n_instances)[:max_mil_instances].sort().values
        mil_pep_tok = mil_pep_tok[keep]
        mil_mhc_a_tok = mil_mhc_a_tok[keep]
        mil_mhc_b_tok = mil_mhc_b_tok[keep]
        mil_mhc_class = [mil_mhc_class[i] for i in keep.tolist()]
        mil_species = [mil_species[i] for i in keep.tolist()]
        if mil_flank_n_tok is not None:
            mil_flank_n_tok = mil_flank_n_tok[keep]
        if mil_flank_c_tok is not None:
            mil_flank_c_tok = mil_flank_c_tok[keep]
        mil_instance_to_bag = mil_instance_to_bag[keep]

    mil_outputs = model(
        pep_tok=mil_pep_tok.to(device),
        mhc_a_tok=mil_mhc_a_tok.to(device),
        mhc_b_tok=mil_mhc_b_tok.to(device),
        mhc_class=mil_mhc_class,
        species=mil_species,
        tcr_a_tok=None,
        tcr_b_tok=None,
        flank_n_tok=(
            mil_flank_n_tok.to(device)
            if mil_flank_n_tok is not None
            else None
        ),
        flank_c_tok=(
            mil_flank_c_tok.to(device)
            if mil_flank_c_tok is not None
            else None
        ),
        tcell_context=None,
    )

    instance_to_bag = mil_instance_to_bag.to(device=device, dtype=torch.long)
    n_bags = int(mil_bag_label.shape[0])

    task_to_output = {
        "elution": "elution_logit",
        "presentation": "presentation_logit",
        "ms": "ms_logit",
    }
    for task_name, output_key in task_to_output.items():
        logits = mil_outputs.get(output_key)
        if not isinstance(logits, torch.Tensor):
            continue
        inst_probs = torch.sigmoid(_as_float_vector(logits))
        prob_matrix, mask_matrix = _build_mil_prob_matrix(
            inst_probs=inst_probs,
            instance_to_bag=instance_to_bag,
            n_bags=n_bags,
        )
        bag_loss, bag_probs = mil_bag_loss(
            inst_probs=prob_matrix,
            bag_labels=mil_bag_label,
            mask=mask_matrix,
        )
        mil_losses[task_name] = bag_loss
        mil_metrics[f"out_mil_{task_name}_prob_mean"] = float(
            bag_probs.detach().mean().item()
        )
        mil_metrics[f"out_mil_{task_name}_prob_var"] = float(
            bag_probs.detach().var(unbiased=False).item()
        )

    return mil_losses, mil_metrics


def _compute_consistency_losses(
    outputs: Dict[str, Any],
    batch,
    regularization: Mapping[str, float],
) -> Dict[str, torch.Tensor]:
    """Compute biologic-prior consistency losses."""
    losses: Dict[str, torch.Tensor] = {}
    cascade_w = float(regularization.get("consistency_cascade_weight", 0.0))
    affinity_w = float(regularization.get("consistency_assay_affinity_weight", 0.0))
    assay_pres_w = float(
        regularization.get("consistency_assay_presentation_weight", 0.0)
    )
    no_b2m_w = float(regularization.get("consistency_no_b2m_weight", 0.0))
    tcell_ctx_w = float(regularization.get("consistency_tcell_context_weight", 0.0))
    tcell_upstream_w = float(regularization.get("consistency_tcell_upstream_weight", 0.0))
    mhc_attn_sparse_w = float(regularization.get("mhc_attention_sparsity_weight", 0.0))
    mhc_attn_sparse_min = float(regularization.get("mhc_attention_sparsity_min_residues", 30.0))
    mhc_attn_sparse_max = float(regularization.get("mhc_attention_sparsity_max_residues", 60.0))
    prob_margin = float(regularization.get("consistency_prob_margin", 0.02))
    parent_low_thr = float(regularization.get("consistency_parent_low_threshold", 0.1))
    pres_high_thr = float(
        regularization.get("consistency_presentation_high_threshold", 0.9)
    )
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

    if cascade_w > 0:
        proc_prob = torch.sigmoid(_as_float_vector(outputs["processing_logit"]))
        bind_prob = torch.sigmoid(_as_float_vector(outputs["binding_logit"]))
        pres_prob = torch.sigmoid(_as_float_vector(outputs["presentation_logit"]))
        parent_min = torch.minimum(proc_prob, bind_prob)
        high_pres = torch.relu(pres_prob - pres_high_thr)
        low_parent = torch.relu(parent_low_thr - parent_min)
        losses["consistency_cascade"] = cascade_w * (
            (high_pres * low_parent).square().mean()
        )

    assays = outputs.get("assays", {})
    if affinity_w > 0 and isinstance(assays, dict):
        kd = assays.get("KD_nM")
        ic50 = assays.get("IC50_nM")
        ec50 = assays.get("EC50_nM")
        if isinstance(kd, torch.Tensor) and isinstance(ic50, torch.Tensor) and isinstance(ec50, torch.Tensor):
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
            losses["consistency_assay_presentation"] = (
                assay_pres_w * (sum(loss_terms) / len(loss_terms))
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
            lower = torch.relu(torch.tensor(mhc_attn_sparse_min, device=effective.device) - effective)
            upper = torch.relu(effective - torch.tensor(mhc_attn_sparse_max, device=effective.device))
            penalty = lower.square() + upper.square()
            reduced = _masked_mean(penalty, valid_mask)
            if reduced is not None:
                losses["consistency_binding_mhc_attention_sparsity"] = (
                    mhc_attn_sparse_w * reduced
                )

    return losses


def compute_loss(
    model,
    batch,
    device,
    uncertainty_weighting=None,
    regularization: Optional[Mapping[str, float]] = None,
    supervised_loss_aggregation: str = "sample_weighted",
    profile_performance: bool = False,
    non_blocking_transfer: bool = False,
    use_amp: bool = False,
    max_mil_instances: int = 0,
):
    """Compute multi-task loss for a batch."""
    # Move batch to device
    try:
        batch = batch.to(device, non_blocking=non_blocking_transfer)
    except TypeError:
        batch = batch.to(device)
    regularization_cfg = _resolve_regularization_config(regularization)
    aggregation_mode = _normalize_supervised_loss_aggregation(
        supervised_loss_aggregation
    )
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
            return_binding_attention=return_binding_attention,
        )
        if profile_performance:
            perf_metrics["perf_forward_main_sec"] = float(time.perf_counter() - forward_start)
        output_metrics = _summarize_outputs(outputs)
        has_mil_elution = (
            getattr(batch, "mil_bag_label", None) is not None
            and getattr(batch, "mil_instance_to_bag", None) is not None
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

        supervised_losses: Dict[str, torch.Tensor] = {}
        supervised_loss_support: Dict[str, float] = {}
        supervised_start = time.perf_counter() if profile_performance else 0.0
        for spec in LOSS_TASK_SPECS:
            if has_mil_elution and spec.name in {"elution", "presentation"}:
                # These tasks are trained at bag-level via Noisy-OR MIL below.
                continue
            target = _get_batch_target(batch, spec)
            mask = _get_batch_mask(batch, spec)
            if target is None or mask is None:
                continue
            mask_float = _as_float_vector(mask)
            support = float(mask_float.sum().detach().item())
            if support <= 0:
                continue

            pred = _resolve_output_tensor(outputs, spec.pred_paths)
            if pred is None:
                continue
            qual_tensor = _get_batch_qual(batch, spec)
            loss_vector = _compute_task_loss_vector(spec, pred, target, qual_tensor=qual_tensor)
            if loss_vector is None:
                continue

            masked_loss = (loss_vector * mask_float).sum() / (mask_float.sum() + 1e-8)
            supervised_losses[spec.name] = masked_loss
            supervised_loss_support[spec.name] = support
        if profile_performance:
            perf_metrics["perf_supervised_loss_sec"] = float(
                time.perf_counter() - supervised_start
            )

        if has_mil_elution:
            mil_start = time.perf_counter() if profile_performance else 0.0
            mil_losses, mil_metrics = _compute_mil_elution_losses(
                model=model,
                batch=batch,
                device=device,
                max_mil_instances=max_mil_instances,
            )
            supervised_losses.update(mil_losses)
            mil_bag_label = getattr(batch, "mil_bag_label", None)
            mil_support = (
                float(mil_bag_label.numel())
                if isinstance(mil_bag_label, torch.Tensor)
                else 1.0
            )
            for name in mil_losses:
                supervised_loss_support[name] = mil_support
            output_metrics.update(mil_metrics)
            if profile_performance:
                perf_metrics["perf_mil_sec"] = float(time.perf_counter() - mil_start)
        elif profile_performance:
            perf_metrics["perf_mil_sec"] = 0.0

        regularization_start = time.perf_counter() if profile_performance else 0.0
        regularization_losses = _compute_consistency_losses(
            outputs=outputs,
            batch=batch,
            regularization=regularization_cfg,
        )
        if profile_performance:
            perf_metrics["perf_regularization_sec"] = float(
                time.perf_counter() - regularization_start
            )
        # Scale auxiliary binding probe loss (shortcut gradient path, not dominant)
        if "binding_affinity_probe" in supervised_losses:
            supervised_losses["binding_affinity_probe"] = (
                0.3 * supervised_losses["binding_affinity_probe"]
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
            for task_name, task_loss in supervised_losses.items():
                if aggregation_mode == "task_mean":
                    task_weight = 1.0
                else:
                    task_weight = max(float(supervised_loss_support.get(task_name, 1.0)), 1e-6)
                total_weight += task_weight

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
    uncertainty_weighting=None,
    pcgrad: PCGrad = None,
    regularization: Optional[Mapping[str, float]] = None,
    show_progress: bool = True,
    profile_performance: bool = False,
    supervised_loss_aggregation: str = "sample_weighted",
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

        if pcgrad is not None and len(loss_dict) > 1:
            step_start = time.perf_counter()
            pcgrad.step(list(loss_dict.values()), model.parameters())
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
            optimizer_elapsed = time.perf_counter() - optim_start
            perf_optimizer_sec += optimizer_elapsed

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
        name: output_totals[name] / max(output_counts[name], 1)
        for name in sorted(output_totals)
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
    supervised_loss_aggregation: str = "sample_weighted",
    use_amp: bool = False,
    max_mil_instances: int = 0,
    max_batches: int = 0,
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
            )
            pep_tok = getattr(batch, "pep_tok", None)
            if isinstance(pep_tok, torch.Tensor) and pep_tok.ndim >= 1:
                total_samples += int(pep_tok.shape[0])
            total_loss += loss.item()
            for name, value in loss_dict.items():
                task_loss_totals[name] = task_loss_totals.get(name, 0.0) + float(value.detach().item())
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
        name: output_totals[name] / max(output_counts[name], 1)
        for name in sorted(output_totals)
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
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, collator=collator)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, collator=collator)

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
                uncertainty_weighting,
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

            print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if run_logger is not None:
                run_logger.log(
                    epoch + 1,
                    "train",
                    {"loss": train_loss, **train_task_losses, **uw_metrics},
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
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory (temp if not specified)")
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
    parser.add_argument("--use-pcgrad", action="store_true", help="Use PCGrad for multi-task gradient conflicts")
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
