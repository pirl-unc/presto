"""Data collation utilities for Presto.

Handles batching of variable-length sequences with proper padding.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from .allele_resolver import normalize_mhc_class
from .tokenizer import Tokenizer
from .vocab import (
    BINDING_ASSAY_METHOD_TO_IDX,
    BINDING_ASSAY_PREP_TO_IDX,
    BINDING_ASSAY_GEOMETRY_TO_IDX,
    BINDING_ASSAY_READOUT_TO_IDX,
    BINDING_ASSAY_TYPE_TO_IDX,
    FOREIGN_CATEGORIES,
    ORGANISM_TO_IDX,
    TCELL_APC_TYPE_TO_IDX,
    TCELL_ASSAY_METHOD_TO_IDX,
    TCELL_ASSAY_READOUT_TO_IDX,
    TCELL_CULTURE_CONTEXT_TO_IDX,
    TCELL_PEPTIDE_FORMAT_TO_IDX,
    TCELL_STIM_CONTEXT_TO_IDX,
    normalize_species,
)

OPTIONAL_MISSING_SEQ_TOKENS = {"NA", "N/A", "NONE", "NULL", "-", "?"}
TCR_EVIDENCE_METHOD_BINS = (
    "multimer_binding",
    "target_cell_functional",
    "functional_readout",
)


@dataclass(frozen=True)
class TargetSpec:
    """How to collate one supervised target from `PrestoSample` fields."""

    task_name: str
    sample_field: str
    target_field: str
    mask_field: str
    unsqueeze_last: bool = False
    transform: Optional[Callable[[float], float]] = None


def _log10_clamp(value: float, min_value: float = 1e-12) -> float:
    return math.log10(max(float(value), min_value))


TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec(
        task_name="binding",
        sample_field="bind_value",
        target_field="bind_target",
        mask_field="bind_mask",
        unsqueeze_last=True,
    ),
    TargetSpec(
        task_name="kon",
        sample_field="kon",
        target_field="kon_target",
        mask_field="kon_mask",
        unsqueeze_last=True,
        transform=lambda value: _log10_clamp(value, min_value=1e-12),
    ),
    TargetSpec(
        task_name="koff",
        sample_field="koff",
        target_field="koff_target",
        mask_field="koff_mask",
        unsqueeze_last=True,
        transform=lambda value: _log10_clamp(value, min_value=1e-12),
    ),
    TargetSpec(
        task_name="t_half",
        sample_field="t_half",
        target_field="t_half_target",
        mask_field="t_half_mask",
        unsqueeze_last=True,
        transform=lambda value: _log10_clamp(value * 60.0, min_value=1e-12),
    ),
    TargetSpec(
        task_name="tm",
        sample_field="tm",
        target_field="tm_target",
        mask_field="tm_mask",
        unsqueeze_last=True,
        transform=lambda value: (float(value) - 50.0) / 15.0,
    ),
    TargetSpec(
        task_name="tcell",
        sample_field="tcell_label",
        target_field="tcell_label",
        mask_field="tcell_mask",
    ),
    TargetSpec(
        task_name="elution",
        sample_field="elution_label",
        target_field="elution_label",
        mask_field="elution_mask",
    ),
    TargetSpec(
        task_name="processing",
        sample_field="processing_label",
        target_field="processing_label",
        mask_field="processing_mask",
    ),
    TargetSpec(
        task_name="tcr_evidence",
        sample_field="tcr_evidence_label",
        target_field="tcr_evidence_target",
        mask_field="tcr_evidence_mask",
    ),
    TargetSpec(
        task_name="foreignness",
        sample_field="foreignness_label",
        target_field="foreignness_target",
        mask_field="foreignness_mask",
        unsqueeze_last=True,
    ),
)


@dataclass
class PrestoSample:
    """A single training/inference sample."""
    # Peptide
    peptide: str
    flank_n: Optional[str] = None
    flank_c: Optional[str] = None

    # MHC
    mhc_a: str = ""  # Groove half 1: alpha1 (class I/II)
    mhc_b: str = ""  # Groove half 2: alpha2 (class I) or beta1 (class II)
    mhc_class: Optional[str] = None

    # Labels (optional, for training)
    # Binding
    bind_value: Optional[float] = None  # nM
    bind_qual: int = 0  # -1=<, 0==, 1=>
    bind_measurement_type: Optional[str] = None  # KD / IC50 / EC50 / unknown
    binding_assay_type: Optional[str] = None
    binding_assay_method: Optional[str] = None
    binding_effector_culture: Optional[str] = None
    binding_apc_culture: Optional[str] = None

    # Kinetics
    kon: Optional[float] = None
    koff: Optional[float] = None

    # Stability
    t_half: Optional[float] = None
    tm: Optional[float] = None

    # T-cell
    tcell_label: Optional[float] = None
    tcell_assay_method: Optional[str] = None
    tcell_assay_readout: Optional[str] = None
    tcell_apc_name: Optional[str] = None
    tcell_effector_culture: Optional[str] = None
    tcell_apc_culture: Optional[str] = None
    tcell_in_vitro_process: Optional[str] = None
    tcell_in_vitro_responder: Optional[str] = None
    tcell_in_vitro_stimulator: Optional[str] = None
    tcell_peptide_format: Optional[str] = None
    tcell_culture_duration_hours: Optional[float] = None

    # Elution
    elution_label: Optional[float] = None
    # Optional multi-allele bag instances for elution/MS supervision.
    # When set, each list element is one allele-instance for this sample.
    mil_mhc_a_list: Optional[List[str]] = None
    mil_mhc_b_list: Optional[List[str]] = None
    mil_mhc_class_list: Optional[List[str]] = None
    mil_species_list: Optional[List[str]] = None
    use_tcell_pathway_mil: bool = False
    tcell_mil_mhc_a_list: Optional[List[str]] = None
    tcell_mil_mhc_b_list: Optional[List[str]] = None
    tcell_mil_mhc_class_list: Optional[List[str]] = None
    tcell_mil_species_list: Optional[List[str]] = None

    # Processing
    processing_label: Optional[float] = None
    core_start: Optional[int] = None
    tcr_evidence_label: Optional[float] = None
    tcr_evidence_method_bins: tuple[str, ...] = ()

    # Peptide source organism (unified 12-class taxonomy)
    species_of_origin: Optional[str] = None
    foreignness_label: Optional[float] = None

    species: Optional[str] = None

    # Metadata
    sample_source: Optional[str] = None
    assay_group: Optional[str] = None
    label_bucket: Optional[str] = None
    primary_allele: Optional[str] = None
    synthetic_kind: Optional[str] = None
    dataset_index: int = -1
    peptide_id: int = -1
    allele_id: int = -1
    bind_target_log10: Optional[float] = None
    sample_id: str = ""


@dataclass
class PrestoBatch:
    """A collated batch of samples."""
    # Tokenized sequences
    pep_tok: torch.Tensor
    mhc_a_tok: torch.Tensor
    mhc_b_tok: torch.Tensor
    mhc_class: List[Optional[str]]

    # Optional sequences
    flank_n_tok: Optional[torch.Tensor] = None
    flank_c_tok: Optional[torch.Tensor] = None

    # Labels
    bind_target: Optional[torch.Tensor] = None
    bind_qual: Optional[torch.Tensor] = None
    kon_target: Optional[torch.Tensor] = None
    koff_target: Optional[torch.Tensor] = None
    t_half_target: Optional[torch.Tensor] = None
    tm_target: Optional[torch.Tensor] = None
    tcell_label: Optional[torch.Tensor] = None
    elution_label: Optional[torch.Tensor] = None
    processing_label: Optional[torch.Tensor] = None
    tcr_evidence_target: Optional[torch.Tensor] = None
    tcr_evidence_method_target: Optional[torch.Tensor] = None

    # Masks for which samples have which labels
    bind_mask: Optional[torch.Tensor] = None
    kon_mask: Optional[torch.Tensor] = None
    koff_mask: Optional[torch.Tensor] = None
    t_half_mask: Optional[torch.Tensor] = None
    tm_mask: Optional[torch.Tensor] = None
    tcell_mask: Optional[torch.Tensor] = None
    elution_mask: Optional[torch.Tensor] = None
    processing_mask: Optional[torch.Tensor] = None
    tcr_evidence_mask: Optional[torch.Tensor] = None
    tcr_evidence_method_mask: Optional[torch.Tensor] = None

    # Optional T-cell assay context (categorical IDs + masks)
    binding_context: Dict[str, torch.Tensor] = field(default_factory=dict)
    tcell_context: Dict[str, torch.Tensor] = field(default_factory=dict)
    tcell_context_masks: Dict[str, torch.Tensor] = field(default_factory=dict)
    tcell_mil_context: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Optional multi-allele MIL bag tensors for elution/presentation/MS tasks.
    mil_pep_tok: Optional[torch.Tensor] = None
    mil_mhc_a_tok: Optional[torch.Tensor] = None
    mil_mhc_b_tok: Optional[torch.Tensor] = None
    mil_mhc_class: Optional[List[str]] = None
    mil_species: Optional[List[str]] = None
    mil_flank_n_tok: Optional[torch.Tensor] = None
    mil_flank_c_tok: Optional[torch.Tensor] = None
    mil_instance_to_bag: Optional[torch.Tensor] = None
    mil_bag_label: Optional[torch.Tensor] = None
    mil_bag_sample_ids: List[str] = field(default_factory=list)
    tcell_mil_pep_tok: Optional[torch.Tensor] = None
    tcell_mil_mhc_a_tok: Optional[torch.Tensor] = None
    tcell_mil_mhc_b_tok: Optional[torch.Tensor] = None
    tcell_mil_mhc_class: Optional[List[str]] = None
    tcell_mil_species: Optional[List[str]] = None
    tcell_mil_flank_n_tok: Optional[torch.Tensor] = None
    tcell_mil_flank_c_tok: Optional[torch.Tensor] = None
    tcell_mil_instance_to_bag: Optional[torch.Tensor] = None
    tcell_mil_bag_label: Optional[torch.Tensor] = None
    tcell_mil_bag_sample_ids: List[str] = field(default_factory=list)

    # Lengths for masking
    pep_lengths: Optional[torch.Tensor] = None
    dataset_index: Optional[torch.Tensor] = None
    peptide_id: Optional[torch.Tensor] = None
    allele_id: Optional[torch.Tensor] = None
    bind_target_log10: Optional[torch.Tensor] = None
    same_peptide_diff_allele_pairs: Optional[torch.Tensor] = None
    same_allele_diff_peptide_pairs: Optional[torch.Tensor] = None

    # Metadata
    processing_species: List[Optional[str]] = field(default_factory=list)
    primary_alleles: List[str] = field(default_factory=list)
    sample_ids: List[str] = field(default_factory=list)
    targets: Dict[str, torch.Tensor] = field(default_factory=dict)
    target_masks: Dict[str, torch.Tensor] = field(default_factory=dict)
    target_quals: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to(self, device: str, non_blocking: bool = False) -> "PrestoBatch":
        """Move batch to device."""
        def _move(t):
            return (
                t.to(device, non_blocking=non_blocking)
                if t is not None
                else None
            )

        return PrestoBatch(
            pep_tok=_move(self.pep_tok),
            mhc_a_tok=_move(self.mhc_a_tok),
            mhc_b_tok=_move(self.mhc_b_tok),
            mhc_class=self.mhc_class,
            flank_n_tok=_move(self.flank_n_tok),
            flank_c_tok=_move(self.flank_c_tok),
            bind_target=_move(self.bind_target),
            bind_qual=_move(self.bind_qual),
            kon_target=_move(self.kon_target),
            koff_target=_move(self.koff_target),
            t_half_target=_move(self.t_half_target),
            tm_target=_move(self.tm_target),
            tcell_label=_move(self.tcell_label),
            elution_label=_move(self.elution_label),
            processing_label=_move(self.processing_label),
            tcr_evidence_target=_move(self.tcr_evidence_target),
            tcr_evidence_method_target=_move(self.tcr_evidence_method_target),
            bind_mask=_move(self.bind_mask),
            kon_mask=_move(self.kon_mask),
            koff_mask=_move(self.koff_mask),
            t_half_mask=_move(self.t_half_mask),
            tm_mask=_move(self.tm_mask),
            tcell_mask=_move(self.tcell_mask),
            elution_mask=_move(self.elution_mask),
            processing_mask=_move(self.processing_mask),
            tcr_evidence_mask=_move(self.tcr_evidence_mask),
            tcr_evidence_method_mask=_move(self.tcr_evidence_method_mask),
            pep_lengths=_move(self.pep_lengths),
            dataset_index=_move(self.dataset_index),
            peptide_id=_move(self.peptide_id),
            allele_id=_move(self.allele_id),
            bind_target_log10=_move(self.bind_target_log10),
            same_peptide_diff_allele_pairs=_move(self.same_peptide_diff_allele_pairs),
            same_allele_diff_peptide_pairs=_move(self.same_allele_diff_peptide_pairs),
            processing_species=self.processing_species,
            primary_alleles=self.primary_alleles,
            sample_ids=self.sample_ids,
            targets={name: _move(tensor) for name, tensor in self.targets.items()},
            target_masks={
                name: _move(tensor) for name, tensor in self.target_masks.items()
            },
            target_quals={
                name: _move(tensor) for name, tensor in self.target_quals.items()
            },
            binding_context={
                name: _move(tensor) for name, tensor in self.binding_context.items()
            },
            tcell_context={
                name: _move(tensor) for name, tensor in self.tcell_context.items()
            },
            tcell_context_masks={
                name: _move(tensor) for name, tensor in self.tcell_context_masks.items()
            },
            tcell_mil_context={
                name: _move(tensor) for name, tensor in self.tcell_mil_context.items()
            },
            mil_pep_tok=_move(self.mil_pep_tok),
            mil_mhc_a_tok=_move(self.mil_mhc_a_tok),
            mil_mhc_b_tok=_move(self.mil_mhc_b_tok),
            mil_mhc_class=self.mil_mhc_class,
            mil_species=self.mil_species,
            mil_flank_n_tok=_move(self.mil_flank_n_tok),
            mil_flank_c_tok=_move(self.mil_flank_c_tok),
            mil_instance_to_bag=_move(self.mil_instance_to_bag),
            mil_bag_label=_move(self.mil_bag_label),
            mil_bag_sample_ids=self.mil_bag_sample_ids,
            tcell_mil_pep_tok=_move(self.tcell_mil_pep_tok),
            tcell_mil_mhc_a_tok=_move(self.tcell_mil_mhc_a_tok),
            tcell_mil_mhc_b_tok=_move(self.tcell_mil_mhc_b_tok),
            tcell_mil_mhc_class=self.tcell_mil_mhc_class,
            tcell_mil_species=self.tcell_mil_species,
            tcell_mil_flank_n_tok=_move(self.tcell_mil_flank_n_tok),
            tcell_mil_flank_c_tok=_move(self.tcell_mil_flank_c_tok),
            tcell_mil_instance_to_bag=_move(self.tcell_mil_instance_to_bag),
            tcell_mil_bag_label=_move(self.tcell_mil_bag_label),
            tcell_mil_bag_sample_ids=self.tcell_mil_bag_sample_ids,
        )


class PrestoCollator:
    """Collates PrestoSamples into PrestoBatches."""

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        max_pep_len: int = 50,
        max_mhc_len: int = 120,
        max_tcr_len: int = 200,
        max_flank_len: int = 25,
    ):
        self.tokenizer = tokenizer or Tokenizer()
        self.max_pep_len = max_pep_len
        self.max_mhc_len = max_mhc_len
        self.max_tcr_len = max_tcr_len
        self.max_flank_len = max_flank_len

    def _collate_targets(
        self, samples: List[PrestoSample]
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        targets: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        for spec in TARGET_SPECS:
            values: List[float] = []
            mask: List[float] = []
            has_any = False
            for sample in samples:
                raw = getattr(sample, spec.sample_field)
                if raw is None:
                    values.append(0.0)
                    mask.append(0.0)
                    continue
                value = spec.transform(raw) if spec.transform is not None else float(raw)
                values.append(float(value))
                is_enabled = not (
                    spec.task_name == "tcell" and sample.use_tcell_pathway_mil
                )
                mask.append(1.0 if is_enabled else 0.0)
                has_any = has_any or is_enabled

            if not has_any:
                continue

            tensor = torch.tensor(values, dtype=torch.float32)
            if spec.unsqueeze_last:
                tensor = tensor.unsqueeze(-1)
            targets[spec.task_name] = tensor
            masks[spec.task_name] = torch.tensor(mask, dtype=torch.float32)
        return targets, masks

    @staticmethod
    def _collate_fixed_metadata(
        samples: List[PrestoSample],
    ) -> Dict[str, torch.Tensor]:
        dataset_index = torch.tensor(
            [int(getattr(sample, "dataset_index", -1)) for sample in samples],
            dtype=torch.long,
        )
        peptide_id = torch.tensor(
            [int(getattr(sample, "peptide_id", -1)) for sample in samples],
            dtype=torch.long,
        )
        allele_id = torch.tensor(
            [int(getattr(sample, "allele_id", -1)) for sample in samples],
            dtype=torch.long,
        )
        bind_target_log10 = torch.tensor(
            [
                float(getattr(sample, "bind_target_log10", 0.0) or 0.0)
                for sample in samples
            ],
            dtype=torch.float32,
        )
        return {
            "dataset_index": dataset_index,
            "peptide_id": peptide_id,
            "allele_id": allele_id,
            "bind_target_log10": bind_target_log10,
        }

    @staticmethod
    def _collate_binding_pair_indices(
        samples: List[PrestoSample],
    ) -> Dict[str, torch.Tensor]:
        by_peptide: Dict[int, List[int]] = {}
        by_allele: Dict[int, List[int]] = {}
        for idx, sample in enumerate(samples):
            peptide_id = int(getattr(sample, "peptide_id", -1))
            allele_id = int(getattr(sample, "allele_id", -1))
            if peptide_id >= 0:
                by_peptide.setdefault(peptide_id, []).append(idx)
            if allele_id >= 0:
                by_allele.setdefault(allele_id, []).append(idx)

        same_peptide_diff_allele_pairs: List[tuple[int, int]] = []
        for indices in by_peptide.values():
            if len(indices) < 2:
                continue
            for pos, idx_i in enumerate(indices):
                allele_i = int(getattr(samples[idx_i], "allele_id", -1))
                for idx_j in indices[pos + 1 :]:
                    allele_j = int(getattr(samples[idx_j], "allele_id", -1))
                    if allele_i >= 0 and allele_j >= 0 and allele_i != allele_j:
                        same_peptide_diff_allele_pairs.append((idx_i, idx_j))

        same_allele_diff_peptide_pairs: List[tuple[int, int]] = []
        for indices in by_allele.values():
            if len(indices) < 2:
                continue
            for pos, idx_i in enumerate(indices):
                peptide_i = int(getattr(samples[idx_i], "peptide_id", -1))
                for idx_j in indices[pos + 1 :]:
                    peptide_j = int(getattr(samples[idx_j], "peptide_id", -1))
                    if peptide_i >= 0 and peptide_j >= 0 and peptide_i != peptide_j:
                        same_allele_diff_peptide_pairs.append((idx_i, idx_j))

        def _tensor(pairs: List[tuple[int, int]]) -> torch.Tensor:
            if not pairs:
                return torch.zeros((0, 2), dtype=torch.long)
            return torch.tensor(pairs, dtype=torch.long)

        return {
            "same_peptide_diff_allele_pairs": _tensor(same_peptide_diff_allele_pairs),
            "same_allele_diff_peptide_pairs": _tensor(same_allele_diff_peptide_pairs),
        }

    @staticmethod
    def _expand_with_fallback(
        values: Optional[List[str]],
        n_instances: int,
        fallback: str,
    ) -> List[str]:
        if values:
            out = [str(v) for v in values if v is not None]
        else:
            out = []
        if not out:
            out = [fallback]
        if len(out) < n_instances:
            out = out + [out[-1]] * (n_instances - len(out))
        return out[:n_instances]

    @staticmethod
    def _sanitize_optional_sequence(value: Optional[str]) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text.upper() in OPTIONAL_MISSING_SEQ_TOKENS:
            return ""
        return text

    @staticmethod
    def _split_indices_by_mhc_class(
        mhc_classes: List[Optional[str]],
    ) -> List[tuple[str, List[int]]]:
        groups: Dict[str, List[int]] = {}
        order: List[str] = []
        for idx, raw_cls in enumerate(mhc_classes):
            cls = normalize_mhc_class(raw_cls, default=None) or ""
            if cls not in groups:
                groups[cls] = []
                order.append(cls)
            groups[cls].append(idx)

        resolved_classes = {cls for cls in groups if cls in {"I", "II"}}
        if len(resolved_classes) < 2:
            whole = list(range(len(mhc_classes)))
            default_label = normalize_mhc_class(
                mhc_classes[0] if mhc_classes else None,
                default=None,
            ) or ""
            return [(default_label, whole)]

        ordered_classes = [cls for cls in ("I", "II") if cls in groups]
        ordered_classes.extend(cls for cls in order if cls not in {"I", "II"})
        return [(cls, groups[cls]) for cls in ordered_classes]

    def _materialize_mil_tensors(
        self,
        *,
        peptides: List[str],
        mhc_as: List[str],
        mhc_bs: List[str],
        mhc_classes: List[str],
        species: List[str],
        flank_ns: List[str],
        flank_cs: List[str],
        instance_to_bag: List[int],
        bag_labels: List[float],
        bag_sample_ids: List[str],
    ) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {
            "pep_tok": None,
            "mhc_a_tok": None,
            "mhc_b_tok": None,
            "mhc_class": None,
            "species": None,
            "flank_n_tok": None,
            "flank_c_tok": None,
            "instance_to_bag": None,
            "bag_label": None,
            "bag_sample_ids": bag_sample_ids,
        }
        if not peptides:
            return outputs

        outputs["pep_tok"] = self.tokenizer.batch_encode(
            peptides,
            max_len=self.max_pep_len,
            pad=True,
        )
        outputs["mhc_a_tok"] = self.tokenizer.batch_encode(
            mhc_as,
            max_len=self.max_mhc_len,
            pad=True,
        )
        outputs["mhc_b_tok"] = self.tokenizer.batch_encode(
            mhc_bs,
            max_len=self.max_mhc_len,
            pad=True,
        )
        outputs["mhc_class"] = mhc_classes
        outputs["species"] = species
        if any(v for v in flank_ns):
            outputs["flank_n_tok"] = self.tokenizer.batch_encode(
                flank_ns,
                max_len=self.max_flank_len,
                pad=True,
            )
        if any(v for v in flank_cs):
            outputs["flank_c_tok"] = self.tokenizer.batch_encode(
                flank_cs,
                max_len=self.max_flank_len,
                pad=True,
            )
        outputs["instance_to_bag"] = torch.tensor(
            instance_to_bag,
            dtype=torch.long,
        )
        outputs["bag_label"] = torch.tensor(
            bag_labels,
            dtype=torch.float32,
        )
        return outputs

    @staticmethod
    def _normalize_binding_measurement(value: Optional[str]) -> str:
        if value is None:
            return "unknown"
        token = str(value).strip().lower()
        if not token:
            return "unknown"
        if "ic50" in token or "inhibitory concentration" in token:
            return "ic50"
        if "ec50" in token or "effective concentration" in token:
            return "ec50"
        if "kd" in token or "dissociation constant" in token:
            return "kd"
        return "unknown"

    def _collate_binding_measurement_targets(
        self,
        samples: List[PrestoSample],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        targets: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        quals: Dict[str, torch.Tensor] = {}

        task_matchers = {
            "binding_kd": lambda measurement_name, assay_family: measurement_name == "kd",
            "binding_kd_direct": lambda measurement_name, assay_family: assay_family == "KD",
            "binding_kd_proxy_ic50": lambda measurement_name, assay_family: assay_family == "KD_PROXY_IC50",
            "binding_kd_proxy_ec50": lambda measurement_name, assay_family: assay_family == "KD_PROXY_EC50",
            "binding_ic50": lambda measurement_name, assay_family: assay_family == "IC50",
            "binding_ec50": lambda measurement_name, assay_family: assay_family == "EC50",
            "binding_unknown": lambda measurement_name, assay_family: measurement_name == "unknown",
        }

        for task_name, matcher in task_matchers.items():
            values: List[float] = []
            mask_values: List[float] = []
            qual_values: List[int] = []
            has_any = False

            for sample in samples:
                raw = sample.bind_value
                measurement_name = self._normalize_binding_measurement(
                    sample.bind_measurement_type
                )
                assay_family = self._categorize_binding_assay_type(
                    sample.binding_assay_type or sample.bind_measurement_type
                )
                if raw is None or not matcher(measurement_name, assay_family):
                    values.append(0.0)
                    mask_values.append(0.0)
                    qual_values.append(0)
                    continue

                values.append(float(raw))
                mask_values.append(1.0)
                qual_values.append(int(sample.bind_qual))
                has_any = True

            if not has_any:
                continue

            targets[task_name] = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)
            masks[task_name] = torch.tensor(mask_values, dtype=torch.float32)
            quals[task_name] = torch.tensor(qual_values, dtype=torch.long).unsqueeze(-1)

        return targets, masks, quals

    def _categorize_binding_assay_type(self, assay_type: Optional[str]) -> str:
        token = self._norm_text(assay_type)
        if not token:
            return "unknown"
        if "kd (~ic50)" in token:
            return "KD_PROXY_IC50"
        if "kd (~ec50)" in token:
            return "KD_PROXY_EC50"
        if "ic50" in token or "inhibitory concentration" in token:
            return "IC50"
        if "ec50" in token or "effective concentration" in token:
            return "EC50"
        if "kd" in token or "dissociation constant" in token:
            return "KD"
        return "OTHER"

    def _categorize_binding_assay_method(self, assay_method: Optional[str]) -> str:
        token = self._norm_text(assay_method)
        if not token:
            return "unknown"
        mapping = {
            "purified mhc/competitive/radioactivity": "PURIFIED_COMPETITIVE_RADIOACTIVITY",
            "purified mhc/direct/fluorescence": "PURIFIED_DIRECT_FLUORESCENCE",
            "purified mhc/competitive/fluorescence": "PURIFIED_COMPETITIVE_FLUORESCENCE",
            "cellular mhc/competitive/fluorescence": "CELLULAR_COMPETITIVE_FLUORESCENCE",
            "cellular mhc/direct/fluorescence": "CELLULAR_DIRECT_FLUORESCENCE",
            "cellular mhc/competitive/radioactivity": "CELLULAR_COMPETITIVE_RADIOACTIVITY",
            "cellular mhc/t cell inhibition": "CELLULAR_TCELL_INHIBITION",
            "lysate mhc/direct/radioactivity": "LYSATE_DIRECT_RADIOACTIVITY",
            "purified mhc/direct/radioactivity": "PURIFIED_DIRECT_RADIOACTIVITY",
        }
        return mapping.get(token, "OTHER")

    def _factorize_binding_assay_method(
        self,
        assay_method: Optional[str],
    ) -> tuple[str, str, str]:
        token = self._norm_text(assay_method)
        if not token:
            return ("unknown", "unknown", "unknown")
        prep = "OTHER"
        geometry = "OTHER"
        readout = "OTHER"
        if "purified" in token:
            prep = "PURIFIED"
        elif "cellular" in token:
            prep = "CELLULAR"
        elif "lysate" in token:
            prep = "LYSATE"
        elif "binding assay" in token:
            prep = "BINDING_ASSAY"

        if "competitive" in token:
            geometry = "COMPETITIVE"
        elif "direct" in token:
            geometry = "DIRECT"
        elif "t cell inhibition" in token:
            geometry = "T_CELL_INHIBITION"

        if "radioactivity" in token:
            readout = "RADIOACTIVITY"
        elif "fluorescence" in token:
            readout = "FLUORESCENCE"

        return (prep, geometry, readout)

    def _collate_binding_context(
        self,
        samples: List[PrestoSample],
    ) -> Dict[str, torch.Tensor]:
        assay_type_idx: List[int] = []
        assay_method_idx: List[int] = []
        assay_prep_idx: List[int] = []
        assay_geometry_idx: List[int] = []
        assay_readout_idx: List[int] = []

        for sample in samples:
            assay_type_label = self._categorize_binding_assay_type(
                sample.binding_assay_type or sample.bind_measurement_type
            )
            assay_method_label = self._categorize_binding_assay_method(
                sample.binding_assay_method
            )
            assay_prep_label, assay_geometry_label, assay_readout_label = (
                self._factorize_binding_assay_method(sample.binding_assay_method)
            )
            assay_type_idx.append(
                BINDING_ASSAY_TYPE_TO_IDX.get(
                    assay_type_label,
                    BINDING_ASSAY_TYPE_TO_IDX["OTHER"],
                )
            )
            assay_method_idx.append(
                BINDING_ASSAY_METHOD_TO_IDX.get(
                    assay_method_label,
                    BINDING_ASSAY_METHOD_TO_IDX["OTHER"],
                )
            )
            assay_prep_idx.append(
                BINDING_ASSAY_PREP_TO_IDX.get(
                    assay_prep_label,
                    BINDING_ASSAY_PREP_TO_IDX["OTHER"],
                )
            )
            assay_geometry_idx.append(
                BINDING_ASSAY_GEOMETRY_TO_IDX.get(
                    assay_geometry_label,
                    BINDING_ASSAY_GEOMETRY_TO_IDX["OTHER"],
                )
            )
            assay_readout_idx.append(
                BINDING_ASSAY_READOUT_TO_IDX.get(
                    assay_readout_label,
                    BINDING_ASSAY_READOUT_TO_IDX["OTHER"],
                )
            )

        return {
            "assay_type_idx": torch.tensor(assay_type_idx, dtype=torch.long),
            "assay_method_idx": torch.tensor(assay_method_idx, dtype=torch.long),
            "assay_prep_idx": torch.tensor(assay_prep_idx, dtype=torch.long),
            "assay_geometry_idx": torch.tensor(assay_geometry_idx, dtype=torch.long),
            "assay_readout_idx": torch.tensor(assay_readout_idx, dtype=torch.long),
        }

    def _collate_tcr_evidence_targets(
        self,
        samples: List[PrestoSample],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        labels: List[float] = []
        masks: List[float] = []
        method_targets: List[List[float]] = []
        method_masks: List[List[float]] = []
        has_label = False
        has_method = False

        for sample in samples:
            raw = sample.tcr_evidence_label
            if raw is None:
                labels.append(0.0)
                masks.append(0.0)
            else:
                labels.append(float(raw))
                masks.append(1.0)
                has_label = True

            bins = {str(token).strip() for token in (sample.tcr_evidence_method_bins or ()) if str(token).strip()}
            if bins:
                method_targets.append(
                    [1.0 if name in bins else 0.0 for name in TCR_EVIDENCE_METHOD_BINS]
                )
                method_masks.append([1.0] * len(TCR_EVIDENCE_METHOD_BINS))
                has_method = True
            else:
                method_targets.append([0.0] * len(TCR_EVIDENCE_METHOD_BINS))
                method_masks.append([0.0] * len(TCR_EVIDENCE_METHOD_BINS))

        targets: Dict[str, torch.Tensor] = {}
        target_masks: Dict[str, torch.Tensor] = {}
        if has_label:
            targets["tcr_evidence"] = torch.tensor(labels, dtype=torch.float32)
            target_masks["tcr_evidence"] = torch.tensor(masks, dtype=torch.float32)
        if has_method:
            targets["tcr_evidence_method"] = torch.tensor(method_targets, dtype=torch.float32)
            target_masks["tcr_evidence_method"] = torch.tensor(method_masks, dtype=torch.float32)
        return targets, target_masks

    @staticmethod
    def _norm_text(value: Optional[str]) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    def _categorize_tcell_assay_method(
        self,
        assay_method: Optional[str],
        assay_readout: Optional[str],
    ) -> str:
        method = self._norm_text(assay_method)
        readout = self._norm_text(assay_readout)
        combined = f"{method} {readout}".strip()
        if not combined:
            return "unknown"
        if "elispot" in combined:
            return "ELISPOT"
        if "ics" in combined or "intracellular cytokine" in combined:
            return "ICS"
        if "multimer" in combined or "tetramer" in combined:
            return "MULTIMER"
        if "elisa" in combined:
            return "ELISA"
        if "51 chromium" in combined or "cytotoxic" in combined:
            return "CYTOTOXICITY_ASSAY"
        if "3h-thymidine" in combined or "prolifer" in combined:
            return "PROLIFERATION_ASSAY"
        if "in vitro assay" in combined:
            return "IN_VITRO_ASSAY"
        if "in vivo assay" in combined:
            return "IN_VIVO_ASSAY"
        if "bioassay" in combined or "biological activity" in combined:
            return "BIOASSAY"
        return "OTHER"

    def _categorize_tcell_assay_readout(
        self,
        assay_readout: Optional[str],
        assay_method: Optional[str],
    ) -> str:
        readout = self._norm_text(assay_readout)
        method = self._norm_text(assay_method)
        combined = f"{method} {readout}".strip()
        if not combined:
            return "unknown"
        if "ifng" in combined or "ifn-g" in combined or "ifnγ" in combined:
            return "IFNG"
        if "tnfa" in combined or "tnf-a" in combined or "tnfα" in combined:
            return "TNFA"
        if "il-2" in combined or "il2" in combined:
            return "IL2"
        if "il-4" in combined or "il4" in combined:
            return "IL4"
        if "il-5" in combined or "il5" in combined:
            return "IL5"
        if "il-10" in combined or "il10" in combined:
            return "IL10"
        if "gm-csf" in combined or "gmcsf" in combined:
            return "GMCSF"
        if "cytotoxic" in combined:
            return "CYTOTOXICITY"
        if "prolifer" in combined:
            return "PROLIFERATION"
        if "activation" in combined or "degranulation" in combined:
            return "ACTIVATION"
        if "dissociation constant kd" in combined or " kd" in f" {combined} ":
            return "KD"
        if "multimer" in combined or "tetramer" in combined:
            return "MULTIMER_BINDING"
        if "binding" in combined:
            return "QUAL_BINDING"
        return "OTHER"

    def _categorize_tcell_apc_type(self, apc_name: Optional[str]) -> str:
        name = self._norm_text(apc_name)
        if not name:
            return "unknown"
        if "dendritic" in name:
            return "DENDRITIC"
        if "pbmc" in name:
            return "PBMC"
        if "splenocyte" in name:
            return "SPLENOCYTE"
        if "t2 cell" in name:
            return "T2_B_CELL"
        if "b-lcl" in name or "ebv transformed" in name or "hmy2.c1r" in name:
            return "B_LCL"
        if "b cell" in name or "plasma cell" in name or "lymphoblast" in name:
            return "B_CELL"
        if "t cell" in name:
            return "T_CELL"
        return "OTHER"

    def _categorize_tcell_culture_context(
        self,
        effector_culture: Optional[str],
        apc_culture: Optional[str],
    ) -> str:
        combined = f"{self._norm_text(effector_culture)} {self._norm_text(apc_culture)}".strip()
        if not combined:
            return "unknown"
        if "direct ex vivo" in combined:
            return "DIRECT_EX_VIVO"
        if "short term restim" in combined or "restim" in combined:
            return "SHORT_RESTIM"
        if "in vivo" in combined:
            return "IN_VIVO"
        if "engineered" in combined:
            return "ENGINEERED"
        if "cell line / clone" in combined or "cell line/clone" in combined:
            return "CELL_LINE_CLONE"
        if "non-antigen specific activation" in combined:
            return "NON_SPECIFIC_ACTIVATION"
        if "in vitro" in combined:
            return "IN_VITRO"
        return "OTHER"

    def _categorize_tcell_stim_context(
        self,
        effector_culture: Optional[str],
        apc_culture: Optional[str],
        in_vitro_process: Optional[str],
        in_vitro_responder: Optional[str],
        in_vitro_stimulator: Optional[str],
    ) -> str:
        culture = self._norm_text(f"{effector_culture} {apc_culture}")
        has_in_vitro_fields = any(
            self._norm_text(v)
            for v in (in_vitro_process, in_vitro_responder, in_vitro_stimulator)
        )
        if not culture and not has_in_vitro_fields:
            return "unknown"
        if "direct ex vivo" in culture:
            return "EX_VIVO"
        if "engineered" in culture:
            return "ENGINEERED"
        if "in vivo" in culture:
            return "IN_VIVO"
        if has_in_vitro_fields or "in vitro" in culture or "restim" in culture:
            return "IN_VITRO_STIM"
        return "OTHER"

    def _categorize_tcell_peptide_format(
        self,
        peptide: Optional[str],
        explicit_format: Optional[str],
        assay_method: Optional[str],
        assay_readout: Optional[str],
    ) -> str:
        explicit = self._norm_text(explicit_format)
        if explicit:
            if "minimal" in explicit or "epitope" in explicit:
                return "MINIMAL_EPITOPE"
            if "long" in explicit:
                return "LONG_PEPTIDE"
            if "pool" in explicit or "mix" in explicit:
                return "PEPTIDE_POOL"
            if "protein" in explicit or "whole" in explicit:
                return "WHOLE_PROTEIN"
            return "OTHER"

        method = self._norm_text(assay_method)
        readout = self._norm_text(assay_readout)
        combined = f"{method} {readout}".strip()
        if "pool" in combined:
            return "PEPTIDE_POOL"
        if "whole protein" in combined:
            return "WHOLE_PROTEIN"

        pep = (peptide or "").strip()
        if not pep:
            return "unknown"
        if any(sep in pep for sep in (";", ",", "|", "/")):
            return "PEPTIDE_POOL"
        pep_len = len(pep)
        if pep_len <= 15:
            return "MINIMAL_EPITOPE"
        return "LONG_PEPTIDE"

    def _infer_tcell_culture_duration_hours(
        self,
        explicit_duration_hours: Optional[float],
        effector_culture: Optional[str],
        apc_culture: Optional[str],
        in_vitro_process: Optional[str],
        in_vitro_responder: Optional[str],
        in_vitro_stimulator: Optional[str],
    ) -> Optional[float]:
        if explicit_duration_hours is not None:
            try:
                value = float(explicit_duration_hours)
                if value > 0.0:
                    return value
            except (TypeError, ValueError):
                pass

        combined = " ".join([
            self._norm_text(effector_culture),
            self._norm_text(apc_culture),
            self._norm_text(in_vitro_process),
            self._norm_text(in_vitro_responder),
            self._norm_text(in_vitro_stimulator),
        ]).strip()
        if not combined:
            return None

        match = re.search(
            r"(\\d+(?:\\.\\d+)?)\\s*(h|hr|hrs|hour|hours|d|day|days|wk|wks|week|weeks)",
            combined,
        )
        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2)
        if unit in {"h", "hr", "hrs", "hour", "hours"}:
            return value
        if unit in {"d", "day", "days"}:
            return value * 24.0
        if unit in {"wk", "wks", "week", "weeks"}:
            return value * 24.0 * 7.0
        return None

    def _collate_tcell_context(
        self,
        samples: List[PrestoSample],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        context_ids: Dict[str, List[int]] = {
            "assay_method_idx": [],
            "assay_readout_idx": [],
            "apc_type_idx": [],
            "culture_context_idx": [],
            "stim_context_idx": [],
            "peptide_format_idx": [],
            "culture_duration_hours": [],
        }
        targets: Dict[str, List[int]] = {
            "tcell_assay_method": [],
            "tcell_assay_readout": [],
            "tcell_apc_type": [],
            "tcell_culture_context": [],
            "tcell_stim_context": [],
            "tcell_peptide_format": [],
        }
        masks: Dict[str, List[float]] = {
            "tcell_assay_method": [],
            "tcell_assay_readout": [],
            "tcell_apc_type": [],
            "tcell_culture_context": [],
            "tcell_stim_context": [],
            "tcell_peptide_format": [],
            "tcell_culture_duration": [],
        }

        for sample in samples:
            method = self._categorize_tcell_assay_method(
                sample.tcell_assay_method,
                sample.tcell_assay_readout,
            )
            readout = self._categorize_tcell_assay_readout(
                sample.tcell_assay_readout,
                sample.tcell_assay_method,
            )
            apc_type = self._categorize_tcell_apc_type(sample.tcell_apc_name)
            culture = self._categorize_tcell_culture_context(
                sample.tcell_effector_culture,
                sample.tcell_apc_culture,
            )
            stim = self._categorize_tcell_stim_context(
                sample.tcell_effector_culture,
                sample.tcell_apc_culture,
                sample.tcell_in_vitro_process,
                sample.tcell_in_vitro_responder,
                sample.tcell_in_vitro_stimulator,
            )
            pep_format = self._categorize_tcell_peptide_format(
                peptide=sample.peptide,
                explicit_format=sample.tcell_peptide_format,
                assay_method=sample.tcell_assay_method,
                assay_readout=sample.tcell_assay_readout,
            )
            culture_duration_hours = self._infer_tcell_culture_duration_hours(
                explicit_duration_hours=sample.tcell_culture_duration_hours,
                effector_culture=sample.tcell_effector_culture,
                apc_culture=sample.tcell_apc_culture,
                in_vitro_process=sample.tcell_in_vitro_process,
                in_vitro_responder=sample.tcell_in_vitro_responder,
                in_vitro_stimulator=sample.tcell_in_vitro_stimulator,
            )

            method_idx = TCELL_ASSAY_METHOD_TO_IDX.get(method, TCELL_ASSAY_METHOD_TO_IDX["OTHER"])
            readout_idx = TCELL_ASSAY_READOUT_TO_IDX.get(readout, TCELL_ASSAY_READOUT_TO_IDX["OTHER"])
            apc_idx = TCELL_APC_TYPE_TO_IDX.get(apc_type, TCELL_APC_TYPE_TO_IDX["OTHER"])
            culture_idx = TCELL_CULTURE_CONTEXT_TO_IDX.get(
                culture, TCELL_CULTURE_CONTEXT_TO_IDX["OTHER"]
            )
            stim_idx = TCELL_STIM_CONTEXT_TO_IDX.get(
                stim, TCELL_STIM_CONTEXT_TO_IDX["OTHER"]
            )
            pep_format_idx = TCELL_PEPTIDE_FORMAT_TO_IDX.get(
                pep_format,
                TCELL_PEPTIDE_FORMAT_TO_IDX["OTHER"],
            )

            context_ids["assay_method_idx"].append(method_idx)
            context_ids["assay_readout_idx"].append(readout_idx)
            context_ids["apc_type_idx"].append(apc_idx)
            context_ids["culture_context_idx"].append(culture_idx)
            context_ids["stim_context_idx"].append(stim_idx)
            context_ids["peptide_format_idx"].append(pep_format_idx)
            context_ids["culture_duration_hours"].append(
                float(culture_duration_hours) if culture_duration_hours is not None else 0.0
            )

            targets["tcell_assay_method"].append(method_idx)
            targets["tcell_assay_readout"].append(readout_idx)
            targets["tcell_apc_type"].append(apc_idx)
            targets["tcell_culture_context"].append(culture_idx)
            targets["tcell_stim_context"].append(stim_idx)
            targets["tcell_peptide_format"].append(pep_format_idx)

            has_tcell = sample.tcell_label is not None and not sample.use_tcell_pathway_mil
            masks["tcell_assay_method"].append(
                1.0 if has_tcell and method_idx != 0 else 0.0
            )
            masks["tcell_assay_readout"].append(
                1.0 if has_tcell and readout_idx != 0 else 0.0
            )
            masks["tcell_apc_type"].append(
                1.0 if has_tcell and apc_idx != 0 else 0.0
            )
            masks["tcell_culture_context"].append(
                1.0 if has_tcell and culture_idx != 0 else 0.0
            )
            masks["tcell_stim_context"].append(
                1.0 if has_tcell and stim_idx != 0 else 0.0
            )
            masks["tcell_peptide_format"].append(
                1.0 if has_tcell and pep_format_idx != 0 else 0.0
            )
            masks["tcell_culture_duration"].append(
                1.0 if has_tcell and culture_duration_hours is not None else 0.0
            )

        context_tensors = {
            key: torch.tensor(values, dtype=torch.long)
            for key, values in context_ids.items()
            if key != "culture_duration_hours"
        }
        context_tensors["culture_duration_hours"] = torch.tensor(
            context_ids["culture_duration_hours"],
            dtype=torch.float32,
        )
        target_tensors = {
            key: torch.tensor(values, dtype=torch.long)
            for key, values in targets.items()
        }
        mask_tensors = {
            key: torch.tensor(values, dtype=torch.float32)
            for key, values in masks.items()
        }
        return context_tensors, target_tensors, mask_tensors

    def __call__(self, samples: List[PrestoSample]) -> PrestoBatch:
        """Collate samples into a batch.

        Args:
            samples: List of PrestoSample objects

        Returns:
            PrestoBatch
        """
        n = len(samples)

        # Tokenize sequences
        pep_tok, pep_lengths = self.tokenizer.batch_encode(
            [s.peptide for s in samples],
            max_len=self.max_pep_len,
            pad=True,
            return_lengths=True,
        )
        mhc_a_values = [self._sanitize_optional_sequence(s.mhc_a) for s in samples]
        mhc_b_values = [self._sanitize_optional_sequence(s.mhc_b) for s in samples]
        mhc_a_tok = self.tokenizer.batch_encode(
            mhc_a_values,
            max_len=self.max_mhc_len,
            pad=True,
        )
        mhc_b_tok = self.tokenizer.batch_encode(
            mhc_b_values,
            max_len=self.max_mhc_len,
            pad=True,
        )
        mhc_class = [s.mhc_class for s in samples]

        # Optional flanks
        flank_n_tok = None
        flank_c_tok = None
        flank_n_values = [self._sanitize_optional_sequence(s.flank_n) for s in samples]
        flank_c_values = [self._sanitize_optional_sequence(s.flank_c) for s in samples]
        if any(flank_n_values):
            flank_n_tok = self.tokenizer.batch_encode(
                flank_n_values,
                max_len=self.max_flank_len,
                pad=True,
            )
        if any(flank_c_values):
            flank_c_tok = self.tokenizer.batch_encode(
                flank_c_values,
                max_len=self.max_flank_len,
                pad=True,
            )

        targets, target_masks = self._collate_targets(samples)
        fixed_metadata = self._collate_fixed_metadata(samples)
        binding_pair_indices = self._collate_binding_pair_indices(samples)
        binding_targets, binding_masks, binding_quals = self._collate_binding_measurement_targets(
            samples
        )
        binding_context = self._collate_binding_context(samples)
        targets.update(binding_targets)
        target_masks.update(binding_masks)
        tcr_evidence_targets, tcr_evidence_masks = self._collate_tcr_evidence_targets(samples)
        targets.update(tcr_evidence_targets)
        target_masks.update(tcr_evidence_masks)
        tcell_context, tcell_context_targets, tcell_context_masks = self._collate_tcell_context(
            samples
        )
        targets.update(tcell_context_targets)
        target_masks.update(tcell_context_masks)

        # Peptide species of origin (categorical → targets dict)
        so_labels: List[int] = []
        so_mask: List[float] = []
        has_so = False
        for s in samples:
            if s.species_of_origin and s.species_of_origin in ORGANISM_TO_IDX:
                so_labels.append(ORGANISM_TO_IDX[s.species_of_origin])
                so_mask.append(1.0)
                has_so = True
            else:
                so_labels.append(0)
                so_mask.append(0.0)
        if has_so:
            targets["species_of_origin"] = torch.tensor(so_labels, dtype=torch.long)
            target_masks["species_of_origin"] = torch.tensor(so_mask, dtype=torch.float32)

        # Optional core-start supervision (class-II core pointer target).
        core_start_labels: List[int] = []
        core_start_mask: List[float] = []
        has_core_start = False
        for idx, sample in enumerate(samples):
            raw = sample.core_start
            if raw is None:
                core_start_labels.append(0)
                core_start_mask.append(0.0)
                continue
            start_idx = int(raw)
            pep_len = int(pep_lengths[idx].item())
            if 0 <= start_idx < pep_len:
                core_start_labels.append(start_idx)
                core_start_mask.append(1.0)
                has_core_start = True
            else:
                core_start_labels.append(0)
                core_start_mask.append(0.0)
        if has_core_start:
            targets["core_start"] = torch.tensor(core_start_labels, dtype=torch.long)
            target_masks["core_start"] = torch.tensor(core_start_mask, dtype=torch.float32)

        target_quals: Dict[str, torch.Tensor] = {}
        if "binding" in targets:
            target_quals["binding"] = torch.tensor(
                [s.bind_qual for s in samples],
                dtype=torch.long,
            ).unsqueeze(-1)
        target_quals.update(binding_quals)

        # Backward-compatible aliases while migrating callers to dict-based access.
        bind_target = targets.get("binding")
        bind_qual = target_quals.get("binding")
        bind_mask = target_masks.get("binding")
        kon_target = targets.get("kon")
        kon_mask = target_masks.get("kon")
        koff_target = targets.get("koff")
        koff_mask = target_masks.get("koff")
        t_half_target = targets.get("t_half")
        t_half_mask = target_masks.get("t_half")
        tm_target = targets.get("tm")
        tm_mask = target_masks.get("tm")
        tcell_label = targets.get("tcell")
        tcell_mask = target_masks.get("tcell")
        elution_label = targets.get("elution")
        elution_mask = target_masks.get("elution")
        processing_label = targets.get("processing")
        processing_mask = target_masks.get("processing")
        tcr_evidence_target = targets.get("tcr_evidence")
        tcr_evidence_mask = target_masks.get("tcr_evidence")
        tcr_evidence_method_target = targets.get("tcr_evidence_method")
        tcr_evidence_method_mask = target_masks.get("tcr_evidence_method")

        # Optional multi-allele MIL bag tensors for elution/presentation/MS.
        mil_peptides: List[str] = []
        mil_mhc_as: List[str] = []
        mil_mhc_bs: List[str] = []
        mil_mhc_classes: List[str] = []
        mil_species: List[str] = []
        mil_flank_ns: List[str] = []
        mil_flank_cs: List[str] = []
        mil_instance_to_bag: List[int] = []
        mil_bag_labels: List[float] = []
        mil_bag_sample_ids: List[str] = []
        tcell_mil_peptides: List[str] = []
        tcell_mil_mhc_as: List[str] = []
        tcell_mil_mhc_bs: List[str] = []
        tcell_mil_mhc_classes: List[str] = []
        tcell_mil_species: List[str] = []
        tcell_mil_flank_ns: List[str] = []
        tcell_mil_flank_cs: List[str] = []
        tcell_mil_instance_to_bag: List[int] = []
        tcell_mil_bag_labels: List[float] = []
        tcell_mil_bag_sample_ids: List[str] = []
        tcell_mil_source_samples: List[PrestoSample] = []

        for sample in samples:
            if sample.elution_label is None:
                continue

            n_instances = max(
                len(sample.mil_mhc_a_list or []),
                len(sample.mil_mhc_b_list or []),
                len(sample.mil_mhc_class_list or []),
                len(sample.mil_species_list or []),
                1,
            )

            mhc_a_list = self._expand_with_fallback(
                sample.mil_mhc_a_list,
                n_instances=n_instances,
                fallback=sample.mhc_a or "",
            )
            mhc_b_list = self._expand_with_fallback(
                sample.mil_mhc_b_list,
                n_instances=n_instances,
                fallback=sample.mhc_b or "",
            )
            mhc_class_list = self._expand_with_fallback(
                sample.mil_mhc_class_list,
                n_instances=n_instances,
                fallback=sample.mhc_class or "I",
            )
            species_list = self._expand_with_fallback(
                sample.mil_species_list,
                n_instances=n_instances,
                fallback=sample.species or "",
            )

            grouped_indices = self._split_indices_by_mhc_class(mhc_class_list)
            for class_label, indices in grouped_indices:
                bag_index = len(mil_bag_labels)
                mil_bag_labels.append(float(sample.elution_label))
                bag_sample_id = sample.sample_id
                if len(grouped_indices) > 1 and class_label:
                    bag_sample_id = f"{sample.sample_id}:{class_label}"
                mil_bag_sample_ids.append(bag_sample_id)

                for i in indices:
                    mil_peptides.append(sample.peptide)
                    mil_mhc_as.append(self._sanitize_optional_sequence(mhc_a_list[i]))
                    mil_mhc_bs.append(self._sanitize_optional_sequence(mhc_b_list[i]))
                    mil_mhc_classes.append(mhc_class_list[i])
                    mil_species.append(species_list[i])
                    mil_flank_ns.append(self._sanitize_optional_sequence(sample.flank_n))
                    mil_flank_cs.append(self._sanitize_optional_sequence(sample.flank_c))
                    mil_instance_to_bag.append(bag_index)

        for sample in samples:
            if sample.tcell_label is None or not sample.use_tcell_pathway_mil:
                continue

            n_instances = max(
                len(sample.tcell_mil_mhc_a_list or []),
                len(sample.tcell_mil_mhc_b_list or []),
                len(sample.tcell_mil_mhc_class_list or []),
                len(sample.tcell_mil_species_list or []),
                1,
            )
            mhc_a_list = self._expand_with_fallback(
                sample.tcell_mil_mhc_a_list,
                n_instances=n_instances,
                fallback=sample.mhc_a or "",
            )
            mhc_b_list = self._expand_with_fallback(
                sample.tcell_mil_mhc_b_list,
                n_instances=n_instances,
                fallback=sample.mhc_b or "",
            )
            mhc_class_list = self._expand_with_fallback(
                sample.tcell_mil_mhc_class_list,
                n_instances=n_instances,
                fallback=sample.mhc_class or "",
            )
            species_list = self._expand_with_fallback(
                sample.tcell_mil_species_list,
                n_instances=n_instances,
                fallback=sample.species or "",
            )

            bag_index = len(tcell_mil_bag_labels)
            tcell_mil_bag_labels.append(float(sample.tcell_label))
            tcell_mil_bag_sample_ids.append(sample.sample_id)

            for i in range(n_instances):
                tcell_mil_peptides.append(sample.peptide)
                tcell_mil_mhc_as.append(self._sanitize_optional_sequence(mhc_a_list[i]))
                tcell_mil_mhc_bs.append(self._sanitize_optional_sequence(mhc_b_list[i]))
                tcell_mil_mhc_classes.append(mhc_class_list[i])
                tcell_mil_species.append(species_list[i])
                tcell_mil_flank_ns.append(self._sanitize_optional_sequence(sample.flank_n))
                tcell_mil_flank_cs.append(self._sanitize_optional_sequence(sample.flank_c))
                tcell_mil_instance_to_bag.append(bag_index)
                tcell_mil_source_samples.append(sample)

        mil_tensors = self._materialize_mil_tensors(
            peptides=mil_peptides,
            mhc_as=mil_mhc_as,
            mhc_bs=mil_mhc_bs,
            mhc_classes=mil_mhc_classes,
            species=mil_species,
            flank_ns=mil_flank_ns,
            flank_cs=mil_flank_cs,
            instance_to_bag=mil_instance_to_bag,
            bag_labels=mil_bag_labels,
            bag_sample_ids=mil_bag_sample_ids,
        )
        tcell_mil_tensors = self._materialize_mil_tensors(
            peptides=tcell_mil_peptides,
            mhc_as=tcell_mil_mhc_as,
            mhc_bs=tcell_mil_mhc_bs,
            mhc_classes=tcell_mil_mhc_classes,
            species=tcell_mil_species,
            flank_ns=tcell_mil_flank_ns,
            flank_cs=tcell_mil_flank_cs,
            instance_to_bag=tcell_mil_instance_to_bag,
            bag_labels=tcell_mil_bag_labels,
            bag_sample_ids=tcell_mil_bag_sample_ids,
        )
        if tcell_mil_source_samples:
            tcell_mil_context, _, _ = self._collate_tcell_context(tcell_mil_source_samples)
        else:
            tcell_mil_context = {}

        return PrestoBatch(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class=mhc_class,
            flank_n_tok=flank_n_tok,
            flank_c_tok=flank_c_tok,
            bind_target=bind_target,
            bind_qual=bind_qual,
            kon_target=kon_target,
            koff_target=koff_target,
            t_half_target=t_half_target,
            tm_target=tm_target,
            tcell_label=tcell_label,
            elution_label=elution_label,
            processing_label=processing_label,
            tcr_evidence_target=tcr_evidence_target,
            tcr_evidence_method_target=tcr_evidence_method_target,
            bind_mask=bind_mask,
            kon_mask=kon_mask,
            koff_mask=koff_mask,
            t_half_mask=t_half_mask,
            tm_mask=tm_mask,
            tcell_mask=tcell_mask,
            elution_mask=elution_mask,
            processing_mask=processing_mask,
            tcr_evidence_mask=tcr_evidence_mask,
            tcr_evidence_method_mask=tcr_evidence_method_mask,
            pep_lengths=pep_lengths,
            dataset_index=fixed_metadata["dataset_index"],
            peptide_id=fixed_metadata["peptide_id"],
            allele_id=fixed_metadata["allele_id"],
            bind_target_log10=fixed_metadata["bind_target_log10"],
            same_peptide_diff_allele_pairs=binding_pair_indices["same_peptide_diff_allele_pairs"],
            same_allele_diff_peptide_pairs=binding_pair_indices["same_allele_diff_peptide_pairs"],
            processing_species=[s.species for s in samples],
            primary_alleles=[s.primary_allele or "" for s in samples],
            sample_ids=[s.sample_id for s in samples],
            targets=targets,
            target_masks=target_masks,
            target_quals=target_quals,
            binding_context=binding_context,
            tcell_context=tcell_context,
            tcell_context_masks=tcell_context_masks,
            tcell_mil_context=tcell_mil_context,
            mil_pep_tok=mil_tensors["pep_tok"],
            mil_mhc_a_tok=mil_tensors["mhc_a_tok"],
            mil_mhc_b_tok=mil_tensors["mhc_b_tok"],
            mil_mhc_class=mil_tensors["mhc_class"],
            mil_species=mil_tensors["species"],
            mil_flank_n_tok=mil_tensors["flank_n_tok"],
            mil_flank_c_tok=mil_tensors["flank_c_tok"],
            mil_instance_to_bag=mil_tensors["instance_to_bag"],
            mil_bag_label=mil_tensors["bag_label"],
            mil_bag_sample_ids=mil_tensors["bag_sample_ids"],
            tcell_mil_pep_tok=tcell_mil_tensors["pep_tok"],
            tcell_mil_mhc_a_tok=tcell_mil_tensors["mhc_a_tok"],
            tcell_mil_mhc_b_tok=tcell_mil_tensors["mhc_b_tok"],
            tcell_mil_mhc_class=tcell_mil_tensors["mhc_class"],
            tcell_mil_species=tcell_mil_tensors["species"],
            tcell_mil_flank_n_tok=tcell_mil_tensors["flank_n_tok"],
            tcell_mil_flank_c_tok=tcell_mil_tensors["flank_c_tok"],
            tcell_mil_instance_to_bag=tcell_mil_tensors["instance_to_bag"],
            tcell_mil_bag_label=tcell_mil_tensors["bag_label"],
            tcell_mil_bag_sample_ids=tcell_mil_tensors["bag_sample_ids"],
        )


def collate_dict_batch(batch: List[Dict[str, Any]], tokenizer: Tokenizer = None) -> Dict[str, Any]:
    """Collate a batch of dict samples (simpler interface).

    Args:
        batch: List of dicts with keys like "peptide", "mhc_a", etc.
        tokenizer: Tokenizer to use

    Returns:
        Dict with tokenized tensors
    """
    tokenizer = tokenizer or Tokenizer()

    result = {}

    # Required fields
    if "peptide" in batch[0]:
        result["pep_tok"] = tokenizer.batch_encode(
            [b["peptide"] for b in batch], max_len=50, pad=True
        )

    if "mhc_a" in batch[0]:
        result["mhc_a_tok"] = tokenizer.batch_encode(
            [PrestoCollator._sanitize_optional_sequence(b.get("mhc_a", "")) for b in batch],
            max_len=400,
            pad=True,
        )

    if "mhc_b" in batch[0]:
        result["mhc_b_tok"] = tokenizer.batch_encode(
            [PrestoCollator._sanitize_optional_sequence(b.get("mhc_b", "")) for b in batch],
            max_len=200,
            pad=True,
        )

    result["mhc_class"] = [b.get("mhc_class", "I") for b in batch]

    targets: Dict[str, torch.Tensor] = {}
    target_masks: Dict[str, torch.Tensor] = {}
    for spec in TARGET_SPECS:
        values: List[float] = []
        masks: List[float] = []
        has_any = False
        for row in batch:
            raw = row.get(spec.sample_field)
            if raw is None:
                values.append(0.0)
                masks.append(0.0)
                continue
            value = spec.transform(raw) if spec.transform is not None else float(raw)
            values.append(float(value))
            masks.append(1.0)
            has_any = True

        if not has_any:
            continue

        target_tensor = torch.tensor(values, dtype=torch.float32)
        if spec.unsqueeze_last:
            target_tensor = target_tensor.unsqueeze(-1)
        mask_tensor = torch.tensor(masks, dtype=torch.float32)

        result[spec.target_field] = target_tensor
        result[spec.mask_field] = mask_tensor
        targets[spec.task_name] = target_tensor
        target_masks[spec.task_name] = mask_tensor

    target_quals: Dict[str, torch.Tensor] = {}
    if "binding" in targets:
        bind_qual = torch.tensor(
            [[int(row.get("bind_qual", 0))] for row in batch],
            dtype=torch.long,
        )
        result["bind_qual"] = bind_qual
        target_quals["binding"] = bind_qual

    result["targets"] = targets
    result["target_masks"] = target_masks
    result["target_quals"] = target_quals

    return result
