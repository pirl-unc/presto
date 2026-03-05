"""Data collation utilities for Presto.

Handles batching of variable-length sequences with proper padding.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from .tokenizer import Tokenizer
from .vocab import (
    CHAIN_TO_IDX,
    CELL_TO_IDX,
    FOREIGN_CATEGORIES,
    ORGANISM_TO_IDX,
    SPECIES_TO_IDX,
    TCELL_APC_TYPE_TO_IDX,
    TCELL_ASSAY_METHOD_TO_IDX,
    TCELL_ASSAY_READOUT_TO_IDX,
    TCELL_CULTURE_CONTEXT_TO_IDX,
    TCELL_PEPTIDE_FORMAT_TO_IDX,
    TCELL_STIM_CONTEXT_TO_IDX,
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
    mhc_a: str = ""  # Alpha chain sequence or allele name
    mhc_b: str = ""  # Beta chain / beta2m
    mhc_class: str = "I"

    # TCR (optional)
    tcr_a: Optional[str] = None
    tcr_b: Optional[str] = None

    # Labels (optional, for training)
    # Binding
    bind_value: Optional[float] = None  # nM
    bind_qual: int = 0  # -1=<, 0==, 1=>
    bind_measurement_type: Optional[str] = None  # KD / IC50 / EC50 / unknown

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

    # Processing
    processing_label: Optional[float] = None

    # Peptide source organism (unified 12-class taxonomy)
    species_of_origin: Optional[str] = None
    foreignness_label: Optional[float] = None

    # Chain classification
    chain_type: Optional[str] = None
    species: Optional[str] = None
    phenotype: Optional[str] = None

    # Metadata
    sample_source: Optional[str] = None
    assay_group: Optional[str] = None
    label_bucket: Optional[str] = None
    primary_allele: Optional[str] = None
    synthetic_kind: Optional[str] = None
    sample_id: str = ""


@dataclass
class PrestoBatch:
    """A collated batch of samples."""
    # Tokenized sequences
    pep_tok: torch.Tensor
    mhc_a_tok: torch.Tensor
    mhc_b_tok: torch.Tensor
    mhc_class: List[str]

    # Optional sequences
    flank_n_tok: Optional[torch.Tensor] = None
    flank_c_tok: Optional[torch.Tensor] = None
    tcr_a_tok: Optional[torch.Tensor] = None
    tcr_b_tok: Optional[torch.Tensor] = None

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

    # Chain attribute labels (for chain-species-phenotype auxiliary supervision)
    chain_species_label: Optional[torch.Tensor] = None
    chain_type_label: Optional[torch.Tensor] = None
    chain_phenotype_label: Optional[torch.Tensor] = None

    # Masks for which samples have which labels
    bind_mask: Optional[torch.Tensor] = None
    kon_mask: Optional[torch.Tensor] = None
    koff_mask: Optional[torch.Tensor] = None
    t_half_mask: Optional[torch.Tensor] = None
    tm_mask: Optional[torch.Tensor] = None
    tcell_mask: Optional[torch.Tensor] = None
    elution_mask: Optional[torch.Tensor] = None
    processing_mask: Optional[torch.Tensor] = None
    chain_species_mask: Optional[torch.Tensor] = None
    chain_type_mask: Optional[torch.Tensor] = None
    chain_phenotype_mask: Optional[torch.Tensor] = None

    # Optional T-cell assay context (categorical IDs + masks)
    tcell_context: Dict[str, torch.Tensor] = field(default_factory=dict)
    tcell_context_masks: Dict[str, torch.Tensor] = field(default_factory=dict)

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

    # Lengths for masking
    pep_lengths: Optional[torch.Tensor] = None

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
            tcr_a_tok=_move(self.tcr_a_tok),
            tcr_b_tok=_move(self.tcr_b_tok),
            bind_target=_move(self.bind_target),
            bind_qual=_move(self.bind_qual),
            kon_target=_move(self.kon_target),
            koff_target=_move(self.koff_target),
            t_half_target=_move(self.t_half_target),
            tm_target=_move(self.tm_target),
            tcell_label=_move(self.tcell_label),
            elution_label=_move(self.elution_label),
            processing_label=_move(self.processing_label),
            chain_species_label=_move(self.chain_species_label),
            chain_type_label=_move(self.chain_type_label),
            chain_phenotype_label=_move(self.chain_phenotype_label),
            bind_mask=_move(self.bind_mask),
            kon_mask=_move(self.kon_mask),
            koff_mask=_move(self.koff_mask),
            t_half_mask=_move(self.t_half_mask),
            tm_mask=_move(self.tm_mask),
            tcell_mask=_move(self.tcell_mask),
            elution_mask=_move(self.elution_mask),
            processing_mask=_move(self.processing_mask),
            chain_species_mask=_move(self.chain_species_mask),
            chain_type_mask=_move(self.chain_type_mask),
            chain_phenotype_mask=_move(self.chain_phenotype_mask),
            pep_lengths=_move(self.pep_lengths),
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
            tcell_context={
                name: _move(tensor) for name, tensor in self.tcell_context.items()
            },
            tcell_context_masks={
                name: _move(tensor) for name, tensor in self.tcell_context_masks.items()
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
        )


class PrestoCollator:
    """Collates PrestoSamples into PrestoBatches."""

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        max_pep_len: int = 50,
        max_mhc_len: int = 400,
        max_tcr_len: int = 200,
        max_flank_len: int = 30,
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
                mask.append(1.0)
                has_any = True

            if not has_any:
                continue

            tensor = torch.tensor(values, dtype=torch.float32)
            if spec.unsqueeze_last:
                tensor = tensor.unsqueeze(-1)
            targets[spec.task_name] = tensor
            masks[spec.task_name] = torch.tensor(mask, dtype=torch.float32)
        return targets, masks

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

        task_to_measurement = {
            "binding_kd": "kd",
            "binding_ic50": "ic50",
            "binding_ec50": "ec50",
            "binding_unknown": "unknown",
        }

        for task_name, measurement_name in task_to_measurement.items():
            values: List[float] = []
            mask_values: List[float] = []
            qual_values: List[int] = []
            has_any = False

            for sample in samples:
                raw = sample.bind_value
                normalized = self._normalize_binding_measurement(
                    sample.bind_measurement_type
                )
                if raw is None or normalized != measurement_name:
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

    @staticmethod
    def _normalize_species(species: Optional[str]) -> Optional[str]:
        if not species:
            return None
        s = str(species).strip().lower()
        if not s:
            return None
        if "human" in s or "homo sapiens" in s:
            return "human"
        if "mouse" in s or "mus musculus" in s:
            return "mouse"
        if "macaque" in s or "macaca" in s:
            return "macaque"
        if s in SPECIES_TO_IDX:
            return s
        return "other"

    def _collate_chain_attribute_labels(
        self, samples: List[PrestoSample]
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        labels: Dict[str, List[int]] = {
            "chain_species_label": [],
            "chain_type_label": [],
            "chain_phenotype_label": [],
        }
        masks: Dict[str, List[float]] = {
            "chain_species_mask": [],
            "chain_type_mask": [],
            "chain_phenotype_mask": [],
        }

        for sample in samples:
            has_chain_seq = bool(sample.tcr_a or sample.tcr_b)

            # Species label
            norm_species = self._normalize_species(sample.species) if has_chain_seq else None
            if norm_species is not None and norm_species in SPECIES_TO_IDX:
                labels["chain_species_label"].append(SPECIES_TO_IDX[norm_species])
                masks["chain_species_mask"].append(1.0)
            else:
                labels["chain_species_label"].append(0)
                masks["chain_species_mask"].append(0.0)

            # Chain type label
            chain_type = (sample.chain_type or "").strip().upper() if has_chain_seq else ""
            if chain_type in CHAIN_TO_IDX:
                labels["chain_type_label"].append(CHAIN_TO_IDX[chain_type])
                masks["chain_type_mask"].append(1.0)
            else:
                labels["chain_type_label"].append(0)
                masks["chain_type_mask"].append(0.0)

            # Phenotype label
            phenotype = (sample.phenotype or "").strip()
            if has_chain_seq and phenotype in CELL_TO_IDX:
                labels["chain_phenotype_label"].append(CELL_TO_IDX[phenotype])
                masks["chain_phenotype_mask"].append(1.0)
            else:
                labels["chain_phenotype_label"].append(0)
                masks["chain_phenotype_mask"].append(0.0)

        label_tensors = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in labels.items()
        }
        mask_tensors = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in masks.items()
        }
        return label_tensors, mask_tensors

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

            has_tcell = sample.tcell_label is not None
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
        mhc_a_tok = self.tokenizer.batch_encode(
            [s.mhc_a for s in samples],
            max_len=self.max_mhc_len,
            pad=True,
        )
        mhc_b_tok = self.tokenizer.batch_encode(
            [s.mhc_b for s in samples],
            max_len=self.max_mhc_len,
            pad=True,
        )
        mhc_class = [s.mhc_class for s in samples]

        # Optional flanks
        flank_n_tok = None
        flank_c_tok = None
        if any(s.flank_n for s in samples):
            flank_n_tok = self.tokenizer.batch_encode(
                [s.flank_n or "" for s in samples],
                max_len=self.max_flank_len,
                pad=True,
            )
        if any(s.flank_c for s in samples):
            flank_c_tok = self.tokenizer.batch_encode(
                [s.flank_c or "" for s in samples],
                max_len=self.max_flank_len,
                pad=True,
            )

        # Optional TCR
        tcr_a_tok = None
        tcr_b_tok = None
        if any(s.tcr_a for s in samples):
            tcr_a_tok = self.tokenizer.batch_encode(
                [s.tcr_a or "" for s in samples],
                max_len=self.max_tcr_len,
                pad=True,
            )
        if any(s.tcr_b for s in samples):
            tcr_b_tok = self.tokenizer.batch_encode(
                [s.tcr_b or "" for s in samples],
                max_len=self.max_tcr_len,
                pad=True,
            )

        targets, target_masks = self._collate_targets(samples)
        binding_targets, binding_masks, binding_quals = self._collate_binding_measurement_targets(
            samples
        )
        targets.update(binding_targets)
        target_masks.update(binding_masks)
        chain_labels, chain_masks = self._collate_chain_attribute_labels(samples)
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

            bag_index = len(mil_bag_labels)
            mil_bag_labels.append(float(sample.elution_label))
            mil_bag_sample_ids.append(sample.sample_id)

            for i in range(n_instances):
                mil_peptides.append(sample.peptide)
                mil_mhc_as.append(mhc_a_list[i])
                mil_mhc_bs.append(mhc_b_list[i])
                mil_mhc_classes.append(mhc_class_list[i])
                mil_species.append(species_list[i])
                mil_flank_ns.append(sample.flank_n or "")
                mil_flank_cs.append(sample.flank_c or "")
                mil_instance_to_bag.append(bag_index)

        mil_pep_tok = None
        mil_mhc_a_tok = None
        mil_mhc_b_tok = None
        mil_mhc_class = None
        mil_species_out = None
        mil_flank_n_tok = None
        mil_flank_c_tok = None
        mil_instance_to_bag_t = None
        mil_bag_label_t = None
        if mil_peptides:
            mil_pep_tok = self.tokenizer.batch_encode(
                mil_peptides,
                max_len=self.max_pep_len,
                pad=True,
            )
            mil_mhc_a_tok = self.tokenizer.batch_encode(
                mil_mhc_as,
                max_len=self.max_mhc_len,
                pad=True,
            )
            mil_mhc_b_tok = self.tokenizer.batch_encode(
                mil_mhc_bs,
                max_len=self.max_mhc_len,
                pad=True,
            )
            mil_mhc_class = mil_mhc_classes
            mil_species_out = mil_species
            if any(v for v in mil_flank_ns):
                mil_flank_n_tok = self.tokenizer.batch_encode(
                    mil_flank_ns,
                    max_len=self.max_flank_len,
                    pad=True,
                )
            if any(v for v in mil_flank_cs):
                mil_flank_c_tok = self.tokenizer.batch_encode(
                    mil_flank_cs,
                    max_len=self.max_flank_len,
                    pad=True,
                )
            mil_instance_to_bag_t = torch.tensor(
                mil_instance_to_bag,
                dtype=torch.long,
            )
            mil_bag_label_t = torch.tensor(
                mil_bag_labels,
                dtype=torch.float32,
            )

        return PrestoBatch(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class=mhc_class,
            flank_n_tok=flank_n_tok,
            flank_c_tok=flank_c_tok,
            tcr_a_tok=tcr_a_tok,
            tcr_b_tok=tcr_b_tok,
            bind_target=bind_target,
            bind_qual=bind_qual,
            kon_target=kon_target,
            koff_target=koff_target,
            t_half_target=t_half_target,
            tm_target=tm_target,
            tcell_label=tcell_label,
            elution_label=elution_label,
            processing_label=processing_label,
            chain_species_label=chain_labels["chain_species_label"],
            chain_type_label=chain_labels["chain_type_label"],
            chain_phenotype_label=chain_labels["chain_phenotype_label"],
            bind_mask=bind_mask,
            kon_mask=kon_mask,
            koff_mask=koff_mask,
            t_half_mask=t_half_mask,
            tm_mask=tm_mask,
            tcell_mask=tcell_mask,
            elution_mask=elution_mask,
            processing_mask=processing_mask,
            chain_species_mask=chain_masks["chain_species_mask"],
            chain_type_mask=chain_masks["chain_type_mask"],
            chain_phenotype_mask=chain_masks["chain_phenotype_mask"],
            pep_lengths=pep_lengths,
            processing_species=[s.species for s in samples],
            primary_alleles=[s.primary_allele or "" for s in samples],
            sample_ids=[s.sample_id for s in samples],
            targets=targets,
            target_masks=target_masks,
            target_quals=target_quals,
            tcell_context=tcell_context,
            tcell_context_masks=tcell_context_masks,
            mil_pep_tok=mil_pep_tok,
            mil_mhc_a_tok=mil_mhc_a_tok,
            mil_mhc_b_tok=mil_mhc_b_tok,
            mil_mhc_class=mil_mhc_class,
            mil_species=mil_species_out,
            mil_flank_n_tok=mil_flank_n_tok,
            mil_flank_c_tok=mil_flank_c_tok,
            mil_instance_to_bag=mil_instance_to_bag_t,
            mil_bag_label=mil_bag_label_t,
            mil_bag_sample_ids=mil_bag_sample_ids,
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
            [b.get("mhc_a", "") for b in batch], max_len=400, pad=True
        )

    if "mhc_b" in batch[0]:
        result["mhc_b_tok"] = tokenizer.batch_encode(
            [b.get("mhc_b", "") for b in batch], max_len=200, pad=True
        )

    result["mhc_class"] = [b.get("mhc_class", "I") for b in batch]

    # Optional TCR
    if any(b.get("tcr_a") for b in batch):
        result["tcr_a_tok"] = tokenizer.batch_encode(
            [b.get("tcr_a", "") for b in batch], max_len=200, pad=True
        )

    if any(b.get("tcr_b") for b in batch):
        result["tcr_b_tok"] = tokenizer.batch_encode(
            [b.get("tcr_b", "") for b in batch], max_len=200, pad=True
        )

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
