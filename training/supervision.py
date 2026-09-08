"""Executable row and panel supervision shared by training, census and export."""

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..models.affinity import DEFAULT_MAX_AFFINITY_NM, normalize_binding_target_log10
from ..data.allele_resolver import (
    PROCESSING_SPECIES_TO_IDX,
    PROCESSING_SPECIES_BUCKETS,
    infer_gene,
    normalize_processing_species_label,
)
from ..data.mhc_index import infer_fine_chain_type
from ..data.collate import TCR_EVIDENCE_METHOD_BINS
from ..data.vocab import (
    MHC_CHAIN_FINE_TO_IDX,
    ORGANISM_CATEGORIES,
    BINDING_ASSAY_TYPES,
    BINDING_ASSAY_PREP,
    BINDING_ASSAY_GEOMETRY,
    BINDING_ASSAY_READOUT,
    APM_PERTURBATIONS,
    PROCESSING_STIMULI,
)
from .losses import censor_aware_loss
from .mil import MIL_TASKS


@dataclass(frozen=True)
class TaskLossSpec:
    """Specification for a supervised training loss."""

    name: str
    target_key: str
    mask_key: str
    pred_paths: Tuple[Tuple[str, ...], ...]
    loss_type: str  # one of: "censor", "bce", "mse", "ce"
    target_attr: Optional[str] = None
    mask_attr: Optional[str] = None
    qual_key: Optional[str] = None
    qual_attr: Optional[str] = None
    target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    base_weight: float = 1.0
    # Metadata selects one response column; it is never a category target.
    selector_key: Optional[str] = None
    selector_context: str = "tcell_context"
    axis: str = ""
    columns: Tuple[str, ...] = ()
    component_names: Tuple[str, ...] = ()
    class_names: Tuple[str, ...] = ()
    loss_group: Optional[str] = None
    raw_unit: str = "response"
    target_unit: str = "response"


LOSS_TASK_SPECS: Tuple[TaskLossSpec, ...] = (
    TaskLossSpec(
        name="binding",
        target_key="binding",
        mask_key="binding",
        pred_paths=(("assays", "KD_nM"),),
        loss_type="censor",
        target_attr="bind_target",
        mask_attr="bind_mask",
        qual_key="binding",
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
        pred_paths=(("tcell_panel_logits", "assay_method"),),
        loss_type="bce",
        selector_key="assay_method_idx",
    ),
    TaskLossSpec(
        name="tcell_assay_readout",
        target_key="tcell_assay_readout",
        mask_key="tcell_assay_readout",
        pred_paths=(("tcell_panel_logits", "assay_readout"),),
        loss_type="bce",
        selector_key="assay_readout_idx",
    ),
    TaskLossSpec(
        name="tcell_apc_type",
        target_key="tcell_apc_type",
        mask_key="tcell_apc_type",
        pred_paths=(("tcell_panel_logits", "apc_type"),),
        loss_type="bce",
        selector_key="apc_type_idx",
    ),
    TaskLossSpec(
        name="tcell_culture_context",
        target_key="tcell_culture_context",
        mask_key="tcell_culture_context",
        pred_paths=(("tcell_panel_logits", "culture_context"),),
        loss_type="bce",
        selector_key="culture_context_idx",
    ),
    TaskLossSpec(
        name="tcell_stim_context",
        target_key="tcell_stim_context",
        mask_key="tcell_stim_context",
        pred_paths=(("tcell_panel_logits", "stim_context"),),
        loss_type="bce",
        selector_key="stim_context_idx",
    ),
    TaskLossSpec(
        name="tcell_peptide_format",
        target_key="tcell_peptide_format",
        mask_key="tcell_peptide_format",
        pred_paths=(("tcell_panel_logits", "peptide_format"),),
        loss_type="bce",
        selector_key="peptide_format_idx",
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
        # Censor-aware, not plain MSE. 51 half-life rows carry an inequality and
        # the qualifier was collected then ignored, so a ">2h" measurement was
        # trained as if it were exactly 2h. The t_half target transform is
        # monotone increasing (unlike the inverting affinity encoding), so the
        # censor codes carry through unchanged.
        name="t_half",
        target_key="t_half",
        mask_key="t_half",
        pred_paths=(("assays", "t_half"),),
        loss_type="censor",
        target_attr="t_half_target",
        mask_attr="t_half_mask",
        qual_key="t_half",
        qual_attr="t_half_qual",
    ),
    TaskLossSpec(
        name="tm",
        target_key="tm",
        mask_key="tm",
        pred_paths=(("assays", "Tm"),),
        loss_type="censor",
        qual_key="tm",
        qual_attr="tm_qual",
        target_attr="tm_target",
        mask_attr="tm_mask",
    ),
    TaskLossSpec(
        name="binding_affinity_probe",
        target_key="binding",
        mask_key="binding",
        pred_paths=(("binding_affinity_probe_kd",),),
        loss_type="censor",
        target_attr="bind_target",
        mask_attr="bind_mask",
        qual_key="binding",
        qual_attr="bind_qual",
        target_transform=lambda t: normalize_binding_target_log10(
            t,
            max_affinity_nM=DEFAULT_MAX_AFFINITY_NM,
            assume_log10=False,
        ),
        base_weight=1.0,
    ),
    TaskLossSpec(
        # Supervises the detectability latent directly. Without it the latent
        # is a free bottleneck that silently absorbs whatever the presentation
        # pathway cannot explain; the shotgun corpus is what makes it
        # identifiable. Targets are graded over the fractionation-depth ladder,
        # so BCE is used with soft targets.
        name="ms_detectability",
        target_key="ms_detectability",
        mask_key="ms_detectability",
        pred_paths=(("ms_detectability_logit",),),
        loss_type="bce",
        base_weight=0.5,
    ),
    TaskLossSpec(
        # Machinery-conditioned excision. Positives are peptides an arm
        # actually observed; negatives relabel a peptide with an enzyme whose
        # cleavage rule its termini violate.
        name="excision",
        target_key="excision",
        mask_key="excision",
        pred_paths=(("excision_logit",),),
        loss_type="bce",
        base_weight=1.0,
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
        name="core_start",
        target_key="core_start",
        mask_key="core_start",
        pred_paths=(("core_start_logit",),),
        loss_type="ce",
    ),
    TaskLossSpec(
        name="mhc_class",
        target_key="mhc_class",
        mask_key="mhc_class",
        pred_paths=(("mhc_class_logits",),),
        loss_type="ce",
        base_weight=0.1,
    ),
    TaskLossSpec(
        name="mhc_species",
        target_key="mhc_species",
        mask_key="mhc_species",
        pred_paths=(("mhc_species_logits",),),
        loss_type="ce",
        base_weight=0.1,
    ),
    TaskLossSpec(
        name="mhc_a_fine_type",
        target_key="mhc_a_fine_type",
        mask_key="mhc_a_fine_type",
        pred_paths=(("mhc_a_type_logits",),),
        loss_type="ce",
        base_weight=0.1,
    ),
    TaskLossSpec(
        name="mhc_b_fine_type",
        target_key="mhc_b_fine_type",
        mask_key="mhc_b_fine_type",
        pred_paths=(("mhc_b_type_logits",),),
        loss_type="ce",
        base_weight=0.1,
    ),
    TaskLossSpec(
        name="tcr_evidence",
        target_key="tcr_evidence",
        mask_key="tcr_evidence",
        pred_paths=(("tcr_evidence_logit",),),
        loss_type="bce",
        target_attr="tcr_evidence_target",
        mask_attr="tcr_evidence_mask",
        base_weight=0.05,
    ),
    TaskLossSpec(
        name="tcr_evidence_method",
        target_key="tcr_evidence_method",
        mask_key="tcr_evidence_method",
        pred_paths=(("tcr_evidence_method_logits",),),
        loss_type="bce",
        target_attr="tcr_evidence_method_target",
        mask_attr="tcr_evidence_method_mask",
        base_weight=0.02,
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

#: Keep legacy uncertainty indices separate from the grouped panel objectives.
PANEL_TASK_BASE_WEIGHTS: Dict[str, float] = {
    "excision_condition_panel": 1.0,
    "binding_assay_panel": 1.0,
}

_ELUTION_SPEC = next(
    (spec for spec in LOSS_TASK_SPECS if getattr(spec, "name", "") == "elution"),
    None,
)


def _describe_spec(spec):
    metadata = {}
    if spec.name.startswith("binding"):
        metadata.update(raw_unit="nM", target_unit="log10(nM)")
    elif spec.name == "kon":
        metadata.update(raw_unit="1/(M*s)", target_unit="log10(1/(M*s))")
    elif spec.name == "koff":
        metadata.update(raw_unit="1/s", target_unit="log10(1/s)")
    elif spec.name == "t_half":
        metadata.update(raw_unit="h", target_unit="log10(min)")
    elif spec.name == "tm":
        metadata.update(raw_unit="degC", target_unit="(degC-50)/15")
    if spec.loss_type == "ce":
        names = {
            "mhc_class": ("I", "II"),
            "mhc_species": tuple(PROCESSING_SPECIES_BUCKETS),
            "mhc_a_fine_type": tuple(MHC_CHAIN_FINE_TO_IDX),
            "mhc_b_fine_type": tuple(MHC_CHAIN_FINE_TO_IDX),
            "species_of_origin": tuple(ORGANISM_CATEGORIES),
        }.get(spec.name, ())
        metadata.update(class_names=names, raw_unit="class_index", target_unit="class_index")
        if spec.name == "core_start":
            metadata.update(raw_unit="residue_index_0based", target_unit="residue_index_0based")
    if spec.name == "tcr_evidence_method":
        metadata["component_names"] = tuple(TCR_EVIDENCE_METHOD_BINS)
    if spec.selector_key:
        bag_spec = next(s for s in MIL_TASKS["tcell_mil"] if s.selector_key == spec.selector_key)
        metadata.update(axis=bag_spec.axis, columns=bag_spec.columns)
    return replace(spec, **metadata)


# Preserve the established order: checkpoint uncertainty weights use these indices.
LOSS_TASK_SPECS = tuple(_describe_spec(spec) for spec in LOSS_TASK_SPECS)
PANEL_TASK_SPECS = (
    *(
        TaskLossSpec(
            name=f"binding_assay_panel_{axis}",
            target_key="binding",
            mask_key="binding",
            pred_paths=((f"binding_assay_panel_{axis}",),),
            loss_type="censor",
            target_attr="bind_target",
            mask_attr="bind_mask",
            qual_key="binding",
            qual_attr="bind_qual",
            target_transform=LOSS_TASK_SPECS[0].target_transform,
            selector_key=f"{axis}_idx",
            selector_context="binding_context",
            axis=axis,
            columns=tuple(columns),
            loss_group="binding_assay_panel",
            raw_unit="nM",
            target_unit="log10(nM)",
        )
        for axis, columns in (
            ("assay_type", BINDING_ASSAY_TYPES),
            ("assay_prep", BINDING_ASSAY_PREP),
            ("assay_geometry", BINDING_ASSAY_GEOMETRY),
            ("assay_readout", BINDING_ASSAY_READOUT),
        )
    ),
    *(
        TaskLossSpec(
            name=name,
            target_key="elution",
            mask_key="elution",
            pred_paths=((name,),),
            loss_type="bce",
            target_attr="elution_label",
            mask_attr="elution_mask",
            selector_key=key,
            selector_context="provenance",
            axis=axis,
            columns=tuple(columns),
            loss_group="excision_condition_panel",
        )
        for name, key, axis, columns in (
            ("excision_panel_apm", "apm_perturbation_idx", "apm_perturbation", APM_PERTURBATIONS),
            (
                "excision_panel_stimulus",
                "processing_stimulus_idx",
                "processing_stimulus",
                PROCESSING_STIMULI,
            ),
        )
    ),
)
ROW_TASK_SPECS = LOSS_TASK_SPECS + PANEL_TASK_SPECS


LOSS_TASK_NAMES: Tuple[str, ...] = tuple(spec.name for spec in LOSS_TASK_SPECS)
LOSS_TASK_NAME_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(LOSS_TASK_NAMES)}
LOSS_TASK_NAME_TO_SPEC: Dict[str, TaskLossSpec] = {spec.name: spec for spec in LOSS_TASK_SPECS}


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


def _resolve_task_prediction(outputs, batch, spec: TaskLossSpec) -> Optional[torch.Tensor]:
    """Resolve the same supervised response for training and held-out dumps."""
    pred = _resolve_output_tensor(outputs, spec.pred_paths)
    if pred is not None and spec.selector_key is not None:
        index = (
            getattr(batch, spec.selector_context)[spec.selector_key]
            .reshape(-1)
            .long()
            .to(pred.device)
        )
        pred = pred.gather(1, index.unsqueeze(1)).squeeze(1)
    return pred


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
            # In groove-half mode, the second MHC segment for class I is alpha2.
            b_labels.append(MHC_CHAIN_FINE_TO_IDX["MHC_I"])
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

    if (transform := getattr(spec, "target_transform", None)) is not None:
        target_vec = transform(target_vec)

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


@dataclass(frozen=True)
class RowTarget:
    """One observation axis, with CE's class axis kept out of observation identity."""

    spec: Any
    target: torch.Tensor
    mask: torch.Tensor
    raw_target: torch.Tensor
    qualifiers: Optional[torch.Tensor]
    source_rows: torch.Tensor
    components: Optional[torch.Tensor]
    selectors: Optional[torch.Tensor]

    @property
    def support(self):
        return int((self.mask > 0).sum().item())

    @property
    def weight_sum(self):
        return float(self.mask.sum().item())

    @property
    def transformed(self):
        transform = getattr(self.spec, "target_transform", None)
        return transform(self.target.float()) if transform is not None else self.target

    @property
    def group(self):
        return getattr(self.spec, "loss_group", None) or self.spec.name


def make_row_target(spec, target, mask, *, raw_target=None, qualifiers=None, selectors=None):
    """Validate shape once, before loss/count/export take different paths."""
    if target.ndim == 0:
        target = target.reshape(1)
    n_rows = target.shape[0]
    width = target.numel() // max(n_rows, 1)
    if spec.loss_type == "ce" and width != 1:
        raise ValueError(f"{spec.name}: categorical targets must have one class index per row")
    names = getattr(spec, "component_names", ())
    if names and width != len(names):
        raise ValueError(f"{spec.name}: expected {len(names)} components, got {width}")
    flat_target = target.reshape(-1)
    mask = mask.to(device=target.device, dtype=torch.float32)
    if mask.numel() == n_rows and width > 1:
        mask = mask.reshape(-1, 1).expand(n_rows, width)
    if mask.numel() != target.numel():
        raise ValueError(f"{spec.name}: target/mask shape mismatch")
    if not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()):
        raise ValueError(f"{spec.name}: invalid observation weights")
    if qualifiers is not None:
        if qualifiers.numel() != target.numel():
            raise ValueError(f"{spec.name}: target/qualifier shape mismatch")
        qualifiers = qualifiers.reshape(-1).long().to(target.device)
    if raw_target is None:
        raw_target = target
    if raw_target.numel() != target.numel():
        raise ValueError(f"{spec.name}: raw target shape mismatch")
    rows = torch.arange(n_rows, device=target.device).repeat_interleave(width)
    components = torch.arange(width, device=target.device).repeat(n_rows) if width > 1 else None
    if selectors is not None:
        selectors = selectors.reshape(-1).long().to(target.device)
        if selectors.numel() != n_rows or bool((selectors < 0).any()):
            raise ValueError(f"{spec.name}: invalid selector shape or index")
        columns = getattr(spec, "columns", ())
        if columns and bool((selectors >= len(columns)).any()):
            raise ValueError(f"{spec.name}: selector outside declared columns")
        selectors = selectors[rows]
    return RowTarget(
        spec,
        flat_target,
        mask.reshape(-1),
        raw_target.reshape(-1),
        qualifiers,
        rows,
        components,
        selectors,
    )


def resolve_row_targets(batch, specs=ROW_TASK_SPECS):
    """Model-independent observation census, including derived targets and panels."""
    replaced = (
        {spec.name for spec in MIL_TASKS["mil"]}
        if getattr(batch, "mil_bag_label", None) is not None
        else set()
    )
    resolved = {}
    for spec in specs:
        if spec.name in replaced:
            continue
        target = _get_batch_target(batch, spec)
        mask = _get_batch_mask(batch, spec)
        if target is None or mask is None:
            continue
        selectors = None
        if spec.selector_key:
            selectors = getattr(batch, spec.selector_context, {}).get(spec.selector_key)
            if selectors is None:
                # An optional panel without selector metadata has no observation.
                continue
        raw_target = getattr(batch, "raw_targets", {}).get(spec.target_key, target)
        resolved[spec.name] = make_row_target(
            spec,
            target,
            mask,
            raw_target=raw_target,
            qualifiers=_get_batch_qual(batch, spec),
            selectors=selectors,
        )
    return resolved


@dataclass(frozen=True)
class RowPrediction:
    target: RowTarget
    values: torch.Tensor
    losses: torch.Tensor
    output_path: tuple[str, ...]

    def loss(self):
        return (self.losses * self.target.mask).sum() / (self.target.mask.sum() + 1e-8)


def pair_row_prediction(target, pred, output_path=()):
    """Validate selected output and compute the exact per-observation objective."""
    spec = target.spec
    n = target.target.numel()
    if spec.loss_type == "ce":
        if pred.ndim != 2 or pred.shape[0] != n:
            raise ValueError(
                f"{spec.name}: expected {n} categorical class vectors, got {tuple(pred.shape)}"
            )
        names = getattr(spec, "class_names", ())
        if names and pred.shape[1] != len(names):
            raise ValueError(f"{spec.name}: class count does not match declared vocabulary")
        pred = pred.float()
    else:
        if pred.numel() != n:
            raise ValueError(
                f"{spec.name}: expected {n} scalar predictions, got {tuple(pred.shape)}"
            )
        pred = pred.reshape(-1).float()
    active = target.mask > 0
    losses = pred.new_zeros(n)
    if bool(active.any()):
        qualifiers = target.qualifiers[active] if target.qualifiers is not None else None
        if qualifiers is not None and not bool(((qualifiers >= -1) & (qualifiers <= 1)).all()):
            raise ValueError(f"{spec.name}: invalid censor qualifier")
        labels = target.target[active]
        if not bool(torch.isfinite(labels).all()):
            raise ValueError(f"{spec.name}: nonfinite active target")
        if spec.loss_type == "ce" and not bool((labels == labels.long()).all()):
            raise ValueError(f"{spec.name}: categorical target must be an integer class index")
        supported_loss = _compute_task_loss_vector(spec, pred[active], labels, qualifiers)
        if supported_loss is None:
            raise ValueError(f"{spec.name}: missing qualifier for active censored observations")
        if not bool(torch.isfinite(supported_loss).all()):
            raise ValueError(f"{spec.name}: nonfinite active observation loss")
        losses = losses.masked_scatter(active, supported_loss)
    return RowPrediction(target, pred, losses, tuple(output_path))


def resolve_row_predictions(outputs, targets):
    predictions = {}
    for name, target in targets.items():
        if not target.support:
            continue
        spec = target.spec
        for path in spec.pred_paths:
            pred = _resolve_output_tensor(outputs, (path,))
            if pred is not None:
                break
        else:
            # Supported configurations/required endpoints are enforced by the
            # prospective output manifest, not guessed from a label's presence.
            continue
        if target.selectors is not None:
            if pred.ndim != 2 or pred.shape[0] != target.target.numel():
                raise ValueError(f"{name}: selected panel shape mismatch")
            if bool((target.selectors >= pred.shape[1]).any()):
                raise ValueError(f"{name}: selector outside predicted panel")
            pred = pred.gather(1, target.selectors.unsqueeze(1)).squeeze(1)
        predictions[name] = pair_row_prediction(target, pred, path)
    return predictions


def reduce_row_predictions(predictions):
    """Preserve ordinary masked means and the existing mean-across-panel-axes loss."""
    grouped = {}
    for prediction in predictions.values():
        grouped.setdefault(prediction.target.group, []).append(prediction)
    losses, support = {}, {}
    for name, group in grouped.items():
        losses[name] = torch.stack([prediction.loss() for prediction in group]).mean()
        # Panel axes select the same source rows. Counting axes as additional
        # support would change the existing sample-weighted training objective.
        weights = [prediction.target.weight_sum for prediction in group]
        if any(weight != weights[0] for weight in weights):
            raise ValueError(f"{name}: panel axes disagree on observation support")
        support[name] = weights[0]
    return losses, support
