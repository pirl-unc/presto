"""Declared public quantities and their actual supervision, including aliases.

The loss specifications remain executable authorities for target selection and
reduction. This registry describes those objectives' meaning and canonicalizes
their output identities; it never creates an additional training objective.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch

from ..data.collate import PrestoCollator
from ..data.vocab import (
    BINDING_ASSAY_METHODS,
    BINDING_ASSAY_PREP,
    BINDING_ASSAY_READOUT,
    BINDING_ASSAY_TYPES,
)
from ..models.heads import AssayHeads
from .mil import MIL_TASKS
from .supervision import PANEL_TASK_BASE_WEIGHTS, ROW_TASK_SPECS


CONTRACT_VERSION = 1
RESIDUAL_MODES = (
    "legacy",
    "pooled_single_output",
    "shared_base_segment_residual",
    "shared_base_factorized_context_residual",
    "shared_base_factorized_context_plus_segment_residual",
    "dag_family",
    "dag_method_leaf",
    "dag_prep_readout_leaf",
)


@dataclass(frozen=True)
class OutputConfiguration:
    latent_topology: str = "expanded"
    affinity_assay_residual_mode: str = "legacy"
    kd_grouping_mode: str = "merged_kd"
    affinity_target_encoding: str = "log10"
    max_affinity_nM: float = 50000.0
    core_window_lengths: tuple[int, ...] = (9,)
    binding_direct_segment_mode: str = "off"
    binding_kinetic_input_mode: str = "affinity_vec"

    def __post_init__(self):
        choices = {
            "latent_topology": ("collapsed", "expanded"),
            "affinity_assay_residual_mode": RESIDUAL_MODES,
            "kd_grouping_mode": ("merged_kd", "split_kd_proxy"),
            "affinity_target_encoding": ("log10", "mhcflurry"),
            "binding_direct_segment_mode": (
                "off",
                "affinity_residual",
                "affinity_stability_residual",
                "gated_affinity",
            ),
            "binding_kinetic_input_mode": ("affinity_vec", "interaction_vec", "fused"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"Unsupported output configuration {name}={getattr(self, name)!r}")
        # The canonical row/panel loss currently fixes its label clamp at 50k.
        if self.max_affinity_nM != 50000.0:
            raise ValueError("The declared training contract requires max_affinity_nM=50000")
        if not self.core_window_lengths or any(
            type(x) is not int or x < 1 for x in self.core_window_lengths
        ):
            raise ValueError("core_window_lengths must contain positive integer lengths")
        object.__setattr__(
            self, "core_window_lengths", tuple(sorted(set(self.core_window_lengths)))
        )

    @classmethod
    def from_object(cls, value):
        values = {
            name: getattr(value, name, f.default) for name, f in cls.__dataclass_fields__.items()
        }
        if values["core_window_lengths"] is None:
            values["core_window_lengths"] = (9,)
        return cls(**values)


@dataclass(frozen=True)
class OutputAlias:
    path: str
    canonical: str
    transform: str = "identity"  # identity permits singleton squeezing


@dataclass(frozen=True)
class ObjectiveBinding:
    id: str
    endpoint: str
    path: str
    channel: str
    target: str
    mask: str
    qualifier: str
    selector: str
    loss_type: str
    loss_group: str
    base_weight: float
    raw_unit: str
    target_unit: str
    source_target: str
    evidence_role: str
    reduction: str


@dataclass(frozen=True)
class OutputSpec:
    id: str
    shape: str = "batch_scalar"
    unit: str = "logit"
    columns: tuple[str, ...] = ()
    role: str = "indirect"
    source_families: tuple[str, ...] = ()
    note: str = "No independent label objective; gradient flow is not direct evidence."
    required: bool = True


@dataclass(frozen=True)
class ParameterRowSpec:
    endpoint: str
    parameter: str
    columns: tuple[str, ...]
    interpretation: str = "column_parameter"


def declared_parameter_rows(contract, model) -> tuple[ParameterRowSpec, ...]:
    """Map output columns to actual parameter rows, without shape guessing.

    Shared/composed outputs can have no dedicated row. A mapped row may also
    participate in other computations; observing its update is not causal
    attribution to a particular label or objective.
    """
    model = getattr(model, "_orig_mod", model)
    params = dict(model.named_parameters())
    declarations = []

    def rows(endpoint, parameter, interpretation="column_parameter"):
        columns = contract.outputs[endpoint].columns or ("",)
        value = params.get(parameter)
        if value is None or value.ndim == 0 or value.shape[0] != len(columns):
            raise ValueError(f"Missing or incompatible declared parameter rows: {parameter}")
        declarations.append(ParameterRowSpec(endpoint, parameter, columns, interpretation))

    def readout(endpoint, path):
        module = model.get_submodule(path)
        linear = [
            (name, part)
            for name, part in module.named_modules()
            if isinstance(part, torch.nn.Linear)
        ]
        if not linear:
            raise ValueError(f"No linear readout in declared module {path}")
        name, layer = linear[-1]
        prefix = f"{path}.{name}" if name else path
        rows(endpoint, f"{prefix}.weight")
        if layer.bias is not None:
            rows(endpoint, f"{prefix}.bias")

    for endpoint in contract.outputs:
        if endpoint.startswith("tcell_panel_logits."):
            axis = endpoint.split(".", 1)[1]
            rows(endpoint, f"tcell_assay_head.{axis}_embed.weight")
        if endpoint.startswith("binding_assay_panel_"):
            axis = endpoint.removeprefix("binding_assay_panel_")
            rows(endpoint, f"affinity_predictor.assay_panel_embed.{axis}.weight")
    for suffix in ("invivo_profile_c", "invivo_profile_n", "invivo_bias"):
        rows("excision_panel_apm", f"excision_head.{suffix}")
    rows("excision_panel_stimulus", "excision_head.stimulus_profile_c")
    for endpoint, path in (
        ("mhc_a_type_logits", "mhc_a_type_head"),
        ("mhc_b_type_logits", "mhc_b_type_head"),
        ("species_of_origin_logits", "species_of_origin_head"),
        ("tcr_evidence_method_logits", "tcr_evidence_method_head"),
        ("tcr_evidence_logit", "tcr_evidence_head"),
        ("binding_affinity_probe_kd", "affinity_predictor.binding_affinity_probe"),
        ("assays.Tm", "affinity_predictor.assay_heads.tm"),
        ("assays.t_half", "affinity_predictor.assay_heads.t_half_residual"),
    ):
        readout(endpoint, path)
    mode = contract.configuration.affinity_assay_residual_mode
    for family in ("ic50", "ec50"):
        base = "affinity_predictor.assay_heads"
        if mode == "pooled_single_output":
            continue
        suffix = "leaf_residual" if mode.startswith("dag_") else "residual"
        readout(f"assays.{family.upper()}_nM", f"{base}.{family}_{suffix}")
        proxy = f"assays.KD_proxy_{family}_nM"
        if proxy in contract.outputs:
            readout(proxy, f"{base}.kd_proxy_{family}_{suffix}")
        if mode.startswith("dag_"):
            readout(f"assays.{family.upper()}_family_anchor_nM", f"{base}.{family}_family_residual")
        if mode == "dag_method_leaf":
            for method in BINDING_ASSAY_METHODS:
                endpoint = f"assays.{AssayHeads.method_output_key(f'{family.upper()}_nM', method)}"
                readout(endpoint, f"{base}.{family}_method_leaf_residuals.{method}")
        if mode == "dag_prep_readout_leaf":
            for prep in BINDING_ASSAY_PREP:
                for readout_name in BINDING_ASSAY_READOUT:
                    key = AssayHeads.prep_readout_output_key(
                        f"{family.upper()}_nM", prep, readout_name
                    )
                    endpoint = f"assays.{key}"
                    readout(endpoint, f"{base}.{family}_prep_leaf_residuals.{prep}")
                    readout(endpoint, f"{base}.{family}_readout_leaf_residuals.{readout_name}")
    return tuple(declarations)


@dataclass
class OutputContract:
    configuration: OutputConfiguration
    outputs: dict[str, OutputSpec] = field(default_factory=dict)
    aliases: dict[str, OutputAlias] = field(default_factory=dict)
    objectives: dict[str, ObjectiveBinding] = field(default_factory=dict)

    def canonical(self, path: str) -> str:
        return self.aliases[path].canonical if path in self.aliases else path

    def to_dict(self) -> dict:
        return {"schema_version": CONTRACT_VERSION, **asdict(self)}

    def validate_outputs(self, outputs: Mapping[str, Any]) -> None:
        """Reject undeclared quantities, wrong shapes and drifting alias values."""
        flat = {}

        def visit(values, prefix=""):
            for name, value in values.items():
                path = f"{prefix}.{name}" if prefix else name
                # Only explicitly declared diagnostic containers may be opaque.
                if path in self.outputs and self.outputs[path].shape == "diagnostic_container":
                    flat[path] = value
                elif isinstance(value, Mapping):
                    visit(value, path)
                else:
                    flat[path] = value

        visit(outputs)
        unknown = set(flat) - self.outputs.keys() - self.aliases.keys()
        missing = {name for name, spec in self.outputs.items() if spec.required} - flat.keys()
        if unknown or missing:
            raise ValueError(
                f"Output contract mismatch: undeclared={sorted(unknown)}, missing={sorted(missing)}"
            )
        for name, spec in self.outputs.items():
            value = flat.get(name)
            if value is None or spec.shape in {"diagnostic", "diagnostic_container"}:
                continue
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{name}: expected a tensor")
            if spec.shape == "batch_scalar" and not (
                value.ndim == 1 or value.ndim == 2 and value.shape[1] == 1
            ):
                raise ValueError(
                    f"{name}: expected one scalar per example, got {tuple(value.shape)}"
                )
            if spec.shape in {"panel", "components", "classes", "positions"}:
                if value.ndim != 2 or spec.columns and value.shape[1] != len(spec.columns):
                    raise ValueError(f"{name}: wrong declared column shape {tuple(value.shape)}")
        for name, alias in self.aliases.items():
            if name not in flat:
                continue
            source = flat.get(alias.canonical)
            value = flat[name]
            if not isinstance(source, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise ValueError(f"{name}: alias requires canonical tensor {alias.canonical}")
            expected = {
                "identity": lambda x: x,
                "sigmoid": torch.sigmoid,
                "softmax": lambda x: x.softmax(-1),
                "argmax": lambda x: x.argmax(-1),
            }[alias.transform](source)
            if value.numel() != expected.numel() or not torch.allclose(
                value.reshape(-1).float(), expected.reshape(-1).float(), atol=2e-6, rtol=2e-5
            ):
                raise ValueError(
                    f"{name}: differs from declared {alias.transform} view of {alias.canonical}"
                )


def _source_contract(name: str) -> tuple[str, str]:
    """Source-label family and endpoint role, before per-record origin checks."""
    if name.startswith("binding"):
        return "binding", "auxiliary" if name == "binding_affinity_probe" else "direct"
    if name in {"kon", "koff", "t_half", "tm", "processing"}:
        return name, "direct"
    if name in {"elution", "ms"}:
        return "elution", "direct"
    if name == "presentation" or name.startswith("excision_panel"):
        return "elution", "proxy"
    if name.startswith("immunogenicity") or name in {"tcell", "tcell_mil"}:
        # The scalar uses the all-unknown assay reference, while labels are
        # observed under heterogeneous assay conditions.
        return "tcell", "proxy"
    if name.startswith("tcell_"):
        return "tcell", "direct"
    if name in {"excision", "ms_detectability"}:
        return name, "proxy"
    if name.startswith("tcr_evidence"):
        return "tcr_evidence", "auxiliary"
    if name in {
        "mhc_class",
        "mhc_species",
        "mhc_a_fine_type",
        "mhc_b_fine_type",
        "core_start",
        "species_of_origin",
    }:
        return "annotation", "auxiliary"
    if name == "foreignness":
        return "organism", "proxy"
    raise ValueError(f"Missing source declaration for objective {name}")


def build_output_contract(configuration: OutputConfiguration | None = None) -> OutputContract:
    config = configuration or OutputConfiguration()
    contract = OutputContract(config)

    def alias(path, canonical, transform="identity"):
        contract.aliases[path] = OutputAlias(path, canonical, transform)

    alias("ms_logit", "elution_logit")
    pooled = config.affinity_assay_residual_mode == "pooled_single_output"
    if pooled:
        for name in ("IC50_nM", "EC50_nM"):
            alias(f"assays.{name}", "assays.KD_nM")
    if pooled or config.kd_grouping_mode == "merged_kd":
        for name in ("KD_proxy_ic50_nM", "KD_proxy_ec50_nM"):
            alias(f"assays.{name}", "assays.KD_nM")

    for spec in ROW_TASK_SPECS:
        path = ".".join(spec.pred_paths[0])
        endpoint = contract.canonical(path)
        source, role = _source_contract(spec.name)
        columns = spec.columns or spec.component_names or spec.class_names
        shape = (
            "panel"
            if spec.columns
            else "components"
            if spec.component_names
            else "classes"
            if spec.class_names
            else "positions"
            if spec.loss_type == "ce"
            else "batch_scalar"
        )
        contract.outputs.setdefault(
            endpoint,
            OutputSpec(
                endpoint,
                shape,
                "logit" if spec.loss_type in {"bce", "ce"} else spec.target_unit,
                tuple(columns),
                role,
                tuple(f"binding:{name}" for name in BINDING_ASSAY_TYPES)
                if source == "binding"
                else {
                    "annotation": ("source_annotation",),
                    "organism": ("organism_derived",),
                    "excision": ("bulk_observed_product",),
                    "ms_detectability": ("bulk_depth_proxy",),
                }.get(source, (source,)),
                "Support depends on selected labels and source provenance; "
                "generated:<kind> families are recorded separately with synthetic role. "
                "No adequacy claim is implied.",
            ),
        )
        group = spec.loss_group or spec.name
        weight = PANEL_TASK_BASE_WEIGHTS.get(group, spec.base_weight)
        contract.objectives[f"row:{spec.name}"] = ObjectiveBinding(
            f"row:{spec.name}",
            endpoint,
            path,
            "row",
            spec.target_key,
            spec.mask_key,
            spec.qual_key or "",
            f"{spec.selector_context}.{spec.selector_key}" if spec.selector_key else "",
            spec.loss_type,
            group,
            weight,
            spec.raw_unit,
            spec.target_unit,
            source,
            role,
            "mean_active_axes_then_masked_mean" if spec.loss_group else "masked_mean",
        )
    for channel, specs in MIL_TASKS.items():
        for spec in specs:
            path = ".".join(spec.output_path)
            source, role = _source_contract(spec.name)
            identity = f"{channel}:{spec.name}"
            contract.objectives[identity] = ObjectiveBinding(
                identity,
                contract.canonical(path),
                path,
                channel,
                "bag_label",
                "resolved_bag_mask",
                "",
                spec.selector_key,
                "bce",
                spec.name,
                spec.base_weight,
                "response",
                "response",
                source,
                role,
                "noisy_or_instances_then_mean_bags",
            )

    for path in ("assays.KD_proxy_ic50_nM", "assays.KD_proxy_ec50_nM"):
        if path not in contract.aliases:
            contract.outputs[path] = OutputSpec(path, unit="log10(nM)")
    mode = config.affinity_assay_residual_mode
    if mode.startswith("dag_"):
        for family in ("IC50", "EC50"):
            path = f"assays.{family}_family_anchor_nM"
            contract.outputs[path] = OutputSpec(path, unit="log10(nM)")
            leaves = []
            if mode == "dag_method_leaf":
                leaves = [
                    AssayHeads.method_output_key(f"{family}_nM", method)
                    for method in BINDING_ASSAY_METHODS
                ]
            if mode == "dag_prep_readout_leaf":
                leaves = [
                    AssayHeads.prep_readout_output_key(f"{family}_nM", prep, readout)
                    for prep in BINDING_ASSAY_PREP
                    for readout in BINDING_ASSAY_READOUT
                ]
            for name in leaves:
                path = f"assays.{name}"
                contract.outputs[path] = OutputSpec(
                    path,
                    unit="log10(nM)",
                    note="Published leaf without a selected-label training objective.",
                )

    # Separately published biological intermediates have no independent labels.
    for stem in (
        "binding",
        "binding_base",
        "binding_class1",
        "binding_class2",
        "processing_class1",
        "processing_class2",
        "presentation_class1",
        "presentation_class2",
        "recognition_cd8",
        "recognition_cd4",
        "recognition_repertoire",
        "immunogenicity_cd8",
        "immunogenicity_cd4",
        "chain_compat",
    ):
        path = f"{stem}_logit"
        contract.outputs[path] = OutputSpec(path)
    for stem in ("binding", "processing", "presentation", "immunogenicity"):
        alias(f"{stem}_mixed_logit", f"{stem}_logit")
        alias(f"{stem}_mixed_prob", f"{stem}_logit", "sigmoid")
    alias("immunogenicity_mixture_logit", "immunogenicity_logit")
    alias("recognition_mixed_logit", "recognition_repertoire_logit")
    alias("recognition_mixed_prob", "recognition_repertoire_logit", "sigmoid")
    for stem in (
        "binding",
        "binding_class1",
        "binding_class2",
        "processing",
        "processing_class1",
        "processing_class2",
        "presentation",
        "presentation_class1",
        "presentation_class2",
        "recognition_cd8",
        "recognition_cd4",
        "recognition_repertoire",
        "immunogenicity",
        "immunogenicity_cd8",
        "immunogenicity_cd4",
        "excision",
        "foreignness",
        "elution",
        "tcell",
        "tcr_evidence",
        "chain_compat",
    ):
        alias(f"{stem}_prob", f"{stem}_logit", "sigmoid")
    alias("ms_prob", "elution_logit", "sigmoid")
    alias("tcr_evidence_method_probs", "tcr_evidence_method_logits", "sigmoid")
    for name in ("mhc_class", "mhc_species"):
        alias(f"{name}_probs_inferred", f"{name}_logits", "softmax")
        alias(f"{name}_pred", f"{name}_logits", "argmax")
    for spec in MIL_TASKS["tcell_mil"]:
        if spec.axis:
            alias(f"tcell_context_logits.{spec.axis}", f"tcell_panel_logits.{spec.axis}")
    alias("binding_affinity_score", "binding_affinity_probe_kd")
    alias("binding_affinity_probe_kd_raw", "binding_affinity_score_raw")
    alias("pmhc_vec", "pmhc_interaction_vec")
    # Multiple window lengths may share a start, so the projected window
    # posterior is not generally softmax(core_start_logit).
    contract.outputs["core_start_prob"] = OutputSpec(
        "core_start_prob",
        "positions",
        "window_posterior_projection",
        role="indirect",
        note="Projected window posterior; no independent label objective.",
    )
    alias("core_start_probs", "core_start_prob")

    # An explicit list prevents a new assay output being swallowed by a prefix
    # rule. Routing probabilities can incorporate caller overrides and are not
    # aliases of the inferred classification probabilities above.
    diagnostics = """
        pep_vec mhc_a_vec mhc_b_vec groove_vec apc_cell_type_context_vec pmhc_interaction_vec
        mhc_a_species_logits mhc_b_species_logits mhc_class_probs mhc_species_probs species_probs
        mhc_is_class1_prob mhc_is_class2_prob binding_direct_segment_mode
        binding_direct_segment_gate_mean binding_direct_affinity_vec binding_direct_stability_vec
        binding_assay_context_vec binding_affinity_score_raw binding_stability_score_raw
        binding_stability_score binding_sequence_summary_vec binding_kinetic_input_mode
        binding_logit_from_core binding_kd_bias_raw binding_kd_bias binding_kd_bias_cap
        binding_probe_mix_weight binding_core_kd_log10 binding_mixed_kd_log10
        core_window_mask core_window_start core_window_length core_window_prior_logit
        core_window_score_logit core_window_logit core_window_posterior_prob core_membership_prob
        core_relative_position_index core_length npfr_length cpfr_length core_length_norm
        npfr_length_norm cpfr_length_norm binding_mhc_attention_effective_residues
        binding_mhc_attention_mass binding_mhc_attention_valid_mask
        binding_mhc_attention_token_count
        excision_n_terminus_score excision_c_terminus_score excision_length_score
        excision_missed_cleavage_score excision_machinery_idx presentation_invivo_excision_term
        immunogenicity_recognition_term
    """.split()
    for path in diagnostics:
        contract.outputs[path] = OutputSpec(
            path, "diagnostic", "implementation_defined", role="diagnostic", required=False
        )
    for path in ("latent_vecs", "binding_latents", "attn_layers"):
        contract.outputs[path] = OutputSpec(
            path, "diagnostic_container", "latent", role="diagnostic", required=False
        )
    return contract


@dataclass(frozen=True)
class LabelEvidence:
    role: str
    family: str
    raw_source: str
    origin: str


def label_evidence(objective: ObjectiveBinding, sample) -> LabelEvidence:
    """Classify an actual label without upgrading indirect/proxy evidence."""
    source = str(getattr(sample, "sample_source", "") or "unknown")
    provenance = getattr(sample, "target_provenance", {})
    origin = provenance.get(objective.source_target, "unknown")
    generated = str(getattr(sample, "synthetic_kind", "") or "")
    if origin.startswith("generated:"):
        generated = origin.removeprefix("generated:")
    if not generated and (source == "synthetic" or source.startswith("synthetic_")):
        generated = source
    family = objective.source_target
    if generated:
        return LabelEvidence("synthetic", f"generated:{generated}", source, origin)
    if family == "annotation":
        return LabelEvidence("auxiliary", "source_annotation", source, origin)
    if family == "organism":
        if origin == "organism_derived":
            return LabelEvidence("proxy", "organism_derived", source, origin)
        return LabelEvidence("unknown", "organism", source, origin)
    if origin == "unknown":
        return LabelEvidence("unknown", family, source, origin)
    role = objective.evidence_role
    if family == "binding":
        assay = PrestoCollator._categorize_binding_assay_type(
            sample.binding_assay_type or sample.bind_measurement_type
        )
        family = f"binding:{assay}"
        if role == "direct" and (
            assay not in {"KD", "IC50", "EC50"}
            or objective.endpoint == "assays.KD_nM"
            and assay != "KD"
        ):
            role = "proxy"
    if origin in {"bulk_observed_product", "bulk_depth_proxy"}:
        role, family = "proxy", origin
    if origin != "assay_record" and origin not in {"bulk_observed_product", "bulk_depth_proxy"}:
        role = "unknown"
    return LabelEvidence(role, family, source, origin)
