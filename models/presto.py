"""The shared observation-based Presto model for all new training.

Register attention is statistical pooling, not physical occupancy. Component
scores in an MS mixture are not independently identified by mixture labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Sequence

import torch
from torch import nn

from ..data.modular import (
    AA,
    CORECEPTORS,
    DELIVERY,
    EXPOSURE_STATES,
    ENDPOINTS,
    IDENTITY_TASKS,
    ROLES,
    STATES,
    Context,
    MHCInput,
    Observation,
    MachineryProfile,
    identity_vocabulary,
    has_identity_input,
    endpoint_spec,
    is_endpoint,
)
from ..data.modular_quantitative import QuantitativeDomain, quantity_domain
from ..data.modular_ordinal import OrdinalScale
from ..data.modular_context import (
    resolve_cell_context,
    resolve_components,
    resolve_field,
    resolve_species,
)
from .apm_state_bottleneck import ApmStateBottleneck
from .apc_state import HarmonizedAPCState
from .junction_processing import JunctionProcessing
from .shared_processing import SharedProcessingModule
from .encoders import SequenceEncoder
from .imgt_groove import (
    IMGT_VOCAB_SIZE,
    POCKET_SLOT_SCHEME_UNION46,
    VALID_POCKET_SLOT_SCHEMES,
    DirectPocketBilinearHead,
    IMGTPocketCrossInteraction,
    imgt_groove_slot_indices,
    imgt_pan_class_aa_ids,
    peptide_core_aa_ids,
)
from .modular_support import feature_inputs, feature_layout
from .modular_dependencies import frozen_output, staged_elution_parameters
from ..data.assay_measurement import AssayTrack, observation_track, observation_method
from .assay_measurement import AFFINITY_ENDPOINTS, TRACKED_ENDPOINTS, AssayMeasurementLayer
from ..data.tcell_assay import TCellTrack
from .tcell_assay import (
    TCellAssayLayer,
    TCellStagedHead,
    tcr_bulge_physicochem,
    tcr_facing_indices,
)

MODEL_VERSION = "presto.shared.v7"
ELUTION_STAGINGS = ("legacy_joint", "affinity_downstream")


@dataclass(frozen=True)
class PrestoConfig:
    endpoints: tuple[str, ...] = ("ic50",)
    assay_tracks: tuple[AssayTrack, ...] = ()
    tcell_tracks: tuple[TCellTrack, ...] = ()
    tcell_ordinal_loss_weight: float = 1.0
    tcell_staged: bool = False
    assay_mode: str = "factorized"
    identity_tasks: tuple[str, ...] = ()
    mhc_classes: tuple[str, ...] = ("I", "II")
    register_pooling: bool = True
    species: tuple[str, ...] = ()
    source_categories: tuple[str, ...] = ()
    auxiliary_context: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    component_activity_units: dict[str, str] = field(default_factory=dict)
    machinery_profiles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    machinery_references: tuple[MachineryProfile, ...] = ()
    cell_types: tuple[str, ...] = ()
    cytokines: tuple[str, ...] = ()
    cytokine_dose_units: dict[str, str] = field(default_factory=dict)
    cytokine_duration_features: tuple[str, ...] = ()
    processing_pathways: tuple[str, ...] = ()
    interventions: tuple[str, ...] = ()
    ordinal_scales: dict[str, OrdinalScale] = field(default_factory=dict)
    context_species: tuple[str, ...] = ()
    enzymes: tuple[str, ...] = ()
    acquisition_features: tuple[str, ...] = ()
    observation_features: tuple[str, ...] = ()
    acquisition_units: dict[str, str] = field(default_factory=dict)
    observation_units: dict[str, str] = field(default_factory=dict)
    preparation_features: tuple[str, ...] = ()
    preparation_units: dict[str, str] = field(default_factory=dict)
    quantity_domains: dict[str, QuantitativeDomain] = field(default_factory=dict)
    source_context: bool = False
    species_defaults: bool = True
    identity_threshold: float = 0.8
    dropout: float = 0.1
    max_length: int = 512
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 512
    peptide_context: str = "shared_core_context_v2"
    # "legacy_joint": elution reads the fused trunk through `cellular_score`.
    # "affinity_downstream": elution = a*(-pooled affinity score) [+ b*pooled
    # half-life score] + processing + ms_response + bias; see docs/modular_model.md.
    # The three staging fields are omitted from `to_dict()` while legacy so frozen
    # checkpoints and their contract digests are unchanged.
    elution_staging: str = "legacy_joint"
    elution_affinity_endpoint: str | None = None
    elution_detach_affinity: bool = False
    elution_pathway_conditioning: bool = False
    elution_junction_processing: bool = False
    elution_shared_processing: bool = False
    interaction_mode: str = "mean"
    groove_segment_embedding: bool = False
    imgt_groove_embedding: bool = False
    imgt_pocket_cross: bool = False
    imgt_pocket_slot_scheme: str = POCKET_SLOT_SCHEME_UNION46
    groove_cross_chain: bool = False
    query_anchor_embedding: bool = False
    groove_pocket_pool: bool = False
    interaction_cooperativity: bool = False
    pmhc_pfr_conditioning: bool = False
    pmhc_refine: bool = False
    register_pooling_mode: str = "normalized"
    elution_pmhc_residual: bool = False
    elution_mhc_abundance: bool = False
    class1_overhang_registers: bool = False
    tcell_class_specific_tcr: bool = False
    tcell_register_aligned_supply: bool = False
    class_conditioned_assay_offset: bool = False
    direct_pocket_bilinear: bool = False
    shared_processing_v2: bool = False
    candidate_junction_mode: str = "masked"
    candidate_junction_temp: float = 1.0
    # Output-side APM state bottleneck (0 = off; omitted from to_dict, contracts unchanged).
    state_bottleneck: int = 0
    state_mode: str = "structured"

    def __post_init__(self):
        if self.interaction_mode not in ("mean", "anchor_v2", "anchor_v3"):
            raise ValueError("interaction_mode must be 'mean', 'anchor_v2', or 'anchor_v3'")
        if type(self.groove_segment_embedding) is not bool:
            raise ValueError("groove_segment_embedding must be boolean")
        if type(self.imgt_groove_embedding) is not bool:
            raise ValueError("imgt_groove_embedding must be boolean")
        if type(self.imgt_pocket_cross) is not bool:
            raise ValueError("imgt_pocket_cross must be boolean")
        if self.imgt_pocket_slot_scheme not in VALID_POCKET_SLOT_SCHEMES:
            raise ValueError(f"imgt_pocket_slot_scheme must be one of {VALID_POCKET_SLOT_SCHEMES}")
        if type(self.groove_cross_chain) is not bool:
            raise ValueError("groove_cross_chain must be boolean")
        if type(self.query_anchor_embedding) is not bool:
            raise ValueError("query_anchor_embedding must be boolean")
        if type(self.groove_pocket_pool) is not bool:
            raise ValueError("groove_pocket_pool must be boolean")
        if type(self.interaction_cooperativity) is not bool:
            raise ValueError("interaction_cooperativity must be boolean")
        if self.interaction_cooperativity and self.interaction_mode not in (
            "anchor_v2",
            "anchor_v3",
        ):
            raise ValueError("interaction_cooperativity requires anchor_v2 or anchor_v3")
        if type(self.pmhc_pfr_conditioning) is not bool:
            raise ValueError("pmhc_pfr_conditioning must be boolean")
        if type(self.pmhc_refine) is not bool:
            raise ValueError("pmhc_refine must be boolean")
        if self.register_pooling_mode not in ("normalized", "thermodynamic"):
            raise ValueError("register_pooling_mode must be 'normalized' or 'thermodynamic'")
        if type(self.elution_pmhc_residual) is not bool:
            raise ValueError("elution_pmhc_residual must be boolean")
        if self.elution_pmhc_residual and self.elution_staging != "affinity_downstream":
            raise ValueError("elution_pmhc_residual requires affinity_downstream staged elution")
        if type(self.elution_mhc_abundance) is not bool:
            raise ValueError("elution_mhc_abundance must be boolean")
        if self.elution_mhc_abundance and self.elution_staging != "affinity_downstream":
            raise ValueError("elution_mhc_abundance requires affinity_downstream staged elution")
        if type(self.class1_overhang_registers) is not bool:
            raise ValueError("class1_overhang_registers must be boolean")
        if type(self.tcell_class_specific_tcr) is not bool:
            raise ValueError("tcell_class_specific_tcr must be boolean")
        if self.tcell_class_specific_tcr and not self.tcell_staged:
            raise ValueError("tcell_class_specific_tcr requires tcell_staged=True")
        if type(self.tcell_register_aligned_supply) is not bool:
            raise ValueError("tcell_register_aligned_supply must be boolean")
        if self.tcell_register_aligned_supply and not self.tcell_staged:
            raise ValueError("tcell_register_aligned_supply requires tcell_staged=True")
        if type(self.class_conditioned_assay_offset) is not bool:
            raise ValueError("class_conditioned_assay_offset must be boolean")
        if type(self.direct_pocket_bilinear) is not bool:
            raise ValueError("direct_pocket_bilinear must be boolean")
        if type(self.shared_processing_v2) is not bool:
            raise ValueError("shared_processing_v2 must be boolean")
        if self.shared_processing_v2 and not self.elution_shared_processing:
            raise ValueError("shared_processing_v2 requires elution_shared_processing")
        if self.candidate_junction_mode not in ("masked", "sample", "mil"):
            raise ValueError("candidate_junction_mode must be 'masked', 'sample', or 'mil'")
        if (
            isinstance(self.candidate_junction_temp, bool)
            or not isinstance(self.candidate_junction_temp, (int, float))
            or not math.isfinite(self.candidate_junction_temp)
            or self.candidate_junction_temp <= 0.0
        ):
            raise ValueError("candidate_junction_temp must be positive finite")
        if type(self.elution_shared_processing) is not bool:
            raise ValueError("elution_shared_processing must be boolean")
        if self.elution_shared_processing and (
            self.elution_staging != "affinity_downstream"
            or not self.source_context
            or (self.state_bottleneck and self.state_mode != "harmonized")
        ):
            raise ValueError(
                "shared processing requires source-aware staged elution and harmonized APC state"
            )
        if type(self.elution_junction_processing) is not bool:
            raise ValueError("elution_junction_processing must be boolean")
        if self.elution_junction_processing and (
            self.elution_staging != "affinity_downstream"
            or not self.source_context
            or (self.state_bottleneck and self.state_mode != "harmonized")
        ):
            raise ValueError(
                "junction processing requires source-aware staged elution and shared APC state"
            )
        if type(self.elution_pathway_conditioning) is not bool:
            raise ValueError("elution_pathway_conditioning must be boolean")
        if self.elution_pathway_conditioning and self.elution_staging != "affinity_downstream":
            raise ValueError("pathway-conditioned processing requires staged elution")
        if type(self.state_bottleneck) is not int or self.state_bottleneck < 0:
            raise ValueError("state_bottleneck must be a nonnegative integer")
        if self.state_mode not in ("structured", "free", "harmonized"):
            raise ValueError("state_mode must be one of structured, free, harmonized")
        if self.state_bottleneck and self.elution_staging != "affinity_downstream":
            raise ValueError("APC state requires staged elution")
        if self.assay_mode != "factorized":
            raise ValueError(
                "new training requires factorized assay_mode; v6 controls are historical"
            )
        if any(not isinstance(track, AssayTrack) for track in self.assay_tracks):
            raise ValueError("normalized assay tracks required")
        if len({track.key for track in self.assay_tracks}) != len(self.assay_tracks):
            raise ValueError("duplicate assay tracks")
        for track in self.assay_tracks:
            if track.endpoint not in TRACKED_ENDPOINTS:
                raise ValueError("unsupported factorized assay-track endpoint")
            spec = endpoint_spec(track.endpoint)
            if (
                track.endpoint not in self.endpoints
                or spec.parts
                or spec.route not in ("pmhc", "cellular", "tcr", "tcr_cellular")
            ):
                raise ValueError("assay track needs an allocated direct measurement output")
            preparation = "purified" if spec.route in ("pmhc", "tcr") else "cellular"
            if track.preparation != preparation:
                raise ValueError("assay preparation conflicts with the biological endpoint route")
        if any(not isinstance(track, TCellTrack) for track in self.tcell_tracks):
            raise ValueError("normalized T-cell tracks required")
        if len({track.key for track in self.tcell_tracks}) != len(self.tcell_tracks):
            raise ValueError("duplicate T-cell tracks")
        if self.tcell_tracks and "tcell_response" not in self.endpoints:
            raise ValueError("T-cell tracks require the T-cell response endpoint")
        if type(self.tcell_staged) is not bool:
            raise ValueError("tcell_staged must be boolean")
        if self.tcell_staged and not self.tcell_tracks:
            raise ValueError("tcell_staged requires declared tcell_tracks")
        if (
            type(self.tcell_ordinal_loss_weight) not in (int, float)
            or not math.isfinite(self.tcell_ordinal_loss_weight)
            or self.tcell_ordinal_loss_weight < 0
        ):
            raise ValueError("T-cell ordinal loss weight must be finite and nonnegative")
        for name in ("d_model", "n_layers", "n_heads", "d_ff"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % 2 or self.d_model % self.n_heads:
            raise ValueError("model width must be even and divisible by attention heads")
        if self.elution_staging not in ELUTION_STAGINGS:
            raise ValueError(f"elution_staging must be one of {ELUTION_STAGINGS}")
        if self.elution_staging == "legacy_joint":
            if self.elution_affinity_endpoint is not None or self.elution_detach_affinity:
                raise ValueError("elution affinity options require elution_staging")
        else:
            if "elution_observation" not in self.endpoints:
                raise ValueError("staged elution needs the elution_observation endpoint")
            if (
                self.elution_affinity_endpoint not in AFFINITY_ENDPOINTS
                or self.elution_affinity_endpoint not in self.endpoints
            ):
                raise ValueError(
                    "staged elution needs an allocated concentration affinity endpoint"
                )
            if type(self.elution_detach_affinity) is not bool:
                raise ValueError("elution_detach_affinity must be boolean")
        if self.peptide_context != "shared_core_context_v2":
            raise ValueError("new training requires the shared core/full ordered context contract")
        if not self.register_pooling:
            raise ValueError("new training requires shared candidate register pooling")
        for name in (
            "endpoints",
            "identity_tasks",
            "mhc_classes",
            "species",
            "source_categories",
            "auxiliary_context",
            "components",
            "cell_types",
            "cytokines",
            "cytokine_duration_features",
            "processing_pathways",
            "interventions",
            "context_species",
            "enzymes",
            "acquisition_features",
            "observation_features",
            "preparation_features",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {name}")
        if any(not is_endpoint(name) for name in self.endpoints) or set(self.identity_tasks) - set(
            IDENTITY_TASKS
        ):
            raise ValueError("unsupported endpoint or identity task")
        domains = {quantity_domain(name) for name in self.endpoints} - {None}
        if domains != set(self.quantity_domains) or any(
            not isinstance(value, QuantitativeDomain) for value in self.quantity_domains.values()
        ):
            raise ValueError(
                "quantity endpoints require exactly their declared measurement domains"
            )
        if "peptide_source" in self.identity_tasks and not self.source_categories:
            raise ValueError("peptide source prediction requires declared category labels")
        for name in self.auxiliary_context:
            if name not in ("mhc_pairing", "tcr_pairing", "tcr_lineage") and not name.startswith(
                "peptide_source:"
            ):
                raise ValueError(f"unsupported auxiliary context slot: {name}")
            identity_vocabulary(name, self.species, self.source_categories)
        if not self.endpoints and not self.identity_tasks:
            raise ValueError("no active supervision")
        if set(self.mhc_classes) - {"I", "II"} or not self.mhc_classes:
            raise ValueError("unsupported class vocabulary")
        if set(self.context_species) - (set(ROLES) - {"mhc1", "mhc2", "antigen"}):
            raise ValueError("unsupported context species slot")
        if any(task.endswith("species") for task in self.identity_tasks) and len(self.species) < 2:
            raise ValueError("species prediction requires a varied, declared vocabulary")
        if self.context_species and not self.species:
            raise ValueError("context species needs a vocabulary")
        for names, units in (
            (self.acquisition_features, self.acquisition_units),
            (self.observation_features, self.observation_units),
            (self.preparation_features, self.preparation_units),
        ):
            if set(names) != set(units) or any(not value for value in units.values()):
                raise ValueError("each numeric feature requires a declared unit/scale")
        if set(self.component_activity_units) - set(self.components):
            raise ValueError("activity units refer to inactive components")
        for species, names in self.machinery_profiles.items():
            if species not in self.species or set(names) - set(self.components):
                raise ValueError("reference profile outside declared vocabulary")
        if any(not name or name == "unknown" for name in (*self.cell_types, *self.cytokines)):
            raise ValueError("cell/cytokine vocabulary requires nonempty, nonreserved labels")
        if any(
            not isinstance(name, str) or not name.strip() or name == "unknown"
            for name in (*self.processing_pathways, *self.interventions)
        ):
            raise ValueError("pathway/intervention vocabulary requires nonempty named claims")
        if (set(self.cytokine_dose_units) | set(self.cytokine_duration_features)) - set(
            self.cytokines
        ):
            raise ValueError("cytokine numeric features require an allocated exposure")
        if any(not unit for unit in self.cytokine_dose_units.values()):
            raise ValueError("cytokine dose features require declared units")
        profile_ids, profile_scopes = set(), set()
        for profile in self.machinery_references:
            if not isinstance(profile, MachineryProfile):
                raise ValueError("machinery references require typed profiles")
            scope = (profile.species, profile.cell_type)
            if profile.profile_id in profile_ids or scope in profile_scopes:
                raise ValueError("duplicate or ambiguous machinery reference profile")
            profile_ids.add(profile.profile_id)
            profile_scopes.add(scope)
            if (
                profile.species not in self.species
                or (profile.cell_type is not None and profile.cell_type not in self.cell_types)
                or set(profile.components) - set(self.components)
            ):
                raise ValueError("machinery reference outside declared vocabulary")
        if not 0 <= self.dropout < 1 or not 0 < self.identity_threshold <= 1:
            raise ValueError("invalid dropout or identity threshold")
        if self.max_length < 9:
            raise ValueError("max_length must support canonical cores")
        routes = {endpoint_spec(name).route for name in self.endpoints}
        cellular = bool(routes & {"cellular", "tcr_cellular"})
        contextual = bool(routes & {"cellular", "tcr", "tcr_cellular"})
        if self.components and not cellular:
            raise ValueError("machinery slots need a cellular outcome")
        if (self.cell_types or self.cytokines or self.machinery_references) and not cellular:
            raise ValueError("APC cell/cytokine context requires a cellular outcome")
        if (self.processing_pathways or self.interventions) and not cellular:
            raise ValueError("pathway/intervention context requires a cellular outcome")
        if self.ordinal_scales and not cellular:
            raise ValueError("ordinal context requires a cellular outcome")
        if any(
            not isinstance(name, str)
            or not name.strip()
            or name == "unknown"
            or not isinstance(scale, OrdinalScale)
            for name, scale in self.ordinal_scales.items()
        ):
            raise ValueError("ordinal context requires named typed scales")
        if self.context_species and not contextual:
            raise ValueError("context species slots need a contextual outcome")
        receptor = bool(routes & {"tcr", "tcr_cellular"})
        for name in self.auxiliary_context:
            if name == "mhc_pairing" and not routes & {"pmhc", "cellular", "tcr", "tcr_cellular"}:
                raise ValueError("MHC pairing context requires a pMHC outcome")
            if name == "tcr_pairing" and not receptor:
                raise ValueError("TCR pairing context requires a receptor outcome")
            if (name == "tcr_lineage" or name.startswith("peptide_source:")) and not cellular:
                raise ValueError("source/lineage priors require a cellular outcome")
            if name == "tcr_lineage" and "tcr_cellular" not in routes:
                raise ValueError("TCR lineage context requires a paired cellular outcome")
        for role in self.context_species:
            if role in ("apc", "immune") and not cellular:
                raise ValueError("APC/immune species slots require cellular outcomes")
            if role in ("tcr_a", "tcr_b", "repertoire") and not receptor:
                raise ValueError("receptor species slots require receptor outcomes")
        if self.source_context and not (routes & {"cellular", "tcr_cellular", "digest", "bulk"}):
            raise ValueError("source context has no active loss path")
        if self.enzymes and not (routes & {"digest", "bulk"}):
            raise ValueError("enzyme features have no digest loss path")
        if self.preparation_features and not routes & {"digest", "bulk"}:
            raise ValueError("preparation features have no digest loss path")
        if self.acquisition_features and not any(
            "ms_response" in endpoint_spec(name).parts for name in self.endpoints
        ):
            raise ValueError("acquisition features have no MS loss path")
        if self.observation_features and not any(
            endpoint_spec(name).observation_covariates for name in self.endpoints
        ):
            raise ValueError("observation covariates have no observation loss path")

    def to_dict(self) -> dict[str, Any]:
        """Serializable contract; legacy elution staging adds no keys."""
        data = asdict(self)
        if not self.elution_shared_processing:
            del data["elution_shared_processing"]
        if self.interaction_mode == "mean":
            del data["interaction_mode"]
        if not self.groove_segment_embedding:
            del data["groove_segment_embedding"]
        if not self.imgt_groove_embedding:
            del data["imgt_groove_embedding"]
        if not self.imgt_pocket_cross:
            del data["imgt_pocket_cross"]
        if not self.groove_cross_chain:
            del data["groove_cross_chain"]
        if not self.query_anchor_embedding:
            del data["query_anchor_embedding"]
        if not self.groove_pocket_pool:
            del data["groove_pocket_pool"]
        if not self.interaction_cooperativity:
            del data["interaction_cooperativity"]
        if not self.pmhc_pfr_conditioning:
            del data["pmhc_pfr_conditioning"]
        if not self.pmhc_refine:
            del data["pmhc_refine"]
        if self.register_pooling_mode == "normalized":
            del data["register_pooling_mode"]
        if not self.elution_pmhc_residual:
            del data["elution_pmhc_residual"]
        if not self.elution_mhc_abundance:
            del data["elution_mhc_abundance"]
        if not self.class1_overhang_registers:
            del data["class1_overhang_registers"]
        if not self.tcell_class_specific_tcr:
            del data["tcell_class_specific_tcr"]
        if not self.tcell_register_aligned_supply:
            del data["tcell_register_aligned_supply"]
        if not self.class_conditioned_assay_offset:
            del data["class_conditioned_assay_offset"]
        if not self.direct_pocket_bilinear:
            del data["direct_pocket_bilinear"]
        if not self.imgt_pocket_cross and not self.direct_pocket_bilinear:
            del data["imgt_pocket_slot_scheme"]
        if not self.shared_processing_v2:
            del data["shared_processing_v2"]
        if self.candidate_junction_mode == "masked":
            del data["candidate_junction_mode"]
            del data["candidate_junction_temp"]
        if not self.elution_junction_processing:
            del data["elution_junction_processing"]
        if not self.elution_pathway_conditioning:
            del data["elution_pathway_conditioning"]
        if not self.tcell_tracks:
            del data["tcell_tracks"]
            del data["tcell_ordinal_loss_weight"]
        if not self.tcell_staged:
            del data["tcell_staged"]
        if self.elution_staging == "legacy_joint":
            for name in ("elution_staging", "elution_affinity_endpoint", "elution_detach_affinity"):
                del data[name]
        if not self.state_bottleneck:
            del data["state_bottleneck"]
            del data["state_mode"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrestoConfig:
        data = dict(data)
        if "assay_tracks" in data:
            data["assay_tracks"] = tuple(AssayTrack(**track) for track in data["assay_tracks"])
        if "tcell_tracks" in data:
            data["tcell_tracks"] = tuple(TCellTrack(**track) for track in data["tcell_tracks"])
        for name in (
            "endpoints",
            "identity_tasks",
            "mhc_classes",
            "species",
            "source_categories",
            "auxiliary_context",
            "components",
            "cell_types",
            "cytokines",
            "cytokine_duration_features",
            "processing_pathways",
            "interventions",
            "context_species",
            "enzymes",
            "acquisition_features",
            "observation_features",
            "preparation_features",
        ):
            if name in data:
                data[name] = tuple(data[name])
        if "machinery_profiles" in data:
            data["machinery_profiles"] = {
                key: tuple(value) for key, value in data["machinery_profiles"].items()
            }
        if "machinery_references" in data:
            data["machinery_references"] = tuple(
                MachineryProfile.from_dict(value) for value in data["machinery_references"]
            )
        if "quantity_domains" in data:
            data["quantity_domains"] = {
                name: QuantitativeDomain(**value)
                for name, value in data["quantity_domains"].items()
            }
        if "ordinal_scales" in data:
            data["ordinal_scales"] = {
                name: OrdinalScale.from_dict(value)
                for name, value in data["ordinal_scales"].items()
            }
        return cls(**data)


@dataclass
class Prediction:
    observation_id: str
    endpoint: str
    value: torch.Tensor | None = None
    identity: dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    assay_values: dict[str, torch.Tensor] = field(default_factory=dict)
    tcell_values: dict[str, torch.Tensor] = field(default_factory=dict)
    tcell_intensity_values: dict[str, torch.Tensor] = field(default_factory=dict)
    tcell_intensity_value: torch.Tensor | None = None
    reference_value: torch.Tensor | None = None


def candidates(
    peptide: str, mhc_class: str, *, class1_overhang: bool = False
) -> list[tuple[int, str, list[float]]]:
    """Return (start, sequence, flank/peptide lengths); no flank sequence cutoff.

    For Class I peptides of length <= 14, returns the canonical single bulging span
    `(0, peptide)`. When `class1_overhang=True` and `len(peptide) >= 15`, scans
    sliding 9-mer sub-registers `(s, peptide[s:s+9], [log1p(left), log1p(right), log1p(L)])`
    to capture nested Class I epitopes and long N/C-terminal overhangs cleanly.
    """
    if mhc_class == "I":
        if class1_overhang and len(peptide) >= 15:
            result = []
            for start in range(len(peptide) - 8):
                left, right = peptide[:start], peptide[start + 9 :]
                pfr = [math.log1p(length) for length in (len(left), len(right), len(peptide))]
                result.append((start, peptide[start : start + 9], pfr))
            return result
        return [(0, peptide, [0.0, 0.0, math.log1p(len(peptide))])]
    if mhc_class != "II" or len(peptide) < 9:
        raise ValueError("canonical class-II prediction requires a peptide of at least 9 residues")
    result = []
    for start in range(len(peptide) - 8):
        left, right = peptide[:start], peptide[start + 9 :]
        pfr = [math.log1p(length) for length in (len(left), len(right), len(peptide))]
        result.append((start, peptide[start : start + 9], pfr))
    return result


class ObservationOffset(nn.Module):
    """One intercept, with optional measured observation covariates."""

    def __init__(self, width: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.covariates = nn.Linear(width, 1, bias=False) if width else None

    def forward(self, values):
        return self.bias if self.covariates is None else self.bias + self.covariates(values)


class Presto(nn.Module):
    """One encoder; construct only modules reachable from the declared tasks."""

    def __init__(self, config: PrestoConfig):
        super().__init__()
        self.config = config
        self.encoder = SequenceEncoder(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            vocab_size=21,
            max_len=config.max_length,
            dropout=config.dropout,
        )
        routes = {endpoint_spec(name).route for name in config.endpoints}
        self.has_pmhc = bool(routes & {"pmhc", "cellular", "tcr", "tcr_cellular"})
        self.has_cellular = bool(routes & {"cellular", "tcr_cellular"})
        self.has_tcr = bool(routes & {"tcr", "tcr_cellular"})
        self.staged_elution = config.elution_staging == "affinity_downstream"
        self.fixed_tcell = bool(config.tcell_tracks)
        # Staged elution bypasses context_fusion; other contextual routes still use it.
        self.has_context = bool(
            {
                endpoint_spec(name).route
                for name in config.endpoints
                if not (self.staged_elution and name == "elution_observation")
                and not (self.fixed_tcell and name == "tcell_response")
            }
            & {"cellular", "tcr", "tcr_cellular"}
        )
        self.elution_stability = self.staged_elution and "half_life" in config.endpoints
        self.has_digest = bool(routes & {"bulk", "digest"})
        self.has_ms = any("ms_response" in endpoint_spec(name).parts for name in config.endpoints)
        self.has_coreceptor = bool(set(config.endpoints) & {"tcell_response", "tcr_response"})
        self.shared_has_coreceptor = self.has_coreceptor and (
            not self.fixed_tcell or "tcr_response" in config.endpoints
        )
        self.identity_heads = nn.ModuleDict()
        for name in config.identity_tasks:
            width, classes = {
                "mhc_class": (2 * config.d_model, 2),
                "mhc_species": (config.d_model, len(config.species)),
                "tcr_species": (config.d_model, len(config.species)),
                "tcr_lineage": (2 * config.d_model, 2),
                "mhc_pairing": (2 * config.d_model, 2),
                "tcr_pairing": (2 * config.d_model, 2),
                "peptide_source": (config.d_model, len(config.source_categories)),
            }[name]
            self.identity_heads[name] = nn.Linear(width, classes)
        self.register_buffer("_class_one_hot", torch.eye(2), persistent=False)
        if self.has_pmhc:
            if config.groove_segment_embedding:
                self.groove_segment = nn.Embedding(2, config.d_model)
                nn.init.zeros_(self.groove_segment.weight)
            if config.imgt_groove_embedding:
                self.imgt_slot_embedding = nn.Embedding(IMGT_VOCAB_SIZE, config.d_model)
                nn.init.zeros_(self.imgt_slot_embedding.weight)
            if config.imgt_pocket_cross:
                with torch.random.fork_rng(devices=[]):
                    self.imgt_pocket_cross = IMGTPocketCrossInteraction(
                        config.d_model,
                        n_heads=config.n_heads,
                        dropout=config.dropout,
                        pocket_slot_scheme=config.imgt_pocket_slot_scheme,
                    )
            if config.groove_cross_chain:
                with torch.random.fork_rng(devices=[]):
                    self.groove_self_attn = nn.MultiheadAttention(
                        config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
                    )
                    self.groove_self_norm = nn.LayerNorm(config.d_model)
                self.groove_self_gate = nn.Parameter(torch.zeros(1))
            if config.query_anchor_embedding:
                self.query_n_anchor = nn.Embedding(5, config.d_model)
                self.query_c_anchor = nn.Embedding(5, config.d_model)
                self.query_class_emb = nn.Embedding(2, config.d_model)
                nn.init.zeros_(self.query_n_anchor.weight)
                nn.init.zeros_(self.query_c_anchor.weight)
                nn.init.zeros_(self.query_class_emb.weight)
            if config.groove_pocket_pool:
                with torch.random.fork_rng(devices=[]):
                    self.groove_pocket_queries = nn.Parameter(torch.randn(8, config.d_model) * 0.02)
                    self.groove_pocket_attn = nn.MultiheadAttention(
                        config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
                    )
                    self.groove_pocket_norm = nn.LayerNorm(config.d_model)
                    self.groove_pocket_proj = nn.Linear(8 * config.d_model, config.d_model)
                    nn.init.zeros_(self.groove_pocket_proj.weight)
                    nn.init.zeros_(self.groove_pocket_proj.bias)
            self.residue_interaction = nn.MultiheadAttention(
                config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
            )
            self.interaction_norm = nn.LayerNorm(config.d_model)
            if config.interaction_mode in ("anchor_v2", "anchor_v3"):
                self.interaction_ffn_norm = nn.LayerNorm(config.d_model)
                self.interaction_ffn = nn.Sequential(
                    nn.Linear(config.d_model, config.d_ff),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_ff, config.d_model),
                    nn.Dropout(config.dropout),
                )
                if config.interaction_cooperativity:
                    with torch.random.fork_rng(devices=[]):
                        self.interaction_self_attn = nn.MultiheadAttention(
                            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
                        )
                        self.interaction_self_norm = nn.LayerNorm(config.d_model)
                    self.interaction_self_gate = nn.Parameter(torch.zeros(1))
                self.residue_interaction_2 = nn.MultiheadAttention(
                    config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
                )
                self.interaction_norm_2 = nn.LayerNorm(config.d_model)
                n_anchor_slots = 9 if config.interaction_mode == "anchor_v3" else 5
                self.interaction_anchor_proj = nn.Linear(
                    n_anchor_slots * config.d_model, config.d_model
                )
                self.interaction_anchor_norm = nn.LayerNorm(config.d_model)
            self.pmhc_fusion = self._fusion(
                3 * config.d_model
                + 2 * len(config.species)
                + 2 * ("mhc_pairing" in config.auxiliary_context)
            )
            if config.pmhc_pfr_conditioning:
                with torch.random.fork_rng(devices=[]):
                    self.pmhc_pfr_mlp = nn.Sequential(
                        nn.Linear(2 * config.d_model + 6, config.d_model),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_model, config.d_model),
                    )
                    nn.init.zeros_(self.pmhc_pfr_mlp[-1].weight)
                    nn.init.zeros_(self.pmhc_pfr_mlp[-1].bias)
            if config.pmhc_refine:
                with torch.random.fork_rng(devices=[]):
                    self.pmhc_refine = nn.Sequential(
                        nn.LayerNorm(config.d_model),
                        nn.Linear(config.d_model, config.d_ff),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_ff, config.d_model),
                    )
                    nn.init.zeros_(self.pmhc_refine[-1].weight)
                    nn.init.zeros_(self.pmhc_refine[-1].bias)
        source_width = (2 * config.d_model + 6) if config.source_context else 0
        self.context_width = (
            (3 * config.d_model if self.has_tcr else 0)
            + (source_width if self.has_cellular else 0)
            + len(config.context_species) * len(config.species)
            + sum(
                5 + (2 if name in config.component_activity_units else 0)
                for name in config.components
            )
            + (len(config.cell_types) + 1 if config.cell_types else 0)
            + len(EXPOSURE_STATES) * len(config.cytokines)
            + 2 * (len(config.cytokine_dose_units) + len(config.cytokine_duration_features))
            + len(EXPOSURE_STATES) * (len(config.processing_pathways) + len(config.interventions))
            + sum(len(scale.levels) + 1 for scale in config.ordinal_scales.values())
            + (len(DELIVERY) if self.has_cellular else 0)
            + (len(CORECEPTORS) if self.shared_has_coreceptor else 0)
            + 2 * sum(name != "mhc_pairing" for name in config.auxiliary_context)
        )
        self.tcell_context_width = self.context_width + (
            len(CORECEPTORS) if self.fixed_tcell and not self.shared_has_coreceptor else 0
        )
        if self.has_context:
            self.context_fusion = self._fusion(config.d_model + self.context_width)
        if self.staged_elution:
            self.harmonized_apc = config.state_bottleneck > 0 and config.state_mode == "harmonized"
            apc_start = (
                (3 * config.d_model if self.has_tcr else 0)
                + source_width
                + len(config.context_species) * len(config.species)
            )
            apc_width = (
                sum(
                    5 + (2 if n in config.component_activity_units else 0)
                    for n in config.components
                )
                + (len(config.cell_types) + 1 if config.cell_types else 0)
                + 3 * len(config.cytokines)
                + 2 * (len(config.cytokine_dose_units) + len(config.cytokine_duration_features))
                + 3 * (len(config.processing_pathways) + len(config.interventions))
            )
            self.apc_context_slice = (apc_start, apc_start + apc_width)
            # Supply-side term: peptide (+ source/flank when source_context) and APC context.
            self.elution_processing = nn.Sequential(
                nn.Linear(
                    config.d_model + self.context_width - (apc_width if self.harmonized_apc else 0),
                    config.d_model,
                ),
                nn.GELU(),
                nn.Linear(config.d_model, 1, bias=False),
            )
            if config.elution_pmhc_residual:
                self.elution_pmhc_mlp = nn.Sequential(
                    nn.Linear(config.d_model + 2, max(32, config.d_model // 2)),
                    nn.GELU(),
                    nn.Linear(max(32, config.d_model // 2), 1),
                )
                nn.init.zeros_(self.elution_pmhc_mlp[-1].weight)
                nn.init.zeros_(self.elution_pmhc_mlp[-1].bias)
                self.elution_binding_linear = nn.Parameter(torch.full((2,), 0.54132485))
                self.elution_binding_sat_gain = nn.Parameter(torch.zeros(2))
                self.elution_binding_sat_center = nn.Parameter(torch.tensor([2.5, 2.7]))
                self.elution_binding_sat_slope = nn.Parameter(torch.tensor([1.5, 1.2]))
                if self.elution_stability:
                    self.elution_stability_class_weight = nn.Parameter(torch.zeros(2))
            else:
                self.elution_affinity_weight = nn.Parameter(torch.tensor(1.0))
                if self.elution_stability:
                    self.elution_stability_weight = nn.Parameter(torch.tensor(0.5))
            if config.elution_mhc_abundance:
                with torch.random.fork_rng(devices=[]):
                    self.elution_mhc_abundance_mlp = nn.Sequential(
                        nn.LayerNorm(2 * config.d_model),
                        nn.Linear(2 * config.d_model, max(32, config.d_model // 2)),
                        nn.GELU(),
                        nn.Linear(max(32, config.d_model // 2), 1),
                    )
                nn.init.zeros_(self.elution_mhc_abundance_mlp[-1].weight)
                nn.init.zeros_(self.elution_mhc_abundance_mlp[-1].bias)
            if config.state_bottleneck:
                self.apm_state = (
                    HarmonizedAPCState(config)
                    if self.harmonized_apc
                    else ApmStateBottleneck(
                        config.state_bottleneck, config.d_model, config.state_mode
                    )
                )
        if self.has_digest:
            self.digest_fusion = self._fusion(
                config.d_model
                + source_width
                + len(config.enzymes)
                + 1
                + 2 * len(config.preparation_features)
            )
        if self.has_ms:
            self.ms_fusion = self._fusion(config.d_model + 2 * len(config.acquisition_features))
        binary_parts = {
            part
            for name in config.endpoints
            if not endpoint_spec(name).quantitative
            for part in endpoint_spec(name).parts
        }
        if self.staged_elution:
            binary_parts.discard("cellular")
        for part in sorted(binary_parts):
            setattr(
                self,
                "ms_score" if part == "ms_response" else f"{part}_score",
                nn.Linear(config.d_model, 1, bias=False),
            )
        self.quantity_scores = nn.ModuleDict()
        for domain in config.quantity_domains:
            parts = {
                part
                for name in config.endpoints
                if quantity_domain(name) == domain
                for part in endpoint_spec(name).parts
            }
            self.quantity_scores[domain] = nn.ModuleDict(
                {part: nn.Linear(config.d_model, 1, bias=False) for part in sorted(parts)}
            )
        self.heads = nn.ModuleDict()
        self.offsets = nn.ModuleDict()
        self.affinity_endpoints = tuple(sorted(set(config.endpoints) & AFFINITY_ENDPOINTS))
        self.assay_endpoints = tuple(
            sorted(set(self.affinity_endpoints) | {track.endpoint for track in config.assay_tracks})
        )
        for name in config.endpoints:
            spec = endpoint_spec(name)
            if spec.parts:
                self.offsets[name] = ObservationOffset(
                    2 * len(config.observation_features) if spec.observation_covariates else 0
                )
            elif name not in self.assay_endpoints and not (
                name == "tcell_response" and config.tcell_tracks
            ):
                self.heads[name] = nn.Linear(config.d_model, 1)
        if self.assay_endpoints:
            with torch.random.fork_rng(devices=[]):
                self.assay_layer = AssayMeasurementLayer(
                    config.d_model, self.assay_endpoints, config.assay_tracks
                )
            if config.direct_pocket_bilinear:
                with torch.random.fork_rng(devices=[]):
                    self.direct_pocket_bilinear = DirectPocketBilinearHead(
                        mhc_classes=config.mhc_classes,
                        d_rank=24,
                        d_hidden=192,
                        dropout=config.dropout,
                        pocket_slot_scheme=config.imgt_pocket_slot_scheme,
                    )
            if config.class_conditioned_assay_offset and "II" in config.mhc_classes:
                self.assay_class_offset = nn.ParameterDict(
                    {
                        f"{ep}__II": nn.Parameter(torch.zeros(()))
                        for ep in ("ic50", "kd_via_ic50", "kd_via_ec50")
                        if ep in self.assay_endpoints
                    }
                )
        if config.tcell_tracks:
            with torch.random.fork_rng(devices=[]):
                self.tcell_assay_layer = TCellAssayLayer(
                    config.d_model + self.tcell_context_width,
                    config.tcell_tracks,
                    hidden_width=config.d_model,
                    include_positivity=not config.tcell_staged,
                )
            if config.tcell_staged:
                with torch.random.fork_rng(devices=[]):
                    self.tcell_staged_head = TCellStagedHead(
                        config.d_model,
                        self.tcell_context_width,
                        config.tcell_tracks,
                        mhc_classes=config.mhc_classes,
                        dropout=config.dropout,
                        has_affinity=bool(self.assay_endpoints),
                        has_stability="half_life" in config.endpoints,
                        has_presentation=bool(config.elution_pmhc_residual),
                        class_specific_tcr=config.tcell_class_specific_tcr,
                    )
        if config.elution_pathway_conditioning:
            # Preserve all existing initial weights and the caller's RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.elution_pathway = nn.Linear(2, config.d_model, bias=False)
            nn.init.zeros_(self.elution_pathway.weight)
        if config.elution_junction_processing:
            with torch.random.fork_rng(devices=[]):
                self.elution_junction = JunctionProcessing(config.d_model, config.state_bottleneck)
        if config.elution_shared_processing:
            with torch.random.fork_rng(devices=[]):
                self.elution_shared = SharedProcessingModule(config)
        self.assay_support = {track.key: 0 for track in config.assay_tracks}
        self.tcell_support = {
            track.key: {"binary": 0, "conditional_ordinal": 0} for track in config.tcell_tracks
        }
        self.trained_elements = {
            name: torch.zeros_like(p, dtype=torch.bool) for name, p in self.named_parameters()
        }
        self.element_history_complete = set(self.trained_elements)
        self.trained_parameters: set[str] = set()
        self.frozen_targets: tuple[str, ...] = ()
        self.frozen_parameters: set[str] = set()
        self.feature_layout = feature_layout(self)
        self.trained_features = {name: set() for name in self.feature_layout}
        self.exposed_features = {name: set() for name in self.feature_layout}
        self.source_label_support = {name: set() for name in config.source_categories}
        self.support = {
            name: {"status": "random/untrained", "observations": 0, "validation": "not_evaluated"}
            for name in (*config.endpoints, *config.identity_tasks)
        }

    def _fusion(self, width: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(width, self.config.d_model), nn.GELU())

    def assay_selection(self, row):
        if row.endpoint not in self.assay_endpoints:
            return None
        track = observation_track(row)
        if track is not None and track not in self.config.assay_tracks:
            raise ValueError(f"assay track outside checkpoint registry: {track.key}")
        return track

    def tcell_selection(self, row):
        if row.endpoint != "tcell_response" or not self.config.tcell_tracks:
            return None
        track = row.tcell_track
        if row.value is not None and track is None:
            raise ValueError("measured T-cell response requires a declared output track")
        if track is not None and track not in self.config.tcell_tracks:
            raise ValueError(f"T-cell track outside checkpoint registry: {track.key}")
        return track

    def assay_capabilities(self):
        return {
            track.key: {
                "observations": self.assay_support[track.key],
                "status": "directly_supervised"
                if self.assay_support[track.key]
                and self.assay_layer.trained(track, self.trained_elements)
                else "indirectly_trained"
                if self.assay_layer.trained(track, self.trained_elements)
                else "random/untrained",
                "combination_exposed": bool(self.assay_support[track.key]),
                "validation": "not_evaluated",
                "descriptor": asdict(track),
            }
            for track in self.config.assay_tracks
        }

    def tcell_capabilities(self):
        if not self.config.tcell_tracks:
            return {}
        return {
            track.key: {
                "observations": self.tcell_support[track.key]["binary"],
                "conditional_ordinal_observations": self.tcell_support[track.key][
                    "conditional_ordinal"
                ],
                "status": "directly_supervised"
                if self.tcell_support[track.key]["binary"]
                and (
                    self.tcell_staged_head.trained(track, self.trained_elements)
                    if self.config.tcell_staged
                    else self.tcell_assay_layer.trained(track, self.trained_elements)
                )
                else "random/untrained",
                "conditional_ordinal_status": "directly_supervised"
                if self.tcell_support[track.key]["conditional_ordinal"]
                and self.tcell_assay_layer.trained(track, self.trained_elements, ordinal=True)
                else "random/untrained",
                "validation": "not_evaluated",
                "descriptor": asdict(track),
            }
            for track in self.config.tcell_tracks
        }

    def train(self, mode=True):
        super().train(mode)
        # Every declared function depends on the shared encoder. Its dropout
        # must remain disabled for frozen behavior even while sibling heads learn.
        if self.frozen_parameters:
            self.encoder.eval()
            if hasattr(self, "residue_interaction") and any(
                name.startswith("residue_interaction.") for name in self.frozen_parameters
            ):
                self.residue_interaction.eval()
        return self

    @property
    def device(self):
        return self.encoder.embedding.weight.device

    def _tensor(self, values) -> torch.Tensor:
        return torch.tensor(values, dtype=self.encoder.embedding.weight.dtype, device=self.device)

    def _zero(self, length: int) -> torch.Tensor:
        return self.encoder.embedding.weight.new_zeros(length)

    def _component_score(self, endpoint, part, features):
        domain = quantity_domain(endpoint)
        head = (
            self.quantity_scores[domain][part]
            if domain is not None
            else getattr(self, "ms_score" if part == "ms_response" else f"{part}_score")
        )
        return head(features).squeeze(-1)

    def _candidate_mean(self, endpoint, scores):
        domain = quantity_domain(endpoint)
        if domain is None:
            factor = 1.0
        else:
            transform = self.config.quantity_domains[domain].transform
            if transform == "identity":
                return scores.mean()
            factor = math.log(2) if transform == "log2" else math.log(10)
        return (torch.logsumexp(scores * factor, 0) - math.log(len(scores))) / factor

    def _encode(self, rows: Sequence[Observation]):
        sequences: dict[str, None] = {}
        for row in rows:
            route = endpoint_spec(row.endpoint).route if row.endpoint != "identity" else "identity"
            receptor = route in ("tcr", "tcr_cellular") or any(
                name.startswith("tcr") for name in self.config.identity_tasks
            )
            for value in (
                row.peptide,
                row.tcr_a if receptor else "",
                row.tcr_b if receptor else "",
            ):
                if value:
                    sequences[value] = None
            pmhc = route in ("pmhc", "cellular", "tcr", "tcr_cellular", "identity")
            for mhc in row.mhc if pmhc else ():
                for value in (mhc.groove1, mhc.groove2):
                    if value:
                        sequences[value] = None
                known = mhc.class_override or mhc.mhc_class
                need_sliding_9mers = ("II" in self.config.mhc_classes and known != "I") or (
                    self.config.class1_overhang_registers and len(row.peptide) >= 15
                )
                if row.peptide and need_sliding_9mers:
                    for start in range(max(0, len(row.peptide) - 8)):
                        sequences[row.peptide[start : start + 9]] = None
            if self.config.source_context and route in (
                "cellular",
                "tcr_cellular",
                "bulk",
                "digest",
            ):
                for value in (row.context.source_n, row.context.source_c):
                    if value:
                        sequences[value] = None
                if self.config.candidate_junction_mode != "masked":
                    for cand_j in row.context.candidate_junctions:
                        for value in (cand_j.source_n, cand_j.source_c):
                            if value:
                                sequences[value] = None
        names = list(sequences)
        if not names:
            return {}, {}
        token_ids = {aa: i + 1 for i, aa in enumerate(AA)}
        buckets: dict[int, list[str]] = {}
        for value in names:
            length = len(value)
            b = 0 if length <= 16 else 1 if length <= 32 else 2 if length <= 128 else 3
            buckets.setdefault(b, []).append(value)
        pooled_map: dict[str, torch.Tensor] = {}
        states_map: dict[str, torch.Tensor] = {}
        for b_names in buckets.values():
            max_len = max(map(len, b_names))
            cpu_tokens = torch.zeros((len(b_names), max_len), dtype=torch.long)
            for index, value in enumerate(b_names):
                cpu_tokens[index, : len(value)] = torch.tensor(
                    [token_ids[aa] for aa in value], dtype=torch.long
                )
            tokens = cpu_tokens.to(self.device, non_blocking=self.device.type == "cuda")
            pooled, states = self.encoder(tokens)
            for index, name in enumerate(b_names):
                pooled_map[name] = pooled[index]
                states_map[name] = states[index, : len(name)]
        return pooled_map, states_map

    def _groove_states(
        self, groove1: str, groove2: str, states: dict[str, torch.Tensor], mhc_class: str = "I"
    ):
        g1 = states[groove1]
        g2 = states[groove2]
        if self.config.groove_segment_embedding:
            g1 = g1 + self.groove_segment.weight[0]
            g2 = g2 + self.groove_segment.weight[1]
        groove = torch.cat([g1, g2], 0)
        if self.config.imgt_groove_embedding:
            s1, s2 = imgt_groove_slot_indices(groove1, groove2, mhc_class)
            slot_idx = torch.tensor((*s1, *s2), dtype=torch.long, device=groove.device)
            groove = groove + self.imgt_slot_embedding(slot_idx)
        if self.config.groove_cross_chain:
            g_in = groove.unsqueeze(0)
            g_attn, _ = self.groove_self_attn(g_in, g_in, g_in, need_weights=False)
            groove = groove + self.groove_self_gate * self.groove_self_norm(g_attn.squeeze(0))
        return groove

    def _groove_pocket_vector(self, groove: torch.Tensor) -> torch.Tensor:
        """K=8 learned pocket-attention summary over cross-chain groove residues."""
        q = self.groove_pocket_queries.unsqueeze(0)
        g_in = groove.unsqueeze(0)
        out, _ = self.groove_pocket_attn(q, g_in, g_in, need_weights=False)
        out = self.groove_pocket_norm(out.squeeze(0))
        return self.groove_pocket_proj(out.reshape(-1))

    def _add_query_anchor_embedding(self, queries: torch.Tensor, mhc_class: str) -> torch.Tensor:
        """Add invariant N-offset, C-offset, and MHC-class pocket-frame embeddings."""
        length = queries.size(1)
        pos = torch.arange(length, device=queries.device)
        n_idx = pos.clamp_max(4)
        c_idx = (length - 1 - pos).clamp_max(4)
        cls_idx = 0 if mhc_class == "I" else 1
        frame = (
            self.query_n_anchor(n_idx)
            + self.query_c_anchor(c_idx)
            + self.query_class_emb.weight[cls_idx]
        )
        return queries + frame.unsqueeze(0)

    def _anchor_readout(self, out: torch.Tensor) -> torch.Tensor:
        if self.config.interaction_mode == "mean":
            return out.mean(1)
        if self.config.interaction_mode == "anchor_v2":
            p1 = out[:, 0]
            p2 = out[:, 1] if out.size(1) > 1 else out[:, 0]
            p_mid = out.mean(1)
            p_m2 = out[:, -2] if out.size(1) > 1 else out[:, -1]
            p_end = out[:, -1]
            anchors = torch.cat([p1, p2, p_mid, p_m2, p_end], dim=-1)
            return self.interaction_anchor_norm(
                nn.functional.gelu(self.interaction_anchor_proj(anchors))
            )
        # anchor_v3: full 9-slot structural pocket readout preserving Class I (P1, P2, P3,
        # central bulge, P-2, P-1) AND Class II (P1, P4, P6, P7, P9 + TCR-facing P2, P3, P5, P8).
        length = out.size(1)
        p1 = out[:, 0]
        p2 = out[:, min(1, length - 1)]
        p3 = out[:, min(2, length - 1)]
        p4 = out[:, min(3, length - 1)]
        if length >= 9:
            p5 = out[:, 4 : length - 4].mean(1)
        elif length == 8:
            p5 = out[:, 3:5].mean(1)
        else:
            p5 = out.mean(1)
        p6 = out[:, max(0, length - 4)]
        p7 = out[:, max(0, length - 3)]
        p8 = out[:, max(0, length - 2)]
        p9 = out[:, length - 1]
        anchors = torch.cat([p1, p2, p3, p4, p5, p6, p7, p8, p9], dim=-1)
        return self.interaction_anchor_norm(
            nn.functional.gelu(self.interaction_anchor_proj(anchors))
        )

    def _direct_pocket_energy(
        self,
        mhc: MHCInput,
        register_candidates: list[tuple[int, str, list[float]]],
        mhc_class: str,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Compute unblurred `9 × 58` IMGT bilinear contact-energy scalar per register."""
        if not self.config.direct_pocket_bilinear or not hasattr(self, "direct_pocket_bilinear"):
            return None
        core_ids = torch.tensor(
            [peptide_core_aa_ids(core) for _, core, _ in register_candidates],
            dtype=torch.long,
            device=device,
        )
        pan_one = imgt_pan_class_aa_ids(mhc.groove1, mhc.groove2, mhc_class)
        pan_ids = torch.tensor(
            [pan_one] * len(register_candidates),
            dtype=torch.long,
            device=device,
        )
        pfr = torch.tensor(
            [pfr_vals for _, _, pfr_vals in register_candidates],
            dtype=torch.float32,
            device=device,
        )
        return self.direct_pocket_bilinear(core_ids, pan_ids, pfr, mhc_class)

    def _interaction_candidates(
        self, peptide, mhc, register_candidates, states, mhc_class: str = "I"
    ):
        """One residue interaction for either class, before any candidate pooling."""
        queries = torch.stack(
            [
                0.5 * (states[core] + states[peptide][start : start + len(core)])
                for start, core, _ in register_candidates
            ]
        )
        if self.config.query_anchor_embedding:
            queries = self._add_query_anchor_embedding(queries, mhc_class)
        groove = self._groove_states(mhc.groove1, mhc.groove2, states, mhc_class=mhc_class)
        groove = groove.unsqueeze(0).expand(len(queries), -1, -1)
        interaction, _ = self.residue_interaction(queries, groove, groove, need_weights=False)
        out = self.interaction_norm(queries + interaction)
        if self.config.interaction_mode in ("anchor_v2", "anchor_v3"):
            out = self.interaction_ffn_norm(out + self.interaction_ffn(out))
            if self.config.interaction_cooperativity:
                self_out, _ = self.interaction_self_attn(out, out, out, need_weights=False)
                out = out + self.interaction_self_gate * self.interaction_self_norm(self_out)
            interaction2, _ = self.residue_interaction_2(out, groove, groove, need_weights=False)
            out = self.interaction_norm_2(out + interaction2)
        pooled = self._anchor_readout(out)
        if self.config.imgt_pocket_cross:
            core_ids = torch.tensor(
                [peptide_core_aa_ids(core) for _, core, _ in register_candidates],
                dtype=torch.long,
                device=out.device,
            )
            pan_ids = (
                torch.tensor(
                    imgt_pan_class_aa_ids(mhc.groove1, mhc.groove2, mhc_class),
                    dtype=torch.long,
                    device=out.device,
                )
                .unsqueeze(0)
                .expand(len(register_candidates), -1)
            )
            pooled = pooled + self.imgt_pocket_cross(out, core_ids, pan_ids, mhc_class)
        return pooled

    def _precompute_interactions(
        self,
        rows: Sequence[Observation],
        states: dict[str, torch.Tensor],
        pocket_cache: dict[tuple[str, str], torch.Tensor] | None = None,
    ) -> dict[tuple[str, str, str, str], torch.Tensor]:
        """Batch known-class residue cross-attentions across rows grouped by (core_len, class)."""
        if not self.has_pmhc:
            return {}
        buckets: dict[tuple[int, str], list[tuple[tuple[str, str, str, str], list, str, str]]] = {}
        seen: set[tuple[str, str, str, str]] = set()
        for row in rows:
            if row.endpoint == "identity":
                continue
            route = endpoint_spec(row.endpoint).route
            if route not in ("pmhc", "cellular", "tcr", "tcr_cellular"):
                continue
            for mhc in row.mhc:
                mhc_class = mhc.class_override or mhc.mhc_class
                if mhc_class not in self.config.mhc_classes:
                    continue
                key = (row.peptide, mhc.groove1, mhc.groove2, mhc_class)
                if key in seen:
                    continue
                seen.add(key)
                reg_cands = candidates(
                    row.peptide,
                    mhc_class,
                    class1_overhang=self.config.class1_overhang_registers,
                )
                if not reg_cands:
                    continue
                q_len = len(reg_cands[0][1])
                buckets.setdefault((q_len, mhc_class), []).append(
                    (key, reg_cands, mhc.groove1, mhc.groove2)
                )
        cache: dict[tuple[str, str, str, str], torch.Tensor] = {}
        groove_cache: dict[tuple[str, str, str], torch.Tensor] = {}
        for (_, mhc_class), items in buckets.items():
            q_tensors, g_tensors, spans = [], [], []
            core_id_rows, pan_id_rows = [], []
            offset = 0
            for key, reg_cands, g1, g2 in items:
                g_key = (g1, g2, mhc_class)
                g_vec = groove_cache.get(g_key)
                if g_vec is None:
                    g_vec = self._groove_states(g1, g2, states, mhc_class=mhc_class)
                    groove_cache[g_key] = g_vec
                    if self.config.groove_pocket_pool and pocket_cache is not None:
                        pocket_cache[(g1, g2)] = self._groove_pocket_vector(g_vec)
                pan_ids_tuple = (
                    imgt_pan_class_aa_ids(g1, g2, mhc_class)
                    if self.config.imgt_pocket_cross
                    else None
                )
                pep = key[0]
                pep_states = states[pep]
                for start, core, _ in reg_cands:
                    q_tensors.append(0.5 * (states[core] + pep_states[start : start + len(core)]))
                    g_tensors.append(g_vec)
                    if pan_ids_tuple is not None:
                        core_id_rows.append(peptide_core_aa_ids(core))
                        pan_id_rows.append(pan_ids_tuple)
                spans.append((key, offset, offset + len(reg_cands)))
                offset += len(reg_cands)
            queries = torch.stack(q_tensors, dim=0)
            if self.config.query_anchor_embedding:
                queries = self._add_query_anchor_embedding(queries, mhc_class)
            g_lens = [g.size(0) for g in g_tensors]
            max_g = max(g_lens)
            if min(g_lens) == max_g:
                grooves = torch.stack(g_tensors, dim=0)
                kpm = None
            else:
                grooves = queries.new_zeros((len(g_tensors), max_g, queries.size(-1)))
                kpm = torch.ones((len(g_tensors), max_g), dtype=torch.bool, device=queries.device)
                for idx_g, g_vec in enumerate(g_tensors):
                    lg = g_vec.size(0)
                    grooves[idx_g, :lg] = g_vec
                    kpm[idx_g, :lg] = False
            interaction, _ = self.residue_interaction(
                queries, grooves, grooves, key_padding_mask=kpm, need_weights=False
            )
            out = self.interaction_norm(queries + interaction)
            if self.config.interaction_mode in ("anchor_v2", "anchor_v3"):
                out = self.interaction_ffn_norm(out + self.interaction_ffn(out))
                if self.config.interaction_cooperativity:
                    self_out, _ = self.interaction_self_attn(out, out, out, need_weights=False)
                    out = out + self.interaction_self_gate * self.interaction_self_norm(self_out)
                interaction2, _ = self.residue_interaction_2(
                    out, grooves, grooves, key_padding_mask=kpm, need_weights=False
                )
                out = self.interaction_norm_2(out + interaction2)
            pooled = self._anchor_readout(out)
            if self.config.imgt_pocket_cross:
                core_ids_t = torch.tensor(core_id_rows, dtype=torch.long, device=out.device)
                pan_ids_t = torch.tensor(pan_id_rows, dtype=torch.long, device=out.device)
                pooled = pooled + self.imgt_pocket_cross(out, core_ids_t, pan_ids_t, mhc_class)
            for key, start_i, end_i in spans:
                cache[key] = pooled[start_i:end_i]
        return cache

    def _precompute_identities(
        self, rows: Sequence[Observation], encoded: dict[str, torch.Tensor]
    ) -> tuple[dict[tuple[str, Any], torch.Tensor], dict[tuple[str, Any], list[float]]]:
        """Batch all identity-head projections and GPU->CPU softmax transfers once per batch."""
        if not self.identity_heads:
            return {}, {}
        peps, grooves, groove_pairs, tcrs, tcr_pairs = {}, {}, {}, {}, {}
        for row in rows:
            if row.peptide and "peptide_source" in self.identity_heads:
                peps[row.peptide] = None
            for mhc in row.mhc:
                if mhc.groove1 and mhc.groove2:
                    if "mhc_class" in self.identity_heads or "mhc_pairing" in self.identity_heads:
                        groove_pairs[(mhc.groove1, mhc.groove2)] = None
                if "mhc_species" in self.identity_heads:
                    if mhc.groove1:
                        grooves[mhc.groove1] = None
                    if mhc.groove2:
                        grooves[mhc.groove2] = None
            if "tcr_species" in self.identity_heads:
                if row.tcr_a:
                    tcrs[row.tcr_a] = None
                if row.tcr_b:
                    tcrs[row.tcr_b] = None
            if (
                row.tcr_a
                and row.tcr_b
                and ("tcr_lineage" in self.identity_heads or "tcr_pairing" in self.identity_heads)
            ):
                tcr_pairs[(row.tcr_a, row.tcr_b)] = None

        logits_cache: dict[tuple[str, Any], torch.Tensor] = {}
        probs_cache: dict[tuple[str, Any], list[float]] = {}

        def _eval_head(head_name: str, keys: list, feats: list[torch.Tensor]) -> None:
            if not keys:
                return
            out = self.identity_heads[head_name](torch.stack(feats, dim=0))
            if head_name == "peptide_source":
                # Independent sigmoids represented as two logits [0, logit] per category.
                stacked = torch.stack([torch.zeros_like(out), out], dim=-1)
                probs = stacked.detach().softmax(-1).cpu().tolist()
                for idx_k, k in enumerate(keys):
                    for idx_c, cat in enumerate(self.config.source_categories):
                        ck = (f"peptide_source:{cat}", k)
                        logits_cache[ck] = stacked[idx_k, idx_c]
                        probs_cache[ck] = probs[idx_k][idx_c]
            else:
                probs = out.detach().softmax(-1).cpu().tolist()
                for idx_k, k in enumerate(keys):
                    ck = (head_name, k)
                    logits_cache[ck] = out[idx_k]
                    probs_cache[ck] = probs[idx_k]

        if peps and "peptide_source" in self.identity_heads:
            ks = list(peps)
            _eval_head("peptide_source", ks, [encoded[k] for k in ks])
        if groove_pairs:
            ks = list(groove_pairs)
            feats = [torch.cat([encoded[g1], encoded[g2]]) for g1, g2 in ks]
            for hname in ("mhc_class", "mhc_pairing"):
                if hname in self.identity_heads:
                    _eval_head(hname, ks, feats)
        if grooves and "mhc_species" in self.identity_heads:
            ks = list(grooves)
            _eval_head("mhc_species", ks, [encoded[k] for k in ks])
        if tcrs and "tcr_species" in self.identity_heads:
            ks = list(tcrs)
            _eval_head("tcr_species", ks, [encoded[k] for k in ks])
        if tcr_pairs:
            ks = list(tcr_pairs)
            feats = [torch.cat([encoded[a], encoded[b]]) for a, b in ks]
            for hname in ("tcr_lineage", "tcr_pairing"):
                if hname in self.identity_heads:
                    _eval_head(hname, ks, feats)
        return logits_cache, probs_cache

    def _register_beta(self) -> float:
        return math.log(10.0) if self.config.register_pooling_mode == "thermodynamic" else 1.0

    def _pool_registers(self, scores, direction):
        """Pool candidate binding registers (`normalized` or physical `thermodynamic`).

        In `thermodynamic` mode, uses physical log10 partition-function pooling
        `direction * logsumexp(ln(10) * direction * scores) / ln(10)` without the
        unphysical `-log(R)` penalty against longer peptides; Class I singletons
        (`R == 1`) are exact identities under both modes.
        """
        beta = self._register_beta()
        signed = direction * scores * beta
        weights = signed.softmax(0)
        if len(scores) == 1:
            return scores[0], weights
        if self.config.register_pooling_mode == "thermodynamic":
            value = direction * torch.logsumexp(signed, 0) / beta
        else:
            value = direction * (torch.logsumexp(signed, 0) - math.log(len(scores)))
        return value, weights

    def _pfr_features(self, peptide: str, register_candidates, token_states):
        """Peptide-only PFR_N, PFR_C (up to 3 residues each) and length/overhang geometry."""
        pep_states = token_states[peptide]
        pep_len = len(peptide)
        zero_d = self._zero(self.config.d_model)
        pfr_n_list, pfr_c_list, geom_rows = [], [], []
        len_scaled = (pep_len - 9.0) / 4.0
        for start, core, pfr in register_candidates:
            n_start = max(0, start - 3)
            pfr_n_list.append(pep_states[n_start:start].mean(0) if start > n_start else zero_d)
            c_start = start + len(core)
            c_end = min(pep_len, c_start + 3)
            pfr_c_list.append(pep_states[c_start:c_end].mean(0) if c_end > c_start else zero_d)
            geom_rows.append(
                [pfr[0], pfr[1], pfr[2], float(start > 0), float(c_start < pep_len), len_scaled]
            )
        return torch.cat(
            [torch.stack(pfr_n_list), torch.stack(pfr_c_list), self._tensor(geom_rows)], dim=-1
        )

    def _identity(self, row, mhc, encoded, id_logits_cache=None):
        result = {}
        if row.peptide and "peptide_source" in self.identity_heads:
            if id_logits_cache is not None:
                for category in self.config.source_categories:
                    key = f"peptide_source:{category}"
                    result[key] = id_logits_cache[(key, row.peptide)]
            else:
                logits = self.identity_heads["peptide_source"](encoded[row.peptide])
                for index, category in enumerate(self.config.source_categories):
                    # Independent sigmoids represented as two logits for the common CE path.
                    result[f"peptide_source:{category}"] = torch.stack(
                        [logits[index] * 0, logits[index]]
                    )
        if mhc is not None:
            if "mhc_class" in self.identity_heads and mhc.groove1 and mhc.groove2:
                result["mhc_class"] = (
                    id_logits_cache[("mhc_class", (mhc.groove1, mhc.groove2))]
                    if id_logits_cache is not None
                    else self.identity_heads["mhc_class"](
                        torch.cat([encoded[mhc.groove1], encoded[mhc.groove2]])
                    )
                )
            if "mhc_species" in self.identity_heads:
                for name, value in (("mhc1", mhc.groove1), ("mhc2", mhc.groove2)):
                    if value:
                        result[name] = (
                            id_logits_cache[("mhc_species", value)]
                            if id_logits_cache is not None
                            else self.identity_heads["mhc_species"](encoded[value])
                        )
            if "mhc_pairing" in self.identity_heads and mhc.groove1 and mhc.groove2:
                result["mhc_pairing"] = (
                    id_logits_cache[("mhc_pairing", (mhc.groove1, mhc.groove2))]
                    if id_logits_cache is not None
                    else self.identity_heads["mhc_pairing"](
                        torch.cat([encoded[mhc.groove1], encoded[mhc.groove2]])
                    )
                )
        if "tcr_species" in self.identity_heads:
            for name in ("tcr_a", "tcr_b"):
                val = getattr(row, name)
                if val:
                    result[name] = (
                        id_logits_cache[("tcr_species", val)]
                        if id_logits_cache is not None
                        else self.identity_heads["tcr_species"](encoded[val])
                    )
        if row.tcr_a and row.tcr_b:
            if "tcr_lineage" in self.identity_heads:
                result["tcr_lineage"] = (
                    id_logits_cache[("tcr_lineage", (row.tcr_a, row.tcr_b))]
                    if id_logits_cache is not None
                    else self.identity_heads["tcr_lineage"](
                        torch.cat([encoded[row.tcr_a], encoded[row.tcr_b]])
                    )
                )
            if "tcr_pairing" in self.identity_heads:
                result["tcr_pairing"] = (
                    id_logits_cache[("tcr_pairing", (row.tcr_a, row.tcr_b))]
                    if id_logits_cache is not None
                    else self.identity_heads["tcr_pairing"](
                        torch.cat([encoded[row.tcr_a], encoded[row.tcr_b]])
                    )
                )
        return result

    def _identity_probs(self, row, mhc, identity, id_probs_cache=None) -> dict[str, list[float]]:
        if id_probs_cache is None:
            return {
                name: value.detach().softmax(-1).cpu().tolist() for name, value in identity.items()
            }
        probs: dict[str, list[float]] = {}
        for name, value in identity.items():
            if name.startswith("peptide_source:"):
                probs[name] = id_probs_cache[(name, row.peptide)]
            elif name in ("mhc_class", "mhc_pairing") and mhc is not None:
                probs[name] = id_probs_cache[(name, (mhc.groove1, mhc.groove2))]
            elif name == "mhc1" and mhc is not None:
                probs[name] = id_probs_cache[("mhc_species", mhc.groove1)]
            elif name == "mhc2" and mhc is not None:
                probs[name] = id_probs_cache[("mhc_species", mhc.groove2)]
            elif name in ("tcr_a", "tcr_b"):
                probs[name] = id_probs_cache[("tcr_species", getattr(row, name))]
            elif name in ("tcr_lineage", "tcr_pairing"):
                probs[name] = id_probs_cache[(name, (row.tcr_a, row.tcr_b))]
            else:
                probs[name] = value.detach().softmax(-1).cpu().tolist()
        return probs

    def _auxiliary(self, row, identity, probabilities=None):
        names = set(self.config.auxiliary_context) | {
            name
            for name in identity
            if name in ("mhc_pairing", "tcr_pairing", "tcr_lineage")
            or name.startswith("peptide_source:")
        }
        return {
            name: resolve_field(
                identity_vocabulary(name, self.config.species, self.config.source_categories),
                supplied=row.context.auxiliary_values.get(name),
                override=row.context.auxiliary_overrides.get(name),
                prediction=probabilities[name]
                if probabilities is not None and name in probabilities
                else identity[name].detach().softmax(-1).cpu().tolist()
                if name in identity
                else (),
                threshold=self.config.identity_threshold,
            )
            for name in sorted(names)
        }

    def _resolve(self, row, mhc, identity, probabilities=None):
        # The classifier is supervised only by its own labels, not by assay gradients.
        if probabilities is None:
            probabilities = {
                name: value.detach().softmax(-1).cpu().tolist() for name, value in identity.items()
            }
        species = resolve_species(
            self.config.species,
            row.context,
            mhc,
            probabilities,
            defaults=self.config.species_defaults,
            threshold=self.config.identity_threshold,
        )
        class_field = None
        if mhc is not None and mhc.groove1 and mhc.groove2:
            class_field = resolve_field(
                ("I", "II"),
                supplied=mhc.mhc_class,
                override=mhc.class_override,
                prediction=probabilities.get("mhc_class", ()),
                threshold=self.config.identity_threshold,
            )
        cell = resolve_cell_context(
            row.context,
            self.config.cell_types,
            self.config.cytokines,
            self.config.cytokine_dose_units,
            self.config.cytokine_duration_features,
            strict=row.endpoint != "identity"
            and endpoint_spec(row.endpoint).route in ("cellular", "tcr_cellular"),
            processing_pathways=self.config.processing_pathways,
            interventions=self.config.interventions,
            ordinal_scales=self.config.ordinal_scales,
            shared_gene_perturbations=self.config.elution_shared_processing,
        )
        components = resolve_components(
            row.context,
            species["apc"],
            self.config.components,
            self.config.machinery_profiles,
            cell_type=cell.cell_type.value,
            reference_profiles=self.config.machinery_references,
        )
        return species, class_field, components, cell

    def _source(self, row, encoded):
        context = row.context
        values = [
            encoded.get(context.source_n, self._zero(self.config.d_model)),
            encoded.get(context.source_c, self._zero(self.config.d_model)),
        ]
        flags = [float(context.source_n is not None), float(context.source_c is not None)]
        for value in (context.n_terminus, context.c_terminus):
            flags.extend([float(value or False), float(value is not None)])
        return torch.cat([*values, self._tensor(flags)])

    def _covariates(self, names, values, units, expected_units):
        for name in names:
            if name in values and units.get(name) != expected_units[name]:
                raise ValueError(f"numeric feature unit/scale mismatch for {name}")
        return self._tensor(
            [
                item
                for name in names
                for item in (float(values.get(name, 0.0)), float(name in values))
            ]
        )

    def _context(self, row, encoded, species, components, auxiliary, cell):
        route = endpoint_spec(row.endpoint).route
        cellular = route in ("cellular", "tcr_cellular")
        receptor = route in ("tcr", "tcr_cellular")
        values = []
        if self.has_tcr:
            values.append(
                torch.cat([encoded[row.peptide], encoded[row.tcr_a], encoded[row.tcr_b]])
                if receptor
                else self._zero(3 * self.config.d_model)
            )
        if self.config.source_context and self.has_cellular:
            values.append(
                self._source(row, encoded) if cellular else self._zero(2 * self.config.d_model + 6)
            )
        allowed_roles = set()
        if cellular:
            allowed_roles.update(("apc", "immune"))
        if receptor:
            allowed_roles.update(("tcr_a", "tcr_b", "repertoire"))
        scalars: list[float] = []
        n_species = len(self.config.species)
        for role in self.config.context_species:
            if role in allowed_roles:
                scalars.extend(species[role].probabilities)
            else:
                scalars.extend((0.0,) * n_species)
        for name in self.config.components:
            component = components[name]
            has_act = name in self.config.component_activity_units
            if has_act and (
                component.activity is not None
                and component.unit != self.config.component_activity_units[name]
            ):
                raise ValueError(f"activity unit mismatch for {name}")
            if cellular:
                scalars.extend(float(component.state == state) for state in STATES)
                if has_act:
                    scalars.extend(
                        (component.activity or 0.0, float(component.activity is not None))
                    )
            else:
                scalars.extend((0.0,) * (len(STATES) + (2 if has_act else 0)))
        if self.config.cell_types:
            if cellular:
                scalars.extend(cell.cell_type.probabilities)
                scalars.append(float(cell.cell_type.source == "unknown"))
            else:
                scalars.extend((0.0,) * (len(self.config.cell_types) + 1))
        for name in self.config.cytokines:
            exposure = cell.cytokines[name]
            has_dose = name in self.config.cytokine_dose_units
            has_dur = name in self.config.cytokine_duration_features
            if cellular:
                scalars.extend(float(exposure.state == state) for state in EXPOSURE_STATES)
                if has_dose:
                    scalars.extend(
                        (math.log1p(exposure.dose or 0), float(exposure.dose is not None))
                    )
                if has_dur:
                    scalars.extend(
                        (
                            math.log1p(exposure.duration_hours or 0),
                            float(exposure.duration_hours is not None),
                        )
                    )
            else:
                scalars.extend(
                    (0.0,) * (len(EXPOSURE_STATES) + (2 if has_dose else 0) + (2 if has_dur else 0))
                )
        for configured, resolved in (
            (self.config.processing_pathways, cell.processing_pathways),
            (self.config.interventions, cell.interventions),
        ):
            for name in configured:
                if cellular:
                    st = resolved[name].state
                    scalars.extend(float(st == state) for state in EXPOSURE_STATES)
                else:
                    scalars.extend((0.0,) * len(EXPOSURE_STATES))
        for name in sorted(self.config.ordinal_scales):
            scale = self.config.ordinal_scales[name]
            if cellular:
                level = cell.ordinal_values[name].level
                scalars.extend(float(level == label) for label in (*scale.levels, None))
            else:
                scalars.extend((0.0,) * (len(scale.levels) + 1))
        if self.has_cellular:
            if cellular:
                scalars.extend(float(row.context.delivery == value) for value in DELIVERY)
            else:
                scalars.extend((0.0,) * len(DELIVERY))
        if self.shared_has_coreceptor:
            if row.endpoint in ("tcell_response", "tcr_response"):
                state = row.context.coreceptor_override or row.context.coreceptor
                scalars.extend(float(state == value) for value in CORECEPTORS)
            else:
                scalars.extend((0.0,) * len(CORECEPTORS))
        for name in self.config.auxiliary_context:
            if name == "mhc_pairing":
                continue
            allowed = (
                (name == "tcr_pairing" and receptor)
                or (name == "tcr_lineage" and row.endpoint == "tcr_response")
                or (name.startswith("peptide_source:") and cellular)
            )
            if allowed:
                scalars.extend(auxiliary[name].probabilities)
            else:
                scalars.extend((0.0, 0.0))
        if scalars:
            values.append(self._tensor(scalars))
        return torch.cat(values) if values else self._zero(0)

    def _tcell_context(self, row, encoded, species, components, auxiliary, cell):
        context = self._context(row, encoded, species, components, auxiliary, cell)
        if self.fixed_tcell and not self.shared_has_coreceptor:
            state = row.context.coreceptor_override or row.context.coreceptor
            context = torch.cat(
                [context, self._tensor([float(state == value) for value in CORECEPTORS])]
            )
        return context

    def _processing_score(self, features, mhc_class):
        if not self.config.elution_pathway_conditioning:
            return self.elution_processing(features).squeeze(-1)
        cls_idx = 0 if mhc_class == "I" else 1
        pathway = self._class_one_hot[cls_idx].to(dtype=features.dtype)
        hidden = self.elution_processing[0](features) + self.elution_pathway(pathway)
        return self.elution_processing[2](self.elution_processing[1](hidden)).squeeze(-1)

    def _staged_elution(
        self,
        hidden,
        processing,
        selection=None,
        precomputed_affinity=None,
        mhc_class: str | None = None,
    ):
        """Elution logit with binding/stability entering only via their own scores.

        The affinity score is the reference-track decoder of the configured affinity
        endpoint, register-pooled with that endpoint's direction (as for the affinity
        endpoints themselves); higher affinity (lower value) raises the logit.
        """
        affinity_endpoint = self.config.elution_affinity_endpoint
        affinity = (
            precomputed_affinity
            if precomputed_affinity is not None
            else self.assay_layer.score(hidden, affinity_endpoint)
        )
        if precomputed_affinity is None and self.config.elution_detach_affinity:
            affinity = affinity.detach()
        pooled, attention = self._pool_registers(
            affinity, endpoint_spec(affinity_endpoint).register_direction
        )
        if self.config.elution_pmhc_residual:
            cls_idx = 0 if (mhc_class or "I") == "I" else 1
            lin_w = nn.functional.softplus(self.elution_binding_linear[cls_idx])
            sat_g = nn.functional.softplus(self.elution_binding_sat_gain[cls_idx])
            sat_s = nn.functional.softplus(self.elution_binding_sat_slope[cls_idx])
            sat_c = self.elution_binding_sat_center[cls_idx]
            binding = -lin_w * pooled + sat_g * torch.sigmoid(-sat_s * (pooled - sat_c))
        else:
            binding = -self.elution_affinity_weight * pooled
        total = binding + processing
        terms = {"binding": binding, "processing": processing}
        log_weights = (
            endpoint_spec(affinity_endpoint).register_direction * affinity * self._register_beta()
        ).log_softmax(0)
        if self.config.elution_pmhc_residual:
            cls_oh = (
                self._class_one_hot[0 if (mhc_class or "I") == "I" else 1]
                .to(dtype=hidden.dtype)
                .unsqueeze(0)
                .expand(len(hidden), -1)
            )
            pmhc_res_scores = self.elution_pmhc_mlp(torch.cat([hidden, cls_oh], -1)).squeeze(-1)
            pmhc_res_pooled = torch.logsumexp(log_weights + pmhc_res_scores, 0) - torch.logsumexp(
                log_weights, 0
            )
            terms["pmhc_residual"] = pmhc_res_pooled
            terms["pmhc_residual_scores"] = pmhc_res_scores
            total = total + pmhc_res_pooled
            log_weights = (log_weights + pmhc_res_scores).log_softmax(0)
            attention = log_weights.exp()
        if selection is not None:
            terms["selection"] = torch.logsumexp(log_weights + selection, 0) - torch.logsumexp(
                log_weights, 0
            )
            terms["selection_scores"] = selection
            terms["affinity_weights"] = attention
            total = total + terms["selection"]
            attention = (log_weights + selection).softmax(0)
        if self.elution_stability:
            stability, _ = self._pool_registers(
                self.heads["half_life"](hidden).squeeze(-1),
                endpoint_spec("half_life").register_direction,
            )
            if self.config.elution_pmhc_residual:
                cls_idx = 0 if (mhc_class or "I") == "I" else 1
                stab_w = self.elution_stability_class_weight[cls_idx]
            else:
                stab_w = self.elution_stability_weight
            terms["stability"] = stab_w * stability
            total = total + terms["stability"]
        return affinity, total, attention, terms

    def _apply_source_junction(self, row: Observation, cand_j) -> Observation:
        return replace(
            row,
            context=replace(
                row.context,
                source_n=cand_j.source_n,
                source_c=cand_j.source_c,
                n_terminus=cand_j.n_terminus,
                c_terminus=cand_j.c_terminus,
                candidate_junctions=(),
            ),
        )

    def _resolve_candidate_junctions(
        self, row: Observation
    ) -> tuple[Observation, tuple[Observation, ...] | None, list[float] | None]:
        """Resolve `Context.candidate_junctions` per `candidate_junction_mode`."""
        mode = self.config.candidate_junction_mode
        cands = row.context.candidate_junctions
        if not self.config.source_context or mode == "masked" or not cands:
            return row, None, None
        raw_w = [float(c.weight) if c.weight is not None else 1.0 for c in cands]
        w_sum = sum(raw_w) or float(len(cands))
        prior = [w / w_sum for w in raw_w]
        if len(cands) == 1:
            return self._apply_source_junction(row, cands[0]), None, None
        if mode == "sample" and self.training:
            probs = torch.tensor(prior, dtype=torch.float32)
            idx = int(torch.multinomial(probs, num_samples=1).item())
            return self._apply_source_junction(row, cands[idx]), None, None
        # "mil" mode (and deterministic eval for "sample" mode)
        variants = tuple(self._apply_source_junction(row, c) for c in cands)
        return row, variants, prior

    def _eval_staged_processing(
        self,
        row_v: Observation,
        processing_context_v: torch.Tensor,
        mhc,
        encoded,
        token_states,
        register_candidates,
        mhc_class: str,
        components,
        cell,
        hidden: torch.Tensor,
        precomputed_affinity: torch.Tensor | None,
    ):
        processing_features_v = torch.cat([encoded[row_v.peptide], processing_context_v])
        class_processing = self._processing_score(processing_features_v, mhc_class)
        selection = None
        state_diagnostics = None
        if self.harmonized_apc:
            source_vector = (
                torch.cat([encoded[row_v.peptide], self._source(row_v, encoded)])
                if self.config.source_context
                else encoded[row_v.peptide]
            )
            shift, selection, state_diagnostics = self.apm_state(
                components, cell, mhc_class, source_vector, hidden
            )
            class_processing = class_processing + shift
        elif self.config.state_bottleneck:
            class_processing = class_processing + self.apm_state(
                row_v.context, mhc_class, encoded[row_v.peptide]
            )
        junction_terms = None
        if self.config.elution_junction_processing:
            correction, junction_terms = self.elution_junction(
                row_v,
                token_states,
                mhc_class,
                state_diagnostics["state"] if state_diagnostics is not None else None,
            )
            class_processing = class_processing + correction
        shared_terms = None
        if self.config.elution_shared_processing:
            aff_ep = self.config.elution_affinity_endpoint
            shared_supply, shared_selection, shared_terms = self.elution_shared(
                row_v,
                mhc,
                encoded,
                token_states,
                register_candidates,
                mhc_class,
                cell,
                precomputed_affinity,
                endpoint_spec(aff_ep).register_direction,
                state_diagnostics["state"] if state_diagnostics is not None else None,
                pmhc_hidden=hidden,
            )
            detached_log_w = (
                endpoint_spec(aff_ep).register_direction
                * precomputed_affinity.detach()
                * self._register_beta()
            ).log_softmax(0)
            shared_terms["selection_detached_pool"] = torch.logsumexp(
                detached_log_w + shared_selection, 0
            )
            class_processing = class_processing + shared_supply
            selection = shared_selection if selection is None else selection + shared_selection
        return class_processing, selection, state_diagnostics, junction_terms, shared_terms

    def _pmhc(
        self,
        row,
        mhc,
        encoded,
        token_states,
        interaction_cache=None,
        pocket_cache=None,
        id_logits_cache=None,
        id_probs_cache=None,
        row_context_cache=None,
        junction_variants=None,
        junction_prior=None,
    ):
        identity = self._identity(row, mhc, encoded, id_logits_cache=id_logits_cache)
        probabilities = self._identity_probs(row, mhc, identity, id_probs_cache=id_probs_cache)
        species, class_field, components, cell = self._resolve(
            row, mhc, identity, probabilities=probabilities
        )
        auxiliary = self._auxiliary(row, identity, probabilities=probabilities)
        if class_field.source in ("unknown", "unsupported"):
            raise ValueError(
                "MHC class must be supplied or predicted by an allocated identity task"
            )
        pooled, register_info = [], []
        panel = {
            track.key: [] for track in self.config.assay_tracks if track.endpoint == row.endpoint
        }
        tcell_panel = {
            track.key: [] for track in self.config.tcell_tracks if row.endpoint == "tcell_response"
        }
        tcell_intensity_panel = {key: [] for key in tcell_panel}
        selected_tcell = self.tcell_selection(row)
        contextual = endpoint_spec(row.endpoint).route != "pmhc"
        if contextual:
            ctx_key = (
                tuple(tuple(species[r].probabilities) for r in self.config.context_species),
                species["apc"].value,
            )
            if row_context_cache is not None and ctx_key in row_context_cache:
                context = row_context_cache[ctx_key]
            else:
                context = (
                    self._tcell_context(row, encoded, species, components, auxiliary, cell)
                    if tcell_panel
                    else self._context(row, encoded, species, components, auxiliary, cell)
                )
                if row_context_cache is not None:
                    row_context_cache[ctx_key] = context
        else:
            context = None
        staged = self.staged_elution and row.endpoint == "elution_observation"
        processing_context = context
        if staged and self.harmonized_apc:
            start, stop = self.apc_context_slice
            processing_context = torch.cat([context[:start], context[stop:]])
        tail_probs = list(species["mhc1"].probabilities) + list(species["mhc2"].probabilities)
        if "mhc_pairing" in self.config.auxiliary_context:
            tail_probs.extend(auxiliary["mhc_pairing"].probabilities)
        static_tail = torch.cat(
            [encoded[mhc.groove1], encoded[mhc.groove2], self._tensor(tail_probs)]
        )
        pocket_vec = None
        if self.config.groove_pocket_pool:
            g_key = (mhc.groove1, mhc.groove2)
            pocket_vec = (
                pocket_cache[g_key]
                if pocket_cache is not None and g_key in pocket_cache
                else self._groove_pocket_vector(
                    self._groove_states(
                        mhc.groove1,
                        mhc.groove2,
                        token_states,
                        mhc_class=mhc.class_override or mhc.mhc_class or "I",
                    )
                )
            )
        for index, weight in enumerate(class_field.probabilities):
            if weight == 0:
                continue
            mhc_class = ("I", "II")[index]
            if mhc_class not in self.config.mhc_classes:
                raise ValueError("effective class is outside this checkpoint's supported branches")
            register_candidates = candidates(
                row.peptide,
                mhc_class,
                class1_overhang=self.config.class1_overhang_registers,
            )
            cache_key = (row.peptide, mhc.groove1, mhc.groove2, mhc_class)
            candidate_vectors = (
                interaction_cache[cache_key]
                if interaction_cache is not None and cache_key in interaction_cache
                else self._interaction_candidates(
                    row.peptide, mhc, register_candidates, token_states, mhc_class=mhc_class
                )
            )
            hidden = self.pmhc_fusion(
                torch.cat(
                    [
                        candidate_vectors,
                        static_tail.unsqueeze(0).expand(len(register_candidates), -1),
                    ],
                    dim=-1,
                )
            )
            if self.config.pmhc_pfr_conditioning:
                hidden = hidden + self.pmhc_pfr_mlp(
                    self._pfr_features(row.peptide, register_candidates, token_states)
                )
            if pocket_vec is not None:
                hidden = hidden + pocket_vec.unsqueeze(0)
            if self.config.pmhc_refine:
                hidden = hidden + self.pmhc_refine(hidden)
            if contextual and not staged and not tcell_panel:
                hidden = self.context_fusion(
                    torch.cat([hidden, context.expand(len(hidden), -1)], -1)
                )
            spec = endpoint_spec(row.endpoint)
            direct_dg = (
                self._direct_pocket_energy(mhc, register_candidates, mhc_class, hidden.device)
                if (staged or tcell_panel or row.endpoint in self.assay_endpoints)
                else None
            )
            terms = None
            if staged:
                precomputed_affinity = None
                if self.config.elution_shared_processing or direct_dg is not None:
                    aff_ep = self.config.elution_affinity_endpoint
                    precomputed_affinity = self.assay_layer.score(hidden, aff_ep)
                    if direct_dg is not None:
                        precomputed_affinity = precomputed_affinity + direct_dg
                    if self.config.elution_detach_affinity:
                        precomputed_affinity = precomputed_affinity.detach()

                junc_attn_weights = None
                if junction_variants and len(junction_variants) > 1:
                    proc_list, sel_list, det_pool_list = [], [], []
                    state_diag_list, junc_terms_list, shared_terms_list = [], [], []
                    for row_m in junction_variants:
                        ctx_m = self._context(row_m, encoded, species, components, auxiliary, cell)
                        if self.harmonized_apc:
                            start_s, stop_s = self.apc_context_slice
                            pctx_m = torch.cat([ctx_m[:start_s], ctx_m[stop_s:]])
                        else:
                            pctx_m = ctx_m
                        cp_m, sel_m, sd_m, jt_m, st_m = self._eval_staged_processing(
                            row_m,
                            pctx_m,
                            mhc,
                            encoded,
                            token_states,
                            register_candidates,
                            mhc_class,
                            components,
                            cell,
                            hidden,
                            precomputed_affinity,
                        )
                        proc_list.append(cp_m)
                        sel_list.append(sel_m)
                        state_diag_list.append(sd_m)
                        junc_terms_list.append(jt_m)
                        shared_terms_list.append(st_m)
                        if st_m is not None and "selection_detached_pool" in st_m:
                            det_pool_list.append(st_m["selection_detached_pool"])

                    proc_stack = torch.stack(proc_list, dim=0)
                    tau = float(self.config.candidate_junction_temp)
                    prior_t = torch.tensor(
                        junction_prior
                        if junction_prior is not None
                        else [1.0 / len(proc_list)] * len(proc_list),
                        dtype=proc_stack.dtype,
                        device=proc_stack.device,
                    )
                    log_prior = torch.log(prior_t.clamp_min(1e-12))
                    # Normalized logmeanexp: zero log(M) bag-size inflation
                    scaled_logits = log_prior + proc_stack / tau
                    class_processing = tau * torch.logsumexp(scaled_logits, dim=0)
                    junc_attn = torch.softmax(scaled_logits, dim=0)
                    junc_attn_weights = junc_attn.detach().tolist()
                    if any(s is not None for s in sel_list):
                        sel_stack = torch.stack(
                            [
                                s if s is not None else hidden.new_zeros(len(register_candidates))
                                for s in sel_list
                            ],
                            dim=0,
                        )
                        selection = torch.einsum("m,mr->r", junc_attn, sel_stack)
                    else:
                        selection = None
                    best_m = int(torch.argmax(junc_attn).item())
                    state_diagnostics = state_diag_list[best_m]
                    junction_terms = junc_terms_list[best_m]
                    shared_terms = (
                        dict(shared_terms_list[best_m])
                        if shared_terms_list[best_m] is not None
                        else None
                    )
                    if shared_terms is not None and len(det_pool_list) == len(proc_list):
                        shared_terms["selection_detached_pool"] = torch.sum(
                            junc_attn * torch.stack(det_pool_list, dim=0)
                        )
                else:
                    (
                        class_processing,
                        selection,
                        state_diagnostics,
                        junction_terms,
                        shared_terms,
                    ) = self._eval_staged_processing(
                        row,
                        processing_context,
                        mhc,
                        encoded,
                        token_states,
                        register_candidates,
                        mhc_class,
                        components,
                        cell,
                        hidden,
                        precomputed_affinity,
                    )
                scores, value, attention, terms = self._staged_elution(
                    hidden,
                    class_processing,
                    selection,
                    precomputed_affinity=precomputed_affinity,
                    mhc_class=mhc_class,
                )
                if self.config.elution_mhc_abundance:
                    mhc_abund = self.elution_mhc_abundance_mlp(
                        torch.cat([encoded[mhc.groove1], encoded[mhc.groove2]], dim=-1)
                    ).squeeze(-1)
                    value = value + mhc_abund
                    terms["mhc_abundance"] = mhc_abund
                if state_diagnostics is not None:
                    terms["apc_state"] = state_diagnostics
                if junction_terms is not None:
                    terms["junction_processing"] = junction_terms
                if shared_terms is not None:
                    terms["shared_processing"] = shared_terms
                if junc_attn_weights is not None:
                    terms["candidate_junction_weights"] = junc_attn_weights
            elif tcell_panel:
                tcell_hidden = torch.cat([hidden, context.expand(len(hidden), -1)], -1)
                track_outputs = self.tcell_assay_layer(tcell_hidden)
                if self.config.tcell_staged:
                    pep_st = token_states[row.peptide]
                    b_means, b_maxs = [], []
                    for start_pos, core_seq, _ in register_candidates:
                        idxs = tcr_facing_indices(len(core_seq), mhc_class)
                        sel_st = pep_st[[start_pos + idx for idx in idxs]]
                        b_means.append(sel_st.mean(dim=0))
                        b_maxs.append(sel_st.max(dim=0).values)
                    bulge_mean = torch.stack(b_means, dim=0)
                    bulge_max = torch.stack(b_maxs, dim=0)
                    phys8 = tcr_bulge_physicochem(
                        register_candidates, mhc_class, device=hidden.device, dtype=hidden.dtype
                    )
                    aff_ep = self.config.elution_affinity_endpoint or (
                        "ic50"
                        if "ic50" in self.assay_endpoints
                        else (self.assay_endpoints[0] if self.assay_endpoints else None)
                    )
                    aff_raw = (
                        self.assay_layer.score(hidden, aff_ep)
                        if aff_ep is not None and self.assay_endpoints
                        else None
                    )
                    if aff_raw is not None and direct_dg is not None:
                        aff_raw = aff_raw + direct_dg
                    det_aff = aff_raw.detach() if aff_raw is not None else None
                    det_stab = (
                        self.heads["half_life"](hidden).squeeze(-1).detach()
                        if "half_life" in self.heads
                        else None
                    )
                    if self.config.tcell_register_aligned_supply:
                        if det_aff is not None:
                            pooled_aff, aff_attn = self._pool_registers(det_aff, -1)
                        else:
                            pooled_aff = None
                            aff_attn = hidden.new_full((len(hidden),), 1.0 / max(1, len(hidden)))
                        pooled_stab = (
                            self._pool_registers(det_stab, 1)[0] if det_stab is not None else None
                        )
                        pooled_pres = None
                        if self.staged_elution and self.has_cellular:
                            with torch.no_grad():
                                nf_row = replace(
                                    row,
                                    endpoint="elution_observation",
                                    context=Context(),
                                )
                                nf_ctx = self._context(
                                    nf_row, encoded, species, components, auxiliary, cell
                                )
                                if self.harmonized_apc:
                                    s_s, e_s = self.apc_context_slice
                                    nf_pctx = torch.cat([nf_ctx[:s_s], nf_ctx[e_s:]])
                                else:
                                    nf_pctx = nf_ctx
                                nf_cp, nf_sel, _, _, _ = self._eval_staged_processing(
                                    nf_row,
                                    nf_pctx,
                                    mhc,
                                    encoded,
                                    token_states,
                                    register_candidates,
                                    mhc_class,
                                    components,
                                    cell,
                                    hidden,
                                    det_aff,
                                )
                                _, nf_val, _, _ = self._staged_elution(
                                    hidden,
                                    nf_cp,
                                    nf_sel,
                                    precomputed_affinity=det_aff,
                                    mhc_class=mhc_class,
                                )
                                if self.config.elution_mhc_abundance:
                                    nf_val = nf_val + self.elution_mhc_abundance_mlp(
                                        torch.cat([encoded[mhc.groove1], encoded[mhc.groove2]], -1)
                                    ).squeeze(-1)
                                pooled_pres = nf_val.detach()
                        elif self.config.elution_pmhc_residual and hasattr(
                            self, "elution_pmhc_mlp"
                        ):
                            cls_oh_tc = (
                                self._class_one_hot[0 if mhc_class == "I" else 1]
                                .to(dtype=hidden.dtype)
                                .unsqueeze(0)
                                .expand(len(hidden), -1)
                            )
                            det_pres_reg = (
                                self.elution_pmhc_mlp(torch.cat([hidden, cls_oh_tc], -1))
                                .squeeze(-1)
                                .detach()
                            )
                            pooled_pres = (aff_attn.detach() * det_pres_reg).sum()
                        pos_by_track, recog_pooled, tcr_pooled, sup_pooled = (
                            self.tcell_staged_head.score_aligned(
                                bulge_mean,
                                bulge_max,
                                phys8,
                                aff_attn,
                                mhc_class=mhc_class,
                                pooled_affinity=pooled_aff,
                                pooled_stability=pooled_stab,
                                pooled_presentation=pooled_pres,
                                context=context,
                            )
                        )
                        scores = recog_pooled.expand(len(hidden))
                        value = recog_pooled
                        attention = aff_attn
                        terms = {
                            "tcell_recognition": recog_pooled,
                            "tcr_bulge_propensity": tcr_pooled,
                            "tcell_pmhc_supply": sup_pooled,
                        }
                        for key, values in track_outputs.items():
                            positivity = pos_by_track[key]
                            ordinal = torch.stack(
                                [
                                    self._pool_registers(
                                        values["conditional_intensity"][:, level],
                                        spec.register_direction,
                                    )[0]
                                    for level in range(values["conditional_intensity"].shape[-1])
                                ]
                            )
                            tcell_panel[key].append(weight * positivity)
                            tcell_intensity_panel[key].append(weight * ordinal)
                    else:
                        det_pres = None
                        if self.config.elution_pmhc_residual and hasattr(self, "elution_pmhc_mlp"):
                            cls_oh_tc = (
                                self._class_one_hot[0 if mhc_class == "I" else 1]
                                .to(dtype=hidden.dtype)
                                .unsqueeze(0)
                                .expand(len(hidden), -1)
                            )
                            det_pres = (
                                self.elution_pmhc_mlp(torch.cat([hidden, cls_oh_tc], -1))
                                .squeeze(-1)
                                .detach()
                            )
                        pos_by_track, recog_scores, tcr_scores, sup_scores = (
                            self.tcell_staged_head.score_registers(
                                hidden,
                                bulge_mean,
                                bulge_max,
                                phys8,
                                mhc_class=mhc_class,
                                detached_affinity=det_aff,
                                detached_stability=det_stab,
                                detached_presentation=det_pres,
                                context=context,
                            )
                        )
                        scores = recog_scores
                        value, attention = self._pool_registers(scores, spec.register_direction)
                        recog_pooled, _ = self._pool_registers(
                            recog_scores, spec.register_direction
                        )
                        tcr_pooled, _ = self._pool_registers(tcr_scores, spec.register_direction)
                        sup_pooled, _ = self._pool_registers(sup_scores, spec.register_direction)
                        terms = {
                            "tcell_recognition": recog_pooled,
                            "tcr_bulge_propensity": tcr_pooled,
                            "tcell_pmhc_supply": sup_pooled,
                        }
                        for key, values in track_outputs.items():
                            positivity, _ = self._pool_registers(
                                pos_by_track[key], spec.register_direction
                            )
                            ordinal = torch.stack(
                                [
                                    self._pool_registers(
                                        values["conditional_intensity"][:, level],
                                        spec.register_direction,
                                    )[0]
                                    for level in range(values["conditional_intensity"].shape[-1])
                                ]
                            )
                            tcell_panel[key].append(weight * positivity)
                            tcell_intensity_panel[key].append(weight * ordinal)
                else:
                    # Diagnostics use a registry-stable reference track so the observed
                    # output descriptor cannot alter biological computation or attention.
                    diagnostic_key = next(iter(tcell_panel))
                    scores = track_outputs[diagnostic_key]["positivity"]
                    value, attention = self._pool_registers(scores, spec.register_direction)
                    for key, values in track_outputs.items():
                        positivity, _ = self._pool_registers(
                            values["positivity"], spec.register_direction
                        )
                        ordinal = torch.stack(
                            [
                                self._pool_registers(
                                    values["conditional_intensity"][:, level],
                                    spec.register_direction,
                                )[0]
                                for level in range(values["conditional_intensity"].shape[-1])
                            ]
                        )
                        tcell_panel[key].append(weight * positivity)
                        tcell_intensity_panel[key].append(weight * ordinal)
            else:
                scores = (
                    self._component_score(row.endpoint, "cellular", hidden)
                    if "cellular" in spec.parts
                    else self.assay_layer.score(hidden, row.endpoint)
                    if row.endpoint in self.assay_endpoints
                    else self.heads[row.endpoint](hidden).squeeze(-1)
                )
                if row.endpoint in self.assay_endpoints and direct_dg is not None:
                    scores = scores + direct_dg
                cls_off_key = f"{row.endpoint}__{mhc_class}"
                if (
                    row.endpoint in self.assay_endpoints
                    and hasattr(self, "assay_class_offset")
                    and cls_off_key in self.assay_class_offset
                ):
                    scores = scores + self.assay_class_offset[cls_off_key]
                value, attention = self._pool_registers(scores, spec.register_direction)
            if not tcell_panel or selected_tcell is not None:
                pooled.append(weight * value)
            if panel:
                cls_off_key = f"{row.endpoint}__{mhc_class}"
                cls_off = (
                    self.assay_class_offset[cls_off_key]
                    if (
                        hasattr(self, "assay_class_offset")
                        and cls_off_key in self.assay_class_offset
                    )
                    else None
                )
                for key, assay_scores in self.assay_layer(hidden, row.endpoint).items():
                    if direct_dg is not None:
                        assay_scores = assay_scores + direct_dg
                    if cls_off is not None:
                        assay_scores = assay_scores + cls_off
                    track_value, _ = self._pool_registers(assay_scores, spec.register_direction)
                    panel[key].append(weight * track_value)
            register_info.append(
                {
                    "class": mhc_class,
                    "class_probability": weight,
                    "flank_representation": self.config.peptide_context,
                    "starts": [x[0] for x in register_candidates],
                    "weights": attention,
                    "candidate_scores": scores,
                    "pooling": (
                        "thermodynamic_logsumexp"
                        if self.config.register_pooling_mode == "thermodynamic"
                        else "normalized_logsumexp_model_units"
                    ),
                    "direction": spec.register_direction,
                    "entropy": -(attention * attention.clamp_min(1e-12).log()).sum(),
                    **({"elution_terms": terms} if (terms is not None and staged) else {}),
                    **({"tcell_terms": terms} if (terms is not None and bool(tcell_panel)) else {}),
                }
            )
        diagnostics = {
            "mhc": asdict(mhc),
            "species": {name: asdict(value) for name, value in species.items()},
            "class": asdict(class_field),
            "components": {name: asdict(value) for name, value in components.items()},
            "cell_context": {
                **asdict(cell),
                "applies_to_endpoint": endpoint_spec(row.endpoint).route
                in ("cellular", "tcr_cellular"),
            },
            "registers": register_info,
            "auxiliary": {name: asdict(value) for name, value in auxiliary.items()},
        }
        return (
            torch.stack(pooled).sum(0) if pooled else None,
            identity,
            diagnostics,
            {key: torch.stack(values).sum(0) for key, values in panel.items()},
            {key: torch.stack(values).sum(0) for key, values in tcell_panel.items()},
            {key: torch.stack(values).sum(0) for key, values in tcell_intensity_panel.items()},
        )

    def forward(self, rows: Sequence[Observation]) -> list[Prediction]:
        if not rows:
            return []
        for row in rows:
            row.validate(self.config.max_length)
            if row.endpoint != "identity" and row.endpoint not in self.config.endpoints:
                raise ValueError(f"endpoint not allocated: {row.endpoint}")
            if row.endpoint != "identity" and endpoint_spec(row.endpoint).quantitative:
                if (
                    row.quantity_contract
                    != self.config.quantity_domains[quantity_domain(row.endpoint)]
                ):
                    raise ValueError("quantity measurement contract differs from checkpoint domain")
        encoded, token_states = self._encode(rows)
        pocket_cache: dict[tuple[str, str], torch.Tensor] = {}
        interaction_cache = self._precompute_interactions(
            rows, token_states, pocket_cache=pocket_cache
        )
        id_logits_cache, id_probs_cache = self._precompute_identities(rows, encoded)
        results = []
        for row in rows:
            output = Prediction(row.observation_id, row.endpoint)
            if row.endpoint == "identity":
                mhc = row.mhc[0] if row.mhc else None
                output.identity = self._identity(row, mhc, encoded, id_logits_cache=id_logits_cache)
                probabilities = self._identity_probs(
                    row, mhc, output.identity, id_probs_cache=id_probs_cache
                )
                species, mhc_class, _, cell = self._resolve(
                    row, mhc, output.identity, probabilities=probabilities
                )
                output.diagnostics = {
                    "species": {key: asdict(value) for key, value in species.items()},
                    "class": asdict(mhc_class) if mhc_class else None,
                    "cell_context": {**asdict(cell), "applies_to_endpoint": False},
                    "auxiliary": {
                        name: asdict(value)
                        for name, value in self._auxiliary(
                            row, output.identity, probabilities=probabilities
                        ).items()
                    },
                }
                results.append(output)
                continue
            spec = endpoint_spec(row.endpoint)
            route = spec.route
            if route in ("bulk", "digest", "detector"):
                output.identity = self._identity(
                    row, None, encoded, id_logits_cache=id_logits_cache
                )
                probabilities = self._identity_probs(
                    row, None, output.identity, id_probs_cache=id_probs_cache
                )
                output.diagnostics["auxiliary"] = {
                    name: asdict(value)
                    for name, value in self._auxiliary(
                        row, output.identity, probabilities=probabilities
                    ).items()
                }
            if "ms_response" in spec.parts:
                ms_features = torch.cat(
                    [
                        encoded[row.peptide],
                        self._covariates(
                            self.config.acquisition_features,
                            row.context.acquisition,
                            row.context.acquisition_units,
                            self.config.acquisition_units,
                        ),
                    ]
                )
                output.components["ms_response"] = self._component_score(
                    row.endpoint, "ms_response", self.ms_fusion(ms_features)
                )
            if route in ("bulk", "digest"):
                enzyme = row.context.enzyme
                if enzyme is not None and enzyme not in self.config.enzymes:
                    raise ValueError(f"enzyme outside checkpoint vocabulary: {enzyme}")
                features = [encoded[row.peptide]]
                if self.config.source_context:
                    features.append(self._source(row, encoded))
                features.append(
                    self._tensor([float(enzyme == name) for name in (*self.config.enzymes, None)])
                )
                features.append(
                    self._covariates(
                        self.config.preparation_features,
                        row.context.preparation,
                        row.context.preparation_units,
                        self.config.preparation_units,
                    )
                )
                output.components["digest"] = self._component_score(
                    row.endpoint, "digest", self.digest_fusion(torch.cat(features))
                )
            if route in ("pmhc", "cellular", "tcr", "tcr_cellular"):
                # Resolve candidate junctions once per row (only for cellular routes;
                # pmhc endpoints stay strictly context-invariant).
                if route in ("cellular", "tcr_cellular"):
                    eff_row, junction_variants, junction_prior = self._resolve_candidate_junctions(
                        row
                    )
                else:
                    eff_row, junction_variants, junction_prior = row, None, None
                # Provenance differences do not duplicate a biological candidate.
                unique = {}
                for mhc in eff_row.mhc:
                    key = (
                        mhc.groove1,
                        mhc.groove2,
                        mhc.mhc_class,
                        mhc.class_override,
                        mhc.species1,
                        mhc.species2,
                    )
                    unique.setdefault(key, mhc)
                vectors, diagnostics = [], []
                row_context_cache: dict = {}
                for mhc in unique.values():
                    (
                        vector,
                        identity,
                        diagnostic,
                        panel,
                        tcell_panel,
                        tcell_intensity_panel,
                    ) = self._pmhc(
                        eff_row,
                        mhc,
                        encoded,
                        token_states,
                        interaction_cache=interaction_cache,
                        pocket_cache=pocket_cache,
                        id_logits_cache=id_logits_cache,
                        id_probs_cache=id_probs_cache,
                        row_context_cache=row_context_cache,
                        junction_variants=junction_variants,
                        junction_prior=junction_prior,
                    )
                    vectors.append(vector)
                    diagnostics.append(diagnostic)
                    if len(unique) == 1:
                        output.identity = identity
                        output.assay_values = panel
                        output.tcell_values = tcell_panel
                        output.tcell_intensity_values = tcell_intensity_panel
                output.diagnostics["candidates"] = diagnostics
                if row.endpoint == "tcell_response" and self.config.tcell_staged:
                    tc_keys = ("tcell_recognition", "tcr_bulge_propensity", "tcell_pmhc_supply")
                    for tc_key in tc_keys:
                        cand_vals = []
                        for diag in diagnostics:
                            reg_vals = [
                                float(reg["class_probability"]) * reg["tcell_terms"][tc_key]
                                for reg in diag.get("registers", ())
                                if "tcell_terms" in reg and tc_key in reg["tcell_terms"]
                            ]
                            if reg_vals:
                                cand_vals.append(torch.stack(reg_vals).sum(0))
                        if cand_vals:
                            output.components[tc_key] = torch.stack(cand_vals).mean(0)
                if "cellular" in spec.parts:
                    scores = torch.stack(vectors)
                    staged = self.staged_elution and row.endpoint == "elution_observation"
                    pooled_score = self._candidate_mean(row.endpoint, scores)
                    if staged and self.config.elution_shared_processing:
                        p_i = sum(d["class"]["probabilities"][0] for d in diagnostics) / len(
                            diagnostics
                        )
                        p_ii = sum(d["class"]["probabilities"][1] for d in diagnostics) / len(
                            diagnostics
                        )
                        pooled_score = self.elution_shared.calibrate_presentation(
                            pooled_score, p_i, p_ii
                        )
                    output.components["elution_presentation" if staged else "cellular"] = (
                        pooled_score
                    )
                    if staged:
                        output.diagnostics["elution_staging"] = {
                            "mode": self.config.elution_staging,
                            "affinity_endpoint": self.config.elution_affinity_endpoint,
                            "affinity_detached": self.config.elution_detach_affinity,
                            "stability_term": self.elution_stability,
                            "processing_term": (
                                "processing"
                                if self.config.source_context
                                else "processing_peptide_only"
                            ),
                            "processing_identified": False,
                        }
                        if self.config.elution_pathway_conditioning:
                            output.diagnostics["elution_staging"]["pathway_conditioned"] = True
                        if self.config.elution_shared_processing:
                            output.diagnostics["elution_staging"]["shared_processing"] = True
                else:
                    output.value = vectors[0]
            if row.endpoint in self.offsets:
                if not spec.observation_covariates:
                    covariates = self._zero(0)
                else:
                    covariates = self._covariates(
                        self.config.observation_features,
                        row.context.observation,
                        row.context.observation_units,
                        self.config.observation_units,
                    )
                output.value = sum(output.components.values()) + self.offsets[row.endpoint](
                    covariates
                ).squeeze(-1)
            output.reference_value = (
                output.components.get("tcell_recognition")
                if row.endpoint == "tcell_response" and self.config.tcell_staged
                else None
                if row.endpoint == "tcell_response" and self.config.tcell_tracks
                else output.value
            )
            selected = self.assay_selection(row)
            if selected is not None:
                output.value = output.assay_values[selected.key]
            selected_tcell = self.tcell_selection(row)
            if selected_tcell is not None:
                output.value = output.tcell_values[selected_tcell.key]
                output.tcell_intensity_value = output.tcell_intensity_values[selected_tcell.key]
            if row.endpoint in self.assay_endpoints:
                output.diagnostics["measurement"] = {
                    "selected_track": selected.key if selected else None,
                    "method": asdict(observation_method(row)),
                    "mode": self.config.assay_mode,
                    "reference_interpretation": (
                        "unknown-method prediction from the shared decoder, "
                        "not thermodynamic affinity"
                    ),
                }
            if row.endpoint == "tcell_response" and self.config.tcell_tracks:
                output.diagnostics["tcell_measurement"] = {
                    "selected_track": selected_tcell.key if selected_tcell else None,
                    "output_descriptor_only": True,
                    "conditional_intensity_levels": ["low", "intermediate", "high"],
                }
            results.append(output)
        return results

    def score_binding_pairs(self, pairs: Sequence[tuple[str, Any, str]]) -> torch.Tensor:
        """Score pure calibrated pMHC binding (`ic50` reference track, lower = tighter)
        for a batch of `(peptide, mhc, mhc_class)` pairs with ZERO processing, ZERO flanks,
        and ZERO MS bias — used by `AlleleSwapPool` cross-allele binding contrasts.
        """
        if not pairs:
            return self._zero(0)
        aff_ep = self.config.elution_affinity_endpoint or (
            "ic50"
            if "ic50" in self.assay_endpoints
            else (self.assay_endpoints[0] if self.assay_endpoints else None)
        )
        if aff_ep is None or not self.assay_endpoints:
            raise ValueError("score_binding_pairs requires an allocated affinity assay endpoint")
        synth_rows = [
            Observation(
                observation_id=f"allele_swap:{idx}",
                endpoint=aff_ep,
                source="allele_swap",
                peptide=pep,
                mhc=(
                    (
                        replace(mhc, mhc_class=cls, class_override=None)
                        if (mhc.class_override or mhc.mhc_class) != cls
                        else mhc
                    ),
                ),
                split="predict",
            )
            for idx, (pep, mhc, cls) in enumerate(pairs)
        ]
        preds = self.forward(synth_rows)
        return torch.stack([pred.value for pred in preds])

    def capabilities(self) -> dict:
        outputs = {
            name: self.support.get(
                name, {"status": "not_allocated", "observations": 0, "validation": "not_evaluated"}
            )
            for name in dict.fromkeys((*ENDPOINTS, *self.config.endpoints, *IDENTITY_TASKS))
        }
        components = {}
        for name, anchor, mixtures, allocated in (
            (
                "ms_response",
                "ms_response",
                ("bulk_observation", "elution_observation"),
                hasattr(self, "ms_score"),
            ),
            ("digest", "digest_observation", ("bulk_observation",), hasattr(self, "digest_score")),
            (
                "cellular",
                None,
                ("elution_observation",),
                "elution_observation" in self.config.endpoints,
            ),
        ):
            status = "random/untrained" if allocated else "not_allocated"
            if anchor and outputs[anchor]["status"] in ("directly_supervised", "pretrained_frozen"):
                status = outputs[anchor]["status"]
            elif any(
                outputs[endpoint]["status"] in ("directly_supervised", "pretrained_frozen")
                for endpoint in mixtures
            ):
                status = "indirectly_trained"
            components[name] = {
                "status": status,
                "validation": "not_evaluated",
                "identifiable_from_mixture_labels_alone": False,
                "frozen": allocated and frozen_output(self, f"component:{name}"),
            }
        source_membership = {
            name: {
                "status": (
                    "not_allocated"
                    if "peptide_source" not in self.identity_heads
                    else "directly_supervised"
                    if values
                    else "random/untrained"
                ),
                "observed_labels": sorted(values),
                "validation": self.support.get("peptide_source", {}).get(
                    "validation", "not_evaluated"
                ),
            }
            for name, values in self.source_label_support.items()
        }
        quantity_components = {}
        for domain, readouts in self.quantity_scores.items():
            quantity_components[domain] = {}
            for component in readouts:
                relevant = [
                    name
                    for name in self.config.endpoints
                    if quantity_domain(name) == domain and component in endpoint_spec(name).parts
                ]
                direct = [
                    name
                    for name in relevant
                    if len(endpoint_spec(name).parts) == 1
                    and outputs[name]["status"] in ("directly_supervised", "pretrained_frozen")
                ]
                indirect = [
                    name
                    for name in relevant
                    if outputs[name]["status"] in ("directly_supervised", "pretrained_frozen")
                ]
                quantity_components[domain][component] = {
                    "status": "directly_supervised"
                    if direct
                    else "indirectly_trained"
                    if indirect
                    else "random/untrained",
                    "anchor_endpoints": direct,
                    "validation": "not_evaluated",
                    "identifiable_from_mixture_labels_alone": False,
                    "contract": asdict(self.config.quantity_domains[domain]),
                    "frozen": frozen_output(self, f"component:{component}:{domain}"),
                }
        return {
            "outputs": outputs,
            "assay_tracks": self.assay_capabilities(),
            "tcell_tracks": self.tcell_capabilities(),
            "components": components,
            "source_membership": source_membership,
            "quantity_components": quantity_components,
            "freeze": {
                "targets": list(self.frozen_targets),
                "parameters": sorted(self.frozen_parameters),
            },
        }

    def predict(self, rows: Sequence[Observation]) -> list[Prediction]:
        """Inference refuses random requested outputs; qualification stays visible."""
        track_support = None
        tcell_support = None
        for row in rows:
            row.validate(self.config.max_length)
            selected = self.assay_selection(row)
            self.tcell_selection(row)
            if selected is not None and track_support is None:
                track_support = self.assay_capabilities()
            if selected is not None and track_support[selected.key]["status"] == "random/untrained":
                raise ValueError(f"assay track: no trained support for {selected.key}")
            if row.endpoint == "tcell_response" and self.config.tcell_tracks:
                if tcell_support is None:
                    tcell_support = self.tcell_capabilities()
                unsupported = [
                    key
                    for key, value in tcell_support.items()
                    if value["status"] == "random/untrained"
                    or (
                        self.config.tcell_ordinal_loss_weight > 0
                        and value["conditional_ordinal_status"] == "random/untrained"
                    )
                ]
                if unsupported:
                    raise ValueError(
                        "T-cell tracks lack direct trained support: " + ", ".join(unsupported)
                    )
            relevant_identity = tuple(
                name for name in self.config.identity_tasks if has_identity_input(row, name)
            )
            requested = (
                relevant_identity
                if row.endpoint == "identity"
                else (row.endpoint, *relevant_identity)
            )
            if not requested:
                raise ValueError("no allocated identity capability for these inputs")
            if "peptide_source" in requested:
                missing_categories = [
                    key for key, values in self.source_label_support.items() if not values
                ]
                if missing_categories:
                    raise ValueError(f"untrained peptide source outputs: {missing_categories}")
            if row.endpoint != "identity" and endpoint_spec(row.endpoint).route in (
                "cellular",
                "tcr_cellular",
            ):
                unsupported = (
                    set(row.context.components) | set(row.context.component_overrides)
                ) - set(self.config.components)
                if unsupported:
                    raise ValueError(
                        "unsupported APC component effects: " + ", ".join(sorted(unsupported))
                    )
            for name in requested:
                if self.support.get(name, {}).get("status", "not_allocated") in (
                    "not_allocated",
                    "random/untrained",
                ):
                    raise ValueError(f"{name}: no trained output support")
            if row.endpoint != "identity" and any(
                not (candidate.class_override or candidate.mhc_class) for candidate in row.mhc
            ):
                if self.support.get("mhc_class", {}).get("status") not in (
                    "directly_supervised",
                    "pretrained_frozen",
                ):
                    raise ValueError("inferred class requires trained identity support")
        required = {name for name, _ in self.named_parameters() if name.startswith("encoder.")}
        prefixes = set()
        for row in rows:
            prefixes.update(
                f"identity_heads.{task}."
                for task in self.config.identity_tasks
                if has_identity_input(row, task)
            )
            if row.endpoint == "identity":
                continue
            route = endpoint_spec(row.endpoint).route
            if route in ("pmhc", "cellular", "tcr", "tcr_cellular"):
                prefixes.update(
                    (
                        "groove_segment.",
                        "imgt_slot_embedding.",
                        "imgt_pocket_cross.",
                        "groove_self_attn.",
                        "groove_self_norm.",
                        "groove_self_gate",
                        "query_n_anchor.",
                        "query_c_anchor.",
                        "query_class_emb.",
                        "groove_pocket_",
                        "pmhc_fusion.",
                        "pmhc_pfr_mlp.",
                        "pmhc_refine.",
                        "residue_interaction.",
                        "residue_interaction_2.",
                        "interaction_norm.",
                        "interaction_norm_2.",
                        "interaction_ffn.",
                        "interaction_ffn_norm.",
                        "interaction_self_attn.",
                        "interaction_self_norm.",
                        "interaction_self_gate",
                        "interaction_anchor_proj.",
                        "interaction_anchor_norm.",
                    )
                )
                if len(row.peptide) > 9 and any(
                    (mhc.class_override or mhc.mhc_class) != "I" for mhc in row.mhc
                ):
                    prefixes.add("register_score.")
            staged_row = self.staged_elution and row.endpoint == "elution_observation"
            if staged_row:
                required.update(staged_elution_parameters(self))
            elif route in ("cellular", "tcr", "tcr_cellular"):
                prefixes.add("context_fusion.")
            spec = endpoint_spec(row.endpoint)
            if "ms_response" in spec.parts:
                prefixes.add("ms_fusion.")
            if route in ("bulk", "digest"):
                prefixes.add("digest_fusion.")
            domain = quantity_domain(row.endpoint)
            for part in spec.parts:
                prefixes.add(
                    f"quantity_scores.{domain}.{part}."
                    if domain is not None
                    else "ms_score."
                    if part == "ms_response"
                    else f"{part}_score."
                )
            prefixes.update((f"heads.{row.endpoint}.", f"offsets.{row.endpoint}."))
            if row.endpoint == "tcell_response" and self.config.tcell_tracks:
                required.update(
                    self.tcell_assay_layer.dependencies(
                        ordinal=self.config.tcell_ordinal_loss_weight > 0
                    )
                )
                if self.config.tcell_staged:
                    required.update(self.tcell_staged_head.dependencies())
        required.update(
            name
            for name, _ in self.named_parameters()
            if any(name.startswith(prefix) for prefix in prefixes)
        )
        missing = required - self.trained_parameters
        if missing:
            raise ValueError(
                "requested path contains parameters without gradient/update evidence: "
                + ", ".join(sorted(missing))
            )
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad(), feature_inputs(self, enforce=True):
                result = self(rows)
                for output in result:
                    output.diagnostics["support"] = self.capabilities()
                return result
        finally:
            self.train(was_training)
