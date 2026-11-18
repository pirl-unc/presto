"""The shared observation-based Presto model for all new training.

Register attention is statistical pooling, not physical occupancy. Component
scores in an MS mixture are not independently identified by mixture labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
from ..models.encoders import SequenceEncoder
from .shared_v4_support import feature_inputs, feature_layout
from ..models.modular_dependencies import frozen_output

MODEL_VERSION = "presto.shared.v4"


@dataclass(frozen=True)
class PrestoConfig:
    endpoints: tuple[str, ...] = ("ic50",)
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
    peptide_context: str = "shared_core_context_v2"

    def __post_init__(self):
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrestoConfig:
        data = dict(data)
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


def candidates(peptide: str, mhc_class: str) -> list[tuple[int, str, list[float]]]:
    """Return (start, sequence, flank/peptide lengths); no flank sequence cutoff."""
    if mhc_class == "I":
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
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            vocab_size=21,
            max_len=config.max_length,
            dropout=config.dropout,
        )
        routes = {endpoint_spec(name).route for name in config.endpoints}
        self.has_pmhc = bool(routes & {"pmhc", "cellular", "tcr", "tcr_cellular"})
        self.has_cellular = bool(routes & {"cellular", "tcr_cellular"})
        self.has_tcr = bool(routes & {"tcr", "tcr_cellular"})
        self.has_context = self.has_cellular or self.has_tcr
        self.has_digest = bool(routes & {"bulk", "digest"})
        self.has_ms = any("ms_response" in endpoint_spec(name).parts for name in config.endpoints)
        self.has_coreceptor = bool(set(config.endpoints) & {"tcell_response", "tcr_response"})
        self.identity_heads = nn.ModuleDict()
        for name in config.identity_tasks:
            width, classes = {
                "mhc_class": (256, 2),
                "mhc_species": (128, len(config.species)),
                "tcr_species": (128, len(config.species)),
                "tcr_lineage": (256, 2),
                "mhc_pairing": (256, 2),
                "tcr_pairing": (256, 2),
                "peptide_source": (128, len(config.source_categories)),
            }[name]
            self.identity_heads[name] = nn.Linear(width, classes)
        if self.has_pmhc:
            self.pmhc_fusion = self._fusion(
                384 + 2 * len(config.species) + 2 * ("mhc_pairing" in config.auxiliary_context)
            )
        source_width = 262 if config.source_context else 0
        self.context_width = (
            (384 if self.has_tcr else 0)
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
            + (len(CORECEPTORS) if self.has_coreceptor else 0)
            + 2 * sum(name != "mhc_pairing" for name in config.auxiliary_context)
        )
        if self.has_context:
            self.context_fusion = self._fusion(128 + self.context_width)
        if self.has_digest:
            self.digest_fusion = self._fusion(
                128 + source_width + len(config.enzymes) + 1 + 2 * len(config.preparation_features)
            )
        if self.has_ms:
            self.ms_fusion = self._fusion(128 + 2 * len(config.acquisition_features))
        binary_parts = {
            part
            for name in config.endpoints
            if not endpoint_spec(name).quantitative
            for part in endpoint_spec(name).parts
        }
        for part in sorted(binary_parts):
            setattr(
                self,
                "ms_score" if part == "ms_response" else f"{part}_score",
                nn.Linear(128, 1, bias=False),
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
                {part: nn.Linear(128, 1, bias=False) for part in sorted(parts)}
            )
        self.heads = nn.ModuleDict()
        self.offsets = nn.ModuleDict()
        for name in config.endpoints:
            spec = endpoint_spec(name)
            if spec.parts:
                self.offsets[name] = ObservationOffset(
                    2 * len(config.observation_features) if spec.observation_covariates else 0
                )
            else:
                self.heads[name] = nn.Linear(128, 1)
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

    @staticmethod
    def _fusion(width: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(width, 128), nn.GELU())

    def train(self, mode=True):
        super().train(mode)
        # Every declared function depends on the shared encoder. Its dropout
        # must remain disabled for frozen behavior even while sibling heads learn.
        if self.frozen_parameters:
            self.encoder.eval()
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
                if row.peptide and "II" in self.config.mhc_classes and known != "I":
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
        names = list(sequences)
        tokens = torch.zeros(
            (len(names), max(map(len, names))), dtype=torch.long, device=self.device
        )
        token_ids = {aa: i + 1 for i, aa in enumerate(AA)}
        for index, value in enumerate(names):
            tokens[index, : len(value)] = torch.tensor(
                [token_ids[aa] for aa in value], device=self.device
            )
        pooled, states = self.encoder(tokens)
        return dict(zip(names, pooled, strict=True)), {
            name: states[index, : len(name)] for index, name in enumerate(names)
        }

    @staticmethod
    def _candidate_representation(core, states, start, length):
        """Shared core-relative and full-context views; no flank-only weights."""
        return (core + states[start : start + length].mean(0)) * 0.5

    @staticmethod
    def _pool_registers(scores, direction):
        """Normalized smooth extremum in model units, not physical occupancy.

        The reported weights are derivatives of the pooled score with respect to
        candidate scores. A singleton trains the very same readout as many cores.
        """
        signed = direction * scores
        weights = signed.softmax(0)
        if len(scores) == 1:
            return scores[0], weights
        value = direction * (torch.logsumexp(signed, 0) - math.log(len(scores)))
        return value, weights

    def _identity(self, row, mhc, encoded):
        result = {}
        if row.peptide and "peptide_source" in self.identity_heads:
            logits = self.identity_heads["peptide_source"](encoded[row.peptide])
            for index, category in enumerate(self.config.source_categories):
                # Independent sigmoids represented as two logits for the common CE path.
                result[f"peptide_source:{category}"] = torch.stack(
                    [logits[index] * 0, logits[index]]
                )
        if mhc is not None:
            if "mhc_class" in self.identity_heads and mhc.groove1 and mhc.groove2:
                result["mhc_class"] = self.identity_heads["mhc_class"](
                    torch.cat([encoded[mhc.groove1], encoded[mhc.groove2]])
                )
            if "mhc_species" in self.identity_heads:
                for name, value in (("mhc1", mhc.groove1), ("mhc2", mhc.groove2)):
                    if value:
                        result[name] = self.identity_heads["mhc_species"](encoded[value])
            if "mhc_pairing" in self.identity_heads and mhc.groove1 and mhc.groove2:
                result["mhc_pairing"] = self.identity_heads["mhc_pairing"](
                    torch.cat([encoded[mhc.groove1], encoded[mhc.groove2]])
                )
        if "tcr_species" in self.identity_heads:
            for name in ("tcr_a", "tcr_b"):
                if getattr(row, name):
                    result[name] = self.identity_heads["tcr_species"](encoded[getattr(row, name)])
        if row.tcr_a and row.tcr_b:
            if "tcr_lineage" in self.identity_heads:
                result["tcr_lineage"] = self.identity_heads["tcr_lineage"](
                    torch.cat([encoded[row.tcr_a], encoded[row.tcr_b]])
                )
            if "tcr_pairing" in self.identity_heads:
                result["tcr_pairing"] = self.identity_heads["tcr_pairing"](
                    torch.cat([encoded[row.tcr_a], encoded[row.tcr_b]])
                )
        return result

    def _auxiliary(self, row, identity):
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
                prediction=identity[name].detach().softmax(-1).cpu().tolist()
                if name in identity
                else (),
                threshold=self.config.identity_threshold,
            )
            for name in sorted(names)
        }

    def _resolve(self, row, mhc, identity):
        # The classifier is supervised only by its own labels, not by assay gradients.
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
            encoded.get(context.source_n, self._zero(128)),
            encoded.get(context.source_c, self._zero(128)),
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
                else self._zero(384)
            )
        if self.config.source_context and self.has_cellular:
            values.append(self._source(row, encoded) if cellular else self._zero(262))
        allowed_roles = set()
        if cellular:
            allowed_roles.update(("apc", "immune"))
        if receptor:
            allowed_roles.update(("tcr_a", "tcr_b", "repertoire"))
        for role in self.config.context_species:
            values.append(
                self._tensor(species[role].probabilities)
                if role in allowed_roles
                else self._zero(len(self.config.species))
            )
        for name in self.config.components:
            component = components[name]
            features = [float(component.state == state) for state in STATES]
            if name in self.config.component_activity_units:
                if (
                    component.activity is not None
                    and component.unit != self.config.component_activity_units[name]
                ):
                    raise ValueError(f"activity unit mismatch for {name}")
                features.extend([component.activity or 0.0, float(component.activity is not None)])
            values.append(self._tensor(features) if cellular else self._zero(len(features)))
        if self.config.cell_types:
            features = [*cell.cell_type.probabilities, float(cell.cell_type.source == "unknown")]
            values.append(self._tensor(features) if cellular else self._zero(len(features)))
        for name in self.config.cytokines:
            exposure = cell.cytokines[name]
            features = [float(exposure.state == state) for state in EXPOSURE_STATES]
            if name in self.config.cytokine_dose_units:
                features.extend([math.log1p(exposure.dose or 0), float(exposure.dose is not None)])
            if name in self.config.cytokine_duration_features:
                features.extend(
                    [
                        math.log1p(exposure.duration_hours or 0),
                        float(exposure.duration_hours is not None),
                    ]
                )
            values.append(self._tensor(features) if cellular else self._zero(len(features)))
        for configured, resolved in (
            (self.config.processing_pathways, cell.processing_pathways),
            (self.config.interventions, cell.interventions),
        ):
            for name in configured:
                features = [float(resolved[name].state == state) for state in EXPOSURE_STATES]
                values.append(self._tensor(features) if cellular else self._zero(len(features)))
        for name in sorted(self.config.ordinal_scales):
            scale = self.config.ordinal_scales[name]
            level = cell.ordinal_values[name].level
            features = [float(level == label) for label in (*scale.levels, None)]
            values.append(self._tensor(features) if cellular else self._zero(len(features)))
        if self.has_cellular:
            values.append(
                self._tensor([float(row.context.delivery == value) for value in DELIVERY])
                if cellular
                else self._zero(len(DELIVERY))
            )
        if self.has_coreceptor:
            state = row.context.coreceptor_override or row.context.coreceptor
            values.append(
                self._tensor([float(state == value) for value in CORECEPTORS])
                if row.endpoint in ("tcell_response", "tcr_response")
                else self._zero(len(CORECEPTORS))
            )
        for name in self.config.auxiliary_context:
            if name == "mhc_pairing":
                continue
            allowed = (
                (name == "tcr_pairing" and receptor)
                or (name == "tcr_lineage" and row.endpoint == "tcr_response")
                or (name.startswith("peptide_source:") and cellular)
            )
            values.append(self._tensor(auxiliary[name].probabilities) if allowed else self._zero(2))
        return torch.cat(values) if values else self._zero(0)

    def _pmhc(self, row, mhc, encoded, token_states):
        identity = self._identity(row, mhc, encoded)
        species, class_field, components, cell = self._resolve(row, mhc, identity)
        auxiliary = self._auxiliary(row, identity)
        if class_field.source in ("unknown", "unsupported"):
            raise ValueError(
                "MHC class must be supplied or predicted by an allocated identity task"
            )
        pooled, register_info = [], []
        contextual = endpoint_spec(row.endpoint).route != "pmhc"
        context = (
            self._context(row, encoded, species, components, auxiliary, cell)
            if contextual
            else None
        )
        for index, weight in enumerate(class_field.probabilities):
            if weight == 0:
                continue
            mhc_class = ("I", "II")[index]
            if mhc_class not in self.config.mhc_classes:
                raise ValueError("effective class is outside this checkpoint's supported branches")
            register_candidates = candidates(row.peptide, mhc_class)
            features = [
                torch.cat(
                    [
                        self._candidate_representation(
                            encoded[core], token_states[row.peptide], start, len(core)
                        ),
                        encoded[mhc.groove1],
                        encoded[mhc.groove2],
                        self._tensor(species["mhc1"].probabilities + species["mhc2"].probabilities),
                        *(
                            [self._tensor(auxiliary["mhc_pairing"].probabilities)]
                            if "mhc_pairing" in self.config.auxiliary_context
                            else []
                        ),
                    ]
                )
                for start, core, _ in register_candidates
            ]
            hidden = self.pmhc_fusion(torch.stack(features))
            if contextual:
                hidden = self.context_fusion(
                    torch.cat([hidden, context.expand(len(hidden), -1)], -1)
                )
            spec = endpoint_spec(row.endpoint)
            scores = (
                self._component_score(row.endpoint, "cellular", hidden)
                if "cellular" in spec.parts
                else self.heads[row.endpoint](hidden).squeeze(-1)
            )
            value, attention = self._pool_registers(scores, spec.register_direction)
            pooled.append(weight * value)
            register_info.append(
                {
                    "class": mhc_class,
                    "class_probability": weight,
                    "flank_representation": self.config.peptide_context,
                    "starts": [x[0] for x in register_candidates],
                    "weights": attention,
                    "candidate_scores": scores,
                    "pooling": "normalized_logsumexp_model_units",
                    "direction": spec.register_direction,
                    "entropy": -(attention * attention.clamp_min(1e-12).log()).sum(),
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
        return torch.stack(pooled).sum(0), identity, diagnostics

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
        results = []
        for row in rows:
            output = Prediction(row.observation_id, row.endpoint)
            if row.endpoint == "identity":
                mhc = row.mhc[0] if row.mhc else None
                output.identity = self._identity(row, mhc, encoded)
                species, mhc_class, _, cell = self._resolve(row, mhc, output.identity)
                output.diagnostics = {
                    "species": {key: asdict(value) for key, value in species.items()},
                    "class": asdict(mhc_class) if mhc_class else None,
                    "cell_context": {**asdict(cell), "applies_to_endpoint": False},
                    "auxiliary": {
                        name: asdict(value)
                        for name, value in self._auxiliary(row, output.identity).items()
                    },
                }
                results.append(output)
                continue
            spec = endpoint_spec(row.endpoint)
            route = spec.route
            if route in ("bulk", "digest", "detector"):
                output.identity = self._identity(row, None, encoded)
                output.diagnostics["auxiliary"] = {
                    name: asdict(value)
                    for name, value in self._auxiliary(row, output.identity).items()
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
                # Provenance differences do not duplicate a biological candidate.
                unique = {}
                for mhc in row.mhc:
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
                for mhc in unique.values():
                    vector, identity, diagnostic = self._pmhc(row, mhc, encoded, token_states)
                    vectors.append(vector)
                    diagnostics.append(diagnostic)
                    if len(unique) == 1:
                        output.identity = identity
                output.diagnostics["candidates"] = diagnostics
                if "cellular" in spec.parts:
                    scores = torch.stack(vectors)
                    output.components["cellular"] = self._candidate_mean(row.endpoint, scores)
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
            results.append(output)
        return results

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
        for row in rows:
            row.validate(self.config.max_length)
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
                prefixes.add("pmhc_fusion.")
                if len(row.peptide) > 9 and any(
                    (mhc.class_override or mhc.mhc_class) != "I" for mhc in row.mhc
                ):
                    prefixes.add("register_score.")
            if route in ("cellular", "tcr", "tcr_cellular"):
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
