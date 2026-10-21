"""Small subset-allocated sequence model specified in Presto issue #130.

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
    ENDPOINTS,
    IDENTITY_TASKS,
    ROLES,
    STATES,
    Observation,
)
from ..data.modular_context import resolve_components, resolve_field, resolve_species
from .encoders import SequenceEncoder
from .modular_support import feature_inputs, feature_layout

MODEL_VERSION = "presto.modular.v2"


@dataclass(frozen=True)
class ModularConfig:
    endpoints: tuple[str, ...] = ("ic50",)
    identity_tasks: tuple[str, ...] = ()
    mhc_classes: tuple[str, ...] = ("I", "II")
    register_pooling: bool = True
    species: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    component_activity_units: dict[str, str] = field(default_factory=dict)
    machinery_profiles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    context_species: tuple[str, ...] = ()
    enzymes: tuple[str, ...] = ()
    acquisition_features: tuple[str, ...] = ()
    observation_features: tuple[str, ...] = ()
    acquisition_units: dict[str, str] = field(default_factory=dict)
    observation_units: dict[str, str] = field(default_factory=dict)
    source_context: bool = False
    species_defaults: bool = True
    identity_threshold: float = 0.8
    dropout: float = 0.1
    max_length: int = 512

    def __post_init__(self):
        for name in (
            "endpoints",
            "identity_tasks",
            "mhc_classes",
            "species",
            "components",
            "context_species",
            "enzymes",
            "acquisition_features",
            "observation_features",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {name}")
        if set(self.endpoints) - ENDPOINTS.keys() or set(self.identity_tasks) - set(IDENTITY_TASKS):
            raise ValueError("unsupported endpoint or identity task")
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
        ):
            if set(names) != set(units) or any(not value for value in units.values()):
                raise ValueError("each numeric feature requires a declared unit/scale")
        if set(self.component_activity_units) - set(self.components):
            raise ValueError("activity units refer to inactive components")
        for species, names in self.machinery_profiles.items():
            if species not in self.species or set(names) - set(self.components):
                raise ValueError("reference profile outside declared vocabulary")
        if not 0 <= self.dropout < 1 or not 0 < self.identity_threshold <= 1:
            raise ValueError("invalid dropout or identity threshold")
        if self.max_length < 9:
            raise ValueError("max_length must support canonical cores")
        routes = {ENDPOINTS[name].route for name in self.endpoints}
        cellular = bool(routes & {"cellular", "tcr_cellular"})
        contextual = bool(routes & {"cellular", "tcr", "tcr_cellular"})
        if self.components and not cellular:
            raise ValueError("machinery slots need a cellular outcome")
        if self.context_species and not contextual:
            raise ValueError("context species slots need a contextual outcome")
        receptor = bool(routes & {"tcr", "tcr_cellular"})
        for role in self.context_species:
            if role in ("apc", "immune") and not cellular:
                raise ValueError("APC/immune species slots require cellular outcomes")
            if role in ("tcr_a", "tcr_b", "repertoire") and not receptor:
                raise ValueError("receptor species slots require receptor outcomes")
        if self.source_context and not (routes & {"cellular", "tcr_cellular", "digest", "bulk"}):
            raise ValueError("source context has no active loss path")
        if self.enzymes and not (routes & {"digest", "bulk"}):
            raise ValueError("enzyme features have no digest loss path")
        if self.acquisition_features and not (
            routes & {"bulk", "detector"} or "elution_observation" in self.endpoints
        ):
            raise ValueError("acquisition features have no MS loss path")
        if self.observation_features and not (
            set(self.endpoints) & {"bulk_observation", "elution_observation"}
        ):
            raise ValueError("observation covariates have no observation loss path")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModularConfig:
        data = dict(data)
        for name in (
            "endpoints",
            "identity_tasks",
            "mhc_classes",
            "species",
            "components",
            "context_species",
            "enzymes",
            "acquisition_features",
            "observation_features",
        ):
            if name in data:
                data[name] = tuple(data[name])
        if "machinery_profiles" in data:
            data["machinery_profiles"] = {
                key: tuple(value) for key, value in data["machinery_profiles"].items()
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
    """Return (start, sequence, PFR43), resetting positions inside every core."""
    if mhc_class == "I":
        return [(0, peptide, [0.0] * 42 + [math.log1p(len(peptide))])]
    if mhc_class != "II" or len(peptide) < 9:
        raise ValueError("canonical class-II prediction requires a peptide of at least 9 residues")
    result = []
    for start in range(len(peptide) - 8):
        left, right = peptide[:start], peptide[start + 9 :]
        pfr = []
        for flank in (left[-3:], right[:3]):
            pfr.extend(flank.count(aa) / max(1, len(flank)) for aa in AA)
        pfr.extend(math.log1p(length) for length in (len(left), len(right), len(peptide)))
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


class ModularPresto(nn.Module):
    """One encoder; construct only modules reachable from the declared tasks."""

    def __init__(self, config: ModularConfig):
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
        routes = {ENDPOINTS[name].route for name in config.endpoints}
        self.has_pmhc = bool(routes & {"pmhc", "cellular", "tcr", "tcr_cellular"})
        self.has_cellular = bool(routes & {"cellular", "tcr_cellular"})
        self.has_tcr = bool(routes & {"tcr", "tcr_cellular"})
        self.has_context = self.has_cellular or self.has_tcr
        self.has_digest = bool(routes & {"bulk", "digest"})
        self.has_ms = (
            bool(routes & {"bulk", "detector"}) or "elution_observation" in config.endpoints
        )
        self.has_coreceptor = bool(set(config.endpoints) & {"tcell_response", "tcr_response"})
        self.identity_heads = nn.ModuleDict()
        for name in config.identity_tasks:
            width, classes = {
                "mhc_class": (256, 2),
                "mhc_species": (128, len(config.species)),
                "tcr_species": (128, len(config.species)),
                "tcr_lineage": (256, 2),
            }[name]
            self.identity_heads[name] = nn.Linear(width, classes)
        if self.has_pmhc:
            self.pmhc_fusion = self._fusion(429 + 2 * len(config.species))
            if "II" in config.mhc_classes and config.register_pooling:
                self.register_score = nn.Linear(128, 1, bias=False)
        source_width = 262 if config.source_context else 0
        self.context_width = (
            (384 if self.has_tcr else 0)
            + (source_width if self.has_cellular else 0)
            + len(config.context_species) * len(config.species)
            + sum(
                5 + (2 if name in config.component_activity_units else 0)
                for name in config.components
            )
            + (len(DELIVERY) if self.has_cellular else 0)
            + (len(CORECEPTORS) if self.has_coreceptor else 0)
        )
        if self.has_context:
            self.context_fusion = self._fusion(128 + self.context_width)
        if self.has_digest:
            self.digest_fusion = self._fusion(128 + source_width + len(config.enzymes) + 1)
            self.digest_score = nn.Linear(128, 1, bias=False)
        if self.has_ms:
            self.ms_fusion = self._fusion(128 + 2 * len(config.acquisition_features))
            self.ms_score = nn.Linear(128, 1, bias=False)
        self.heads = nn.ModuleDict()
        self.offsets = nn.ModuleDict()
        for name in config.endpoints:
            if name in ("bulk_observation", "elution_observation"):
                self.offsets[name] = ObservationOffset(2 * len(config.observation_features))
                if name == "elution_observation":
                    self.cellular_score = nn.Linear(128, 1, bias=False)
            elif name in ("digest_observation", "ms_response"):
                self.offsets[name] = ObservationOffset(0)
            else:
                self.heads[name] = nn.Linear(128, 1)
        self.trained_parameters: set[str] = set()
        self.feature_layout = feature_layout(self)
        self.trained_features = {name: set() for name in self.feature_layout}
        self.exposed_features = {name: set() for name in self.feature_layout}
        self.support = {
            name: {"status": "random/untrained", "observations": 0, "validation": "not_evaluated"}
            for name in (*config.endpoints, *config.identity_tasks)
        }

    @staticmethod
    def _fusion(width: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(width, 128), nn.GELU())

    @property
    def device(self):
        return self.encoder.embedding.weight.device

    def _tensor(self, values) -> torch.Tensor:
        return torch.tensor(values, dtype=self.encoder.embedding.weight.dtype, device=self.device)

    def _zero(self, length: int) -> torch.Tensor:
        return self.encoder.embedding.weight.new_zeros(length)

    def _encode(self, rows: Sequence[Observation]) -> dict[str, torch.Tensor]:
        sequences: dict[str, None] = {}
        for row in rows:
            route = ENDPOINTS[row.endpoint].route if row.endpoint != "identity" else "identity"
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
                sequences[mhc.groove1] = sequences[mhc.groove2] = None
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
        pooled, _ = self.encoder(tokens)
        return dict(zip(names, pooled, strict=True))

    def _identity(self, row, mhc, encoded):
        result = {}
        if mhc is not None:
            if "mhc_class" in self.identity_heads:
                result["mhc_class"] = self.identity_heads["mhc_class"](
                    torch.cat([encoded[mhc.groove1], encoded[mhc.groove2]])
                )
            if "mhc_species" in self.identity_heads:
                for name, value in (("mhc1", mhc.groove1), ("mhc2", mhc.groove2)):
                    result[name] = self.identity_heads["mhc_species"](encoded[value])
        if row.tcr_a and row.tcr_b:
            if "tcr_species" in self.identity_heads:
                for name in ("tcr_a", "tcr_b"):
                    result[name] = self.identity_heads["tcr_species"](encoded[getattr(row, name)])
            if "tcr_lineage" in self.identity_heads:
                result["tcr_lineage"] = self.identity_heads["tcr_lineage"](
                    torch.cat([encoded[row.tcr_a], encoded[row.tcr_b]])
                )
        return result

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
        if mhc is not None:
            class_field = resolve_field(
                ("I", "II"),
                supplied=mhc.mhc_class,
                override=mhc.class_override,
                prediction=probabilities.get("mhc_class", ()),
                threshold=self.config.identity_threshold,
            )
        components = resolve_components(
            row.context, species["apc"], self.config.components, self.config.machinery_profiles
        )
        return species, class_field, components

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

    def _context(self, row, encoded, species, components):
        route = ENDPOINTS[row.endpoint].route
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
        return torch.cat(values) if values else self._zero(0)

    def _pmhc(self, row, mhc, encoded):
        identity = self._identity(row, mhc, encoded)
        species, class_field, components = self._resolve(row, mhc, identity)
        if class_field.source in ("unknown", "unsupported"):
            raise ValueError(
                "MHC class must be supplied or predicted by an allocated identity task"
            )
        pooled, register_info = [], []
        contextual = ENDPOINTS[row.endpoint].route != "pmhc"
        context = self._context(row, encoded, species, components) if contextual else None
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
                        encoded[core],
                        encoded[mhc.groove1],
                        encoded[mhc.groove2],
                        self._tensor(pfr),
                        self._tensor([float(index == 0), float(index == 1)]),
                        self._tensor(species["mhc1"].probabilities + species["mhc2"].probabilities),
                    ]
                )
                for _, core, pfr in register_candidates
            ]
            hidden = self.pmhc_fusion(torch.stack(features))
            if contextual:
                hidden = self.context_fusion(
                    torch.cat([hidden, context.expand(len(hidden), -1)], -1)
                )
            if len(hidden) == 1:
                attention = hidden.new_ones(1)
            else:
                if not hasattr(self, "register_score"):
                    raise ValueError("checkpoint has no multi-register pooling capability")
                attention = self.register_score(hidden).squeeze(-1).softmax(0)
            pooled.append(weight * (attention[:, None] * hidden).sum(0))
            register_info.append(
                {
                    "class": mhc_class,
                    "class_probability": weight,
                    "starts": [x[0] for x in register_candidates],
                    "weights": attention,
                    "entropy": -(attention * attention.clamp_min(1e-12).log()).sum(),
                }
            )
        diagnostics = {
            "mhc": asdict(mhc),
            "species": {name: asdict(value) for name, value in species.items()},
            "class": asdict(class_field),
            "components": {name: asdict(value) for name, value in components.items()},
            "registers": register_info,
        }
        return torch.stack(pooled).sum(0), identity, diagnostics

    def forward(self, rows: Sequence[Observation]) -> list[Prediction]:
        if not rows:
            return []
        for row in rows:
            row.validate(self.config.max_length)
            if row.endpoint != "identity" and row.endpoint not in self.config.endpoints:
                raise ValueError(f"endpoint not allocated: {row.endpoint}")
        encoded = self._encode(rows)
        results = []
        for row in rows:
            output = Prediction(row.observation_id, row.endpoint)
            if row.endpoint == "identity":
                mhc = row.mhc[0] if row.mhc else None
                output.identity = self._identity(row, mhc, encoded)
                species, mhc_class, _ = self._resolve(row, mhc, output.identity)
                output.diagnostics = {
                    "species": {key: asdict(value) for key, value in species.items()},
                    "class": asdict(mhc_class) if mhc_class else None,
                }
                results.append(output)
                continue
            route = ENDPOINTS[row.endpoint].route
            if route in ("bulk", "detector") or row.endpoint == "elution_observation":
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
                output.components["ms_response"] = self.ms_score(
                    self.ms_fusion(ms_features)
                ).squeeze(-1)
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
                output.components["digest"] = self.digest_score(
                    self.digest_fusion(torch.cat(features))
                ).squeeze(-1)
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
                    vector, identity, diagnostic = self._pmhc(row, mhc, encoded)
                    vectors.append(vector)
                    diagnostics.append(diagnostic)
                    if len(unique) == 1:
                        output.identity = identity
                output.diagnostics["candidates"] = diagnostics
                if row.endpoint == "elution_observation":
                    scores = self.cellular_score(torch.stack(vectors)).squeeze(-1)
                    output.components["cellular"] = torch.logsumexp(scores, 0) - math.log(
                        len(scores)
                    )
                else:
                    output.value = self.heads[row.endpoint](vectors[0]).squeeze(-1)
            if row.endpoint in self.offsets:
                if row.endpoint in ("ms_response", "digest_observation"):
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
            for name in (*ENDPOINTS, *IDENTITY_TASKS)
        }
        components = {}
        for name, anchor, mixtures, allocated in (
            (
                "ms_response",
                "ms_response",
                ("bulk_observation", "elution_observation"),
                self.has_ms,
            ),
            ("digest", "digest_observation", ("bulk_observation",), self.has_digest),
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
            elif any(outputs[endpoint]["status"] == "directly_supervised" for endpoint in mixtures):
                status = "indirectly_trained"
            components[name] = {
                "status": status,
                "validation": "not_evaluated",
                "identifiable_from_mixture_labels_alone": False,
            }
        return {"outputs": outputs, "components": components}

    def predict(self, rows: Sequence[Observation]) -> list[Prediction]:
        """Inference refuses random requested outputs; qualification stays visible."""
        for row in rows:
            relevant_identity = tuple(
                name
                for name in self.config.identity_tasks
                if (name.startswith("mhc") and row.mhc)
                or (name.startswith("tcr") and row.tcr_a and row.tcr_b)
            )
            requested = (
                relevant_identity
                if row.endpoint == "identity"
                else (row.endpoint, *relevant_identity)
            )
            if not requested:
                raise ValueError("no allocated identity capability for these inputs")
            if row.endpoint in ENDPOINTS and ENDPOINTS[row.endpoint].route in (
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
            if row.mhc:
                prefixes.update(
                    f"identity_heads.{task}."
                    for task in self.config.identity_tasks
                    if task.startswith("mhc")
                )
            if row.tcr_a and row.tcr_b:
                prefixes.update(
                    f"identity_heads.{task}."
                    for task in self.config.identity_tasks
                    if task.startswith("tcr")
                )
            if row.endpoint == "identity":
                continue
            route = ENDPOINTS[row.endpoint].route
            if route in ("pmhc", "cellular", "tcr", "tcr_cellular"):
                prefixes.add("pmhc_fusion.")
                if len(row.peptide) > 9 and any(
                    (mhc.class_override or mhc.mhc_class) != "I" for mhc in row.mhc
                ):
                    prefixes.add("register_score.")
            if route in ("cellular", "tcr", "tcr_cellular"):
                prefixes.add("context_fusion.")
            if route in ("bulk", "detector") or row.endpoint == "elution_observation":
                prefixes.update(("ms_fusion.", "ms_score."))
            if route in ("bulk", "digest"):
                prefixes.update(("digest_fusion.", "digest_score."))
            if row.endpoint == "elution_observation":
                prefixes.add("cellular_score.")
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
