"""Versioned inputs and targets for the small, directly supervised model.

This contract is deliberately separate from legacy clipped targets. Provenance
is retained but never flattened into predictive features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable, Mapping

from .modular_quantitative import QuantitativeDomain, QUANTITY_PARTS, quantity_domain
from .modular_ordinal import OrdinalValue

SCHEMA = "presto.modular.observation.v1"
AA = "ACDEFGHIKLMNPQRSTVWY"
STATES = ("reference", "intact", "reduced", "lost", "unknown")
EXPOSURE_STATES = ("present", "absent", "unknown")
ROLES = ("mhc1", "mhc2", "immune", "apc", "repertoire", "tcr_a", "tcr_b", "antigen")
CORECEPTORS = ("CD4", "CD8", "both", "neither", "unknown")
DELIVERY = ("endogenous", "protein", "peptide", "preformed_pmhc", "unknown")
IDENTITY_TASKS = (
    "mhc_class",
    "mhc_species",
    "tcr_species",
    "tcr_lineage",
    "peptide_source",
    "mhc_pairing",
    "tcr_pairing",
)
IDENTITY_HEADS = {
    "mhc_class": "mhc_class",
    "mhc1": "mhc_species",
    "mhc2": "mhc_species",
    "tcr_a": "tcr_species",
    "tcr_b": "tcr_species",
    "tcr_lineage": "tcr_lineage",
    "mhc_pairing": "mhc_pairing",
    "tcr_pairing": "tcr_pairing",
}


def identity_head(name: str) -> str | None:
    return "peptide_source" if name.startswith("peptide_source:") else IDENTITY_HEADS.get(name)


def identity_vocabulary(name: str, species=(), source_categories=()) -> tuple[str, ...]:
    if name.startswith("peptide_source:"):
        if name.split(":", 1)[1] not in source_categories:
            raise ValueError(f"unsupported peptide source category: {name}")
        return ("absent", "present")
    if name in ("mhc_pairing", "tcr_pairing"):
        return ("matched", "mismatched")
    if name == "mhc_class":
        return ("I", "II")
    if name == "tcr_lineage":
        return ("CD4", "CD8")
    if name in ("mhc1", "mhc2", "tcr_a", "tcr_b"):
        return tuple(species)
    raise ValueError(f"unsupported auxiliary outcome: {name}")


def has_identity_input(row: Observation, name: str) -> bool:
    """Availability of a supervised output, distinct from its label/support."""
    mhc = row.mhc[0] if len(row.mhc) == 1 else None
    if name in ("mhc1", "mhc2"):
        return bool(mhc and (mhc.groove1 if name == "mhc1" else mhc.groove2))
    if name == "mhc_species":
        return any(mhc.groove1 or mhc.groove2 for mhc in row.mhc)
    if name in ("mhc_class", "mhc_pairing"):
        return any(mhc.groove1 and mhc.groove2 for mhc in row.mhc)
    if name in ("tcr_a", "tcr_b"):
        return bool(getattr(row, name))
    if name == "tcr_species":
        return bool(row.tcr_a or row.tcr_b)
    if name in ("tcr_lineage", "tcr_pairing"):
        return bool(row.tcr_a and row.tcr_b)
    if name == "peptide_source" or name.startswith("peptide_source:"):
        return bool(row.peptide)
    return False


@dataclass(frozen=True)
class Endpoint:
    route: str
    unit: str
    log10: bool = False
    binary: bool = False
    parts: tuple[str, ...] = ()
    quantitative: bool = False
    register_direction: int = 1

    @property
    def observation_covariates(self):
        return self.quantitative or len(self.parts) > 1


ENDPOINTS = {
    "ic50": Endpoint("pmhc", "nM", True, register_direction=-1),
    "kd": Endpoint("pmhc", "nM", True, register_direction=-1),
    "ec50": Endpoint("pmhc", "nM", True, register_direction=-1),
    "kd_via_ic50": Endpoint("pmhc", "nM", True, register_direction=-1),
    "kd_via_ec50": Endpoint("pmhc", "nM", True, register_direction=-1),
    "koff": Endpoint("pmhc", "1/s", True, register_direction=-1),
    "kon": Endpoint("pmhc", "1/(M*s)", True),
    "half_life": Endpoint("pmhc", "min", True),
    "tm": Endpoint("pmhc", "C"),
    "td50": Endpoint("pmhc", "C"),
    "qualitative_binding": Endpoint("pmhc", "binary", binary=True),
    "bulk_observation": Endpoint("bulk", "binary", binary=True, parts=("digest", "ms_response")),
    "elution_observation": Endpoint(
        "cellular", "binary", binary=True, parts=("cellular", "ms_response")
    ),
    "ms_response": Endpoint("detector", "binary", binary=True, parts=("ms_response",)),
    "digest_observation": Endpoint("digest", "binary", binary=True, parts=("digest",)),
    "tcell_response": Endpoint("cellular", "binary", binary=True),
    "tcr_binding": Endpoint("tcr", "binary", binary=True),
    "tcr_response": Endpoint("tcr_cellular", "binary", binary=True),
}


def endpoint_spec(name: str) -> Endpoint:
    if name in ENDPOINTS:
        return ENDPOINTS[name]
    if quantity_domain(name) is not None:
        route, parts = QUANTITY_PARTS[name.split(":", 1)[0]]
        return Endpoint(route, "declared_quantity", parts=parts, quantitative=True)
    raise ValueError(f"unsupported endpoint: {name}")


def is_endpoint(name: str) -> bool:
    try:
        endpoint_spec(name)
    except ValueError:
        return False
    return True


def sequence(value: str, *, required: bool = True, max_length: int = 512) -> str:
    """Validate, never normalize away unsupported residues or truncate."""
    if not isinstance(value, str) or (required and not value):
        raise ValueError("required canonical amino-acid sequence is missing")
    if set(value) - set(AA) or len(value) > max_length:
        raise ValueError("unsupported residues or sequence length")
    return value


@dataclass(frozen=True)
class MHCInput:
    groove1: str
    groove2: str
    mhc_class: str | None = None
    class_override: str | None = None
    species1: str | None = None
    species2: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self, max_length: int = 512, *, allow_partial: bool = False) -> None:
        sequence(self.groove1, required=not allow_partial, max_length=max_length)
        sequence(self.groove2, required=not allow_partial, max_length=max_length)
        if not (self.groove1 or self.groove2):
            raise ValueError("at least one actual MHC groove sequence required")
        for value in (self.mhc_class, self.class_override):
            if value is not None and value not in ("I", "II"):
                raise ValueError(f"unsupported MHC class: {value}")


@dataclass(frozen=True)
class Component:
    state: str = "unknown"
    activity: float | None = None
    unit: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError(f"unsupported component state: {self.state}")
        if self.activity is not None and (not math.isfinite(self.activity) or not self.unit):
            raise ValueError("measured component activity requires a finite value and unit")


@dataclass(frozen=True)
class Exposure:
    state: str = "unknown"
    dose: float | None = None
    unit: str | None = None
    duration_hours: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.state not in EXPOSURE_STATES:
            raise ValueError("unsupported cytokine exposure state")
        for value in (self.dose, self.duration_hours):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("exposure dose and duration must be finite and nonnegative")
        if self.dose is not None and not self.unit:
            raise ValueError("exposure dose requires a declared unit")
        if self.state == "unknown" and (self.dose is not None or self.duration_hours is not None):
            raise ValueError("unknown exposure cannot supply a measured dose or duration")
        if self.state == "absent" and self.dose not in (None, 0):
            raise ValueError("absent exposure cannot have a positive dose")


@dataclass(frozen=True)
class ContextAssertion:
    """A scoped claim, separate from measured activity or exposure dose."""

    state: str = "unknown"
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.state not in EXPOSURE_STATES:
            raise ValueError("unsupported context assertion state")
        if not isinstance(self.evidence, dict):
            raise ValueError("context assertion evidence must be a mapping")


@dataclass(frozen=True)
class MachineryProfile:
    profile_id: str
    species: str
    components: dict[str, Component]
    reference: str
    version: str
    cell_type: str | None = None

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.profile_id, self.species, self.reference, self.version)
        ):
            raise ValueError("machinery profile requires ID, species, reference and version")
        if self.cell_type is not None and not self.cell_type:
            raise ValueError("profile cell type cannot be empty")
        if not self.components or any(
            not name or not isinstance(value, Component) for name, value in self.components.items()
        ):
            raise ValueError("machinery profile requires typed component priors")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["components"] = {
            name: Component(**value) for name, value in values.get("components", {}).items()
        }
        return cls(**values)


@dataclass(frozen=True)
class Context:
    species: dict[str, str] = field(default_factory=dict)
    species_overrides: dict[str, str] = field(default_factory=dict)
    components: dict[str, Component] = field(default_factory=dict)
    component_overrides: dict[str, Component] = field(default_factory=dict)
    source_n: str | None = None
    source_c: str | None = None
    n_terminus: bool | None = None
    c_terminus: bool | None = None
    enzyme: str | None = None
    delivery: str = "unknown"
    coreceptor: str = "unknown"
    coreceptor_override: str | None = None
    acquisition: dict[str, float] = field(default_factory=dict)
    observation: dict[str, float] = field(default_factory=dict)
    acquisition_units: dict[str, str] = field(default_factory=dict)
    observation_units: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    auxiliary_values: dict[str, str] = field(default_factory=dict)
    auxiliary_overrides: dict[str, str] = field(default_factory=dict)
    cell_type: str | None = None
    cell_type_override: str | None = None
    cytokines: dict[str, Exposure] = field(default_factory=dict)
    cytokine_overrides: dict[str, Exposure] = field(default_factory=dict)
    processing_pathways: dict[str, ContextAssertion] = field(default_factory=dict)
    pathway_overrides: dict[str, ContextAssertion] = field(default_factory=dict)
    interventions: dict[str, ContextAssertion] = field(default_factory=dict)
    intervention_overrides: dict[str, ContextAssertion] = field(default_factory=dict)
    ordinal_values: dict[str, OrdinalValue] = field(default_factory=dict)
    ordinal_overrides: dict[str, OrdinalValue] = field(default_factory=dict)
    preparation: dict[str, float] = field(default_factory=dict)
    preparation_units: dict[str, str] = field(default_factory=dict)

    def validate(self, max_length: int = 512) -> None:
        for values in (self.ordinal_values, self.ordinal_overrides):
            if any(
                not isinstance(name, str) or not name.strip() or not isinstance(value, OrdinalValue)
                for name, value in values.items()
            ):
                raise ValueError("ordinal context requires named typed values")
        for values in (
            self.processing_pathways,
            self.pathway_overrides,
            self.interventions,
            self.intervention_overrides,
        ):
            if any(
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(value, ContextAssertion)
                for name, value in values.items()
            ):
                raise ValueError("pathways/interventions require named context assertions")
        for value in (self.cell_type, self.cell_type_override):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("cell type must be a nonempty vocabulary label")
        for values in (self.cytokines, self.cytokine_overrides):
            if any(not name or not isinstance(value, Exposure) for name, value in values.items()):
                raise ValueError("cytokines require typed exposure records")
        for values in (self.auxiliary_values, self.auxiliary_overrides):
            for name, value in values.items():
                categories = (name.split(":", 1)[1],) if name.startswith("peptide_source:") else ()
                if name not in ("mhc_pairing", "tcr_pairing", "tcr_lineage") and not categories:
                    raise ValueError(f"unsupported auxiliary context: {name}")
                if value not in identity_vocabulary(name, source_categories=categories):
                    raise ValueError(f"unsupported auxiliary context value: {name}={value}")
        if (set(self.species) | set(self.species_overrides)) - set(ROLES):
            raise ValueError("unknown species role")
        if self.delivery not in DELIVERY:
            raise ValueError("unsupported antigen delivery")
        if self.coreceptor not in CORECEPTORS or (
            self.coreceptor_override is not None and self.coreceptor_override not in CORECEPTORS
        ):
            raise ValueError("unsupported actual coreceptor state")
        for value in (self.source_n, self.source_c):
            if value is not None:
                sequence(value, required=False, max_length=max_length)
        for features in (self.acquisition, self.observation, self.preparation):
            if any(not math.isfinite(value) for value in features.values()):
                raise ValueError("nonfinite context covariate")


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source: str
    endpoint: str
    peptide: str = ""
    mhc: tuple[MHCInput, ...] = ()
    tcr_a: str = ""
    tcr_b: str = ""
    value: float | None = None
    unit: str | None = None
    qualifier: str = "="
    split: str = "train"
    group_id: str = ""
    context: Context = field(default_factory=Context)
    identity_labels: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA
    quantity_contract: QuantitativeDomain | None = None
    value_encoding: str = "linear"

    def validate(self, max_length: int = 512) -> None:
        if self.schema != SCHEMA or not self.observation_id or not self.source:
            raise ValueError("observation requires versioned schema, source and stable ID")
        if self.split not in ("train", "validation", "test", "predict"):
            raise ValueError("unsupported split")
        if self.qualifier not in ("=", "<", "<=", ">", ">="):
            raise ValueError("unsupported qualifier")
        for candidate in self.mhc:
            candidate.validate(max_length, allow_partial=self.endpoint == "identity")
        self.context.validate(max_length)
        if self.value_encoding not in ("linear", "log2", "log10"):
            raise ValueError("unsupported raw value encoding")
        quantitative = self.endpoint != "identity" and endpoint_spec(self.endpoint).quantitative
        if quantitative:
            if not isinstance(self.quantity_contract, QuantitativeDomain):
                raise ValueError("quantity endpoint requires a measurement contract")
            if self.unit is not None and self.unit != self.quantity_contract.unit:
                raise ValueError("quantitative measurement unit mismatch")
        elif self.quantity_contract is not None or self.value_encoding != "linear":
            raise ValueError("quantity contracts/encoded values require a quantity endpoint")
        if self.endpoint == "identity":
            sequence(self.peptide, required=False, max_length=max_length)
            if self.value is not None:
                raise ValueError("identity outcomes are labels, not scalar assay values")
            if not self.peptide and not self.mhc and not (self.tcr_a or self.tcr_b):
                raise ValueError("identity requires a peptide, MHC grooves or receptor sequence")
            if len(self.mhc) > 1:
                raise ValueError("identity labels require one molecular input")
            for name in self.identity_labels:
                if identity_head(name) is None:
                    raise ValueError(f"unsupported identity label: {name}")
                if not has_identity_input(self, name):
                    raise ValueError(f"identity input missing for {name}")
        else:
            spec = endpoint_spec(self.endpoint)
            route = spec.route
            sequence(self.peptide, max_length=max_length)
            if route in ("pmhc", "cellular", "tcr", "tcr_cellular") and not self.mhc:
                raise ValueError("endpoint requires resolved MHC grooves")
            if len(self.mhc) > 1 and "cellular" not in spec.parts:
                raise ValueError("only elution observations support candidate MHC bags")
            if route in ("tcr", "tcr_cellular") and not (self.tcr_a and self.tcr_b):
                raise ValueError("paired TCR sequences required")
            if self.value is not None:
                observation_target(self)
            if self.identity_labels:
                raise ValueError("identity supervision must use separately sampled identity rows")
        for value in (self.tcr_a, self.tcr_b):
            if value:
                sequence(value, max_length=max_length)
        if self.provenance.get("modifications"):
            raise ValueError("modified peptidoforms are not supported in schema v1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Observation:
        if row.get("schema") != SCHEMA:
            raise ValueError("explicit supported observation schema required")
        data = dict(row)
        data["mhc"] = tuple(MHCInput(**candidate) for candidate in data.get("mhc", ()))
        context = dict(data.get("context", {}))
        for key in ("components", "component_overrides"):
            context[key] = {
                name: Component(**value) for name, value in context.get(key, {}).items()
            }
        for key in ("cytokines", "cytokine_overrides"):
            context[key] = {name: Exposure(**value) for name, value in context.get(key, {}).items()}
        for key in ("ordinal_values", "ordinal_overrides"):
            context[key] = {
                name: OrdinalValue(**value) for name, value in context.get(key, {}).items()
            }
        for key in (
            "processing_pathways",
            "pathway_overrides",
            "interventions",
            "intervention_overrides",
        ):
            context[key] = {
                name: ContextAssertion(**value) for name, value in context.get(key, {}).items()
            }
        data["context"] = Context(**context)
        if data.get("quantity_contract") is not None:
            data["quantity_contract"] = QuantitativeDomain(**data["quantity_contract"])
        result = cls(**data)
        result.validate()
        return result


def transformed_target(
    endpoint: str,
    value: float,
    unit: str | None,
    qualifier: str = "=",
    *,
    quantity_contract: QuantitativeDomain | None = None,
    value_encoding: str = "linear",
) -> float:
    spec = endpoint_spec(endpoint)
    if spec.quantitative:
        if quantity_contract is None:
            raise ValueError("quantity target requires a measurement contract")
        return quantity_contract.convert(value, unit, value_encoding)
    if not math.isfinite(value):
        raise ValueError("nonfinite target")
    if spec.binary:
        if unit != "binary" or value not in (0, 1) or qualifier != "=":
            raise ValueError("binary outcome requires an exact 0/1 measured or search-scoped label")
        return float(value)
    conversions = {
        "nM": {"nM": 1.0, "uM": 1e3, "mM": 1e6, "M": 1e9, "pM": 1e-3},
        "min": {"min": 1.0, "h": 60.0, "s": 1 / 60},
        "1/s": {"1/s": 1.0, "1/min": 1 / 60},
        "1/(M*s)": {"1/(M*s)": 1.0, "M^-1s^-1": 1.0, "nM^-1s^-1": 1e9},
        "C": {"C": 1.0},
    }
    if unit not in conversions[spec.unit]:
        raise ValueError(f"unsupported unit {unit!r} for {endpoint}")
    canonical = value * conversions[spec.unit][unit]
    if spec.log10:
        if canonical <= 0:
            raise ValueError("log target must be positive")
        return math.log10(canonical)
    return canonical


def observation_target(row: Observation) -> float:
    return transformed_target(
        row.endpoint,
        row.value,
        row.unit,
        row.qualifier,
        quantity_contract=row.quantity_contract,
        value_encoding=row.value_encoding,
    )


@dataclass(frozen=True)
class TargetScale:
    mean: float
    scale: float
    exact_count: int
    floor: float = 1e-3

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.mean, self.scale, self.floor)):
            raise ValueError("nonfinite target scaler")
        if self.scale < self.floor or self.floor <= 0 or self.exact_count < 2:
            raise ValueError("invalid training target scaler")

    def encode(self, value: float) -> float:
        return (value - self.mean) / self.scale

    def decode(self, value: float) -> float:
        return value * self.scale + self.mean


def fit_target_scales(rows: Iterable[Observation], floor: float = 1e-3) -> dict[str, TargetScale]:
    """Only exact TRAIN outcomes fit preprocessing; bounds never set the scale."""
    moments: dict[str, list] = {}
    quantity_contracts = {}
    active = set()
    for row in rows:
        if row.endpoint == "identity" or endpoint_spec(row.endpoint).binary or row.split != "train":
            continue
        if row.value is None:
            continue
        if endpoint_spec(row.endpoint).quantitative:
            if (
                row.endpoint in quantity_contracts
                and quantity_contracts[row.endpoint] != row.quantity_contract
            ):
                raise ValueError("cannot fit a target scale across incompatible quantity contracts")
            quantity_contracts[row.endpoint] = row.quantity_contract
        active.add(row.endpoint)
        if row.qualifier == "=":
            value = observation_target(row)
            count, mean, m2, first, varied = moments.get(row.endpoint, [0, 0.0, 0.0, value, False])
            count += 1
            delta = value - mean
            mean += delta / count
            m2 += delta * (value - mean)
            moments[row.endpoint] = [count, mean, m2, first, varied or value != first]
    result = {}
    for endpoint in sorted(active):
        count, mean, m2, _, varied = moments.get(endpoint, [0, 0.0, 0.0, None, False])
        if count < 2 or not varied:
            raise ValueError(f"{endpoint}: need two distinct exact training values")
        result[endpoint] = TargetScale(
            mean, max(floor, math.sqrt(max(0.0, m2) / count)), count, floor
        )
    return result
