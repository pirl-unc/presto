"""Versioned inputs and targets for the small, directly supervised model.

This contract is deliberately separate from legacy clipped targets. Provenance
is retained but never flattened into predictive features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable, Mapping

SCHEMA = "presto.modular.observation.v1"
AA = "ACDEFGHIKLMNPQRSTVWY"
STATES = ("reference", "intact", "reduced", "lost", "unknown")
ROLES = ("mhc1", "mhc2", "immune", "apc", "repertoire", "tcr_a", "tcr_b", "antigen")
CORECEPTORS = ("CD4", "CD8", "both", "neither", "unknown")
DELIVERY = ("endogenous", "protein", "peptide", "preformed_pmhc", "unknown")
IDENTITY_TASKS = ("mhc_class", "mhc_species", "tcr_species", "tcr_lineage")


@dataclass(frozen=True)
class Endpoint:
    route: str
    unit: str
    log10: bool = False
    binary: bool = False


ENDPOINTS = {
    "ic50": Endpoint("pmhc", "nM", True),
    "kd": Endpoint("pmhc", "nM", True),
    "ec50": Endpoint("pmhc", "nM", True),
    "kd_via_ic50": Endpoint("pmhc", "nM", True),
    "kd_via_ec50": Endpoint("pmhc", "nM", True),
    "koff": Endpoint("pmhc", "1/s", True),
    "half_life": Endpoint("pmhc", "min", True),
    "tm": Endpoint("pmhc", "C"),
    "bulk_observation": Endpoint("bulk", "binary", binary=True),
    "elution_observation": Endpoint("cellular", "binary", binary=True),
    "ms_response": Endpoint("detector", "binary", binary=True),
    "digest_observation": Endpoint("digest", "binary", binary=True),
    "tcell_response": Endpoint("cellular", "binary", binary=True),
    "tcr_binding": Endpoint("tcr", "binary", binary=True),
    "tcr_response": Endpoint("tcr_cellular", "binary", binary=True),
}


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

    def validate(self, max_length: int = 512) -> None:
        sequence(self.groove1, max_length=max_length)
        sequence(self.groove2, max_length=max_length)
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

    def validate(self, max_length: int = 512) -> None:
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
        for features in (self.acquisition, self.observation):
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

    def validate(self, max_length: int = 512) -> None:
        if self.schema != SCHEMA or not self.observation_id or not self.source:
            raise ValueError("observation requires versioned schema, source and stable ID")
        if self.split not in ("train", "validation", "test", "predict"):
            raise ValueError("unsupported split")
        if self.qualifier not in ("=", "<", "<=", ">", ">="):
            raise ValueError("unsupported qualifier")
        for candidate in self.mhc:
            candidate.validate(max_length)
        self.context.validate(max_length)
        if self.endpoint == "identity":
            sequence(self.peptide, required=False, max_length=max_length)
            if self.value is not None:
                raise ValueError("identity outcomes are labels, not scalar assay values")
            if not self.mhc and not (self.tcr_a and self.tcr_b):
                raise ValueError("identity requires MHC grooves or paired receptor sequences")
            if len(self.mhc) > 1:
                raise ValueError("identity labels require one resolved molecular pair")
        else:
            if self.endpoint not in ENDPOINTS:
                raise ValueError(f"unsupported endpoint: {self.endpoint}")
            route = ENDPOINTS[self.endpoint].route
            sequence(self.peptide, max_length=max_length)
            if route in ("pmhc", "cellular", "tcr", "tcr_cellular") and not self.mhc:
                raise ValueError("endpoint requires resolved MHC grooves")
            if len(self.mhc) > 1 and self.endpoint != "elution_observation":
                raise ValueError("only elution observations support candidate MHC bags")
            if route in ("tcr", "tcr_cellular") and not (self.tcr_a and self.tcr_b):
                raise ValueError("paired TCR sequences required")
            if self.value is not None:
                transformed_target(self.endpoint, self.value, self.unit, self.qualifier)
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
        data["context"] = Context(**context)
        result = cls(**data)
        result.validate()
        return result


def transformed_target(
    endpoint: str, value: float, unit: str | None, qualifier: str = "="
) -> float:
    spec = ENDPOINTS[endpoint]
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
    active = set()
    for row in rows:
        if row.endpoint == "identity" or ENDPOINTS[row.endpoint].binary or row.split != "train":
            continue
        if row.value is None:
            continue
        active.add(row.endpoint)
        if row.qualifier == "=":
            value = transformed_target(row.endpoint, row.value, row.unit)
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
