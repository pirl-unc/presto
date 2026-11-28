"""Deterministic, provenance-preserving identity and machinery resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Mapping, Sequence

from .modular import (
    Component,
    Context,
    ContextAssertion,
    Exposure,
    MachineryProfile,
    MHCInput,
    ROLES,
)
from .modular_ordinal import OrdinalScale, OrdinalValue

PERTURBATION_KINDS = (
    "knockout",
    "knockdown",
    "overexpression",
    "transduction",
    "transfection",
)

HLA_LOCUS_GENES = frozenset({"HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "HLA-G"})

APM_HOST_GENES = HLA_LOCUS_GENES | frozenset(
    {
        "B2M",
        "BATF3",
        "CALR",
        "CANX",
        "CD74",
        "CIITA",
        "CTSB",
        "CTSL",
        "CTSS",
        "CYBB",
        "ERAP1",
        "ERAP2",
        "GANAB",
        "HLA-DM",
        "HLA-DO",
        "HLA-DPA1",
        "HLA-DPB1",
        "HLA-DQA1",
        "HLA-DQB1",
        "HLA-DRA",
        "HLA-DRB1",
        "IFI30",
        "IRF2",
        "LGMN",
        "NLRC5",
        "PDIA3",
        "PSMB5",
        "PSMB8",
        "PSMB9",
        "PSMB10",
        "PSME1",
        "PSME2",
        "SEC22B",
        "SEC61",
        "SPPL3",
        "TAP1",
        "TAP2",
        "TAPBP",
        "TAPBPL",
        "TAX1BP1",
    }
)


def canonical_perturbation_gene(raw_gene: str) -> str | None:
    """Return canonical host APM/HLA gene token, or None for non-host constructs."""
    token = raw_gene.strip()
    if not token:
        return None
    upper_hyphen = token.upper().replace("_", "-")
    alias = {
        "TAPBPR": "TAPBPL",
        "ERAAP": "ERAP1",
        "GILT": "IFI30",
    }.get(upper_hyphen, upper_hyphen)
    return alias if alias in APM_HOST_GENES else None


def parse_gene_perturbation(claim_name: str) -> tuple[str, str] | None:
    """Parse `<kind>:<gene>` into `(kind, canonical_gene)` if it names a host APM/HLA gene."""
    if ":" not in claim_name:
        return None
    kind, rest = claim_name.split(":", 1)
    if kind not in PERTURBATION_KINDS:
        return None
    gene = canonical_perturbation_gene(rest)
    if gene is None:
        return None
    return kind, gene


def supported_perturbation_genes(interventions: Sequence[str]) -> tuple[str, ...]:
    """Return sorted canonical host APM/HLA genes backed by configured interventions."""
    genes = set()
    for item in interventions:
        parsed = parse_gene_perturbation(item)
        if parsed is not None:
            genes.add(parsed[1])
    return tuple(sorted(genes))


def is_supported_gene_perturbation(
    claim_name: str, supported_genes: Sequence[str] | set[str] | frozenset[str]
) -> bool:
    parsed = parse_gene_perturbation(claim_name)
    return parsed is not None and parsed[1] in supported_genes


@dataclass(frozen=True)
class ResolvedField:
    value: str | None
    probabilities: tuple[float, ...]
    source: str
    supplied: str | None = None
    override: str | None = None
    predicted: tuple[float, ...] = ()
    confidence: float | None = None


def resolve_field(
    vocabulary: Sequence[str],
    *,
    supplied: str | None = None,
    override: str | None = None,
    prediction: Sequence[float] = (),
    threshold: float = 0.8,
) -> ResolvedField:
    predicted = tuple(float(x) for x in prediction)
    if predicted and len(predicted) != len(vocabulary):
        raise ValueError("identity prediction/vocabulary mismatch")
    value = override if override is not None else supplied
    if value is not None:
        source = "override" if override is not None else "supplied"
        if value not in vocabulary:
            return ResolvedField(
                None, (0.0,) * len(vocabulary), "unsupported", supplied, override, predicted
            )
        return ResolvedField(
            value,
            tuple(float(x == value) for x in vocabulary),
            source,
            supplied,
            override,
            predicted,
            1.0,
        )
    if predicted:
        index = max(range(len(predicted)), key=predicted.__getitem__)
        confidence = predicted[index]
        return ResolvedField(
            vocabulary[index] if confidence >= threshold else None,
            predicted,
            "predicted",
            predicted=predicted,
            confidence=confidence,
        )
    return ResolvedField(None, (0.0,) * len(vocabulary), "unknown")


def resolve_species(
    vocabulary: Sequence[str],
    context: Context,
    mhc: MHCInput | None = None,
    predictions: Mapping[str, Sequence[float]] | None = None,
    *,
    defaults: bool = True,
    threshold: float = 0.8,
) -> dict[str, ResolvedField]:
    predictions = predictions or {}
    supplied = dict(context.species)
    if mhc is not None:
        for role, value in (("mhc1", mhc.species1), ("mhc2", mhc.species2)):
            if value is not None:
                if role in supplied and supplied[role] != value:
                    raise ValueError(f"conflicting supplied {role} species")
                supplied[role] = value
    result = {
        role: resolve_field(
            vocabulary,
            supplied=supplied.get(role),
            override=context.species_overrides.get(role),
            prediction=predictions.get(role, ()),
            threshold=threshold,
        )
        for role in ROLES
    }
    if defaults:
        common = result["mhc1"].value
        if common is not None and common == result["mhc2"].value:
            _default(result, "immune", common, vocabulary, "default:consistent_mhc")
        for role in ("apc", "repertoire"):
            if result["immune"].value is not None:
                _default(result, role, result["immune"].value, vocabulary, "default:immune")
    return result


def _default(result, role, value, vocabulary, source):
    # An unsupported explicit value must not be overwritten by a convenience prior.
    if result[role].source == "unknown":
        result[role] = ResolvedField(value, tuple(float(x == value) for x in vocabulary), source)


def resolve_components(
    context: Context,
    apc_species: ResolvedField,
    components: Sequence[str],
    profiles: Mapping[str, Sequence[str]],
    *,
    cell_type: str | None = None,
    reference_profiles: Sequence[MachineryProfile] = (),
) -> dict[str, Component]:
    reference = {
        name: Component(
            "reference", evidence={"source": "species_reference_profile", "qualified": False}
        )
        for name in profiles.get(apc_species.value, ())
    }
    for scope in (None, cell_type) if cell_type is not None else (None,):
        matches = [
            profile
            for profile in reference_profiles
            if profile.species == apc_species.value and profile.cell_type == scope
        ]
        if len(matches) > 1:
            raise ValueError("ambiguous machinery reference profile scope")
        if matches:
            profile = matches[0]
            for name, value in profile.components.items():
                reference[name] = replace(
                    value,
                    evidence={
                        **value.evidence,
                        "source": "reference_profile",
                        "assumption": True,
                        "profile_id": profile.profile_id,
                        "reference": profile.reference,
                        "version": profile.version,
                        "species": profile.species,
                        "cell_type": profile.cell_type,
                    },
                )
    return {
        name: context.component_overrides.get(name)
        or context.components.get(name)
        or reference.get(name, Component())
        for name in components
    }


@dataclass(frozen=True)
class ResolvedCellContext:
    cell_type: ResolvedField
    cytokines: dict[str, Exposure]
    cytokine_sources: dict[str, str]
    supplied_cytokines: dict[str, Exposure]
    cytokine_overrides: dict[str, Exposure]
    ignored_fields: list[str] = field(default_factory=list)
    processing_pathways: dict[str, ContextAssertion] = field(default_factory=dict)
    pathway_sources: dict[str, str] = field(default_factory=dict)
    supplied_pathways: dict[str, ContextAssertion] = field(default_factory=dict)
    pathway_overrides: dict[str, ContextAssertion] = field(default_factory=dict)
    interventions: dict[str, ContextAssertion] = field(default_factory=dict)
    intervention_sources: dict[str, str] = field(default_factory=dict)
    supplied_interventions: dict[str, ContextAssertion] = field(default_factory=dict)
    intervention_overrides: dict[str, ContextAssertion] = field(default_factory=dict)
    ordinal_values: dict[str, OrdinalValue] = field(default_factory=dict)
    ordinal_sources: dict[str, str] = field(default_factory=dict)
    supplied_ordinals: dict[str, OrdinalValue] = field(default_factory=dict)
    ordinal_overrides: dict[str, OrdinalValue] = field(default_factory=dict)


def project_apc_context(
    context: Context,
    *,
    cell_types: Sequence[str] = (),
    cytokines: Sequence[str] = (),
    processing_pathways: Sequence[str] = (),
    interventions: Sequence[str] = (),
    shared_gene_perturbations: bool = False,
) -> Context:
    """Explicitly omit unrepresented APC categories, retaining their source claims.

    This is an opt-in input projection, not vocabulary learning or an absent-control
    imputation. Supported doses, units and ordinal references still resolve strictly.
    Apply to original source context for each checkpoint, not a previous projection.
    """
    omitted, changes = {}, {}
    supported_genes = (
        frozenset(supported_perturbation_genes(interventions))
        if shared_gene_perturbations
        else frozenset()
    )
    effective = context.cell_type_override or context.cell_type
    if effective is not None and effective not in cell_types:
        # An unsupported override cannot expose a superseded supplied identity.
        omitted["cell_type"] = context.cell_type
        omitted["cell_type_override"] = context.cell_type_override
        changes.update(cell_type=None, cell_type_override=None)
    for supplied, override, vocabulary in (
        ("cytokines", "cytokine_overrides", cytokines),
        ("processing_pathways", "pathway_overrides", processing_pathways),
        ("interventions", "intervention_overrides", interventions),
    ):
        is_interv = supplied == "interventions"
        for name in (supplied, override):
            values = getattr(context, name)
            unsupported = {
                k: asdict(v)
                for k, v in values.items()
                if k not in vocabulary
                and not (is_interv and is_supported_gene_perturbation(k, supported_genes))
            }
            if unsupported:
                omitted[name] = unsupported
                changes[name] = {
                    k: v
                    for k, v in values.items()
                    if k in vocabulary
                    or (is_interv and is_supported_gene_perturbation(k, supported_genes))
                }
    if not omitted:
        return context
    previous = context.provenance.get("unrepresented_apc_context", [])
    if not isinstance(previous, list):
        raise ValueError("unrepresented_apc_context provenance must be a list")
    return replace(
        context,
        **changes,
        provenance={
            **context.provenance,
            "unrepresented_apc_context": [*previous, omitted],
        },
    )


def _assertions(
    supplied,
    overrides,
    vocabulary,
    *,
    family,
    strict,
    ignored,
    supported_genes: frozenset[str] = frozenset(),
):
    raw_unsupported = (set(supplied) | set(overrides)) - set(vocabulary)
    unsupported = {
        name
        for name in raw_unsupported
        if not (family == "intervention" and is_supported_gene_perturbation(name, supported_genes))
    }
    if strict and vocabulary and unsupported:
        raise ValueError(f"{family} outside checkpoint vocabulary: {sorted(unsupported)}")
    ignored.extend(f"{family}:{name}" for name in sorted(unsupported))
    values, sources = {}, {}
    vocab_set = set(vocabulary)
    for name in vocabulary:
        direct = overrides.get(name) or supplied.get(name)
        if direct is not None:
            values[name] = direct
            sources[name] = "override" if name in overrides else "supplied"
            continue
        # Complete knockout entails loss-of-function for knockdown-only discrete slots
        if family == "intervention" and supported_genes and name.startswith("knockdown:"):
            parsed = parse_gene_perturbation(name)
            if parsed is not None:
                ko_key = f"knockout:{parsed[1]}"
                ko_claim = overrides.get(ko_key) or supplied.get(ko_key)
                if ko_key not in vocab_set and ko_claim is not None and ko_claim.state == "present":
                    values[name] = ContextAssertion("present")
                    sources[name] = "override" if ko_key in overrides else "supplied"
                    continue
        values[name] = ContextAssertion()
        sources[name] = "unknown"
    return values, sources


def resolve_cell_context(
    context: Context,
    cell_types: Sequence[str],
    cytokines: Sequence[str],
    dose_units: Mapping[str, str],
    duration_features: Sequence[str],
    *,
    strict: bool = True,
    processing_pathways: Sequence[str] = (),
    interventions: Sequence[str] = (),
    ordinal_scales: Mapping[str, OrdinalScale] | None = None,
    shared_gene_perturbations: bool = False,
) -> ResolvedCellContext:
    """Resolve annotations without treating missing exposures as negative controls."""
    cell = resolve_field(
        cell_types, supplied=context.cell_type, override=context.cell_type_override
    )
    if strict and cell_types and cell.source == "unsupported":
        raise ValueError("APC cell type outside checkpoint vocabulary")
    supplied = set(context.cytokines) | set(context.cytokine_overrides)
    if strict and cytokines and supplied - set(cytokines):
        raise ValueError("cytokine outside checkpoint vocabulary")
    ignored = []
    if not cell_types and (context.cell_type is not None or context.cell_type_override is not None):
        ignored.append("cell_type")
    ignored.extend(f"cytokine:{name}" for name in sorted(supplied - set(cytokines)))
    resolved, sources = {}, {}
    for name in cytokines:
        value = context.cytokine_overrides.get(name) or context.cytokines.get(name) or Exposure()
        sources[name] = (
            "override"
            if name in context.cytokine_overrides
            else "supplied"
            if name in context.cytokines
            else "unknown"
        )
        if value.dose is not None:
            if name in dose_units:
                if strict and value.unit != dose_units[name]:
                    raise ValueError(f"cytokine dose unit mismatch: {name}")
            else:
                ignored.append(f"cytokine:{name}:dose")
        if value.duration_hours is not None and name not in duration_features:
            ignored.append(f"cytokine:{name}:duration_hours")
        resolved[name] = value
    pathways, pathway_sources = _assertions(
        context.processing_pathways,
        context.pathway_overrides,
        processing_pathways,
        family="pathway",
        strict=strict,
        ignored=ignored,
    )
    supported_genes = (
        frozenset(supported_perturbation_genes(interventions))
        if shared_gene_perturbations
        else frozenset()
    )
    factors, factor_sources = _assertions(
        context.interventions,
        context.intervention_overrides,
        interventions,
        family="intervention",
        strict=strict,
        ignored=ignored,
        supported_genes=supported_genes,
    )
    ordinal_scales = ordinal_scales or {}
    unsupported = (set(context.ordinal_values) | set(context.ordinal_overrides)) - set(
        ordinal_scales
    )
    if strict and ordinal_scales and unsupported:
        raise ValueError(f"ordinal context outside checkpoint vocabulary: {sorted(unsupported)}")
    ignored.extend(f"ordinal:{name}" for name in sorted(unsupported))
    ordinals, ordinal_sources = {}, {}
    for name in sorted(ordinal_scales):
        value = (
            context.ordinal_overrides.get(name)
            or context.ordinal_values.get(name)
            or OrdinalValue()
        )
        scale = ordinal_scales[name]
        if strict and value.level is not None:
            if value.scale != scale.reference:
                raise ValueError(f"ordinal scale reference mismatch: {name}")
            if value.level not in scale.levels:
                raise ValueError(f"ordinal level outside checkpoint vocabulary: {name}")
        ordinals[name] = value
        ordinal_sources[name] = (
            "override"
            if name in context.ordinal_overrides
            else "supplied"
            if name in context.ordinal_values
            else "unknown"
        )
    return ResolvedCellContext(
        cell,
        resolved,
        sources,
        context.cytokines,
        context.cytokine_overrides,
        ignored,
        pathways,
        pathway_sources,
        context.processing_pathways,
        context.pathway_overrides,
        factors,
        factor_sources,
        context.interventions,
        context.intervention_overrides,
        ordinals,
        ordinal_sources,
        context.ordinal_values,
        context.ordinal_overrides,
    )
