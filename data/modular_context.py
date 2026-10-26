"""Deterministic, provenance-preserving identity and machinery resolution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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


def _assertions(supplied, overrides, vocabulary, *, family, strict, ignored):
    unsupported = (set(supplied) | set(overrides)) - set(vocabulary)
    if strict and vocabulary and unsupported:
        raise ValueError(f"{family} outside checkpoint vocabulary: {sorted(unsupported)}")
    ignored.extend(f"{family}:{name}" for name in sorted(unsupported))
    values, sources = {}, {}
    for name in vocabulary:
        values[name] = overrides.get(name) or supplied.get(name) or ContextAssertion()
        sources[name] = (
            "override" if name in overrides else "supplied" if name in supplied else "unknown"
        )
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
    factors, factor_sources = _assertions(
        context.interventions,
        context.intervention_overrides,
        interventions,
        family="intervention",
        strict=strict,
        ignored=ignored,
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
    )
