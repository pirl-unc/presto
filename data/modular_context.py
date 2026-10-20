"""Deterministic, provenance-preserving identity and machinery resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .modular import Component, Context, MHCInput, ROLES


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
) -> dict[str, Component]:
    reference = set(profiles.get(apc_species.value, ()))
    return {
        name: context.component_overrides.get(name)
        or context.components.get(name)
        or Component("reference", evidence={"source": "species_reference_profile"})
        if name in reference or name in context.components or name in context.component_overrides
        else Component()
        for name in components
    }
