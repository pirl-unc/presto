"""Inspection-only applicability of cellular processing to the source assay."""

import json
import re
from collections.abc import Mapping

from .species_context import processing_species_context

ASSAY_CONTEXT_POLICY = "explicit_preparation_and_elution_applicability_v1"
ASSAY_CONTEXT_FIELDS = (
    "processing_context_applicability",
    "processing_context_applicability_basis",
    "processing_context_availability",
    "assay_context_origin",
)


def _value(source, name):
    value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
    return "" if value is None else str(value).strip()


def assay_preparation_applicability(method, evidence_kind):
    """Use explicit preparation and observation modality, never a numeric endpoint."""
    mentions = set(re.findall(r"\b(?:purified mhc|cellular mhc)\b", method.casefold()))
    preparations = set(
        re.findall(r"(?:^|[;|])\s*(purified mhc|cellular mhc)(?=/|$)", method.casefold())
    )
    if len(mentions) > 1 or ("purified mhc" in preparations and evidence_kind == "ms"):
        return "unknown", "conflicting_assay_preparation"
    if "purified mhc" in preparations:
        return "not_applicable", "explicit_purified_mhc_preparation"
    if "cellular mhc" in preparations:
        return "applicable", "explicit_cellular_mhc_preparation"
    if evidence_kind == "ms":
        return "applicable", "native_elution_evidence"
    return "unknown", "unspecified_assay_preparation"


def source_assay_contexts(source):
    """Preserve the measured preparation leaves of a generated-parent graph."""
    if _value(source, "generated_id"):
        text = _value(source, "parent_assay_contexts")
        if not text:
            return [{"method": "", "evidence_kind": ""}]
        values = json.loads(text)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, dict)
                or set(value) != {"method", "evidence_kind"}
                or any(not isinstance(v, str) for v in value.values())
                for value in values
            )
        ):
            raise ValueError("Generated assay contexts require method/evidence-kind objects")
        return values
    return [
        {
            "method": _value(source, "source_assay_method"),
            "evidence_kind": _value(source, "source_evidence_kind"),
        }
    ]


def assay_context_fields(source):
    """Report applicability without altering processing species or model features."""
    contexts = source_assay_contexts(source)
    results = [
        assay_preparation_applicability(value["method"], value["evidence_kind"])
        for value in contexts
    ]
    generated = bool(_value(source, "generated_id"))
    if generated:
        states = {state for state, _ in results}
        if len(states) == 1 and "unknown" not in states:
            applicability = next(iter(states))
            basis = "generated_parent_applicability_consensus"
        else:
            applicability, basis = "unknown", "generated_parent_unresolved_or_conflicting"
    else:
        applicability, basis = results[0]
    availability = (
        "not_applicable"
        if applicability == "not_applicable"
        else "applicability_unknown"
        if applicability == "unknown"
        else "resolved"
        if processing_species_context(source)["processing_species_status"] == "resolved"
        else "unresolved"
    )
    return dict(
        zip(
            ASSAY_CONTEXT_FIELDS,
            (
                applicability,
                basis,
                availability,
                "generated_parent_evidence" if generated else "source_observation",
            ),
        )
    )
