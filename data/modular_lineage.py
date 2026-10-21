"""Original measured-outcome identity, independent of exported row identifiers."""

import hashlib
import json

from .modular import Observation, transformed_target


def measurement_key(row: Observation) -> tuple[str, str, str, str] | None:
    """Require an original ID; publication IDs never identify a measurement."""
    provenance = row.provenance
    declared = provenance.get("measurement_lineage", {})
    if not isinstance(declared, dict):
        raise ValueError("measurement_lineage must be a mapping")
    namespace = declared.get("namespace", "")
    measurement = declared.get("measurement_id", "")
    replicate = declared.get("replicate_id", provenance.get("replicate_id", ""))
    if not measurement and provenance.get("original_measurement_id"):
        measurement = provenance["original_measurement_id"]
        namespace = provenance.get("measurement_namespace", row.source)
    raw = provenance.get("raw_record", {})
    if not isinstance(raw, dict):
        raise ValueError("raw_record must be a mapping")
    if not measurement and raw.get("assay_iri"):
        measurement, namespace = raw["assay_iri"], "assay_iri"
    if not measurement and raw.get("evidence_row_id"):
        measurement = raw["evidence_row_id"]
        namespace = raw.get("source") or row.source
    if not measurement:
        return None
    if not namespace or any(
        not isinstance(value, str) for value in (namespace, measurement, replicate)
    ):
        raise ValueError(
            "original measurement identity requires string namespace, ID and replicate"
        )
    return namespace, measurement, replicate, row.endpoint


def semantic_fingerprint(row: Observation, *, include_target=True) -> str:
    """Ignore export provenance; retain all effective inputs and the canonical outcome."""
    data = row.to_dict()
    for key in ("observation_id", "source", "split", "group_id", "provenance"):
        data.pop(key)
    for candidate in data["mhc"]:
        candidate.pop("provenance", None)
    data["mhc"] = sorted(data["mhc"], key=lambda value: json.dumps(value, sort_keys=True))
    data["context"].pop("provenance", None)
    for family in ("components", "component_overrides", "cytokines", "cytokine_overrides"):
        for component in data["context"][family].values():
            component.pop("evidence", None)
    value = data.pop("value")
    unit = data.pop("unit")
    qualifier = data.pop("qualifier")
    labels = data.pop("identity_labels")
    if include_target:
        data["outcome"] = (
            labels
            if row.endpoint == "identity"
            else {
                "value": transformed_target(row.endpoint, value, unit, qualifier)
                if value is not None
                else None,
                "qualifier": qualifier,
            }
        )
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()
