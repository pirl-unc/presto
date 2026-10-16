"""Publication and original-observation metadata, never predictive features."""

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from typing import Any

SOURCE_LINEAGE_FIELDS = (
    "evidence_row_id",
    "assay_iri",
    "reference_iri",
    "pmid",
    "doi",
    "reference_text",
    "source_bundle_id",
    "provenance_id",
    "provenance_status",
    "source_endpoint",
    "source_units",
    "source_inequality",
    "source_value",
    "source_partition",
    "generated_id",
    "generation_kind",
    "generation_seed",
    "generation_draw_index",
    "parent_ids",
    "root_observation_ids",
    "generation_lineage_status",
)

SCOPED_SOURCE_LINEAGE_FIELDS = (
    "scoped_peptide_id",
    "candidate_id",
    "candidate_lookup_key",
    "source_n_candidate_occurrences",
    "source_label_evidence",
    "source_search_space_id",
    "source_protocol_id",
    "source_reference_sha256",
    "source_parent_assignment_basis",
)


ALL_SOURCE_LINEAGE_FIELDS = SOURCE_LINEAGE_FIELDS + SCOPED_SOURCE_LINEAGE_FIELDS


def source_lineage_fields(source: Any, *, include_scoped: bool = False) -> dict[str, str]:
    """Copy available source values without deriving IDs or conflating their roles.

    Merged rows and typed records share this policy: absent/None becomes an
    empty string and surrounding whitespace is removed. Publication identifiers
    and reference text never substitute for original assay/observation identity.
    Scoped candidate fields require explicit opt-in because ordinary assay-record
    constructors accept only the common observation/publication contract.
    """
    values = {}
    fields = ALL_SOURCE_LINEAGE_FIELDS if include_scoped else SOURCE_LINEAGE_FIELDS
    for name in fields:
        value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
        values[name] = "" if value is None else str(value).strip()
    return values


def record_lineage_id(record: Any) -> str:
    """Identify a generated parent or a source-qualified measured observation."""
    if getattr(record, "generated_id", ""):
        return record.generated_id
    evidence = str(getattr(record, "evidence_row_id", "") or "").strip()
    if not evidence:
        return ""
    return f"observation:{getattr(record, 'source_bundle_id', '')}:{evidence}"


def generated_record(record: Any, *, parents, kind: str, seed: int, draw_index: int):
    """Attach complete parent metadata to a newly constructed control record.

    Frozen inputs cannot create an unresolved or cross-partition parent edge.
    Legacy rows without original IDs retain explicit unresolved lineage. None
    of these identifiers become predictive inputs or measured assay metadata.
    """
    parents = list(parents)
    if not parents or not kind:
        raise ValueError("Generated records require parents and a generation kind")
    if getattr(record, "evidence_row_id", "") or getattr(record, "assay_iri", ""):
        raise ValueError("A generated target must not carry a measured assay identity")
    bundles = {str(getattr(parent, "source_bundle_id", "") or "") for parent in parents}
    partitions = {str(getattr(parent, "source_partition", "") or "") for parent in parents}
    if len(bundles) > 1 or len(partitions) > 1:
        raise ValueError("Generated parents span different bundles or partitions")
    bundle = next(iter(bundles))
    partition = next(iter(partitions))
    parent_ids, roots = set(), set()
    resolved = True
    for parent in parents:
        identity = record_lineage_id(parent)
        if identity:
            parent_ids.add(identity)
        else:
            resolved = False
        if getattr(parent, "generated_id", ""):
            values = json.loads(parent.root_observation_ids or "[]")
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError("Generated-parent roots must be a JSON list of observation IDs")
            roots.update(values)
            resolved &= bool(values) and parent.generation_lineage_status == "resolved"
        elif getattr(parent, "evidence_row_id", ""):
            roots.add(parent.evidence_row_id)
        else:
            resolved = False
    if bundle and not resolved:
        raise ValueError("Frozen generated records require every measured parent identity")
    if hasattr(record, "species_context"):
        from .species_context import SPECIES_ROLE_FIELDS, reported_species_fields

        contexts = [reported_species_fields(parent) for parent in parents]
        inherited = {}
        for name in SPECIES_ROLE_FIELDS:
            values = {context[name] for context in contexts}
            inherited[name] = next(iter(values)) if len(values) == 1 else ""
        inherited["species_context_origin"] = "generated_parent_consensus"
        record.species_context = inherited
    lineage = {
        "source_bundle_id": bundle,
        "source_partition": partition,
        "generation_kind": kind,
        "generation_seed": str(seed),
        "generation_draw_index": str(draw_index),
        "parent_ids": json.dumps(sorted(parent_ids)),
        "root_observation_ids": json.dumps(sorted(roots)),
        "generation_lineage_status": "resolved" if resolved else "unresolved",
    }
    payload = {
        name: value
        for name, value in asdict(record).items()
        if name not in ALL_SOURCE_LINEAGE_FIELDS
    }
    identity = json.dumps(
        {"record_type": type(record).__name__, "payload": payload, "lineage": lineage},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    lineage["generated_id"] = "generated:v1:" + hashlib.sha256(identity.encode()).hexdigest()
    for name, value in lineage.items():
        setattr(record, name, value)
    return record
