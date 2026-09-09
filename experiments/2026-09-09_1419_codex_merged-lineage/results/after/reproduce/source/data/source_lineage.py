"""Publication and original-observation metadata, never predictive features."""

from collections.abc import Mapping
from typing import Any

SOURCE_LINEAGE_FIELDS = (
    "evidence_row_id",
    "assay_iri",
    "reference_iri",
    "pmid",
    "doi",
    "reference_text",
)


def source_lineage_fields(source: Any) -> dict[str, str]:
    """Copy available source values without deriving IDs or conflating their roles.

    Merged rows and typed records share this policy: absent/None becomes an
    empty string and surrounding whitespace is removed. Publication identifiers
    and reference text never substitute for original assay/observation identity.
    """
    values = {}
    for name in SOURCE_LINEAGE_FIELDS:
        value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
        values[name] = "" if value is None else str(value).strip()
    return values
