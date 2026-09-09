"""Evidence origins stamped at record conversion, before provenance is lost."""

from typing import Optional


def record_target_provenance(source: Optional[str], *targets: str) -> dict[str, str]:
    """Identify assay records separately from generated records and missing origins.

    Call only at a typed assay-record conversion boundary. Hand-constructed
    samples with a plausible source name do not acquire measured provenance.
    Endpoint-specific proxy/auxiliary roles are declared in the output contract.
    """
    source = (source or "").strip()
    if source == "synthetic" or source.startswith("synthetic_"):
        origin = f"generated:{source}"
    elif source.lower() in {"", "unknown", "none"}:
        origin = "unknown"
    else:
        origin = "assay_record"
    return dict.fromkeys(targets, origin)


def bulk_target_provenance(observed: bool, generated_kind: str = "") -> dict[str, str]:
    """Depth is an ordinal proxy; wrong-enzyme labels are generated controls."""
    if observed:
        return {"excision": "bulk_observed_product", "ms_detectability": "bulk_depth_proxy"}
    return {
        "excision": f"generated:{generated_kind}" if generated_kind else "unknown",
        "ms_detectability": "unknown",
    }
