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


def bulk_target_provenance(
    observed: bool, generated_kind: str = "", *, scoped_evidence: str = ""
) -> dict[str, str]:
    """Keep verified scoped detection proxies separate from raw excision controls."""
    if scoped_evidence:
        from .scoped_bulk import EVIDENCE

        return {"excision": "unknown", "ms_detectability": EVIDENCE[scoped_evidence]}
    if observed:
        # A manually supplied detectability label has no verified source
        # contract merely because its record says the peptide was observed.
        return {"excision": "bulk_observed_product", "ms_detectability": "unknown"}
    return {
        "excision": f"generated:{generated_kind}" if generated_kind else "unknown",
        "ms_detectability": "unknown",
    }
