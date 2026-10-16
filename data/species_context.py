"""Independent native species roles and molecular MHC supervision identity."""

from functools import lru_cache

from .allele_resolver import infer_species_identity, normalize_species_label
from .vocab import is_missing_scalar

# Prefix original source claims; reviewed role fields retain Hitlist's names.
NATIVE_SPECIES_COLUMNS = {
    "reported_host_species": "host",
    "reported_antigen_species": "source_organism",
    "reported_mhc_species": "mhc_species",
    "reported_mhc_species_basis": "mhc_species_source",
    "reported_mhc_species_context_disagrees": "mhc_species_context_disagrees",
    **{
        name: name
        for name in (
            "species_context_id",
            "species_context_status",
            "species_context_scope",
            "species_context_kind",
            "species_context_reference",
            "presenting_species",
            "in_vivo_host_species",
            "lineage_host_species",
            "introduced_mhc_species",
            "supported_foreign_species",
            "reviewed_candidate_species",
            "species_context_note",
        )
    },
}
SPECIES_ROLE_FIELDS = (*NATIVE_SPECIES_COLUMNS, "species_context_origin")
MOLECULAR_SPECIES_FIELDS = (
    "molecular_mhc_species",
    "molecular_mhc_species_bucket",
    "molecular_mhc_species_status",
)
SPECIES_LINEAGE_FIELDS = SPECIES_ROLE_FIELDS + MOLECULAR_SPECIES_FIELDS
MOLECULAR_SPECIES_POLICY = "explicit_allele_identity_consensus_v1"


def _text(value):
    return "" if is_missing_scalar(value) else str(value).strip()


def native_species_context(row):
    """Preserve source claims and reviewed roles without cross-role defaults."""
    return {
        **{name: _text(row.get(column)) for name, column in NATIVE_SPECIES_COLUMNS.items()},
        "species_context_origin": "native_observation",
    }


@lru_cache(maxsize=16384)
def _molecular_species(alleles):
    if not alleles:
        return "", "", "unknown"
    identities = [infer_species_identity(allele) for allele in alleles]
    if any(not identity for identity in identities):
        return "", "", "unresolved"
    if len({identity.casefold() for identity in identities}) != 1:
        return "", "", "mixed"
    identity = identities[0]
    bucket = normalize_species_label(identity)
    return identity, bucket or "", "resolved" if bucket else "unsupported"


def molecular_mhc_species(alleles):
    """Require a known, consistent molecular species for every explicit allele."""
    if isinstance(alleles, str):
        alleles = (alleles,)
    values = tuple(sorted({_text(value) for value in (alleles or ()) if _text(value)}))
    return dict(zip(MOLECULAR_SPECIES_FIELDS, _molecular_species(values)))


def sample_mhc_species(sample):
    # Resolved identities describe actual retained model inputs; original genotype
    # candidates remain separately inspectable. A legacy primary name is explicit
    # molecular evidence too, but a host/processing label never is.
    alleles = (
        getattr(sample, "resolved_mhc_alleles", ())
        or getattr(sample, "source_mhc_alleles", ())
        or (getattr(sample, "primary_allele", None),)
    )
    return molecular_mhc_species(alleles)


def reported_species_fields(source):
    context = getattr(source, "species_context", {}) or {}
    return {name: _text(context.get(name)) for name in SPECIES_ROLE_FIELDS}


def species_lineage_fields(sample):
    return {
        **reported_species_fields(sample),
        **sample_mhc_species(sample),
    }
