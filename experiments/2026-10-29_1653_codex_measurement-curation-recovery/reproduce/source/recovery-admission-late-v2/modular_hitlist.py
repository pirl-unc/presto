"""Strict raw Hitlist quantitative-assay admission against a caller-pinned human catalog.

This adapter preserves physical measurements and source identity. Partitioning,
original-outcome deduplication and experiment qualification remain explicit steps.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
import re

from .modular import AA, MHCInput, Observation
from .assay_reviews import assay_review_reason
from .measurement_curation import (
    MeasurementCurationError,
    assay_method_context,
    curated_measurement,
    screen_censoring,
)


class AdmissionError(ValueError):
    """A source disposition, distinct from an unexpected implementation failure."""

    def __init__(self, reason: str, *, deferred: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.disposition = "deferred" if deferred else "excluded"


@dataclass(frozen=True)
class MHCReferenceAssumption:
    """Explicit modeled restriction, never a replacement for reported typing."""

    restriction: str
    reason: str
    reference: str

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.restriction, self.reason, self.reference)
        ):
            raise AdmissionError("reference_assumption_requires_restriction_reason_reference")


def human_dr_reference_assumption(restriction, *, reference=None):
    """Optionally complete ordinary, allele-resolved human DR beta typing.

    The caller explicitly supplies the frozen curation-policy reference. Complete
    pairs, DP/DQ, serotypes, nonhuman names and modified construct names retain the
    strict resolver behavior. Reported alpha typing is never overwritten.
    """
    if reference is None:
        return None
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("DRA policy requires a nonempty provenance reference")
    if (
        not isinstance(restriction, str)
        or re.fullmatch(r"HLA-DRB[1345]\*[0-9]{2,3}:[0-9]{2,3}", restriction) is None
    ):
        return None
    return MHCReferenceAssumption(
        f"HLA-DRA*01:01/{restriction}",
        "Declared DRA*01:01 reference for ordinary human DR with unreported alpha typing",
        reference,
    )


def _restriction_names(restriction):
    if not isinstance(restriction, str) or not restriction.startswith("HLA-"):
        raise AdmissionError("not_explicit_human_restriction")
    return [part if part.startswith("HLA-") else "HLA-" + part for part in restriction.split("/")]


def _reference_assignment(row):
    if row.get("source") != "imgt" or row.get("species") != "Homo sapiens":
        raise AdmissionError("catalog_outside_human_imgt")
    if not row.get("source_id") or not row.get("two_field_allele"):
        raise AdmissionError("catalog_missing_identifier")
    if row.get("is_functional") != "True" or any(
        row.get(name) != "False" for name in ("is_null", "is_questionable", "is_pseudogene")
    ):
        raise AdmissionError("catalog_functional_flags")
    if row.get("groove_status") != "ok" or row.get("groove_flags"):
        raise AdmissionError("catalog_groove_flags")
    cls, chain = row.get("mhc_class"), row.get("chain")
    slots = {
        ("I", "alpha"): ("groove1", "groove2"),
        ("II", "alpha"): ("groove1",),
        ("II", "beta"): ("groove2",),
    }.get((cls, chain))
    if slots is None:
        raise AdmissionError("catalog_class_chain")
    protein = row.get("sequence")
    if not isinstance(protein, str):
        raise AdmissionError("catalog_missing_protein")
    end = 0
    for slot in ("groove1", "groove2"):
        value = row.get(slot)
        if slot not in slots:
            if value:
                raise AdmissionError("catalog_unexpected_domain")
            continue
        if not isinstance(value, str) or not 60 <= len(value) <= 130 or set(value) - set(AA):
            raise AdmissionError("catalog_noncanonical_domain")
        try:
            declared_length = int(row.get(slot + "_len", ""))
        except (TypeError, ValueError) as error:
            raise AdmissionError("catalog_domain_length") from error
        if declared_length != len(value):
            raise AdmissionError("catalog_domain_length")
        position = protein.find(value, end)
        if position < 0:
            raise AdmissionError("catalog_domain_not_in_ordered_protein")
        end = position + len(value)
    return cls, chain, row.get("groove1") or "", row.get("groove2") or ""


class HumanMHCCatalog:
    """Resolve explicit two-field names only when all eligible aliases agree."""

    def __init__(self, rows):
        self.assignments = defaultdict(lambda: defaultdict(list))
        self.dispositions = Counter()
        self.used = set()
        for ordinal, raw in enumerate(rows):
            row = dict(raw)
            try:
                assignment = _reference_assignment(row)
                name = row["two_field_allele"]
                if not name.startswith("HomoSapiens-"):
                    raise AdmissionError("catalog_noncanonical_name")
                name = name.replace("HomoSapiens-", "HLA-", 1)
            except AdmissionError as error:
                self.dispositions[error.reason] += 1
                continue
            self.assignments[name][assignment].append({"source_row": ordinal, **row})
            self.dispositions["eligible_alias"] += 1

    def resolve(
        self,
        restriction: str | None,
        mhc_class: str,
        *,
        reference_assumption: MHCReferenceAssumption | None = None,
    ) -> MHCInput:
        """Resolve reported typing, or an explicit reference completing unknown typing.

        The default requires a complete source-reported restriction. With an
        assumption, ``restriction`` remains the actual reported typing (possibly
        partial, or None for unknown); every reported allele must be retained.
        No reference is inferred from class, source, or missingness.
        """
        if reference_assumption is not None and not isinstance(
            reference_assumption, MHCReferenceAssumption
        ):
            raise AdmissionError("untyped_reference_assumption")
        names = _restriction_names(
            restriction if reference_assumption is None else reference_assumption.restriction
        )
        assumed = []
        if reference_assumption is not None:
            reported = [] if restriction is None else _restriction_names(restriction)
            if len(reported) != len(set(reported)) or not set(reported) <= set(names):
                raise AdmissionError("reference_assumption_changes_reported_typing", deferred=True)
            assumed = [name for name in names if name not in reported]
        if len(names) != (1 if mhc_class == "I" else 2):
            raise AdmissionError("missing_explicit_mhc_partner", deferred=True)
        assignments = []
        for name in names:
            values = self.assignments.get(name)
            if not values:
                raise AdmissionError("unresolved_catalog_name", deferred=True)
            if len(values) != 1:
                raise AdmissionError("ambiguous_catalog_name", deferred=True)
            assignment = next(iter(values))
            if assignment[0] != mhc_class:
                raise AdmissionError("source_catalog_class_disagreement", deferred=True)
            assignments.append(assignment)
        by_chain = {assignment[1]: assignment for assignment in assignments}
        if mhc_class == "I":
            if set(by_chain) != {"alpha"}:
                raise AdmissionError("class_i_not_heavy_chain", deferred=True)
            first, second = by_chain["alpha"][2:]
        else:
            if set(by_chain) != {"alpha", "beta"}:
                raise AdmissionError("class_ii_not_alpha_beta_pair", deferred=True)
            first, second = by_chain["alpha"][2], by_chain["beta"][3]
        pair = MHCInput(
            first,
            second,
            mhc_class,
            species1="human",
            species2="human",
            provenance={
                "reported_restriction": restriction,
                "catalog_alleles": names,
                "partner_imputation": mhc_class == "II" and bool(assumed),
                **(
                    {
                        "reference_assumption": asdict(reference_assumption),
                        "assumed_alleles": assumed,
                    }
                    if reference_assumption is not None
                    else {}
                ),
            },
        )
        pair.validate()
        self.used.update(names)
        return pair

    def aliases(self, *, used_only=True):
        names = self.used if used_only else self.assignments.keys()
        return {
            name: [row for rows in self.assignments[name].values() for row in rows]
            for name in sorted(names)
        }


PURIFIED_ASSAYS = {
    "half maximal inhibitory concentration (IC50)": ("ic50", "concentration"),
    "dissociation constant KD": ("kd", "concentration"),
    "half maximal effective concentration (EC50)": ("ec50", "concentration"),
    "dissociation constant KD (~IC50)": ("kd_via_ic50", "concentration"),
    "dissociation constant KD (~EC50)": ("kd_via_ec50", "concentration"),
    "half life": ("half_life", "min"),
    "off rate": ("koff", "1/s"),
    "on rate": ("kon", "association_rate"),
    "50% dissociation temperature": ("td50", "C"),
    "qualitative binding": ("qualitative_binding", "binary"),
}


def purified_ic50_observation(raw, catalog: HumanMHCCatalog, *, source_sha256, ordinal):
    """Historical strict IC50 contract, preserved without admitting proxies."""
    if raw.get("source") not in ("iedb", "cedar") or raw.get("response_measured") != (
        "half maximal inhibitory concentration (IC50)"
    ):
        raise AdmissionError("outside_direct_iedb_cedar_ic50")
    return purified_assay_observation(raw, catalog, source_sha256=source_sha256, ordinal=ordinal)


def purified_assay_observation(
    raw,
    catalog: HumanMHCCatalog,
    *,
    source_sha256,
    ordinal,
    dra_reference=None,
    censor_decision=None,
):
    """Preserve literal assay identity; no relabeling between measured endpoints."""
    assay = PURIFIED_ASSAYS.get(raw.get("response_measured"))
    if raw.get("source") not in ("iedb", "cedar") or assay is None:
        raise AdmissionError("outside_supported_purified_quantitative_assay")
    review_reason = assay_review_reason(raw.get("assay_iri"))
    if review_reason is not None:
        raise AdmissionError("source_review:" + review_reason, deferred=True)
    endpoint, domain = assay
    method = raw.get("assay_method")
    if not isinstance(method, str) or not (
        method == "purified MHC" or method.startswith("purified MHC/")
    ):
        raise AdmissionError("not_explicit_purified_mhc")
    if raw.get("has_ptm") is not False or raw.get("peptide_modifications"):
        raise AdmissionError("modified_or_unspecified_peptidoform")
    if raw.get("is_non_peptide_ligand"):
        raise AdmissionError("non_peptide_ligand")
    peptide, cls = raw.get("peptide"), raw.get("mhc_class")
    if (
        not isinstance(peptide, str)
        or set(peptide) - set(AA)
        or cls not in ("I", "II")
        or not (9 if cls == "II" else 1) <= len(peptide) <= 512
    ):
        raise AdmissionError("unsupported_peptide_or_class")
    unit, value = raw.get("measurement_units"), raw.get("quantitative_value")
    if domain == "concentration":
        if unit not in ("pM", "nM", "uM", "mM", "M"):
            raise AdmissionError("unsupported_concentration_unit")
    elif domain == "association_rate":
        if unit not in ("1/(M*s)", "M^-1s^-1", "nM^-1s^-1"):
            raise AdmissionError("unsupported_assay_unit")
    elif domain == "C":
        if unit not in ("C", "°C"):
            raise AdmissionError("unsupported_assay_unit")
        unit = "C"
    elif domain == "binary":
        outcome = raw.get("qualitative_measurement")
        outcome = outcome.strip().lower() if isinstance(outcome, str) else ""
        if re.fullmatch(r"positive(?:[- :].*)?", outcome):
            value = 1.0
        elif re.fullmatch(r"negative(?:[- :].*)?", outcome):
            value = 0.0
        else:
            raise AdmissionError("unestablished_qualitative_binding_outcome")
        unit = "binary"
    elif unit != domain:
        raise AdmissionError("unsupported_assay_unit")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (domain not in ("binary", "C") and value <= 0)
        or (domain == "C" and value <= -273.15)
    ):
        raise AdmissionError("nonpositive_or_nonfinite_measurement")
    if domain == "binary":
        if censor_decision is not None:
            raise AdmissionError("censor_review_on_binary_outcome", deferred=True)
        measurement = {"effective_qualifier": "=", "status": "reported_qualitative_binary"}
    else:
        try:
            measurement = curated_measurement(raw, censor_decision)
        except MeasurementCurationError as error:
            raise AdmissionError(str(error), deferred=True) from error
    qualifier = measurement["effective_qualifier"]
    assay = raw.get("assay_iri")
    if not isinstance(assay, str) or not assay.strip():
        raise AdmissionError("missing_original_assay_iri", deferred=True)
    restriction = raw.get("mhc_restriction")
    pair = catalog.resolve(
        restriction,
        cls,
        reference_assumption=human_dr_reference_assumption(
            restriction, reference=dra_reference if cls == "II" else None
        ),
    )
    result = Observation(
        f"hitlist-v4-row:{ordinal}",
        f"hitlist-v4-{raw['source']}",
        endpoint,
        peptide,
        (pair,),
        value=float(value),
        unit=unit,
        qualifier=qualifier,
        provenance={
            "measurement_namespace": "assay_iri",
            "original_measurement_id": assay,
            "source_sha256": source_sha256,
            "source_ordinal": ordinal,
            "target_policy": "reported_qualitative_binary_v1"
            if domain == "binary"
            else "physical_unclipped_v1",
            "measurement_curation": measurement,
            "assay_method_context": assay_method_context(raw),
            "missing_qualifier_convention": (
                "unspecified treated as exact only without a censor review; "
                "raw missingness retained"
            ),
            "raw_record": dict(raw),
        },
    )
    result.validate()
    return result


def curated_purified_assays(records, catalog, *, source_sha256, dra_reference=None):
    """Run common source admission with a reproducible floor/ceiling screen.

    ``records`` is a repeatable zero-argument iterator factory over one frozen,
    caller-declared curation scope. Validation/test/model predictions are never
    consulted. Only source rows passing ordinary admission enter the screen.
    Returns an observation iterator, a live disposition counter and frozen panel
    decisions. Exhaust the iterator before saving disposition totals. Observations
    are streamed so a large source does not retain full provenance rows in memory.
    """
    dispositions = Counter()

    def eligible():
        for ordinal, raw in enumerate(records()):
            try:
                purified_assay_observation(
                    raw,
                    catalog,
                    source_sha256=source_sha256,
                    ordinal=ordinal,
                    dra_reference=dra_reference,
                )
            except AdmissionError:
                continue
            yield raw

    decisions, panels = screen_censoring(eligible())

    def admitted():
        for ordinal, raw in enumerate(records()):
            try:
                row = purified_assay_observation(
                    raw,
                    catalog,
                    source_sha256=source_sha256,
                    ordinal=ordinal,
                    dra_reference=dra_reference,
                    censor_decision=decisions.get(raw.get("assay_iri")),
                )
                dispositions["admitted:" + row.provenance["measurement_curation"]["status"]] += 1
                yield row
            except AdmissionError as error:
                dispositions[error.disposition + ":" + error.reason] += 1

    return admitted(), dispositions, decisions, panels
